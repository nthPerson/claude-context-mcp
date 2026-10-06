"""Hybrid search: FTS5 keyword ranking + sqlite-vec semantic ranking, fused with RRF.

Both rankers return up to 50 candidates. A keyword hit on a transcript message and the
semantic chunk of the turn containing it are the same *item*, as are a memory/note doc and
its chunks. Items are scored by reciprocal-rank fusion times a mild recency factor, capped
at three per session, and enriched with titles/refs in a handful of batched queries.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from . import db
from .embed import Embedder
from .models import SearchHit

MODES = ("hybrid", "keyword", "semantic")
DEFAULT_KINDS = tuple(k for k in db.KINDS if k != db.KIND_TOOL)
CANDIDATES = 50
RRF_K = 60
RECENCY_WEIGHT = 0.2
RECENCY_HALF_LIFE_DAYS = 90
MAX_PER_SESSION = 3
MAX_KNN = 4096  # sqlite-vec's limit on k

_FTS_ERRORS = ("fts5:", "unterminated string", "no such column", "unknown special query")
_HL_OPEN, _HL_CLOSE = "\x02", "\x03"  # private markers so existing "**" in text isn't mistaken for a match


@dataclass
class SearchParams:
    query: str
    mode: str = "hybrid"  # hybrid | keyword | semantic
    project_ids: list[int] | None = None
    kinds: list[str] | None = None  # subset of db.KINDS; None → all kinds except "tool"
    machine: str | None = None
    since: str | None = None  # ISO-8601 UTC bounds on ts
    until: str | None = None
    include_automated: bool = False
    limit: int = 10
    context_chars: int = 300


class SearchError(Exception):
    """Invalid search request; the message is safe to show to the caller."""


@dataclass
class _Item:
    key: Hashable
    kind: str
    doc_type: str
    project_id: int
    session_id: str | None
    ts: str | None
    machine: str | None
    matched: list[str] = field(default_factory=list)
    doc_id: int | None = None  # best keyword doc
    uuid: str | None = None
    role: str | None = None
    chunk_id: int | None = None  # best semantic chunk
    memory_id: int | None = None
    note_id: int | None = None


def fts_query(q: str) -> str:
    """Quote every whitespace-separated token so FTS5 treats the query as plain terms (implicit AND)."""
    return " ".join('"' + tok.replace('"', '""') + '"' for tok in q.split())


def rrf_fuse(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> dict[Hashable, float]:
    """Reciprocal-rank fusion: sum of 1/(k + rank) with 1-based ranks; repeats within a ranking count once."""
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        seen: set[Hashable] = set()
        for rank, key in enumerate(ranking, 1):
            if key not in seen:
                seen.add(key)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def recency_factor(ts: str | None, now: datetime) -> float:
    if not ts:
        return 1 - RECENCY_WEIGHT
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return 1 - RECENCY_WEIGHT
    age_days = max(0.0, (now - (t if t.tzinfo else t.replace(tzinfo=UTC))).total_seconds() / 86400)
    return (1 - RECENCY_WEIGHT) + RECENCY_WEIGHT * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def _bound(value: str, name: str) -> tuple[str, int]:
    """ISO-8601 → (canonical ``...T..:..:..mmmZ`` string, epoch seconds)."""
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise SearchError(f"{name} must be an ISO-8601 timestamp, e.g. 2026-01-31T00:00:00Z") from None
    dt = (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z", int(dt.timestamp())


def _marks(values: Sequence[Any]) -> str:
    return ",".join("?" * len(values))


# --- keyword side -----------------------------------------------------------------------------

def _keyword(conn: sqlite3.Connection, p: SearchParams, kinds: Sequence[str],
             since: tuple[str, int] | None, until: tuple[str, int] | None) -> tuple[list[sqlite3.Row], str]:
    """Top keyword docs and the MATCH expression that parsed."""
    where, args = [f"d.kind IN ({_marks(kinds)})"], list(kinds)
    if p.project_ids is not None:
        where.append(f"d.project_id IN ({_marks(p.project_ids)})")
        args += p.project_ids
    if p.machine:
        where.append("d.machine = ?")
        args.append(p.machine)
    if since:
        where.append("d.ts >= ?")
        args.append(since[0])
    if until:
        where.append("d.ts <= ?")
        args.append(until[0])
    if not p.include_automated:
        where.append("d.automated = 0")
    sql = ("SELECT d.id, d.doc_type, d.kind, d.project_id, d.session_id, d.ts, d.machine "
           "FROM fts_docs JOIN docs d ON d.id = fts_docs.rowid "
           f"WHERE fts_docs MATCH ? AND {' AND '.join(where)} ORDER BY bm25(fts_docs) LIMIT {CANDIDATES}")
    expr = p.query
    try:
        return conn.execute(sql, [expr, *args]).fetchall(), expr
    except sqlite3.OperationalError as e:
        if not str(e).startswith(_FTS_ERRORS):
            raise
    expr = fts_query(p.query)
    try:
        return conn.execute(sql, [expr, *args]).fetchall(), expr
    except sqlite3.OperationalError as e:
        if not str(e).startswith(_FTS_ERRORS):
            raise
        raise SearchError("could not parse the search query") from None


def _snippets(conn: sqlite3.Connection, expr: str, doc_ids: list[int], context_chars: int) -> dict[int, str]:
    if not doc_ids:
        return {}
    n_tokens = max(5, min(64, context_chars // 6))
    rows = conn.execute(
        f"SELECT rowid, snippet(fts_docs, 0, '{_HL_OPEN}', '{_HL_CLOSE}', '…', {n_tokens}), substr(text, 1, ?) "
        f"FROM fts_docs WHERE fts_docs MATCH ? AND rowid IN ({_marks(doc_ids)})",
        [context_chars, expr, *doc_ids],
    ).fetchall()
    out = {}
    for rowid, snip, head in rows:
        if _HL_OPEN in snip:
            out[rowid] = snip.replace(_HL_OPEN, "**").replace(_HL_CLOSE, "**")
        else:  # matched only in the title column
            out[rowid] = head + ("…" if len(head) == context_chars else "")
    return out


# --- semantic side ----------------------------------------------------------------------------

def _semantic(conn: sqlite3.Connection, p: SearchParams, embedder: Embedder, kinds: Sequence[str],
              since: tuple[str, int] | None, until: tuple[str, int] | None) -> list[sqlite3.Row]:
    """Top chunks by cosine distance (KNN with metadata filters, then machine/automated post-filter)."""
    where, args = ["embedding MATCH ?", "k = ?"], [db.pack_vector(embedder.embed_query(p.query)), 0]
    if p.project_ids is not None:
        where.append(f"project_id IN ({_marks(p.project_ids)})")
        args += p.project_ids
    where.append(f"kind IN ({_marks(kinds)})")
    args += kinds
    if since:
        where.append("ts >= ?")
        args.append(since[1])
    if until:
        where.append("ts <= ?")
        args.append(until[1])
    post, post_args = ["c.id IS NOT NULL"], []
    if p.machine:
        post.append("c.machine = ?")
        post_args.append(p.machine)
    if not p.include_automated:
        post.append("NOT (c.doc_type = 'session' AND coalesce(s.kind, '') = 'automated')")
    sql = (f"WITH knn AS (SELECT chunk_id, distance FROM chunk_vec WHERE {' AND '.join(where)}) "
           f"SELECT c.id, ({' AND '.join(post)}) AS ok FROM knn "
           "LEFT JOIN chunks c ON c.id = knn.chunk_id "
           "LEFT JOIN sessions s ON c.doc_type = 'session' AND s.id = c.ref_id "
           "ORDER BY knn.distance")
    k = CANDIDATES if len(post) == 1 and p.include_automated else CANDIDATES * 4
    while True:
        args[1] = k
        rows = conn.execute(sql, [*args, *post_args]).fetchall()
        ids = [r["id"] for r in rows if r["ok"]][:CANDIDATES]
        if len(ids) == CANDIDATES or len(rows) < k or k >= MAX_KNN:
            break
        k = min(k * 4, MAX_KNN)
    if not ids:
        return []
    by_id = {r["id"]: r for r in conn.execute(
        "SELECT c.id, c.doc_type, c.ref_id, c.project_id, c.session_id, c.anchor_uuid, c.ts, c.kind, c.machine, "
        "c.text, CASE WHEN c.doc_type = 'session' THEN (SELECT min(c2.id) FROM chunks c2 WHERE "
        "c2.doc_type = 'session' AND c2.ref_id = c.ref_id AND c2.seq_start = c.seq_start) END AS first_id "
        f"FROM chunks c WHERE c.id IN ({_marks(ids)})", ids)}
    return [by_id[i] for i in ids if i in by_id]


def _semantic_snippet(text: str, query: str, context_chars: int) -> str:
    """Up to ``context_chars`` of ``text``, centred on the first query-term occurrence."""
    if len(text) <= context_chars:
        return text
    terms = {t.lower() for t in re.findall(r"\w+", query) if t not in ("AND", "OR", "NOT", "NEAR")}
    m = re.search(r"\b(" + "|".join(map(re.escape, sorted(terms, key=len, reverse=True))) + ")", text,
                  re.IGNORECASE) if terms else None
    start = max(0, min(len(text) - context_chars, m.start() - context_chars // 2)) if m else 0
    end = start + context_chars
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


# --- search -----------------------------------------------------------------------------------

def _validate(p: SearchParams) -> tuple[list[str], tuple[str, int] | None, tuple[str, int] | None]:
    if not p.query or not p.query.strip():
        raise SearchError("query must not be empty")
    if p.mode not in MODES:
        raise SearchError(f"mode must be one of: {', '.join(MODES)}")
    kinds = list(DEFAULT_KINDS) if p.kinds is None else list(p.kinds)
    bad = [k for k in kinds if k not in db.KINDS]
    if bad or not kinds:
        raise SearchError(f"kinds must be a non-empty subset of: {', '.join(db.KINDS)}")
    if not 1 <= p.limit <= 50:
        raise SearchError("limit must be between 1 and 50")
    if not 50 <= p.context_chars <= 2000:
        raise SearchError("context_chars must be between 50 and 2000")
    since = _bound(p.since, "since") if p.since else None
    until = _bound(p.until, "until") if p.until else None
    return kinds, since, until


def _semantic_ready(conn: sqlite3.Connection, embedder: Embedder) -> bool:
    return (db.get_meta(conn, "embed_model") == embedder.model
            and db.get_meta(conn, "embed_dim") == str(embedder.dim)
            and conn.execute("SELECT 1 FROM chunks WHERE embedded = 1 LIMIT 1").fetchone() is not None)


def search(conn: sqlite3.Connection, params: SearchParams, embedder: Embedder | None = None, *,
           now: datetime | None = None) -> list[SearchHit]:
    """Run a search; see the module docstring for ranking. Raises SearchError on bad input."""
    p = params
    kinds, since, until = _validate(p)
    if p.mode == "semantic" and embedder is None:
        raise SearchError("semantic search is not available (embeddings are disabled)")
    if p.project_ids is not None and not p.project_ids:
        return []
    now = now or datetime.now(UTC)
    items: dict[Hashable, _Item] = {}
    rankings: list[list[Hashable]] = []
    expr = p.query

    if p.mode in ("hybrid", "keyword"):
        rows, expr = _keyword(conn, p, kinds, since, until)
        keys = _keyword_keys(conn, rows)
        ranking = []
        for r in rows:
            key, extra = keys[r["id"]]
            ranking.append(key)
            if key not in items:
                items[key] = _Item(key, r["kind"], r["doc_type"], r["project_id"], r["session_id"], r["ts"],
                                   r["machine"], ["keyword"], doc_id=r["id"], **extra)
        rankings.append(ranking)

    chunk_text: dict[int, str] = {}
    if p.mode in ("hybrid", "semantic") and embedder is not None and _semantic_ready(conn, embedder):
        ranking = []
        for c in _semantic(conn, p, embedder, kinds, since, until):
            if c["doc_type"] == "session":
                key: Hashable = ("chunk", c["first_id"])
            else:
                key = (c["doc_type"], int(c["ref_id"]))
            ranking.append(key)
            item = items.get(key)
            if item is None:
                item = items[key] = _Item(
                    key, c["kind"], "message" if c["doc_type"] == "session" else c["doc_type"], c["project_id"],
                    c["session_id"], c["ts"], c["machine"], uuid=c["anchor_uuid"])
                if c["doc_type"] == "memory":
                    item.memory_id = int(c["ref_id"])
                elif c["doc_type"] == "note":
                    item.note_id = int(c["ref_id"])
            if "semantic" not in item.matched:
                item.matched.append("semantic")
                item.chunk_id = c["id"]
                chunk_text[c["id"]] = c["text"]
        rankings.append(ranking)

    scores = rrf_fuse(rankings)
    order = {key: i for i, key in enumerate(scores)}
    ranked = sorted(scores, key=lambda key: (-scores[key] * recency_factor(items[key].ts, now), order[key]))
    per_session: dict[str, int] = {}
    final: list[_Item] = []
    for key in ranked:
        item = items[key]
        if item.session_id and item.doc_type not in ("memory", "note"):
            if per_session.get(item.session_id, 0) >= MAX_PER_SESSION:
                continue
            per_session[item.session_id] = per_session.get(item.session_id, 0) + 1
        final.append(item)
        if len(final) == p.limit:
            break
    return _build_hits(conn, final, scores, now, expr, chunk_text, p)


def _keyword_keys(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> dict[int, tuple[Hashable, dict]]:
    """Fusion key (and message/memory/note details) for each keyword doc."""
    out: dict[int, tuple[Hashable, dict]] = {r["id"]: (("doc", r["id"]), {}) for r in rows}
    by_type: dict[str, list[int]] = {}
    for r in rows:
        by_type.setdefault(r["doc_type"], []).append(r["id"])
    if ids := by_type.get("message"):
        for m in conn.execute(
                "SELECT m.doc_id, m.uuid, m.role, (SELECT min(c.id) FROM chunks c WHERE c.doc_type = 'session' "
                "AND c.ref_id = m.session_id AND c.seq_start <= m.seq AND c.seq_end >= m.seq) AS chunk_id "
                "FROM docs d JOIN messages m ON m.session_id = d.session_id AND m.doc_id = d.id "
                f"WHERE d.id IN ({_marks(ids)})", ids):
            key = ("chunk", m["chunk_id"]) if m["chunk_id"] is not None else ("doc", m["doc_id"])
            out[m["doc_id"]] = (key, {"uuid": m["uuid"], "role": m["role"]})
    for table, field_ in (("memories", "memory_id"), ("notes", "note_id")):
        if ids := by_type.get(field_.removesuffix("_id")):
            for r in conn.execute(f"SELECT id, doc_id FROM {table} WHERE doc_id IN ({_marks(ids)})", ids):
                out[r["doc_id"]] = ((field_.removesuffix("_id"), r["id"]), {field_: r["id"]})
    return out


def _build_hits(conn: sqlite3.Connection, final: list[_Item], scores: dict[Hashable, float], now: datetime,
                expr: str, chunk_text: dict[int, str], p: SearchParams) -> list[SearchHit]:
    kw_snips = _snippets(conn, expr, [i.doc_id for i in final if i.doc_id is not None], p.context_chars)
    sids = sorted({i.session_id for i in final if i.session_id})
    sessions = {r["id"]: r for r in conn.execute(
        f"SELECT id, title, source FROM sessions WHERE id IN ({_marks(sids)})", sids)} if sids else {}
    mem_ids = [i.memory_id for i in final if i.memory_id is not None]
    memories = {r["id"]: r for r in conn.execute(
        "SELECT m.id, m.stem, coalesce(nullif(m.title, ''), nullif(m.name, ''), nullif(f.title, ''), m.stem) "
        f"AS title FROM memories m LEFT JOIN fts_docs f ON f.rowid = m.doc_id WHERE m.id IN ({_marks(mem_ids)})",
        mem_ids)} if mem_ids else {}
    note_ids = [i.note_id for i in final if i.note_id is not None]
    notes = {r["id"]: r for r in conn.execute(
        f"SELECT id, note_id, coalesce(nullif(title, ''), note_id) AS title FROM notes WHERE id IN "
        f"({_marks(note_ids)})", note_ids)} if note_ids else {}

    hits = []
    for item in final:
        title, ref, source = "", None, None
        if item.memory_id is not None and item.memory_id in memories:
            ref, title = memories[item.memory_id]["stem"], memories[item.memory_id]["title"]
        elif item.note_id is not None and item.note_id in notes:
            ref, title = notes[item.note_id]["note_id"], notes[item.note_id]["title"]
        elif item.session_id in sessions:
            title, source = sessions[item.session_id]["title"] or "", sessions[item.session_id]["source"]
        snippet = kw_snips.get(item.doc_id) if item.doc_id is not None else None
        if snippet is None and item.chunk_id is not None:
            snippet = _semantic_snippet(chunk_text[item.chunk_id], p.query, p.context_chars)
        hits.append(SearchHit(
            kind=item.kind, doc_type=item.doc_type, project_id=item.project_id,
            score=round(scores[item.key] * recency_factor(item.ts, now), 6), snippet=snippet or "", title=title,
            session_id=item.session_id, uuid=item.uuid, ref=ref, ts=item.ts, machine=item.machine,
            source=source, role=item.role, matched=tuple(item.matched)))
    return hits
