"""Embeddings: chunking transcripts/memories/notes and filling ``chunk_vec``.

Session transcripts are chunked per *turn* (a human prompt plus everything up to the next
prompt); memories and notes are chunked per Markdown heading section. Chunks are written
with ``embedded = 0`` and vectorised later, newest first, by ``embed_pending``.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from . import db
from .config import Config
from .models import ROLE_ASSISTANT, ROLE_SUMMARY, ROLE_TOOL_CALL, ROLE_TOOL_RESULT, ROLE_USER

TURN_CHARS = 1500
TURN_OVERLAP = 200


class Embedder(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """fastembed ``TextEmbedding``, loaded on first use (downloads into ``cache_dir``)."""

    def __init__(self, model: str, dim: int, *, cache_dir: Path, threads: int = 8, batch: int = 64) -> None:
        self.model = model
        self.dim = dim
        self.cache_dir = cache_dir
        self.threads = threads
        self.batch = batch
        self._te: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._te is None:
            with self._lock:
                if self._te is None:
                    from fastembed import TextEmbedding

                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    te = TextEmbedding(model_name=self.model, cache_dir=str(self.cache_dir), threads=self.threads)
                    if te.embedding_size != self.dim:
                        raise ValueError(f"model {self.model} has dim {te.embedding_size}, config says {self.dim}")
                    self._te = te
        return self._te

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in self._load().passage_embed(list(texts), batch_size=self.batch)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().query_embed(text))).tolist()


class HashEmbedder:
    """Deterministic hashed bag-of-words embedder (tests, local test mode); no model needed."""

    def __init__(self, dim: int = 384) -> None:
        self.model = f"hash-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v))
        if not norm:  # no words: a fixed unit vector keeps cosine distance defined
            v[0], norm = 1.0, 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def embedder_from_config(cfg: Config) -> Embedder | None:
    """The configured embedder, or None when embeddings are disabled. ``model = "hash"`` → HashEmbedder."""
    e = cfg.embedding
    if not e.enabled:
        return None
    if e.model == "hash":
        return HashEmbedder(e.dim)
    return FastEmbedder(e.model, e.dim, cache_dir=cfg.models_dir, threads=e.threads, batch=e.batch)


# --- chunking ------------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?:;][\"')\]]?\s|\n")


def _break_at(text: str, lo: int, hi: int) -> int:
    """Best end position in ``text[lo:hi]``: paragraph, then sentence/line, then word boundary."""
    window = text[lo:hi]
    i = window.rfind("\n\n")
    if i >= 0:
        return lo + i
    ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    if ends:
        return lo + ends[-1]
    i = max(window.rfind(" "), window.rfind("\t"))
    return lo + i if i > 0 else hi


def split_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Split into stripped, non-empty pieces of at most ``max_chars``, consecutive pieces sharing
    about ``overlap`` chars. Cuts prefer paragraph, then sentence, then word boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    overlap = max(0, min(overlap, max_chars // 4))
    out: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = n if n - start <= max_chars else _break_at(text, start + max_chars // 2, start + max_chars)
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        nxt = end - overlap
        ws = text.find(" ", nxt, end)  # start the overlap on a word boundary
        nxt = ws + 1 if ws >= 0 else nxt
        start = nxt if nxt > start else end
    return out


_HEADING = re.compile(r"^#{1,6}\s")
_FENCE = re.compile(r"^\s*(```|~~~)")


def chunk_markdown(body: str, max_chars: int = 1500) -> list[str]:
    """One chunk per heading section (heading kept with its text); oversize sections are split.

    Headings with no text of their own (e.g. a title right above a subheading) are merged into
    the next section. Headings inside fenced code blocks are ignored.
    """
    sections: list[list[str]] = [[]]
    fenced = False
    for line in body.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and _HEADING.match(line):
            prev = sections[-1]
            if not any(ln.strip() and not _HEADING.match(ln) for ln in prev):
                prev.append(line)  # heading-only (or empty) section: carry into this one
                continue
            sections.append([line])
            continue
        sections[-1].append(line)
    out: list[str] = []
    for sec in sections:
        out.extend(split_text("\n".join(sec), max_chars, TURN_OVERLAP))
    return out


@dataclass
class TurnChunk:
    anchor_uuid: str | None
    ts: str | None
    seq_start: int
    seq_end: int
    text: str


def _is_prompt(row: Mapping) -> bool:
    return row["role"] == ROLE_USER and bool(row["is_prompt"])


def _split_turns(rows: Iterable[Mapping]) -> list[list[Mapping]]:
    turns: list[list[Mapping]] = []
    for row in rows:
        if not turns or _is_prompt(row):
            turns.append([])
        turns[-1].append(row)
    return turns


def _turn_text(turn: Sequence[Mapping]) -> str:
    parts: list[tuple[str, list[str]]] = []
    for r in turn:
        text = (r["text"] or "").strip()
        if not text:
            continue
        if _is_prompt(r):
            label = "User"
        elif r["role"] == ROLE_ASSISTANT:
            label = "Assistant"
        elif r["role"] == ROLE_SUMMARY:
            label = "Summary"
        else:  # tool rows and non-prompt user text (injected context) are not embedded
            continue
        if parts and label == "Assistant" == parts[-1][0]:
            parts[-1][1].append(text)
        else:
            parts.append((label, [text]))
    return "\n\n".join(f"{label}: " + "\n\n".join(texts) for label, texts in parts)


def chunk_turns(rows: Iterable[Mapping], *, include_open_tail: bool) -> list[TurnChunk]:
    """Turn chunks for messages ordered by seq (keys: seq, uuid, ts, role, is_prompt, text).

    Tool rows are excluded from the text but included in the seq range. The last turn may still
    grow, so it is only chunked when ``include_open_tail``.
    """
    turns = _split_turns(rows)
    if turns and not include_open_tail:
        turns.pop()
    out: list[TurnChunk] = []
    for turn in turns:
        anchor = next((r["uuid"] for r in turn if r["uuid"]), None)
        ts = next((r["ts"] for r in turn if r["ts"]), None)
        for piece in split_text(_turn_text(turn), TURN_CHARS, TURN_OVERLAP):
            out.append(TurnChunk(anchor, ts, turn[0]["seq"], turn[-1]["seq"], piece))
    return out


# --- writing chunks --------------------------------------------------------------------------

def _delete_chunk_ids(conn: sqlite3.Connection, ids: Iterable[int]) -> None:
    rows = [(i,) for i in ids]
    conn.executemany("DELETE FROM chunk_vec WHERE chunk_id = ?", rows)
    conn.executemany("DELETE FROM chunks WHERE id = ?", rows)


def _load_messages(conn: sqlite3.Connection, session_id: str, after_seq: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT m.seq, m.uuid, m.ts, m.role, m.is_prompt, f.text FROM messages m "
        f"LEFT JOIN fts_docs f ON f.rowid = m.doc_id AND m.role NOT IN ('{ROLE_TOOL_CALL}', '{ROLE_TOOL_RESULT}') "
        "WHERE m.session_id = ? AND m.seq > ? ORDER BY m.seq",
        (session_id, after_seq),
    ).fetchall()


def _is_quiet(ended_at: str | None, quiet_before: datetime) -> bool:
    if not ended_at:
        return True
    try:
        ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    return (ended if ended.tzinfo else ended.replace(tzinfo=UTC)) < quiet_before


def _chunk_session(conn: sqlite3.Connection, s: sqlite3.Row, include_tail: bool) -> int:
    sid, done = s["id"], s["chunked_seq"]
    rows = _load_messages(conn, sid, done)
    if rows and done >= 0 and not _is_prompt(rows[0]):
        # New rows continue the turn that was chunked as a quiet tail: re-chunk that turn whole.
        old = conn.execute(
            "SELECT id, seq_start FROM chunks WHERE doc_type = 'session' AND ref_id = ? AND seq_end = ?",
            (sid, done)).fetchall()
        if old:
            _delete_chunk_ids(conn, [r["id"] for r in old])
            done = min(r["seq_start"] for r in old) - 1
            rows = _load_messages(conn, sid, done)
    if not rows:  # gap in seq numbers: nothing to do, don't look again
        conn.execute("UPDATE sessions SET chunked_seq = ? WHERE id = ?", (s["next_seq"] - 1, sid))
        return 0
    kind = db.KIND_CLAUDEAI if s["source"] == "claude.ai" else db.KIND_TRANSCRIPT
    chunks = chunk_turns(rows, include_open_tail=include_tail)
    conn.executemany(
        "INSERT INTO chunks(doc_type, ref_id, project_id, session_id, anchor_uuid, seq_start, seq_end, ts, kind,"
        " machine, text) VALUES ('session', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(sid, s["project_id"], sid, c.anchor_uuid, c.seq_start, c.seq_end, c.ts, kind, s["machine"], c.text)
         for c in chunks],
    )
    # Everything before the open turn is settled (chunked, or had no text); the open turn is
    # re-read next time. With the tail included, everything loaded is settled.
    new_done = rows[-1]["seq"] if include_tail else _split_turns(rows)[-1][0]["seq"] - 1
    conn.execute("UPDATE sessions SET chunked_seq = ? WHERE id = ?", (new_done, sid))
    return len(chunks)


def chunk_pending_sessions(conn: sqlite3.Connection, cfg: Config, *, now: datetime | None = None,
                           limit: int = 200, quiet_minutes: int = 10) -> int:
    """Chunk new transcript turns of up to ``limit`` sessions (newest first). Returns chunks created.

    A session's last turn is chunked only once the session has been quiet for ``quiet_minutes``;
    if it grows afterwards, that turn's chunks are replaced. Commits per session.
    """
    quiet_before = (now or datetime.now(UTC)) - timedelta(minutes=quiet_minutes)
    sessions = conn.execute(
        "SELECT id, project_id, machine, source, ended_at, next_seq, chunked_seq FROM sessions "
        "WHERE is_subagent = 0 AND (kind = 'interactive' OR ?) AND next_seq - 1 > chunked_seq "
        "ORDER BY ended_at DESC LIMIT ?",
        (int(cfg.embedding.embed_automated), limit),
    ).fetchall()
    total = 0
    for s in sessions:
        total += _chunk_session(conn, s, _is_quiet(s["ended_at"], quiet_before))
        conn.commit()
    return total


def add_doc_chunks(conn: sqlite3.Connection, *, doc_type: str, ref_id: str, project_id: int, kind: str,
                   ts: str | None, machine: str | None, body: str, title: str = "") -> int:
    """Replace the chunks of a memory/note (no commit). Returns the number of chunks written."""
    db.delete_chunks(conn, doc_type, ref_id)
    pieces = [f"{title}\n\n{p}" if title else p for p in chunk_markdown(body)]
    if not pieces and title:
        pieces = [title]
    conn.executemany(
        "INSERT INTO chunks(doc_type, ref_id, project_id, ts, kind, machine, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(doc_type, ref_id, project_id, ts, kind, machine, p) for p in pieces],
    )
    return len(pieces)


# --- vectors ------------------------------------------------------------------------------------

def _vec_dim(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'chunk_vec'").fetchone()
    m = re.search(r"float\[(\d+)\]", row[0]) if row else None
    return int(m.group(1)) if m else None


def ensure_model(conn: sqlite3.Connection, embedder: Embedder) -> bool:
    """Reset ``chunk_vec`` when the model or dimension changed. Returns True if it was reset."""
    model, dim = db.get_meta(conn, "embed_model"), db.get_meta(conn, "embed_dim")
    if model is None:  # first run: vectors of unknown origin can't be trusted
        stale = conn.execute("SELECT 1 FROM chunks WHERE embedded = 1 LIMIT 1").fetchone() is not None
    else:
        stale = model != embedder.model or dim != str(embedder.dim)
    stale = stale or _vec_dim(conn) != embedder.dim
    if stale:
        conn.execute("DROP TABLE IF EXISTS chunk_vec")
        conn.execute(db.vec_schema(embedder.dim))
        conn.execute("UPDATE chunks SET embedded = 0 WHERE embedded != 0")
    if stale or model is None:
        db.set_meta(conn, "embed_model", embedder.model)
        db.set_meta(conn, "embed_dim", str(embedder.dim))
        conn.commit()
    return stale


def embed_pending(conn: sqlite3.Connection, embedder: Embedder, *, batch: int = 64,
                  max_chunks: int | None = None) -> int:
    """Embed chunks with ``embedded = 0``, newest first, committing per batch. Returns the count.

    The model runs outside any transaction; a chunk deleted or rewritten meanwhile is skipped.
    """
    done = 0
    while max_chunks is None or done < max_chunks:
        n = batch if max_chunks is None else min(batch, max_chunks - done)
        rows = conn.execute(
            "SELECT id, text FROM chunks WHERE embedded = 0 ORDER BY ts DESC LIMIT ?", (n,)).fetchall()
        if not rows:
            break
        vecs = dict(zip((r["id"] for r in rows), embedder.embed_documents([r["text"] for r in rows])))
        texts = {r["id"]: r["text"] for r in rows}
        ids = list(texts)
        marks = ",".join("?" * len(ids))
        conn.executemany("DELETE FROM chunk_vec WHERE chunk_id = ?", [(i,) for i in ids])  # takes the write lock
        live = [r for r in conn.execute(
            f"SELECT id, text, project_id, kind, ts FROM chunks WHERE embedded = 0 AND id IN ({marks})", ids)
            if r["text"] == texts[r["id"]]]
        conn.executemany(
            "INSERT INTO chunk_vec(chunk_id, embedding, project_id, kind, ts) VALUES (?, ?, ?, ?, ?)",
            [(r["id"], db.pack_vector(vecs[r["id"]]), r["project_id"], r["kind"], db.iso_to_epoch(r["ts"]))
             for r in live],
        )
        conn.executemany("UPDATE chunks SET embedded = 1 WHERE id = ?", [(r["id"],) for r in live])
        conn.commit()
        done += len(live)
    return done


def backlog(conn: sqlite3.Connection, *, embed_automated: bool = False) -> dict:
    """Embedding backlog for status reporting."""
    pending, oldest = conn.execute("SELECT COUNT(*), MIN(ts) FROM chunks WHERE embedded = 0").fetchone()
    sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE is_subagent = 0 AND (kind = 'interactive' OR ?) "
        "AND next_seq - 1 > chunked_seq", (int(embed_automated),)).fetchone()[0]
    return {"pending_chunks": pending, "oldest_pending_ts": oldest, "pending_sessions": sessions}
