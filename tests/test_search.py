from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from claude_context import db
from claude_context.config import Config, EmbeddingConfig
from claude_context.embed import HashEmbedder, add_doc_chunks, chunk_pending_sessions, embed_pending, ensure_model
from claude_context.search import SearchError, SearchParams, fts_query, recency_factor, rrf_fuse, search

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
EMB = HashEmbedder(384)


@pytest.fixture
def conn(tmp_path: Path):
    c = db.connect(tmp_path / "index.db")
    db.init_schema(c)
    yield c
    c.close()


def add_session(conn, sid, title, *, project_id=1, machine="alpha", kind="interactive", source="claude-code",
                ts="2026-02-20T10:00:00Z"):
    conn.execute("INSERT INTO sessions(id, project_id, root, machine, source, kind, title, ended_at) "
                 "VALUES (?,?,?,?,?,?,?,?)", (sid, project_id, "main", machine, source, kind, title, ts))
    db.add_doc(conn, doc_type="session", kind=db.KIND_TRANSCRIPT, project_id=project_id, text=title, title=title,
               session_id=sid, ts=ts, machine=machine, automated=kind == "automated")


def add_msg(conn, sid, role, text, *, prompt=False, ts=None):
    s = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
    seq = s["next_seq"]
    ts = ts or s["ended_at"]
    kind = db.KIND_TOOL if role.startswith("tool") else db.KIND_TRANSCRIPT
    doc = db.add_doc(conn, doc_type="message", kind=kind, project_id=s["project_id"], text=text, session_id=sid,
                     ts=ts, machine=s["machine"], automated=s["kind"] == "automated")
    conn.execute("INSERT INTO messages(session_id, seq, uuid, ts, role, is_prompt, doc_id) VALUES (?,?,?,?,?,?,?)",
                 (sid, seq, f"{sid}-m{seq}", ts, role, int(prompt), doc))
    conn.execute("UPDATE sessions SET next_seq = next_seq + 1 WHERE id = ?", (sid,))
    return f"{sid}-m{seq}"


def add_memory(conn, stem, title, body, *, project_id=1, ts="2026-02-25T08:00:00Z", chunk_body=None):
    doc = db.add_doc(conn, doc_type="memory", kind=db.KIND_MEMORY, project_id=project_id, text=body, title=title,
                     ts=ts, machine="alpha")
    mid = conn.execute("INSERT INTO memories(project_id, root, stem, title, doc_id, modified_at) "
                       "VALUES (?,?,?,?,?,?)", (project_id, "main", stem, title, doc, ts)).lastrowid
    if chunk_body is not False:
        add_doc_chunks(conn, doc_type="memory", ref_id=str(mid), project_id=project_id, kind=db.KIND_MEMORY, ts=ts,
                       machine="alpha", body=chunk_body or body, title=title)
    return mid


def add_note(conn, note_id, title, body, *, ts="2026-02-26T08:00:00Z", chunk_body=None):
    doc = db.add_doc(conn, doc_type="note", kind=db.KIND_NOTE, project_id=1, text=body, title=title, ts=ts,
                     machine="beta")
    nid = conn.execute("INSERT INTO notes(project_id, root, project_key, note_id, title, doc_id) "
                       "VALUES (1, 'main', 'proj', ?, ?, ?)", (note_id, title, doc)).lastrowid
    if chunk_body is not False:
        add_doc_chunks(conn, doc_type="note", ref_id=str(nid), project_id=1, kind=db.KIND_NOTE, ts=ts,
                       machine="beta", body=chunk_body or body)
    return nid


def embed_all(conn, embed_automated=False):
    ensure_model(conn, EMB)
    chunk_pending_sessions(conn, Config(embedding=EmbeddingConfig(embed_automated=embed_automated)), now=NOW)
    embed_pending(conn, EMB)


@pytest.fixture
def corpus(conn):
    add_session(conn, "s-db", "Postgres upgrade")
    add_msg(conn, "s-db", "user", "How do I upgrade postgres to version 16?", prompt=True)
    add_msg(conn, "s-db", "assistant", "Use pg_upgrade with the link option; it keeps the sqlite-vec index too.")
    add_msg(conn, "s-db", "tool_call", "Bash: pg_upgrade --check")
    add_msg(conn, "s-db", "tool_result", "cluster check passed")
    add_session(conn, "s-garden", "Garden planning", project_id=2, machine="beta", ts="2025-06-01T09:00:00Z")
    add_msg(conn, "s-garden", "user", "Which tomatoes grow well in shade?", prompt=True)
    add_msg(conn, "s-garden", "assistant", "Cherry tomatoes tolerate partial shade.")
    add_session(conn, "s-auto", "Nightly report", kind="automated", ts="2026-02-27T03:00:00Z")
    add_msg(conn, "s-auto", "user", "Summarise the postgres logs from tonight", prompt=True)
    add_msg(conn, "s-auto", "assistant", "The logs show three slow queries.")
    add_memory(conn, "backup-policy", "Backup policy", "Postgres backups run nightly at 02:00 and are kept 30 days.")
    add_note(conn, "2026-02-26-tomatoes.md", "Seed order", "Order tomatoes and basil seeds before spring.")
    conn.commit()
    return conn


def run(conn, query, embedder=None, **kw):
    return search(conn, SearchParams(query=query, **kw), embedder, now=NOW)


# --- pure helpers -----------------------------------------------------------------------------

def test_fts_query_quotes_tokens():
    assert fts_query('foo"bar  C++ (draft') == '"foo""bar" "C++" "(draft"'
    assert fts_query("NOT") == '"NOT"'


def test_rrf_fuse_hand_computed():
    scores = rrf_fuse([["a", "b", "c", "a"], ["c", "a"]])
    assert math.isclose(scores["a"], 1 / 61 + 1 / 62)
    assert math.isclose(scores["b"], 1 / 62)
    assert math.isclose(scores["c"], 1 / 63 + 1 / 61)
    assert sorted(scores, key=scores.get, reverse=True) == ["a", "c", "b"]
    assert rrf_fuse([["x"]], k=0) == {"x": 1.0}


def test_recency_factor():
    assert recency_factor(None, NOW) == pytest.approx(0.8)
    assert recency_factor("2026-03-01T12:00:00Z", NOW) == pytest.approx(1.0)
    assert recency_factor("2025-12-01T12:00:00Z", NOW) == pytest.approx(0.9)  # 90 days: half the bonus


# --- query syntax -----------------------------------------------------------------------------

def test_fts_syntax_passthrough(corpus):
    hits = run(corpus, '"upgrade postgres"')
    assert [(h.session_id, h.role) for h in hits] == [("s-db", "user")]
    assert {h.session_id for h in run(corpus, "tomatoes OR pg_upgrade")} == {"s-db", "s-garden", None}
    assert {h.ref for h in run(corpus, "postg*")} >= {"backup-policy"}
    title_hits = run(corpus, "title:garden")
    assert [(h.doc_type, h.session_id, h.title) for h in title_hits] == [("session", "s-garden", "Garden planning")]
    assert all(h.doc_type != "memory" for h in run(corpus, "postgres NOT backups"))


@pytest.mark.parametrize("query", ['foo"bar', "C++ (draft", "NOT", "(", "-x", "a/b", "it's", "x@y.z"])
def test_fts_quoting_fallback(corpus, query):
    assert run(corpus, query) == []


def test_fallback_still_matches(corpus):
    hits = run(corpus, "sqlite-vec index")  # passthrough fails ("no such column"), quoted phrase matches
    assert [h.uuid for h in hits] == ["s-db-m1"]
    assert run(corpus, 'tomatoes "shade')[0].session_id == "s-garden"
    assert {h.ref for h in run(corpus, "AND")} == {"2026-02-26-tomatoes.md", "backup-policy"}  # plain word "and"


@pytest.mark.parametrize("kw, msg", [
    ({"query": "  "}, "empty"), ({"query": "x", "mode": "fuzzy"}, "mode"),
    ({"query": "x", "kinds": ["bogus"]}, "kinds"), ({"query": "x", "kinds": []}, "kinds"),
    ({"query": "x", "limit": 0}, "limit"), ({"query": "x", "limit": 51}, "limit"),
    ({"query": "x", "context_chars": 49}, "context_chars"), ({"query": "x", "since": "yesterday"}, "since"),
])
def test_validation(conn, kw, msg):
    with pytest.raises(SearchError, match=msg):
        search(conn, SearchParams(**kw))


# --- filters ----------------------------------------------------------------------------------

def test_filters(corpus):
    assert {h.project_id for h in run(corpus, "tomatoes")} == {1, 2}
    assert {h.project_id for h in run(corpus, "tomatoes", project_ids=[2])} == {2}
    assert run(corpus, "tomatoes", project_ids=[]) == []
    # tool output is excluded unless asked for
    assert run(corpus, "cluster") == []
    tool = run(corpus, "cluster", kinds=["tool"])
    assert [(h.kind, h.role, h.session_id) for h in tool] == [("tool", "tool_result", "s-db")]
    assert {h.kind for h in run(corpus, "tomatoes", kinds=["note"])} == {"note"}
    assert {h.machine for h in run(corpus, "tomatoes", machine="beta")} == {"beta"}
    assert {h.session_id for h in run(corpus, "tomatoes", since="2026-01-01")} == {None}  # just the note
    assert {h.session_id for h in run(corpus, "tomatoes", until="2025-12-31T23:59:59Z")} == {"s-garden"}
    # automated sessions are hidden by default
    assert {h.session_id for h in run(corpus, "postgres")} == {"s-db", None}
    assert "s-auto" in {h.session_id for h in run(corpus, "postgres", include_automated=True)}


def test_semantic_filters(corpus):
    embed_all(corpus, embed_automated=True)
    sem = lambda **kw: run(corpus, "tomatoes postgres logs", EMB, mode="semantic", **kw)  # noqa: E731
    assert {h.session_id for h in sem()} == {"s-db", "s-garden", None}  # automated session excluded
    assert "s-auto" in {h.session_id for h in sem(include_automated=True)}
    assert {h.project_id for h in sem(project_ids=[2])} == {2}
    assert {h.machine for h in sem(machine="beta")} == {"beta"}
    assert {h.kind for h in sem(kinds=["memory"])} == {"memory"}
    assert sem(kinds=["tool"]) == []
    assert {h.session_id for h in sem(since="2026-01-01T00:00:00Z")} == {"s-db", None}
    assert {h.session_id for h in sem(until="2025-12-31")} == {"s-garden"}
    assert all(h.matched == ("semantic",) for h in sem())


# --- fusion & ranking -------------------------------------------------------------------------

def test_keyword_and_semantic_fuse_into_one_hit(corpus):
    embed_all(corpus)
    hits = run(corpus, "upgrade postgres version", EMB)
    top = hits[0]
    assert (top.session_id, top.uuid, top.role, top.doc_type) == ("s-db", "s-db-m0", "user", "message")
    assert top.matched == ("keyword", "semantic")
    assert top.title == "Postgres upgrade" and top.source == "claude-code"
    assert "**upgrade**" in top.snippet and "**postgres**" in top.snippet
    assert sum(h.session_id == "s-db" and h.doc_type == "message" for h in hits) == 1
    assert top.score == pytest.approx((1 / 61 + 1 / 61) * recency_factor("2026-02-20T10:00:00Z", NOW), rel=1e-4)


def test_memory_doc_and_chunk_fuse(corpus):
    embed_all(corpus)
    mem = [h for h in run(corpus, "postgres backups nightly", EMB) if h.doc_type == "memory"]
    assert len(mem) == 1
    assert (mem[0].ref, mem[0].title, mem[0].matched) == ("backup-policy", "Backup policy", ("keyword", "semantic"))
    note = [h for h in run(corpus, "basil seeds", EMB) if h.doc_type == "note"]
    assert [(h.ref, h.title, h.matched) for h in note] == [("2026-02-26-tomatoes.md", "Seed order",
                                                           ("keyword", "semantic"))]


def test_semantic_only_hit_uses_chunk_anchor(corpus):
    embed_all(corpus)
    hits = [h for h in run(corpus, "cherry partial", EMB, mode="semantic") if h.session_id == "s-garden"]
    assert [(h.uuid, h.doc_type, h.role, h.matched) for h in hits] == [
        ("s-garden-m0", "message", None, ("semantic",))]
    assert hits[0].snippet.startswith("User: Which tomatoes")


def test_recency_breaks_equal_rrf(conn):
    # keyword-only and semantic-only items, both rank 1 in their ranking → equal RRF
    add_memory(conn, "kw-only", "Old memory", "the quokka lives on an island", ts="2025-01-01T00:00:00Z",
               chunk_body=False)
    add_note(conn, "sem-only.md", "Fresh note", "unrelated words", ts="2026-02-28T00:00:00Z", chunk_body="quokka")
    conn.commit()
    embed_all(conn)
    hits = run(conn, "quokka", EMB)
    assert [(h.ref, h.matched) for h in hits] == [("sem-only.md", ("semantic",)), ("kw-only", ("keyword",))]
    conn.execute("UPDATE docs SET ts = '2026-03-01T00:00:00Z' WHERE doc_type = 'memory'")
    assert [h.ref for h in run(conn, "quokka", EMB)] == ["kw-only", "sem-only.md"]


def test_at_most_three_hits_per_session(conn):
    add_session(conn, "s1", "Fruit chat")
    for i in range(5):
        add_msg(conn, "s1", "user", f"kiwi question {i}", prompt=True)
        add_msg(conn, "s1", "assistant", f"kiwi answer {i}")
    add_memory(conn, "m1", "Kiwi", "kiwi facts")
    add_memory(conn, "m2", "Kiwi 2", "more kiwi facts")
    conn.commit()
    hits = run(conn, "kiwi", limit=20)
    assert sum(h.session_id == "s1" for h in hits) == 3
    assert sum(h.doc_type == "memory" for h in hits) == 2
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(run(conn, "kiwi", limit=2)) == 2


def test_modes_without_embedder(corpus):
    with pytest.raises(SearchError, match="semantic"):
        run(corpus, "postgres", mode="semantic")
    hits = run(corpus, "postgres")  # hybrid degrades to keyword
    assert hits and all(h.matched == ("keyword",) for h in hits)
    hits = run(corpus, "postgres", EMB)  # embedder but nothing embedded yet
    assert hits and all(h.matched == ("keyword",) for h in hits)
    assert run(corpus, "postgres", EMB, mode="semantic") == []


def test_keyword_mode_ignores_vectors(corpus):
    embed_all(corpus)
    assert all(h.matched == ("keyword",) for h in run(corpus, "postgres", EMB, mode="keyword"))


# --- snippets ---------------------------------------------------------------------------------

def test_title_only_match_falls_back_to_text(conn):
    add_memory(conn, "roadmap", "Quarterly roadmap", "Some **bold** words about planning.", chunk_body=False)
    conn.commit()
    (hit,) = run(conn, "roadmap")
    assert hit.snippet == "Some **bold** words about planning."
    (hit,) = run(conn, "planning")
    assert hit.snippet == "Some **bold** words about **planning**."


def test_snippet_lengths(conn):
    long = " ".join(f"filler{i}" for i in range(60)) + " needle " + " ".join(f"pad{i}" for i in range(60))
    add_note(conn, "long.md", "Long", long, chunk_body=long)
    conn.commit()
    (kw,) = run(conn, "needle", context_chars=120)
    assert "**needle**" in kw.snippet and len(kw.snippet) < 400
    embed_all(conn)
    (sem,) = run(conn, "needle", EMB, mode="semantic", context_chars=120)
    assert sem.snippet.startswith("…") and sem.snippet.endswith("…") and "needle" in sem.snippet
    assert len(sem.snippet) <= 122
