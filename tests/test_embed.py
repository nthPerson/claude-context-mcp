from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from claude_context import db
from claude_context.config import Config, EmbeddingConfig
from claude_context.embed import (
    FastEmbedder,
    HashEmbedder,
    add_doc_chunks,
    backlog,
    chunk_markdown,
    chunk_pending_sessions,
    chunk_turns,
    embed_pending,
    embedder_from_config,
    ensure_model,
    split_text,
)

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
ACTIVE = (NOW - timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
QUIET = (NOW - timedelta(hours=2)).isoformat().replace("+00:00", "Z")


@pytest.fixture
def conn(tmp_path: Path):
    c = db.connect(tmp_path / "index.db")
    db.init_schema(c)
    yield c
    c.close()


def add_session(conn, sid: str, *, ended_at: str = QUIET, kind: str = "interactive", source: str = "claude-code",
                is_subagent: int = 0, project_id: int = 1, machine: str = "alpha") -> None:
    conn.execute(
        "INSERT INTO sessions(id, project_id, root, machine, source, kind, is_subagent, ended_at) "
        "VALUES (?,?,?,?,?,?,?,?)", (sid, project_id, "main", machine, source, kind, is_subagent, ended_at))


def add_msg(conn, sid: str, role: str, text: str, *, prompt: bool = False, ts: str | None = None) -> int:
    seq = conn.execute("SELECT next_seq FROM sessions WHERE id = ?", (sid,)).fetchone()[0]
    kind = db.KIND_TOOL if role.startswith("tool") else db.KIND_TRANSCRIPT
    ts = ts or f"2026-02-01T10:{seq // 60:02d}:{seq % 60:02d}.000Z"
    doc = db.add_doc(conn, doc_type="message", kind=kind, project_id=1, text=text, session_id=sid, ts=ts)
    conn.execute("INSERT INTO messages(session_id, seq, uuid, ts, role, is_prompt, doc_id) VALUES (?,?,?,?,?,?,?)",
                 (sid, seq, f"{sid}-u{seq}", ts, role, int(prompt), doc))
    conn.execute("UPDATE sessions SET next_seq = next_seq + 1 WHERE id = ?", (sid,))
    return seq


def turn(conn, sid: str, prompt: str, *replies: str, tools: int = 0) -> None:
    add_msg(conn, sid, "user", prompt, prompt=True)
    for _ in range(tools):
        add_msg(conn, sid, "tool_call", "Bash: ls -la")
        add_msg(conn, sid, "tool_result", "total 0 secretfile")
    for r in replies:
        add_msg(conn, sid, "assistant", r)


def session_chunks(conn, sid: str) -> list[tuple[int, int, str]]:
    return [tuple(r) for r in conn.execute(
        "SELECT seq_start, seq_end, text FROM chunks WHERE ref_id = ? ORDER BY seq_start, id", (sid,))]


def row(seq, role, text, *, prompt=False, uuid=None, ts=None) -> dict:
    return {"seq": seq, "uuid": uuid or f"u{seq}", "ts": ts, "role": role, "is_prompt": prompt, "text": text}


# --- split_text / chunk_markdown --------------------------------------------------------------

def test_split_text_short_and_empty():
    assert split_text("  hello world  ") == ["hello world"]
    assert split_text("   \n\n ") == []


def test_split_text_prefers_paragraphs_and_overlaps():
    paras = [f"Paragraph {i} " + " ".join(f"word{i}x{j}." for j in range(60)) for i in range(6)]
    text = "\n\n".join(paras)
    pieces = split_text(text, 1500, 200)
    assert len(pieces) > 1
    assert all(0 < len(p) <= 1500 for p in pieces)
    assert all(p.endswith("x59.") for p in pieces)  # every cut lands on a paragraph end
    for a, b in zip(pieces, pieces[1:]):
        assert a[-60:].split()[-1] in b[:400]  # consecutive pieces overlap
    joined = " ".join(pieces)
    assert all(w in joined for w in text.split())  # nothing lost


def test_split_text_sentence_and_hard_cuts():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    pieces = split_text(text, 500, 100)
    assert all(p.endswith(".") for p in pieces[:-1])
    blob = "x" * 3200
    hard = split_text(blob, 1000, 100)
    assert all(0 < len(p) <= 1000 for p in hard) and len(hard) >= 4


def test_chunk_markdown_sections():
    body = ("# Title\n## Setup\nInstall the thing.\n\n## Usage\nRun it.\n```\n# not a heading\n```\n"
            "## Empty\n\n## Last\nDone.")
    chunks = chunk_markdown(body)
    assert chunks == [
        "# Title\n## Setup\nInstall the thing.",
        "## Usage\nRun it.\n```\n# not a heading\n```",
        "## Empty\n\n## Last\nDone.",
    ]


def test_chunk_markdown_preamble_and_oversize():
    body = "Intro line.\n\n# Big\n" + "\n\n".join("Lorem ipsum dolor sit amet. " * 8 for _ in range(10))
    chunks = chunk_markdown(body, max_chars=500)
    assert chunks[0] == "Intro line."
    assert chunks[1].startswith("# Big\n")
    assert len(chunks) > 3 and all(len(c) <= 500 for c in chunks)


# --- chunk_turns ------------------------------------------------------------------------------

def test_chunk_turns_format_tools_and_tail():
    rows = [
        row(0, "assistant", "Leading reply."),
        row(1, "user", "First question?", prompt=True, ts="2026-01-01T00:00:01Z"),
        row(2, "assistant", "Let me check."),
        row(3, "tool_call", "Bash: rm -rf build"),
        row(4, "tool_result", "removed"),
        row(5, "user", "<injected context>"),  # not a prompt: excluded
        row(6, "assistant", "Done."),
        row(7, "summary", "Conversation compacted."),
        row(8, "user", "Only tools here", prompt=True),
        row(9, "tool_call", "Read: file"),
        row(10, "user", "Open question", prompt=True),
        row(11, "assistant", "Working…"),
    ]
    chunks = chunk_turns(rows, include_open_tail=False)
    assert [(c.seq_start, c.seq_end, c.anchor_uuid) for c in chunks] == [(0, 0, "u0"), (1, 7, "u1"), (8, 9, "u8")]
    assert chunks[0].text == "Assistant: Leading reply."
    assert chunks[1].text == ("User: First question?\n\nAssistant: Let me check.\n\nDone.\n\n"
                              "Summary: Conversation compacted.")
    assert chunks[1].ts == "2026-01-01T00:00:01Z"
    assert "rm -rf" not in chunks[1].text and "injected" not in chunks[1].text
    assert chunks[2].text == "User: Only tools here"
    with_tail = chunk_turns(rows, include_open_tail=True)
    assert (with_tail[-1].seq_start, with_tail[-1].seq_end) == (10, 11)


def test_chunk_turns_no_text_and_split():
    assert chunk_turns([row(0, "tool_call", "x"), row(1, "tool_result", "y")], include_open_tail=True) == []
    long = " ".join(f"Statement {i} holds." for i in range(300))
    rows = [row(4, "user", "Explain", prompt=True), row(5, "assistant", long), row(6, "tool_call", "z")]
    chunks = chunk_turns(rows, include_open_tail=True)
    assert len(chunks) > 1
    assert {(c.seq_start, c.seq_end, c.anchor_uuid) for c in chunks} == {(4, 6, "u4")}
    assert all(len(c.text) <= 1500 for c in chunks)


# --- chunk_pending_sessions -------------------------------------------------------------------

def test_incremental_session_chunking(conn):
    cfg = Config()
    add_session(conn, "s1", ended_at=ACTIVE)
    turn(conn, "s1", "alpha prompt", "alpha reply", tools=1)  # seq 0-3
    turn(conn, "s1", "beta prompt", "beta reply")  # seq 4-5 (open)
    conn.commit()

    assert chunk_pending_sessions(conn, cfg, now=NOW) == 1
    assert session_chunks(conn, "s1") == [(0, 3, "User: alpha prompt\n\nAssistant: alpha reply")]
    assert conn.execute("SELECT chunked_seq FROM sessions").fetchone()[0] == 3
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 0  # open tail re-examined, nothing new

    turn(conn, "s1", "gamma prompt", "gamma reply")  # seq 6-7 closes beta
    conn.commit()
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 1
    assert [c[:2] for c in session_chunks(conn, "s1")] == [(0, 3), (4, 5)]
    assert conn.execute("SELECT chunked_seq FROM sessions").fetchone()[0] == 5

    # the session goes quiet: the tail is chunked
    conn.execute("UPDATE sessions SET ended_at = ?", (QUIET,))
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 1
    assert [c[:2] for c in session_chunks(conn, "s1")] == [(0, 3), (4, 5), (6, 7)]
    assert conn.execute("SELECT chunked_seq FROM sessions").fetchone()[0] == 7
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 0

    # the quiet tail turn continues (no new prompt): it is re-chunked whole, not duplicated
    add_msg(conn, "s1", "tool_call", "Bash: make")  # 8
    add_msg(conn, "s1", "assistant", "gamma follow-up")  # 9
    conn.commit()
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 1
    chunks = session_chunks(conn, "s1")
    assert [c[:2] for c in chunks] == [(0, 3), (4, 5), (6, 9)]
    assert chunks[-1][2] == "User: gamma prompt\n\nAssistant: gamma reply\n\ngamma follow-up"

    # ...and a new prompt starts a fresh turn
    turn(conn, "s1", "delta prompt", "delta reply")
    conn.commit()
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 1
    assert [c[:2] for c in session_chunks(conn, "s1")] == [(0, 3), (4, 5), (6, 9), (10, 11)]


def test_reopened_tail_while_active_waits_for_quiet(conn):
    cfg = Config()
    add_session(conn, "s1")
    turn(conn, "s1", "first", "one")
    conn.commit()
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 1
    conn.execute("UPDATE sessions SET ended_at = ?", (ACTIVE,))
    add_msg(conn, "s1", "assistant", "two")
    conn.commit()
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 0  # old tail chunk dropped, turn open again
    assert session_chunks(conn, "s1") == []
    assert conn.execute("SELECT chunked_seq FROM sessions").fetchone()[0] == -1
    conn.execute("UPDATE sessions SET ended_at = ?", (QUIET,))
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 1
    assert session_chunks(conn, "s1") == [(0, 2, "User: first\n\nAssistant: one\n\ntwo")]


def test_session_selection_and_kind(conn):
    add_session(conn, "web", source="claude.ai")
    add_session(conn, "auto", kind="automated")
    add_session(conn, "sub", is_subagent=1)
    for sid in ("web", "auto", "sub"):
        turn(conn, sid, f"{sid} prompt", "reply")
    conn.commit()
    assert backlog(conn) == {"pending_chunks": 0, "oldest_pending_ts": None, "pending_sessions": 1}
    assert chunk_pending_sessions(conn, Config(), now=NOW) == 1
    assert [tuple(r) for r in conn.execute("SELECT ref_id, kind, machine, project_id FROM chunks")] == [
        ("web", "claudeai", "alpha", 1)]
    cfg = Config(embedding=EmbeddingConfig(embed_automated=True))
    assert chunk_pending_sessions(conn, cfg, now=NOW) == 1
    assert {r[0] for r in conn.execute("SELECT ref_id FROM chunks")} == {"web", "auto"}
    b = backlog(conn, embed_automated=True)
    assert b["pending_chunks"] == 2 and b["pending_sessions"] == 0 and b["oldest_pending_ts"]


def test_add_doc_chunks_replaces(conn):
    n = add_doc_chunks(conn, doc_type="memory", ref_id="7", project_id=1, kind="memory", ts=None, machine="alpha",
                       body="## A\nfirst\n## B\nsecond", title="Deploy notes")
    assert n == 2
    assert [r[0] for r in conn.execute("SELECT text FROM chunks ORDER BY id")] == [
        "Deploy notes\n\n## A\nfirst", "Deploy notes\n\n## B\nsecond"]
    embed_pending(conn, HashEmbedder())
    add_doc_chunks(conn, doc_type="memory", ref_id="7", project_id=1, kind="memory", ts=None, machine="alpha",
                   body="just one", title="")
    assert [r[0] for r in conn.execute("SELECT text FROM chunks")] == ["just one"]
    assert conn.execute("SELECT count(*) FROM chunk_vec").fetchone()[0] == 0


# --- vectors ----------------------------------------------------------------------------------

def test_hash_embedder():
    e = HashEmbedder(64)
    a, b, c = e.embed_documents(["the docker volume backup", "backup of a docker volume", "pasta recipe"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))  # noqa: E731
    assert math.isclose(dot(a, a), 1.0, rel_tol=1e-6)
    assert dot(a, b) > 0.5 > dot(a, c)
    assert e.embed_query("the docker volume backup") == a
    assert math.isclose(dot(*e.embed_documents(["", ""])), 1.0)


def test_embedder_from_config(tmp_path):
    assert embedder_from_config(Config(embedding=EmbeddingConfig(enabled=False))) is None
    assert isinstance(embedder_from_config(Config(embedding=EmbeddingConfig(model="hash", dim=32))), HashEmbedder)
    e = embedder_from_config(Config(data_dir=tmp_path))
    assert isinstance(e, FastEmbedder) and e.cache_dir == tmp_path / "models" and e._te is None  # lazy


def test_ensure_model_reset(conn):
    e384 = HashEmbedder(384)
    assert ensure_model(conn, e384) is False  # fresh db: just records the model
    assert db.get_meta(conn, "embed_model") == "hash-384"
    add_doc_chunks(conn, doc_type="note", ref_id="1", project_id=1, kind="note", ts=None, machine=None, body="x y")
    assert embed_pending(conn, e384) == 1
    assert ensure_model(conn, e384) is False
    e32 = HashEmbedder(32)
    assert ensure_model(conn, e32) is True
    assert conn.execute("SELECT embedded FROM chunks").fetchone()[0] == 0
    assert "float[32]" in conn.execute("SELECT sql FROM sqlite_master WHERE name = 'chunk_vec'").fetchone()[0]
    assert db.get_meta(conn, "embed_dim") == "32"
    assert embed_pending(conn, e32) == 1


def test_ensure_model_unknown_vectors(conn):
    add_doc_chunks(conn, doc_type="note", ref_id="1", project_id=1, kind="note", ts=None, machine=None, body="x")
    conn.execute("UPDATE chunks SET embedded = 1")
    assert ensure_model(conn, HashEmbedder(384)) is True  # no meta, vectors of unknown origin


def test_embed_pending_newest_first_and_idempotent(conn):
    e = HashEmbedder()
    ensure_model(conn, e)
    for i, ts in enumerate(["2026-01-03T00:00:00Z", None, "2026-01-05T00:00:00Z", "2026-01-04T00:00:00Z"]):
        add_doc_chunks(conn, doc_type="note", ref_id=str(i), project_id=1, kind="note", ts=ts, machine=None,
                       body=f"note {i}")
    conn.commit()
    assert embed_pending(conn, e, batch=2, max_chunks=2) == 2
    done = [r[0] for r in conn.execute("SELECT ref_id FROM chunks WHERE embedded = 1 ORDER BY ref_id")]
    assert done == ["2", "3"]
    assert backlog(conn)["oldest_pending_ts"] == "2026-01-03T00:00:00Z"
    assert embed_pending(conn, e, batch=1) == 2  # the NULL-ts chunk comes last
    assert embed_pending(conn, e) == 0
    conn.execute("UPDATE chunks SET embedded = 0")
    conn.commit()
    assert embed_pending(conn, e) == 4  # vectors replaced, no duplicate-key error
    assert conn.execute("SELECT count(*) FROM chunk_vec").fetchone()[0] == 4
    row_ = conn.execute("SELECT project_id, kind, ts FROM chunk_vec WHERE chunk_id = "
                        "(SELECT id FROM chunks WHERE ref_id = '2')").fetchone()
    assert tuple(row_) == (1, "note", db.iso_to_epoch("2026-01-05T00:00:00Z"))


@pytest.mark.skipif(not os.environ.get("CLAUDE_CONTEXT_REAL_MODEL"), reason="set CLAUDE_CONTEXT_REAL_MODEL=1")
def test_real_model(tmp_path, capsys):
    e = FastEmbedder("BAAI/bge-small-en-v1.5", 384, cache_dir=tmp_path / "models", threads=8)
    t0 = time.perf_counter()
    q = e.embed_query("how do I back up a docker volume")
    load = time.perf_counter() - t0
    docs = [f"Turn {i}: we discussed backing up docker volumes with tar and restoring them on another host. "
            "The assistant suggested a cron job and checksum verification. " * 3 for i in range(256)]
    t0 = time.perf_counter()
    vecs = e.embed_documents(docs)
    rate = len(docs) / (time.perf_counter() - t0)
    assert len(q) == 384 and len(vecs) == 256 and len(vecs[0]) == 384
    a, b = e.embed_documents(["restore a docker volume from a tar backup", "a recipe for tomato soup"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))  # noqa: E731
    assert dot(q, a) > dot(q, b)
    with capsys.disabled():
        print(f"\nreal model: first query (load+download) {load:.1f}s, {rate:.0f} docs/s")
