"""index.db: schema, connections and low-level write helpers.

The indexer is the only writer (WAL mode); the server opens the database read-only.

Text storage: every searchable unit (a transcript message, a memory, a note) is one row in
``docs`` (metadata used for filtering) plus one row with the same rowid in the FTS5 table
``fts_docs`` (which also *stores* the redacted text, so there is no second copy).
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = 1

# docs.kind values (also the `kinds` filter of the search tool)
KIND_TRANSCRIPT = "transcript"  # user / assistant / summary text of Claude Code sessions
KIND_TOOL = "tool"  # tool calls and tool results
KIND_MEMORY = "memory"
KIND_NOTE = "note"
KIND_CLAUDEAI = "claudeai"  # claude.ai conversations and claude.ai memory
KINDS = (KIND_TRANSCRIPT, KIND_TOOL, KIND_MEMORY, KIND_NOTE, KIND_CLAUDEAI)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS projects(
    id INTEGER PRIMARY KEY,
    alias TEXT NOT NULL UNIQUE,
    display TEXT NOT NULL,
    created_by TEXT NOT NULL              -- config | auto | hub
);
CREATE TABLE IF NOT EXISTS project_keys(
    key TEXT PRIMARY KEY,                 -- folder name under a root, e.g. "-home-user-proj"
    project_id INTEGER NOT NULL REFERENCES projects(id),
    is_primary INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY,
    root TEXT NOT NULL,                   -- root label
    rel_path TEXT NOT NULL,               -- path relative to the root, forward slashes
    kind TEXT NOT NULL,                   -- transcript | subagent | memory | note | claudeai
    size INTEGER NOT NULL DEFAULT 0,
    mtime_ns INTEGER NOT NULL DEFAULT 0,
    byte_offset INTEGER NOT NULL DEFAULT 0,   -- transcripts: bytes ingested so far
    head_sha TEXT,                        -- sha256 of the first line; detects rewrites
    last_ingest_at TEXT,
    missing_since TEXT,                   -- set when the source file disappeared
    machine TEXT,
    UNIQUE(root, rel_path)
);

CREATE TABLE IF NOT EXISTS sessions(
    id TEXT PRIMARY KEY,                  -- session uuid; subagents: "<parent>:<agent-id>"
    project_id INTEGER NOT NULL REFERENCES projects(id),
    root TEXT NOT NULL,
    machine TEXT NOT NULL DEFAULT 'unknown',
    source TEXT NOT NULL,                 -- claude-code | claude.ai | recovered
    entrypoint TEXT,
    kind TEXT NOT NULL DEFAULT 'interactive',   -- interactive | automated
    is_subagent INTEGER NOT NULL DEFAULT 0,
    parent_session_id TEXT,
    agent_type TEXT,
    agent_description TEXT,
    title TEXT,
    summary TEXT,
    first_prompt TEXT,
    last_prompt TEXT,
    final_reply_excerpt TEXT,
    cwd TEXT,
    git_branch TEXT,
    cc_version TEXT,
    started_at TEXT,                      -- ISO-8601 UTC
    ended_at TEXT,
    n_user INTEGER NOT NULL DEFAULT 0,
    n_assistant INTEGER NOT NULL DEFAULT 0,
    n_tool_calls INTEGER NOT NULL DEFAULT 0,
    next_seq INTEGER NOT NULL DEFAULT 0,
    chunked_seq INTEGER NOT NULL DEFAULT -1,  -- messages up to this seq are chunked for embedding
    file_id INTEGER REFERENCES files(id),
    updated_at TEXT                       -- source `updated_at` (claude.ai upserts)
);
CREATE INDEX IF NOT EXISTS sessions_project_ended ON sessions(project_id, ended_at);
CREATE INDEX IF NOT EXISTS sessions_ended ON sessions(ended_at);
CREATE INDEX IF NOT EXISTS sessions_parent ON sessions(parent_session_id);

CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    seq INTEGER NOT NULL,
    uuid TEXT,
    ts TEXT,
    role TEXT NOT NULL,                   -- user | assistant | tool_call | tool_result | summary
    tool_name TEXT,
    tool_use_id TEXT,
    is_error INTEGER NOT NULL DEFAULT 0,
    is_prompt INTEGER NOT NULL DEFAULT 0,
    text_len INTEGER NOT NULL DEFAULT 0,  -- full length before index truncation
    doc_id INTEGER NOT NULL,              -- rowid in docs / fts_docs (holds the text)
    file_id INTEGER,                      -- raw source for read_message
    line_offset INTEGER,
    block_index INTEGER NOT NULL DEFAULT 0,
    UNIQUE(session_id, seq)
);
CREATE INDEX IF NOT EXISTS messages_uuid ON messages(uuid);
CREATE INDEX IF NOT EXISTS messages_tool_use ON messages(tool_use_id);
CREATE INDEX IF NOT EXISTS messages_doc ON messages(doc_id);

CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    root TEXT NOT NULL,
    project_key TEXT,                     -- folder the file lives in (NULL for claude.ai memory)
    file_id INTEGER UNIQUE REFERENCES files(id),
    stem TEXT NOT NULL,                   -- filename without .md: the memory's identity
    name TEXT,                            -- frontmatter name
    title TEXT,                           -- link text from MEMORY.md
    description TEXT,
    type TEXT,                            -- user | feedback | project | reference | claudeai-memory
    sha256 TEXT,
    modified_at TEXT,
    modified_by TEXT,                     -- machine
    archived INTEGER NOT NULL DEFAULT 0,
    is_index INTEGER NOT NULL DEFAULT 0,  -- 1 for MEMORY.md itself
    doc_id INTEGER                        -- NULL for archived memories (not searchable)
);
CREATE INDEX IF NOT EXISTS memories_project ON memories(project_id, modified_at);
CREATE INDEX IF NOT EXISTS memories_doc ON memories(doc_id);

CREATE TABLE IF NOT EXISTS notes(
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    root TEXT NOT NULL,
    project_key TEXT NOT NULL,
    file_id INTEGER UNIQUE REFERENCES files(id),
    note_id TEXT NOT NULL,                -- filename
    title TEXT,
    surface TEXT,
    created_at TEXT,
    related_sessions TEXT,                -- JSON list
    machine TEXT,
    doc_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_project ON notes(project_id, created_at);
CREATE INDEX IF NOT EXISTS notes_doc ON notes(doc_id);

CREATE TABLE IF NOT EXISTS docs(
    id INTEGER PRIMARY KEY,
    doc_type TEXT NOT NULL,               -- message | memory | note
    kind TEXT NOT NULL,                   -- see KINDS
    project_id INTEGER NOT NULL,
    session_id TEXT,
    ts TEXT,
    machine TEXT,
    automated INTEGER NOT NULL DEFAULT 0,
    file_id INTEGER
);
CREATE INDEX IF NOT EXISTS docs_session ON docs(session_id);
CREATE INDEX IF NOT EXISTS docs_file ON docs(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_docs USING fts5(
    text, title, tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    doc_type TEXT NOT NULL,               -- session | memory | note
    ref_id TEXT NOT NULL,                 -- session id, or memories.id / notes.id as text
    project_id INTEGER NOT NULL,
    session_id TEXT,
    anchor_uuid TEXT,                     -- message uuid the chunk starts at
    seq_start INTEGER,                    -- session chunks: message seq range covered
    seq_end INTEGER,
    ts TEXT,
    kind TEXT NOT NULL,
    machine TEXT,
    text TEXT NOT NULL,
    embedded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS chunks_ref ON chunks(doc_type, ref_id);
CREATE INDEX IF NOT EXISTS chunks_pending ON chunks(embedded, ts);

CREATE TABLE IF NOT EXISTS conflicts(
    root TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    project_id INTEGER,
    seen_at TEXT NOT NULL,
    PRIMARY KEY(root, rel_path)
);

CREATE TABLE IF NOT EXISTS drift(          -- format drift made visible in hub_status
    category TEXT NOT NULL,               -- record_type | entrypoint
    value TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    last_seen TEXT,
    PRIMARY KEY(category, value)
);

CREATE TABLE IF NOT EXISTS imports(
    name TEXT PRIMARY KEY,                -- processed zip file name
    ingested_at TEXT NOT NULL,
    conversations INTEGER NOT NULL DEFAULT 0,
    updated INTEGER NOT NULL DEFAULT 0,
    result TEXT
);
"""


def vec_schema(dim: int) -> str:
    return (
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding float[{dim}] distance_metric=cosine, "
        "project_id INTEGER, kind TEXT, ts INTEGER)"
    )


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def iso_to_epoch(ts: str | None) -> int:
    """ISO-8601 → epoch seconds (0 when missing/unparseable)."""
    if not ts:
        return 0
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def pack_vector(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _load_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def connect(path: Path, *, readonly: bool = False, vec: bool = True) -> sqlite3.Connection:
    """Open index.db. Writers get WAL + a long busy timeout; readers open ``mode=ro``."""
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=60, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=OFF")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    if vec:
        _load_vec(conn)
    return conn


def init_schema(conn: sqlite3.Connection, *, dim: int = 384) -> None:
    conn.executescript(SCHEMA)
    conn.execute(vec_schema(dim))
    conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                 (key, value))


# --- docs ------------------------------------------------------------------------------

def add_doc(conn: sqlite3.Connection, *, doc_type: str, kind: str, project_id: int, text: str, title: str = "",
            session_id: str | None = None, ts: str | None = None, machine: str | None = None,
            automated: bool = False, file_id: int | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO docs(doc_type, kind, project_id, session_id, ts, machine, automated, file_id) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (doc_type, kind, project_id, session_id, ts, machine, int(automated), file_id),
    )
    doc_id = cur.lastrowid
    conn.execute("INSERT INTO fts_docs(rowid, text, title) VALUES (?,?,?)", (doc_id, text, title))
    return doc_id


def doc_text(conn: sqlite3.Connection, doc_id: int) -> str:
    row = conn.execute("SELECT text FROM fts_docs WHERE rowid = ?", (doc_id,)).fetchone()
    return row[0] if row else ""


def delete_docs(conn: sqlite3.Connection, doc_ids: Iterable[int]) -> None:
    ids = [(i,) for i in doc_ids]
    conn.executemany("DELETE FROM fts_docs WHERE rowid = ?", ids)
    conn.executemany("DELETE FROM docs WHERE id = ?", ids)


def delete_chunks(conn: sqlite3.Connection, doc_type: str, ref_id: str) -> None:
    ids = [(r[0],) for r in conn.execute("SELECT id FROM chunks WHERE doc_type = ? AND ref_id = ?", (doc_type, ref_id))]
    conn.executemany("DELETE FROM chunk_vec WHERE chunk_id = ?", ids)
    conn.executemany("DELETE FROM chunks WHERE id = ?", ids)


def delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Remove a session and everything derived from it (messages, docs, chunks, vectors)."""
    delete_docs(conn, [r[0] for r in conn.execute("SELECT id FROM docs WHERE session_id = ?", (session_id,))])
    delete_chunks(conn, "session", session_id)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def bump_drift(conn: sqlite3.Connection, category: str, value: str, count: int = 1) -> None:
    conn.execute(
        "INSERT INTO drift(category, value, count, last_seen) VALUES (?,?,?,?) "
        "ON CONFLICT(category, value) DO UPDATE SET count = count + excluded.count, last_seen = excluded.last_seen",
        (category, value, count, utcnow()),
    )
