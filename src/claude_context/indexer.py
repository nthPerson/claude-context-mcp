"""Ingestion: turn files under the configured roots into index.db rows.

``Indexer.reconcile()`` stats every file and ingests what changed; ``Indexer.ingest_path()``
handles one path (watcher / Syncthing event / server queue). Transcripts are append-only,
so each file keeps a byte offset and only the new complete lines are parsed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import db, memfiles
from .config import Config, RootConfig
from .models import ROLE_ASSISTANT, ROLE_SUMMARY, ROLE_TOOL_CALL, ROLE_TOOL_RESULT, TranscriptChunk
from .parsers.transcript import load_subagent_meta, parse_chunk
from .projects import project_for_key, sync_config_projects
from .redact import redact
from .roots import root_dir
from .syncthing import MachineResolver, SyncthingClient

log = logging.getLogger(__name__)

READ_BYTES = 32 * 1024 * 1024
EXCERPT_CHARS = 500
HUB_MACHINE = "hub"
UNKNOWN_MACHINE = "unknown"
RECENT_SECONDS = 30 * 60  # attribution of files ingested this recently is re-checked at reconcile

# files.kind values
F_TRANSCRIPT, F_SUBAGENT, F_MEMORY, F_NOTE = "transcript", "subagent", "memory", "note"


@dataclass
class Classified:
    kind: str  # transcript | subagent | memory | memory_index | memory_archived | note | conflict
    key: str  # project key (first path component)
    session_id: str | None = None
    parent_session_id: str | None = None
    stem: str | None = None


def is_temp_name(name: str) -> bool:
    return name.startswith((".syncthing.", "~syncthing~")) or name.endswith(".tmp")


def classify(rel: str) -> Classified | None:
    """Decide what a path (relative to a root, forward slashes) is; None = not indexed."""
    parts = rel.split("/")
    if len(parts) < 2 or parts[0].startswith(".") or is_temp_name(parts[-1]):
        return None
    key, name = parts[0], parts[-1]
    if len(parts) == 2 and name.endswith(".jsonl"):
        return Classified("transcript", key, session_id=name[:-6])
    if len(parts) >= 4 and parts[2] == "subagents" and name.startswith("agent-") and name.endswith(".jsonl"):
        return Classified("subagent", key, session_id=f"{parts[1]}:{name[6:-6]}", parent_session_id=parts[1])
    if parts[1] == "memory" and name.endswith(".md"):
        if ".sync-conflict-" in name:
            return Classified("conflict", key) if len(parts) == 3 else None
        if len(parts) == 3:
            return Classified("memory_index" if name == "MEMORY.md" else "memory", key, stem=name[:-3])
        if len(parts) == 4 and parts[2] == ".archived":
            return Classified("memory_archived", key, stem=name[:-3])
        return None
    if parts[1] == "remote-notes" and len(parts) == 3 and name.endswith(".md"):
        return Classified("note", key, stem=name[:-3])
    return None


def scan(base: Path) -> Iterator[tuple[str, os.stat_result]]:
    """Yield (rel_path, stat) for every regular file that might be indexable."""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not d.startswith(".st") and d != "tool-results"]
        for name in filenames:
            if not name.endswith((".jsonl", ".md")):
                continue
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            yield os.path.relpath(path, base).replace(os.sep, "/"), st


def mtime_iso(st: os.stat_result) -> str:
    return datetime.fromtimestamp(st.st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class _Session:
    """In-memory copy of a sessions row while a transcript file is being ingested."""

    id: str
    project_id: int
    root: str
    source: str
    machine: str
    is_subagent: bool = False
    parent_session_id: str | None = None
    agent_type: str | None = None
    agent_description: str | None = None
    entrypoint: str | None = None
    kind: str = "interactive"
    title: str | None = None
    summary: str | None = None
    first_prompt: str | None = None
    last_prompt: str | None = None
    final_reply_excerpt: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    cc_version: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    n_user: int = 0
    n_assistant: int = 0
    n_tool_calls: int = 0
    next_seq: int = 0
    file_id: int | None = None
    exists: bool = False
    title_dirty: bool = False


@dataclass
class ReconcileStats:
    scanned: int = 0
    ingested: int = 0
    missing: int = 0
    removed: int = 0
    errors: int = 0
    seconds: float = 0.0
    per_root: dict[str, int] = field(default_factory=dict)


class Indexer:
    def __init__(self, cfg: Config, conn: sqlite3.Connection, *, syncthing: SyncthingClient | None = None):
        self.cfg = cfg
        self.conn = conn
        self.syncthing = syncthing
        self._resolvers: dict[str, MachineResolver] = {}
        self._embed_chunks = cfg.embedding.enabled

    # --- helpers -----------------------------------------------------------------------
    def _resolver(self, folder: str) -> MachineResolver:
        resolver = self._resolvers.get(folder)
        if resolver is None:
            resolver = self._resolvers[folder] = MachineResolver(self.syncthing, folder)
        return resolver

    def _machine(self, root: RootConfig, rel: str, st: os.stat_result) -> str:
        folder = root.syncthing_folder
        if folder is None:
            return root.machine
        return self._resolver(folder).machine_for(rel, st.st_mtime_ns)

    def refresh_machine(self, root: RootConfig, rel: str) -> bool:
        """Correct an attribution made before Syncthing knew the file's current version.

        A file is usually ingested (inotify) before Syncthing has scanned it, when Syncthing
        still reports the previous version's device. Called when Syncthing announces the
        change, and for recently ingested files at each reconcile.
        """
        folder = root.syncthing_folder
        row = self._file_row(root, rel)
        if folder is None or row is None or row["kind"] == F_NOTE:
            return False
        machine = self._resolver(folder).refresh(rel, row["mtime_ns"])
        if machine == UNKNOWN_MACHINE or machine == row["machine"]:
            return False
        self.conn.execute("UPDATE files SET machine = ? WHERE id = ?", (machine, row["id"]))
        if row["kind"] == F_MEMORY:
            self.conn.execute("UPDATE memories SET modified_by = ? WHERE file_id = ?", (machine, row["id"]))
            self.conn.execute("UPDATE docs SET machine = ? WHERE file_id = ? AND doc_type = 'memory'",
                              (machine, row["id"]))
            self.conn.execute("UPDATE chunks SET machine = ? WHERE doc_type = 'memory' AND ref_id IN "
                              "(SELECT CAST(id AS TEXT) FROM memories WHERE file_id = ?)", (machine, row["id"]))
        else:  # a session belongs to the machine that created it: only fill in an unknown one
            s = self.conn.execute("SELECT id FROM sessions WHERE file_id = ? AND machine = ?",
                                  (row["id"], UNKNOWN_MACHINE)).fetchone()
            if s is not None:
                self.conn.execute("UPDATE sessions SET machine = ? WHERE id = ?", (machine, s["id"]))
                self.conn.execute("UPDATE docs SET machine = ? WHERE session_id = ?", (machine, s["id"]))
                self.conn.execute("UPDATE chunks SET machine = ? WHERE session_id = ?", (machine, s["id"]))
        self.conn.commit()
        return True

    def _refresh_recent_machines(self, root: RootConfig) -> None:
        if root.syncthing_folder is None or self.syncthing is None:
            return
        cutoff = (datetime.now(UTC) - timedelta(seconds=RECENT_SECONDS)).strftime("%Y-%m-%dT%H:%M:%S")
        for r in self.conn.execute("SELECT rel_path FROM files WHERE root = ? AND last_ingest_at >= ? "
                                   "AND missing_since IS NULL", (root.label, cutoff)).fetchall():
            self.refresh_machine(root, r["rel_path"])

    def _file_row(self, root: RootConfig, rel: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE root = ? AND rel_path = ?", (root.label, rel)).fetchone()

    def _upsert_file(self, root: RootConfig, rel: str, kind: str, st: os.stat_result, *, byte_offset: int = 0,
                     head_sha: str | None = None, machine: str | None = None) -> int:
        self.conn.execute(
            "INSERT INTO files(root, rel_path, kind, size, mtime_ns, byte_offset, head_sha, last_ingest_at, machine) "
            "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(root, rel_path) DO UPDATE SET kind = excluded.kind, "
            "size = excluded.size, mtime_ns = excluded.mtime_ns, byte_offset = excluded.byte_offset, "
            "head_sha = excluded.head_sha, last_ingest_at = excluded.last_ingest_at, "
            "machine = COALESCE(excluded.machine, machine), missing_since = NULL",
            (root.label, rel, kind, st.st_size, st.st_mtime_ns, byte_offset, head_sha, db.utcnow(), machine),
        )
        return self.conn.execute("SELECT id FROM files WHERE root = ? AND rel_path = ?", (root.label, rel)).fetchone()[0]

    def _session_kind(self, entrypoint: str | None) -> str:
        if entrypoint in self.cfg.automated_entrypoints:
            return "automated"
        if entrypoint and entrypoint not in self.cfg.interactive_entrypoints:
            db.bump_drift(self.conn, "entrypoint", entrypoint)
        return "interactive"

    # --- reconcile ---------------------------------------------------------------------
    def reconcile(self, only_root: str | None = None) -> ReconcileStats:
        """Stat everything under every root; ingest new/changed files; flag vanished ones."""
        stats, t0 = ReconcileStats(), time.monotonic()
        sync_config_projects(self.conn, self.cfg)
        for root in self.cfg.roots:
            if only_root and root.label != only_root:
                continue
            base = root_dir(self.cfg, root)
            if not base.is_dir():
                if root.kind == "claude-code":
                    log.warning("root %s: %s is not a directory", root.label, base)
                continue
            seen: set[str] = set()
            conflicts: list[tuple[str, str]] = []
            for rel, st in scan(base):
                c = classify(rel)
                if c is None:
                    continue
                stats.scanned += 1
                if c.kind == "conflict":
                    conflicts.append((rel, c.key))
                    continue
                seen.add(rel)
                try:
                    if self._ingest_file(root, base, rel, st, c):
                        stats.ingested += 1
                        self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    stats.errors += 1
                    log.exception("ingest failed: %s/%s", root.label, rel)
            m, r = self._handle_vanished(root, seen)
            stats.missing += m
            stats.removed += r
            self._set_conflicts(root, conflicts)
            self._refresh_recent_machines(root)
            stats.per_root[root.label] = len(seen)
            db.set_meta(self.conn, f"reconcile:{root.label}", db.utcnow())
            self.conn.commit()
        if not only_root:
            db.set_meta(self.conn, "last_reconcile_at", db.utcnow())
            self.conn.commit()
        stats.seconds = time.monotonic() - t0
        return stats

    def ingest_path(self, root: RootConfig, rel: str) -> bool:
        """(Re)index one path. Returns True if anything changed."""
        c = classify(rel)
        if c is None:
            return False
        base = root_dir(self.cfg, root)
        path = base / rel
        try:
            try:
                st = path.stat()
            except FileNotFoundError:
                changed = self._vanished(root, rel, c)
            else:
                if c.kind == "conflict":
                    self._add_conflict(root, rel, c.key)
                    changed = True
                else:
                    changed = self._ingest_file(root, base, rel, st, c)
            self.conn.commit()
            return changed
        except Exception:
            self.conn.rollback()
            log.exception("ingest failed: %s/%s", root.label, rel)
            return False

    def _ingest_file(self, root: RootConfig, base: Path, rel: str, st: os.stat_result, c: Classified) -> bool:
        row = self._file_row(root, rel)
        if row and row["size"] == st.st_size and row["mtime_ns"] == st.st_mtime_ns and row["missing_since"] is None:
            return False
        if c.kind in ("transcript", "subagent"):
            return self._ingest_transcript(root, base, rel, st, c, row)
        if c.kind == "note":
            return self._ingest_note(root, base, rel, st, c)
        return self._ingest_memory(root, base, rel, st, c)

    # --- transcripts -------------------------------------------------------------------
    def _load_session(self, session_id: str) -> _Session | None:
        r = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if r is None:
            return None
        s = _Session(id=r["id"], project_id=r["project_id"], root=r["root"], source=r["source"], machine=r["machine"],
                     exists=True)
        for name in ("parent_session_id", "agent_type", "agent_description", "entrypoint", "kind", "title", "summary",
                     "first_prompt", "last_prompt", "final_reply_excerpt", "cwd", "git_branch", "cc_version",
                     "started_at", "ended_at", "n_user", "n_assistant", "n_tool_calls", "next_seq", "file_id"):
            setattr(s, name, r[name])
        s.is_subagent = bool(r["is_subagent"])
        return s

    def _ingest_transcript(self, root: RootConfig, base: Path, rel: str, st: os.stat_result, c: Classified,
                           row: sqlite3.Row | None) -> bool:
        kind = F_SUBAGENT if c.kind == "subagent" else F_TRANSCRIPT
        if row is None and st.st_mtime < time.time() - self.cfg.retention_days * 86400:
            return False  # older than retention: never indexed (the pruner would delete it again)
        session_id = c.session_id
        assert session_id is not None
        sess = self._load_session(session_id)
        file_id = row["id"] if row else None
        if sess is not None and sess.file_id is not None and sess.file_id != file_id:
            # The same session already came from another file (e.g. live copy vs. recovered copy).
            self._upsert_file(root, rel, kind, st, byte_offset=st.st_size)
            return False

        offset = row["byte_offset"] if row else 0
        path = base / rel
        with open(path, "rb") as f:
            head = f.readline(1 << 20)
            head_sha = hashlib.sha256(head).hexdigest() if head.endswith(b"\n") else None
            if offset and (st.st_size < offset or (row["head_sha"] and head_sha != row["head_sha"])):
                log.info("transcript rewritten, re-ingesting: %s/%s", root.label, rel)
                db.delete_session(self.conn, session_id)
                sess, offset = None, 0
            machine = sess.machine if sess and sess.machine != "unknown" else self._machine(root, rel, st)
            file_id = self._upsert_file(root, rel, kind, st, byte_offset=offset, head_sha=head_sha, machine=machine)
            if sess is None:
                sess = _Session(id=session_id, project_id=0, root=root.label, machine=machine,
                                source="recovered" if root.kind == "stversions" else "claude-code",
                                is_subagent=c.kind == "subagent", parent_session_id=c.parent_session_id,
                                file_id=file_id)
                if sess.is_subagent:
                    meta = load_subagent_meta(path.with_name(path.name[:-6] + ".meta.json"))
                    sess.agent_type = meta.get("agentType")
                    sess.agent_description = meta.get("description")
            else:
                sess.machine = machine
            f.seek(offset)
            while True:
                data = f.read(READ_BYTES)
                if not data:
                    break
                while b"\n" not in data:  # a single line longer than the read size
                    more = f.read(READ_BYTES)
                    if not more:
                        break
                    data += more
                chunk = parse_chunk(data, offset)
                if chunk.consumed == 0:
                    break
                self._apply_chunk(sess, chunk, c.key, file_id)
                offset += chunk.consumed
                f.seek(offset)
            st = os.fstat(f.fileno())
        if sess.project_id and (sess.next_seq or sess.exists):
            self._save_session(sess)
        self._upsert_file(root, rel, kind, st, byte_offset=offset, head_sha=head_sha, machine=machine)
        return True

    def _apply_chunk(self, s: _Session, chunk: TranscriptChunk, key: str, file_id: int) -> None:
        f = chunk.facts
        s.cwd = f.cwd or s.cwd
        s.git_branch = f.git_branch or s.git_branch
        s.cc_version = f.cc_version or s.cc_version
        if f.title and f.title != s.title:
            s.title, s.title_dirty = redact(f.title), True
        s.summary = f.summary or s.summary
        if f.entrypoint and f.entrypoint != s.entrypoint:
            s.entrypoint = f.entrypoint
            kind = self._session_kind(f.entrypoint)
            if s.exists and kind != s.kind:
                self.conn.execute("UPDATE docs SET automated = ? WHERE session_id = ?", (int(kind == "automated"), s.id))
            s.kind = kind
        if f.first_ts and (s.started_at is None or f.first_ts < s.started_at):
            s.started_at = f.first_ts
        if f.last_ts and (s.ended_at is None or f.last_ts > s.ended_at):
            s.ended_at = f.last_ts
        if not s.project_id:
            s.project_id = project_for_key(self.conn, key, s.cwd)
        for t, n in chunk.unknown_types.items():
            db.bump_drift(self.conn, "record_type", t, n)
        if chunk.malformed_lines:
            db.bump_drift(self.conn, "malformed", "lines", chunk.malformed_lines)

        automated = s.kind == "automated"
        last_ts = s.ended_at
        rows = []
        for m in chunk.messages:
            last_ts = m.ts or last_ts
            tool = m.role in (ROLE_TOOL_CALL, ROLE_TOOL_RESULT)
            doc_id = db.add_doc(self.conn, doc_type="message", kind=db.KIND_TOOL if tool else db.KIND_TRANSCRIPT,
                                project_id=s.project_id, text=m.text, session_id=s.id, ts=last_ts,
                                machine=s.machine, automated=automated, file_id=file_id)
            rows.append((s.id, s.next_seq, m.uuid, last_ts, m.role, m.tool_name, m.tool_use_id, int(m.is_error),
                         int(m.is_prompt), m.text_len, doc_id, file_id, m.line_offset, m.block_index))
            s.next_seq += 1
            if m.is_prompt:
                s.n_user += 1
                s.last_prompt = m.text[:EXCERPT_CHARS]
                s.first_prompt = s.first_prompt or s.last_prompt
            elif m.role == ROLE_ASSISTANT:
                s.n_assistant += 1
                s.final_reply_excerpt = m.text[:EXCERPT_CHARS]
            elif m.role == ROLE_TOOL_CALL:
                s.n_tool_calls += 1
            elif m.role == ROLE_SUMMARY:
                s.summary = m.text
        self.conn.executemany(
            "INSERT INTO messages(session_id, seq, uuid, ts, role, tool_name, tool_use_id, is_error, is_prompt, "
            "text_len, doc_id, file_id, line_offset, block_index) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def _save_session(self, s: _Session) -> None:
        self.conn.execute(
            "INSERT INTO sessions(id, project_id, root, machine, source, entrypoint, kind, is_subagent, "
            "parent_session_id, agent_type, agent_description, title, summary, first_prompt, last_prompt, "
            "final_reply_excerpt, cwd, git_branch, cc_version, started_at, ended_at, n_user, n_assistant, "
            "n_tool_calls, next_seq, file_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET machine = excluded.machine, entrypoint = excluded.entrypoint, "
            "kind = excluded.kind, title = excluded.title, summary = excluded.summary, "
            "first_prompt = excluded.first_prompt, last_prompt = excluded.last_prompt, "
            "final_reply_excerpt = excluded.final_reply_excerpt, cwd = excluded.cwd, git_branch = excluded.git_branch, "
            "cc_version = excluded.cc_version, started_at = excluded.started_at, ended_at = excluded.ended_at, "
            "n_user = excluded.n_user, n_assistant = excluded.n_assistant, n_tool_calls = excluded.n_tool_calls, "
            "next_seq = excluded.next_seq, file_id = excluded.file_id",
            (s.id, s.project_id, s.root, s.machine, s.source, s.entrypoint, s.kind, int(s.is_subagent),
             s.parent_session_id, s.agent_type, s.agent_description, s.title, s.summary, s.first_prompt, s.last_prompt,
             s.final_reply_excerpt, s.cwd, s.git_branch, s.cc_version, s.started_at, s.ended_at, s.n_user,
             s.n_assistant, s.n_tool_calls, s.next_seq, s.file_id),
        )
        if s.title_dirty and s.title:  # one extra doc so sessions are findable by title
            old = [r[0] for r in self.conn.execute(
                "SELECT id FROM docs WHERE session_id = ? AND doc_type = 'session'", (s.id,))]
            db.delete_docs(self.conn, old)
            db.add_doc(self.conn, doc_type="session", kind=db.KIND_TRANSCRIPT, project_id=s.project_id, text=s.title,
                       title=s.title, session_id=s.id, ts=s.ended_at, machine=s.machine,
                       automated=s.kind == "automated", file_id=s.file_id)
        s.exists = True

    # --- memories ----------------------------------------------------------------------
    def _delete_memory(self, file_id: int) -> None:
        r = self.conn.execute("SELECT id, doc_id FROM memories WHERE file_id = ?", (file_id,)).fetchone()
        if r is None:
            return
        if r["doc_id"] is not None:
            db.delete_docs(self.conn, [r["doc_id"]])
        db.delete_chunks(self.conn, "memory", str(r["id"]))
        self.conn.execute("DELETE FROM memories WHERE id = ?", (r["id"],))

    def _index_titles(self, memory_dir: Path) -> dict[str, str]:
        try:
            text = (memory_dir / "MEMORY.md").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {}
        return {line.target: line.title for line in memfiles.parse_index(text)}

    def _ingest_memory(self, root: RootConfig, base: Path, rel: str, st: os.stat_result, c: Classified) -> bool:
        path = base / rel
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        machine = self._machine(root, rel, st)
        file_id = self._upsert_file(root, rel, F_MEMORY, st, machine=machine)
        self._delete_memory(file_id)
        pid = project_for_key(self.conn, c.key)
        stem = c.stem or ""
        archived, is_index = c.kind == "memory_archived", c.kind == "memory_index"
        ts = mtime_iso(st)
        if is_index:
            name, title, description, mtype, body = "MEMORY", "MEMORY.md (memory index)", "", "index", text
        else:
            pm = memfiles.parse_memory(text, stem)
            name, description, mtype = pm.name, pm.description, pm.type
            title = (self._index_titles(path.parent).get(f"{stem}.md") if not archived else None) or pm.name
            body = f"{pm.description}\n\n{pm.body}" if pm.description else pm.body
        body, title, description = redact(body), redact(title), redact(description)
        doc_id = None
        if not archived:
            doc_id = db.add_doc(self.conn, doc_type="memory", kind=db.KIND_MEMORY, project_id=pid, text=body,
                                title=title, ts=ts, machine=machine, file_id=file_id)
        mem_id = self.conn.execute(
            "INSERT INTO memories(project_id, root, project_key, file_id, stem, name, title, description, type, "
            "sha256, modified_at, modified_by, archived, is_index, doc_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, root.label, c.key, file_id, stem, name, title, description, mtype,
             hashlib.sha256(raw).hexdigest(), ts, machine, int(archived), int(is_index), doc_id),
        ).lastrowid
        if self._embed_chunks and not archived and not is_index:
            from .embed import add_doc_chunks
            add_doc_chunks(self.conn, doc_type="memory", ref_id=str(mem_id), project_id=pid, kind=db.KIND_MEMORY,
                           ts=ts, machine=machine, body=body, title=title)
        if is_index:  # link text may have changed: refresh the titles of this folder's memories
            titles = {line.target: line.title for line in memfiles.parse_index(text)}
            for r in self.conn.execute(
                    "SELECT id, stem, title, doc_id FROM memories WHERE root = ? AND project_key = ? AND is_index = 0 "
                    "AND archived = 0", (root.label, c.key)).fetchall():
                new = titles.get(f"{r['stem']}.md")
                if new and (new := redact(new)) != r["title"]:
                    self.conn.execute("UPDATE memories SET title = ? WHERE id = ?", (new, r["id"]))
                    self.conn.execute("UPDATE fts_docs SET title = ? WHERE rowid = ?", (new, r["doc_id"]))
        return True

    # --- notes -------------------------------------------------------------------------
    def _delete_note(self, file_id: int) -> None:
        r = self.conn.execute("SELECT id, doc_id FROM notes WHERE file_id = ?", (file_id,)).fetchone()
        if r is None:
            return
        db.delete_docs(self.conn, [r["doc_id"]])
        db.delete_chunks(self.conn, "note", str(r["id"]))
        self.conn.execute("DELETE FROM notes WHERE id = ?", (r["id"],))

    def _ingest_note(self, root: RootConfig, base: Path, rel: str, st: os.stat_result, c: Classified) -> bool:
        path = base / rel
        note = memfiles.parse_note(path.read_text(encoding="utf-8", errors="replace"))
        file_id = self._upsert_file(root, rel, F_NOTE, st, machine=HUB_MACHINE)
        self._delete_note(file_id)
        pid = project_for_key(self.conn, c.key)
        title = redact(note.title or c.stem or "")
        body = redact(note.body)
        created = note.created or mtime_iso(st)
        doc_id = db.add_doc(self.conn, doc_type="note", kind=db.KIND_NOTE, project_id=pid, text=body, title=title,
                            ts=created, machine=HUB_MACHINE, file_id=file_id)
        note_pk = self.conn.execute(
            "INSERT INTO notes(project_id, root, project_key, file_id, note_id, title, surface, created_at, "
            "related_sessions, machine, doc_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (pid, root.label, c.key, file_id, path.name, title, note.surface, created,
             json.dumps(list(note.related_sessions)), HUB_MACHINE, doc_id),
        ).lastrowid
        if self._embed_chunks:
            from .embed import add_doc_chunks
            add_doc_chunks(self.conn, doc_type="note", ref_id=str(note_pk), project_id=pid, kind=db.KIND_NOTE,
                           ts=created, machine=HUB_MACHINE, body=body, title=title)
        return True

    # --- vanished files and conflicts --------------------------------------------------
    def _vanished(self, root: RootConfig, rel: str, c: Classified) -> bool:
        if c.kind == "conflict":
            return self.conn.execute("DELETE FROM conflicts WHERE root = ? AND rel_path = ?",
                                     (root.label, rel)).rowcount > 0
        row = self._file_row(root, rel)
        if row is None:
            return False
        if row["kind"] in (F_TRANSCRIPT, F_SUBAGENT):
            # Never drop a session because its file disappeared: it is served from the archive.
            if row["missing_since"] is None:
                self.conn.execute("UPDATE files SET missing_since = ? WHERE id = ?", (db.utcnow(), row["id"]))
                return True
            return False
        self._delete_memory(row["id"]) if row["kind"] == F_MEMORY else self._delete_note(row["id"])
        self.conn.execute("DELETE FROM files WHERE id = ?", (row["id"],))
        return True

    def _handle_vanished(self, root: RootConfig, seen: set[str]) -> tuple[int, int]:
        missing = removed = 0
        rows = self.conn.execute("SELECT rel_path, kind, missing_since FROM files WHERE root = ?",
                                 (root.label,)).fetchall()
        for r in rows:
            if r["rel_path"] in seen or r["kind"] == "claudeai":
                continue
            c = classify(r["rel_path"])
            if c is None or not self._vanished(root, r["rel_path"], c):
                continue
            if r["kind"] in (F_TRANSCRIPT, F_SUBAGENT):
                missing += 1
            else:
                removed += 1
        return missing, removed

    def _add_conflict(self, root: RootConfig, rel: str, key: str) -> None:
        pid = project_for_key(self.conn, key)
        self.conn.execute("INSERT OR IGNORE INTO conflicts(root, rel_path, project_id, seen_at) VALUES (?,?,?,?)",
                          (root.label, rel, pid, db.utcnow()))

    def _set_conflicts(self, root: RootConfig, conflicts: list[tuple[str, str]]) -> None:
        current = {rel for rel, _ in conflicts}
        for r in self.conn.execute("SELECT rel_path FROM conflicts WHERE root = ?", (root.label,)).fetchall():
            if r["rel_path"] not in current:
                self.conn.execute("DELETE FROM conflicts WHERE root = ? AND rel_path = ?", (root.label, r["rel_path"]))
        for rel, key in conflicts:
            self._add_conflict(root, rel, key)
