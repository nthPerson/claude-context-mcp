"""claude.ai export importer: conversations become sessions with source ``claude.ai``."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import db
from .config import Config
from .models import ROLE_ASSISTANT, ROLE_USER, ClaudeAiConversation, ClaudeAiExport
from .parsers.claudeai import ExportError, parse_export
from .projects import CLAUDEAI_ALIAS, ensure_project, slug
from .redact import redact

log = logging.getLogger(__name__)

SOURCE = "claude.ai"  # also the root label, machine and entrypoint of imported sessions
EXCERPT_CHARS = 500


@dataclass
class ImportResult:
    name: str
    conversations: int = 0  # conversations in the export
    updated: int = 0  # new or changed conversations written
    memories: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return f"{self.name}: FAILED: {self.error}"
        return (f"{self.name}: {self.conversations} conversations, {self.updated} new/updated, "
                f"{self.memories} memories, {self.skipped} skipped, {len(self.warnings)} warnings")


def _project_ids(conn: sqlite3.Connection, cfg: Config) -> tuple[int, dict[str, int]]:
    default = ensure_project(conn, CLAUDEAI_ALIAS, "claude.ai conversations", "auto")
    mapped = {name.lower(): ensure_project(conn, alias.lower(), alias, "hub")
              for name, alias in cfg.claudeai_project_map.items()}
    return default, mapped


def _import_conversation(conn: sqlite3.Connection, conv: ClaudeAiConversation, pid: int) -> bool:
    row = conn.execute("SELECT source, updated_at FROM sessions WHERE id = ?", (conv.uuid,)).fetchone()
    if row is not None:
        if row["source"] != SOURCE or (row["updated_at"] or "") >= (conv.updated_at or ""):
            return False
        db.delete_session(conn, conv.uuid)
    title = redact(conv.name) or None
    first = last = final = None
    n_user = n_assistant = 0
    ts = conv.created_at
    rows = []
    for seq, m in enumerate(conv.messages):
        ts = m.created_at or ts
        text = redact("\n".join([m.text, *(f"[attachment: {a}]" for a in m.attachments)]).strip())
        human = m.sender == "human"
        doc_id = db.add_doc(conn, doc_type="message", kind=db.KIND_CLAUDEAI, project_id=pid, text=text,
                            session_id=conv.uuid, ts=ts, machine=SOURCE)
        rows.append((conv.uuid, seq, m.uuid, ts, ROLE_USER if human else ROLE_ASSISTANT, int(human), len(text), doc_id))
        if human:
            n_user += 1
            last = text[:EXCERPT_CHARS]
            first = first or last
        else:
            n_assistant += 1
            final = text[:EXCERPT_CHARS]
    conn.executemany("INSERT INTO messages(session_id, seq, uuid, ts, role, is_prompt, text_len, doc_id) "
                     "VALUES (?,?,?,?,?,?,?,?)", rows)
    ended = conv.updated_at or ts
    conn.execute(
        "INSERT INTO sessions(id, project_id, root, machine, source, entrypoint, kind, title, summary, first_prompt, "
        "last_prompt, final_reply_excerpt, started_at, ended_at, n_user, n_assistant, next_seq, updated_at) "
        "VALUES (?,?,?,?,?,?,'interactive',?,?,?,?,?,?,?,?,?,?,?)",
        (conv.uuid, pid, SOURCE, SOURCE, SOURCE, SOURCE, title, redact(conv.summary) or None, first, last, final,
         conv.created_at, ended, n_user, n_assistant, len(rows), conv.updated_at),
    )
    if title:
        db.add_doc(conn, doc_type="session", kind=db.KIND_CLAUDEAI, project_id=pid, text=title, title=title,
                   session_id=conv.uuid, ts=ended, machine=SOURCE)
    return True


def _import_memories(conn: sqlite3.Connection, cfg: Config, export: ClaudeAiExport, default_pid: int,
                     mapped: dict[str, int]) -> int:
    for r in conn.execute("SELECT id, doc_id FROM memories WHERE root = ?", (SOURCE,)).fetchall():
        db.delete_docs(conn, [r["doc_id"]])
        db.delete_chunks(conn, "memory", str(r["id"]))
    conn.execute("DELETE FROM memories WHERE root = ?", (SOURCE,))
    for mem in export.memories:
        pid = mapped.get(mem.scope.lower(), default_pid)
        title = f"claude.ai memory ({mem.scope})"
        body = redact(mem.text)
        ts = mem.updated_at or db.utcnow()
        doc_id = db.add_doc(conn, doc_type="memory", kind=db.KIND_CLAUDEAI, project_id=pid, text=body, title=title,
                            ts=ts, machine=SOURCE)
        mem_id = conn.execute(
            "INSERT INTO memories(project_id, root, stem, name, title, description, type, modified_at, modified_by, "
            "doc_id) VALUES (?,?,?,?,?,?,'claudeai-memory',?,?,?)",
            (pid, SOURCE, f"claudeai-{slug(mem.scope) or 'account'}", title, title, "Memory exported from claude.ai",
             ts, SOURCE, doc_id),
        ).lastrowid
        if cfg.embedding.enabled:
            from .embed import add_doc_chunks
            add_doc_chunks(conn, doc_type="memory", ref_id=str(mem_id), project_id=pid, kind=db.KIND_CLAUDEAI, ts=ts,
                           machine=SOURCE, body=body, title=title)
    return len(export.memories)


def import_export(cfg: Config, conn: sqlite3.Connection, zip_path: Path) -> ImportResult:
    """Upsert every conversation of an export zip (by uuid; newer ``updated_at`` wins)."""
    result = ImportResult(name=zip_path.name)
    try:
        export = parse_export(zip_path)
    except ExportError as e:
        result.error = str(e)
        return result
    try:
        default_pid, mapped = _project_ids(conn, cfg)
        for conv in export.conversations:
            pid = mapped.get((conv.project_name or "").lower(), default_pid)
            result.updated += _import_conversation(conn, conv, pid)
        result.memories = _import_memories(conn, cfg, export, default_pid, mapped)
        conn.commit()
    except Exception as e:
        conn.rollback()
        log.exception("claude.ai import failed: %s", zip_path.name)
        result.error = f"{type(e).__name__}: {e}"
        return result
    result.conversations = len(export.conversations)
    result.skipped = export.skipped
    result.warnings = export.warnings
    return result


def process_drop(cfg: Config, conn: sqlite3.Connection, zip_path: Path) -> ImportResult:
    """Import a zip from the drop folder, then move it out to ``imports/processed/``."""
    result = import_export(cfg, conn, zip_path)
    cfg.imports_processed_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = ".failed.zip" if result.error else ".zip"
    dest = cfg.imports_processed_dir / f"{stamp}-{zip_path.stem}{suffix}"
    shutil.move(str(zip_path), dest)
    dest.with_suffix(".log").write_text(result.summary() + "\n" + "\n".join(result.warnings) + "\n", encoding="utf-8")
    conn.execute("INSERT OR REPLACE INTO imports(name, ingested_at, conversations, updated, result) VALUES (?,?,?,?,?)",
                 (dest.name, db.utcnow(), result.conversations, result.updated, result.summary()))
    conn.commit()
    log.info("claude.ai import: %s", result.summary())
    return result
