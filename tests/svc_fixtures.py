"""Synthetic index.db + on-disk files for the service and health tests.

Everything here is invented. Raw transcripts are written to disk and parsed with the real
transcript parser, so ``messages.line_offset`` points at the real byte offsets.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from claude_context import db, memfiles
from claude_context.config import Config, RootConfig
from claude_context.models import ROLE_ASSISTANT, ROLE_TOOL_CALL, ROLE_TOOL_RESULT, ROLE_USER, ParsedMessage
from claude_context.parsers.transcript import parse_chunk

NOW = datetime(2026, 1, 31, 22, 0, tzinfo=UTC)  # 14:00 PST
TZ = "America/Los_Angeles"
FAKE_SECRET = "sk-" + "ant-" + "FAKEtestKEY0123456789abcdefXYZ"  # split so scanners don't flag it
THINKING = "PRIVATE-THINKING-MARKER"

KEY_A, KEY_A2, KEY_B, KEY_G = "-home-user-alpha", "-mnt-c-work-alpha", "-home-user-beta", "-home-user-gamma"
S1 = "11111111-aaaa-4aaa-8aaa-000000000001"  # alpha, interactive, raw transcript on disk
SUB = f"{S1}:agentabc1"  # subagent of S1
S2 = "22222222-bbbb-4bbb-8bbb-000000000002"  # alpha, interactive, missing upstream, no raw copy
S7 = "22222222-cccc-4ccc-8ccc-000000000007"  # alpha, automated (shares S2's 8-char prefix)
S3 = "33333333-dddd-4ddd-8ddd-000000000003"  # beta, automated
S4 = "44444444-eeee-4eee-8eee-000000000004"  # alpha, recovered (raw copy only in the archive)
S5 = "55555555-ffff-4fff-8fff-000000000005"  # alpha, claude.ai conversation
S6 = "66666666-0000-4000-8000-000000000006"  # gamma, old (inactive project)
ALPHA, BETA, CLAUDEAI, GAMMA, ALPHA_DOCS = 1, 2, 3, 4, 5
NOTE_ID = "2026-01-31-120000-claude.ai-deck-plan.md"
LONG_OUTPUT = "".join(f"output line {i:04d}\n" for i in range(250)) + f"config: {FAKE_SECRET}\nend of output\n"


@dataclass
class Fixture:
    cfg: Config
    sync: Path  # the writable "synced" root

    def rw(self) -> sqlite3.Connection:
        return db.connect(self.cfg.index_db)


def ts(minutes: int, day: int = 30, hour: int = 18) -> str:
    return (datetime(2026, 1, day, hour, tzinfo=UTC) + timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z")


def make_config(tmp: Path) -> Config:
    sync = tmp / "sync"
    return Config(data_dir=tmp / "data", timezone=TZ, roots=[
        RootConfig(label="synced", path=sync, machine="syncthing:folder1", writable=True),
        RootConfig(label="recovered", path=sync / ".stversions", kind="stversions"),
    ])


def _record(rtype: str, uuid: str, when: str, content: object, sid: str = S1, **extra: object) -> dict:
    return {"type": rtype, "uuid": uuid, "timestamp": when, "sessionId": sid, "entrypoint": "cli",
            "cwd": "/home/user/alpha", "gitBranch": "main", "version": "2.0.0",
            "message": {"role": rtype, "content": content}, **extra}


def _tool_use(tid: str, name: str, inp: dict) -> list[dict]:
    return [{"type": "tool_use", "id": tid, "name": name, "input": inp}]


def _tool_result(tid: str, content: str) -> list[dict]:
    return [{"type": "tool_result", "tool_use_id": tid, "content": content}]


S1_RECORDS = [
    _record("user", "u-01", ts(0), "Please fix the parser bug"),
    _record("assistant", "u-02", ts(1), [
        {"type": "thinking", "thinking": THINKING, "signature": "x"},
        {"type": "text", "text": "Looking at the parser now."},
        *_tool_use("toolu_01AAA", "Bash", {"command": "cat parser.py", "description": "Show parser"})]),
    _record("user", "u-03", ts(2), _tool_result("toolu_01AAA", LONG_OUTPUT)),
    _record("assistant", "u-04", ts(3), _tool_use("toolu_01BBB", "Read", {"file_path": "/home/user/alpha/big.log"})),
    _record("user", "u-05", ts(4), _tool_result(
        "toolu_01BBB", "<persisted-output>\nOutput too large (9KB). Full output saved to: "
        f"/home/user/.claude/projects/{KEY_A}/{S1}/tool-results/bshort01.txt\n\nPreview (first 2KB):\nlog 0\n"
        "</persisted-output>")),
    _record("assistant", "u-06", ts(5), _tool_use("toolu_01CCC", "Grep", {"pattern": "TODO", "path": "src"})),
    _record("user", "u-07", ts(6), _tool_result("toolu_01CCC", "grep preview")),
    _record("assistant", "u-08", ts(60), [{"type": "text", "text": "The parser is fixed; all tests pass."}]),
    {"type": "ai-title", "aiTitle": "Fix the parser", "sessionId": S1},
]
SUB_RECORDS = [
    _record("user", "u-s1", ts(1, hour=19), "Find the parser files", isSidechain=True, agentId="agentabc1"),
    _record("assistant", "u-s2", ts(2, hour=19), [{"type": "text", "text": "Found src/parser.py."}],
            isSidechain=True, agentId="agentabc1"),
]
S4_RECORDS = [
    _record("user", "u-41", ts(0, day=28), "Recover the lost notes", sid=S4),
    _record("assistant", "u-42", ts(5, day=28), [{"type": "text", "text": "Recovered them from backups."}], sid=S4),
]


def _add_file(conn: sqlite3.Connection, root: str, rel: str, kind: str, *, size: int = 0, machine: str = "laptop",
              missing_since: str | None = None) -> int:
    return conn.execute("INSERT INTO files(root, rel_path, kind, size, machine, missing_since) VALUES (?,?,?,?,?,?)",
                        (root, rel, kind, size, machine, missing_since)).lastrowid


def _write_raw(conn: sqlite3.Connection, base: Path, root: str, rel: str, records: list[dict],
               kind: str = "transcript", **file_kw: object) -> tuple[int, list[ParsedMessage]]:
    data = b"".join(json.dumps(r).encode() + b"\n" for r in records)
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    file_id = _add_file(conn, root, rel, kind, size=len(data), **file_kw)
    return file_id, parse_chunk(data).messages


def add_session(conn: sqlite3.Connection, sid: str, pid: int, msgs: list[ParsedMessage], *,
                file_id: int | None = None, **cols: object) -> None:
    """Insert a session row and its messages (docs + messages), deriving counts like the indexer."""
    row = {"machine": "laptop", "source": "claude-code", "root": "synced", "entrypoint": "cli",
           "kind": "interactive", **cols}
    automated = row["kind"] == "automated"
    doc_kind = db.KIND_CLAUDEAI if row["source"] == "claude.ai" else None
    for seq, m in enumerate(msgs):
        tool = m.role in (ROLE_TOOL_CALL, ROLE_TOOL_RESULT)
        doc_id = db.add_doc(conn, doc_type="message", kind=doc_kind or (db.KIND_TOOL if tool else db.KIND_TRANSCRIPT),
                            project_id=pid, text=m.text, session_id=sid, ts=m.ts, machine=row["machine"],
                            automated=automated, file_id=file_id)
        conn.execute(
            "INSERT INTO messages(session_id, seq, uuid, ts, role, tool_name, tool_use_id, is_error, is_prompt, "
            "text_len, doc_id, file_id, line_offset, block_index) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, seq, m.uuid, m.ts, m.role, m.tool_name, m.tool_use_id, int(m.is_error), int(m.is_prompt),
             m.text_len, doc_id, file_id, m.line_offset if file_id else None, m.block_index))
    prompts = [m for m in msgs if m.is_prompt]
    replies = [m for m in msgs if m.role == ROLE_ASSISTANT]
    times = [m.ts for m in msgs if m.ts]
    row.update(project_id=pid, file_id=file_id, n_user=len(prompts), n_assistant=len(replies),
               n_tool_calls=sum(m.role == ROLE_TOOL_CALL for m in msgs), next_seq=len(msgs),
               started_at=row.get("started_at") or min(times, default=None),
               ended_at=row.get("ended_at") or max(times, default=None),
               first_prompt=prompts[0].text[:500] if prompts else None,
               last_prompt=prompts[-1].text[:500] if prompts else None,
               final_reply_excerpt=replies[-1].text[:500] if replies else None)
    row["id"] = sid
    conn.execute(f"INSERT INTO sessions({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))


def msg(role: str, text: str, uuid: str, when: str, *, text_len: int | None = None, **kw: object) -> ParsedMessage:
    return ParsedMessage(role=role, text=text, text_len=text_len or len(text), uuid=uuid, ts=when,
                         is_prompt=kw.pop("is_prompt", role == ROLE_USER), **kw)


def _memory_file(name: str, description: str, mtype: str, body: str) -> str:
    return memfiles.render_frontmatter({"name": name, "description": description, "metadata": {"type": mtype}}, body)


def add_memory(conn: sqlite3.Connection, base: Path, pid: int, key: str, stem: str, text: str | None, *,
               title: str = "", description: str = "", mtype: str = "project", modified: str = ts(0, day=29),
               machine: str = "laptop", archived: bool = False, root: str = "synced") -> None:
    """A memory row (plus its file on disk unless ``text`` is None, as for claude.ai memory)."""
    file_id, sha = None, None
    if text is not None:
        rel = f"{key}/memory/{'.archived/' if archived else ''}{stem}.md"
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text(text, encoding="utf-8")
        file_id = _add_file(conn, root, rel, "memory", size=len(text), machine=machine)
        sha = hashlib.sha256(text.encode()).hexdigest()
    body = text if text is not None else description
    doc_id = None if archived else db.add_doc(  # archived memories are not searchable (no doc)
        conn, doc_type="memory", kind=db.KIND_CLAUDEAI if text is None else db.KIND_MEMORY, project_id=pid,
        text=body, title=title, ts=modified, machine=machine, file_id=file_id)
    conn.execute(
        "INSERT INTO memories(project_id, root, project_key, file_id, stem, name, title, description, type, sha256, "
        "modified_at, modified_by, archived, is_index, doc_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (pid, root, key if text is not None else None, file_id, stem, stem, title, description, mtype, sha, modified,
         machine, int(archived), int(stem == "MEMORY"), doc_id))


def write_status(cfg: Config, **overrides: object) -> dict:
    """A healthy status.json relative to NOW, with top-level overrides."""
    def iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    status = {
        "written_at": iso(NOW - timedelta(minutes=1)), "started_at": iso(NOW - timedelta(days=1)),
        "version": "0.1.0", "pid": 1234, "last_reconcile_at": iso(NOW - timedelta(minutes=5)),
        "roots": {"synced": {"last_event_at": iso(NOW - timedelta(minutes=2)),
                             "last_reconcile_at": iso(NOW - timedelta(minutes=5)), "files": 12}},
        "syncthing": {"reachable": True,
                      "devices": {"laptop": {"connected": True, "last_seen": iso(NOW)},
                                  "desktop": {"connected": False, "last_seen": iso(NOW - timedelta(hours=2))}},
                      "folders": {"folder1": {"state": "idle", "need_files": 0, "completion": {"laptop": 100.0}}}},
        "embedding": {"model": "test-model", "pending_chunks": 3, "oldest_pending_ts": iso(NOW - timedelta(hours=1)),
                      "pending_sessions": 1},
        "conflicts": 0, "disk_free_gb": 120.5,
        "last_prune": {"at": iso(NOW - timedelta(hours=10)), "sessions_deleted": 2},
        "last_maintenance_at": iso(NOW - timedelta(hours=10)), "errors": [],
    }
    status.update(overrides)
    cfg.status_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.status_file.write_text(json.dumps(status), encoding="utf-8")
    return status


def build(tmp: Path) -> Fixture:
    cfg = make_config(tmp)
    sync = cfg.roots[0].path
    for key in (KEY_A, KEY_A2, KEY_B, KEY_G):
        (sync / key).mkdir(parents=True)
    conn = db.connect(cfg.index_db)
    db.init_schema(conn)
    for pid, alias, display in ((ALPHA, "alpha", "Alpha Project"), (BETA, "beta", "Beta"),
                                (CLAUDEAI, "claude-ai", "claude.ai conversations"), (GAMMA, "gamma", "Gamma"),
                                (ALPHA_DOCS, "alpha-docs", "Alpha docs")):
        conn.execute("INSERT INTO projects(id, alias, display, created_by) VALUES (?,?,?,'config')",
                     (pid, alias, display))
    for key, pid, primary in ((KEY_A, ALPHA, 1), (KEY_A2, ALPHA, 0), (KEY_B, BETA, 1), (KEY_G, GAMMA, 1)):
        conn.execute("INSERT INTO project_keys(key, project_id, is_primary) VALUES (?,?,?)", (key, pid, primary))

    # Sessions with raw transcripts (live, subagent, and archive-only recovered copy).
    fid, msgs = _write_raw(conn, sync, "synced", f"{KEY_A}/{S1}.jsonl", S1_RECORDS)
    add_session(conn, S1, ALPHA, msgs, file_id=fid, title="Fix the parser", cwd="/home/user/alpha", git_branch="main",
                cc_version="2.0.0")
    tr = sync / KEY_A / S1 / "tool-results"
    tr.mkdir(parents=True)
    (tr / "bshort01.txt").write_text("".join(f"log {i}\n" for i in range(400)) + f"token {FAKE_SECRET}\n")
    (tr / "toolu_01CCC.txt").write_text("src/a.py: TODO full grep output\n")
    fid, msgs = _write_raw(conn, sync, "synced", f"{KEY_A}/{S1}/subagents/agent-agentabc1.jsonl", SUB_RECORDS,
                           kind="subagent")
    add_session(conn, SUB, ALPHA, msgs, file_id=fid, is_subagent=1, parent_session_id=S1, agent_type="Explore",
                agent_description="find parser files")
    fid, msgs = _write_raw(conn, cfg.archive_dir / "recovered", "recovered", f"{KEY_A}/{S4}.jsonl", S4_RECORDS,
                           machine="unknown")
    add_session(conn, S4, ALPHA, msgs, file_id=fid, root="recovered", source="recovered", machine="unknown",
                title="Recover notes")

    # Index-only sessions.
    fid = _add_file(conn, "synced", f"{KEY_A}/{S2}.jsonl", "transcript", machine="desktop",
                    missing_since=ts(0, day=31))
    add_session(conn, S2, ALPHA, [
        msg(ROLE_USER, "How do I deploy alpha?", "u-21", ts(0, day=31, hour=20)),
        msg(ROLE_TOOL_CALL, "Bash: deploy", "u-22", ts(1, day=31, hour=20), tool_name="Bash", tool_use_id="toolu_02X"),
        msg(ROLE_TOOL_RESULT, "d" * 2000, "u-23", ts(2, day=31, hour=20), text_len=6000, tool_use_id="toolu_02X"),
        msg(ROLE_ASSISTANT, "Run the deploy script.", "u-24", ts(3, day=31, hour=20)),
    ], file_id=fid, machine="desktop", entrypoint="claude-vscode", title="Deploy alpha")
    add_session(conn, S7, ALPHA, [msg(ROLE_USER, "nightly lint", "u-71", ts(0, day=31, hour=10), is_prompt=False)],
                kind="automated", entrypoint="sdk-cli", title="Nightly lint")
    add_session(conn, S3, BETA, [msg(ROLE_USER, "beta pipeline run", "u-31", ts(0, day=29), is_prompt=False),
                                 msg(ROLE_ASSISTANT, "pipeline done", "u-32", ts(1, day=29))],
                kind="automated", entrypoint="sdk-py", title="Beta pipeline")
    add_session(conn, S5, ALPHA, [msg(ROLE_USER, "Ideas for the alpha slide deck?", "u-51", ts(0, day=31, hour=16)),
                                  msg(ROLE_ASSISTANT, "Three slides: problem, fix, results.", "u-52",
                                      ts(1, day=31, hour=16))],
                root="claude.ai", source="claude.ai", machine="claude.ai", entrypoint="claude.ai",
                title="Slide deck ideas")
    old = "2025-06-01T12:00:00.000Z"
    add_session(conn, S6, GAMMA, [msg(ROLE_USER, "old gamma work", "u-61", old)], title="Old gamma work")

    # Memories: two key folders for alpha, an archived one, and claude.ai memory.
    add_memory(conn, sync, ALPHA, KEY_A, "MEMORY", "- [Parser facts](parser-facts.md) — how the parser works\n"
               "- [Deploy steps](deploy-steps.md) — how to deploy\n", title="MEMORY.md (memory index)", mtype="index")
    add_memory(conn, sync, ALPHA, KEY_A, "parser-facts",
               _memory_file("parser-facts", "how the parser works", "project",
                            f"The parser lives in src/parser.py.\nTest key: {FAKE_SECRET}\n"),
               title="Parser facts", description="how the parser works", modified=ts(0, day=31))
    add_memory(conn, sync, ALPHA, KEY_A, "deploy-steps",
               _memory_file("deploy-steps", "how to deploy", "reference", "Run deploy.sh.\n"),
               title="Deploy steps", description="how to deploy", mtype="reference", machine="desktop")
    add_memory(conn, sync, ALPHA, KEY_A2, "MEMORY", "- [Windows paths](win-paths.md) — drive mapping\n",
               title="MEMORY.md (memory index)", mtype="index")
    add_memory(conn, sync, ALPHA, KEY_A2, "win-paths",
               _memory_file("win-paths", "drive mapping", "user", "C: is mounted at /mnt/c.\n"),
               title="Windows paths", description="drive mapping", mtype="user", modified=ts(0, day=20))
    add_memory(conn, sync, ALPHA, KEY_A, "old-fact--20260101-000000",
               _memory_file("old-fact", "an outdated fact", "project", "Obsolete.\n"),
               title="old-fact", description="an outdated fact", archived=True, modified=ts(0, day=2))
    add_memory(conn, sync, CLAUDEAI, "", "account-memory", None, title="claude.ai memory",
               description="Prefers concise answers.", mtype="claudeai-memory", machine="claude.ai", root="claude.ai")

    # A note written by log_note.
    note_rel = f"{KEY_A}/remote-notes/{NOTE_ID}"
    note_text = memfiles.render_note(title="Deck plan", surface="claude.ai", created="2026-01-31T20:00:00Z",
                                     project="alpha", related_sessions=[S5], summary="Planned the alpha deck.",
                                     decisions=["three slides"], next_steps=["draft slides"], open_questions=[],
                                     details=f"Shared key was {FAKE_SECRET}.")
    (sync / note_rel).parent.mkdir(parents=True)
    (sync / note_rel).write_text(note_text, encoding="utf-8")
    nfid = _add_file(conn, "synced", note_rel, "note", machine="hub")
    ndoc = db.add_doc(conn, doc_type="note", kind=db.KIND_NOTE, project_id=ALPHA, text="Planned the alpha deck.",
                      title="Deck plan", ts="2026-01-31T20:00:00Z", machine="hub", file_id=nfid)
    conn.execute("INSERT INTO notes(project_id, root, project_key, file_id, note_id, title, surface, created_at, "
                 "related_sessions, machine, doc_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (ALPHA, "synced", KEY_A, nfid, NOTE_ID, "Deck plan", "claude.ai", "2026-01-31T20:00:00Z",
                  json.dumps([S5]), "hub", ndoc))
    conn.execute("INSERT INTO imports(name, ingested_at, conversations, updated, result) VALUES (?,?,?,?,?)",
                 ("export-1.zip", "2026-01-20T10:00:00Z", 12, 3, "ok"))
    conn.commit()
    conn.close()
    write_status(cfg)
    return Fixture(cfg=cfg, sync=sync)
