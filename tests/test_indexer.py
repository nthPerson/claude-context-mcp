"""Indexer, importer and maintenance tests on synthetic trees."""

from __future__ import annotations

import json
import os
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from claude_context import db, importer, maintenance
from claude_context.config import Config, ProjectConfig, RootConfig
from claude_context.indexer import Indexer, classify
from claude_context.projects import list_projects

KEY = "-home-user-demo"
SID = "11111111-2222-3333-4444-555555555555"


def rec(type_: str, uuid: str, content, *, ts: str, **extra) -> str:
    base = {"type": type_, "uuid": uuid, "timestamp": ts, "sessionId": SID, "cwd": "/home/user/demo",
            "gitBranch": "main", "version": "2.0.0", "entrypoint": "cli",
            "message": {"role": type_, "content": content}}
    base.update(extra)
    return json.dumps(base) + "\n"


def ts(minutes: int = 0) -> str:
    t = datetime.now(UTC) - timedelta(hours=2) + timedelta(minutes=minutes)
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


@pytest.fixture
def env(tmp_path: Path):
    root = tmp_path / "sessions"
    (root / KEY / "memory").mkdir(parents=True)
    (root / ".stversions").mkdir()
    cfg = Config(
        data_dir=tmp_path / "data",
        roots=[RootConfig(label="synced", path=root, machine="laptop", writable=True),
               RootConfig(label="recovered", path=root / ".stversions", kind="stversions")],
        projects={"demo": ProjectConfig(keys=[KEY], display="Demo project")},
    )
    cfg.embedding.enabled = False
    cfg.ensure_dirs()
    conn = db.connect(cfg.index_db)
    db.init_schema(conn)
    yield cfg, conn, root
    conn.close()


def write_transcript(root: Path, lines: list[str], sid: str = SID, key: str = KEY) -> Path:
    path = root / key / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_classify():
    assert classify(f"{KEY}/{SID}.jsonl").kind == "transcript"
    sub = classify(f"{KEY}/{SID}/subagents/agent-abc.jsonl")
    assert (sub.kind, sub.session_id, sub.parent_session_id) == ("subagent", f"{SID}:abc", SID)
    assert classify(f"{KEY}/{SID}/subagents/workflows/wf_1/agent-x.jsonl").kind == "subagent"
    assert classify(f"{KEY}/memory/MEMORY.md").kind == "memory_index"
    assert classify(f"{KEY}/memory/fact.md").kind == "memory"
    assert classify(f"{KEY}/memory/.archived/fact--20260101-000000.md").kind == "memory_archived"
    assert classify(f"{KEY}/memory/MEMORY.sync-conflict-20260101-000000-ABCDEFG.md").kind == "conflict"
    assert classify(f"{KEY}/remote-notes/2026-01-01-000000-desktop-x.md").kind == "note"
    for ignored in (f".stversions/{KEY}/{SID}.jsonl", f"{KEY}/{SID}/tool-results/toolu_1.txt",
                    f"{KEY}/memory/.syncthing.fact.md.tmp", f"{KEY}/{SID}/subagents/agent-abc.meta.json", "stray.md"):
        assert classify(ignored) is None, ignored


def test_incremental_ingest_and_partial_line(env):
    cfg, conn, root = env
    full = [rec("user", "u1", "please fix the flaky login test", ts=ts(0)),
            rec("assistant", "a1", [{"type": "thinking", "thinking": "SECRET-THOUGHT"},
                                    {"type": "text", "text": "Looking at the test now."},
                                    {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                     "input": {"command": "pytest -k login", "description": "Run login tests"}}],
                ts=ts(1)),
            rec("user", "u2", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "1 passed"}], ts=ts(2)),
            json.dumps({"type": "ai-title", "aiTitle": "Fix flaky login test", "sessionId": SID}) + "\n"]
    partial = rec("assistant", "a2", [{"type": "text", "text": "All green."}], ts=ts(3))
    path = write_transcript(root, full + [partial[:40]])
    idx = Indexer(cfg, conn)
    assert idx.reconcile().ingested == 1
    s = conn.execute("SELECT * FROM sessions WHERE id = ?", (SID,)).fetchone()
    assert (s["title"], s["kind"], s["machine"], s["n_user"], s["n_assistant"], s["n_tool_calls"]) == \
        ("Fix flaky login test", "interactive", "laptop", 1, 1, 1)
    assert s["first_prompt"] == "please fix the flaky login test" and s["next_seq"] == 4
    assert [p.alias for p in list_projects(conn)] == ["demo"]
    assert not conn.execute("SELECT 1 FROM fts_docs WHERE fts_docs MATCH '\"SECRET-THOUGHT\"'").fetchone()
    assert conn.execute("SELECT COUNT(*) FROM fts_docs WHERE fts_docs MATCH 'flaky'").fetchone()[0] == 2  # msg + title

    assert idx.reconcile().ingested == 0  # unchanged file is skipped
    path.write_text("".join(full) + partial, encoding="utf-8")  # the partial line completes
    assert idx.reconcile().ingested == 1
    s = conn.execute("SELECT next_seq, n_assistant, final_reply_excerpt FROM sessions WHERE id = ?", (SID,)).fetchone()
    assert tuple(s) == (5, 2, "All green.")
    seqs = [r[0] for r in conn.execute("SELECT seq FROM messages WHERE session_id = ? ORDER BY seq", (SID,))]
    assert seqs == [0, 1, 2, 3, 4]
    # every message points at the start of its raw line
    data = path.read_bytes()
    for r in conn.execute("SELECT line_offset FROM messages WHERE session_id = ?", (SID,)):
        assert r[0] == 0 or data[r[0] - 1:r[0]] == b"\n"


def test_rewritten_file_is_reingested(env):
    cfg, conn, root = env
    path = write_transcript(root, [rec("user", "u1", "first version", ts=ts(0)),
                                   rec("assistant", "a1", [{"type": "text", "text": "ok"}], ts=ts(1))])
    idx = Indexer(cfg, conn)
    idx.reconcile()
    path.write_text(rec("user", "u9", "second version entirely", ts=ts(5)), encoding="utf-8")
    idx.reconcile()
    texts = [r[0] for r in conn.execute(
        "SELECT f.text FROM messages m JOIN fts_docs f ON f.rowid = m.doc_id WHERE m.session_id = ?", (SID,))]
    assert texts == ["second version entirely"]


def test_automated_and_subagent_sessions(env):
    cfg, conn, root = env
    write_transcript(root, [rec("user", "u1", "nightly job", ts=ts(0), entrypoint="sdk-py")])
    sub = root / KEY / SID / "subagents" / "agent-abc.jsonl"
    sub.parent.mkdir(parents=True)
    sub.write_text(rec("user", "s1", "explore the repo", ts=ts(1), agentId="abc", isSidechain=True), encoding="utf-8")
    sub.with_name("agent-abc.meta.json").write_text(json.dumps({"agentType": "Explore", "description": "Repo scan"}))
    write_transcript(root, [rec("user", "x1", "hi", ts=ts(0), entrypoint="brand-new-surface")],
                     sid="99999999-0000-0000-0000-000000000000")
    Indexer(cfg, conn).reconcile()
    main = conn.execute("SELECT kind FROM sessions WHERE id = ?", (SID,)).fetchone()
    assert main["kind"] == "automated"
    assert conn.execute("SELECT automated FROM docs WHERE session_id = ?", (SID,)).fetchone()[0] == 1
    s = conn.execute("SELECT * FROM sessions WHERE id = ?", (f"{SID}:abc",)).fetchone()
    assert (s["is_subagent"], s["parent_session_id"], s["agent_type"], s["agent_description"]) == \
        (1, SID, "Explore", "Repo scan")
    assert [tuple(r) for r in conn.execute("SELECT category, value FROM drift")] == [("entrypoint", "brand-new-surface")]


def test_memory_notes_conflicts_and_vanishing(env):
    cfg, conn, root = env
    mem = root / KEY / "memory"
    (mem / "MEMORY.md").write_text("# Index\n- [Deploy steps](deploy_steps.md) — how to ship\n")
    (mem / "deploy_steps.md").write_text(
        "---\nname: deploy-steps\ndescription: How to ship\nmetadata:\n  type: project\n---\n\n"
        "Run the pipeline. token=abcdef1234567890abcdef\n")
    (mem / "MEMORY.sync-conflict-20260101-010101-ABCDEFG.md").write_text("- [X](x.md) — y\n")
    (mem / ".archived").mkdir()
    (mem / ".archived" / "old--20260101-000000.md").write_text("---\nname: old\ndescription: d\n---\n\nbody\n")
    notes = root / KEY / "remote-notes"
    notes.mkdir()
    (notes / "2026-01-02-030405-desktop-kickoff.md").write_text(
        "---\ntitle: Kickoff\nsurface: desktop\ncreated: '2026-01-02T03:04:05Z'\nproject: demo\n"
        "related_sessions: []\n---\n\n## Summary\n\nWe agreed on the plan.\n")
    path = write_transcript(root, [rec("user", "u1", "hello", ts=ts(0))])
    idx = Indexer(cfg, conn)
    idx.reconcile()

    m = conn.execute("SELECT * FROM memories WHERE stem = 'deploy_steps'").fetchone()
    assert (m["title"], m["name"], m["type"], m["modified_by"], m["archived"]) == \
        ("Deploy steps", "deploy-steps", "project", "laptop", 0)
    assert "abcdef1234567890abcdef" not in db.doc_text(conn, m["doc_id"])  # redacted at ingest
    archived = conn.execute("SELECT archived, doc_id FROM memories WHERE stem LIKE 'old--%'").fetchone()
    assert tuple(archived) == (1, None)
    assert conn.execute("SELECT COUNT(*) FROM memories WHERE is_index = 1").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 1
    n = conn.execute("SELECT * FROM notes").fetchone()
    assert (n["title"], n["surface"], n["note_id"]) == ("Kickoff", "desktop", "2026-01-02-030405-desktop-kickoff.md")

    # retitle through MEMORY.md; delete a memory; lose a transcript
    (mem / "MEMORY.md").write_text("- [Shipping guide](deploy_steps.md) — how to ship\n")
    (mem / "MEMORY.sync-conflict-20260101-010101-ABCDEFG.md").unlink()
    path.unlink()
    idx.reconcile()
    assert conn.execute("SELECT title FROM memories WHERE stem = 'deploy_steps'").fetchone()[0] == "Shipping guide"
    assert conn.execute("SELECT COUNT(*) FROM fts_docs WHERE fts_docs MATCH 'title:shipping'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 0
    assert conn.execute("SELECT missing_since FROM files WHERE kind = 'transcript'").fetchone()[0] is not None
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1  # the session survives

    (mem / "deploy_steps.md").unlink()
    assert idx.ingest_path(cfg.roots[0], f"{KEY}/memory/deploy_steps.md")
    assert conn.execute("SELECT COUNT(*) FROM memories WHERE stem = 'deploy_steps'").fetchone()[0] == 0


def test_recovery_and_prune(env):
    cfg, conn, root = env
    sv = root / ".stversions" / KEY
    sv.mkdir(parents=True)
    lost = "aaaaaaaa-0000-0000-0000-000000000000"
    (sv / f"{lost}~20260101-000000.jsonl").write_text(rec("user", "o1", "older copy", ts=ts(0)))
    (sv / f"{lost}~20260102-000000.jsonl").write_text(rec("user", "o1", "older copy", ts=ts(0))
                                                      + rec("user", "o2", "newest copy", ts=ts(1)))
    write_transcript(root, [rec("user", "u1", "still live", ts=ts(0))])
    (sv / f"{SID}~20260101-000000.jsonl").write_text(rec("user", "u0", "stale version of a live file", ts=ts(0)))
    (sv / "memory").mkdir()
    (sv / "memory" / "MEMORY~20260101-000000.md").write_text("old index")

    assert maintenance.recover_stversions(cfg, None, dry_run=True) == [f"recovered/{KEY}/{lost}.jsonl"]
    assert not (cfg.archive_dir / "recovered").exists()
    maintenance.recover_stversions(cfg, conn)
    s = conn.execute("SELECT source, next_seq FROM sessions WHERE id = ?", (lost,)).fetchone()
    assert tuple(s) == ("recovered", 2)
    assert maintenance.recover_stversions(cfg, conn) == []  # idempotent

    Indexer(cfg, conn).reconcile()
    assert maintenance.prune(cfg, conn, dry_run=True) == (0, 0)
    future = datetime.now(UTC) + timedelta(days=cfg.retention_days + 1)
    assert maintenance.prune(cfg, conn, dry_run=True, now=future)[0] == 2
    assert maintenance.prune(cfg, conn, now=future)[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0
    assert not (cfg.archive_dir / "recovered" / KEY / f"{lost}.jsonl").exists()
    assert (root / KEY / f"{SID}.jsonl").exists()  # live files are never deleted by the hub
    assert Indexer(cfg, conn).reconcile().ingested == 0  # and are not re-ingested after pruning


def test_old_transcripts_are_not_indexed(env):
    cfg, conn, root = env
    path = write_transcript(root, [rec("user", "u1", "ancient", ts="2020-01-01T00:00:00.000Z")])
    old = time.time() - (cfg.retention_days + 5) * 86400
    os.utime(path, (old, old))
    Indexer(cfg, conn).reconcile()
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_claudeai_import(env, tmp_path: Path):
    cfg, conn, _root = env
    cfg.claudeai_project_map = {"Demo chats": "demo"}
    drop = tmp_path / "drop"
    drop.mkdir()

    def make_zip(updated: str, reply: str) -> Path:
        conversations = [{
            "uuid": "c0000000-0000-0000-0000-000000000001", "name": "Planning the launch", "created_at": ts(0),
            "updated_at": updated, "project": {"uuid": "p1", "name": "Demo chats"},
            "chat_messages": [
                {"uuid": "m1", "sender": "human", "text": "What should the launch checklist contain?",
                 "created_at": ts(0)},
                {"uuid": "m2", "sender": "assistant", "created_at": ts(1),
                 "content": [{"type": "thinking", "thinking": "HIDDEN"}, {"type": "text", "text": reply}]}]}]
        path = drop / "export.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("conversations.json", json.dumps(conversations))
        return path

    Indexer(cfg, conn).reconcile()  # registers the config project
    result = importer.process_drop(cfg, conn, make_zip(ts(1), "Start with a rollback plan."))
    assert (result.conversations, result.updated, result.error) == (1, 1, None)
    assert not (drop / "export.zip").exists() and len(list(cfg.imports_processed_dir.glob("*.zip"))) == 1
    s = conn.execute("SELECT s.*, p.alias FROM sessions s JOIN projects p ON p.id = s.project_id").fetchone()
    assert (s["source"], s["machine"], s["alias"], s["title"], s["n_user"]) == \
        ("claude.ai", "claude.ai", "demo", "Planning the launch", 1)
    assert not conn.execute("SELECT 1 FROM fts_docs WHERE fts_docs MATCH 'HIDDEN'").fetchone()

    assert importer.import_export(cfg, conn, make_zip(ts(1), "ignored: not newer")).updated == 0
    assert importer.import_export(cfg, conn, make_zip(ts(9), "Now with a newer answer.")).updated == 1
    assert conn.execute("SELECT COUNT(*) FROM fts_docs WHERE fts_docs MATCH 'rollback'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert Indexer(cfg, conn).reconcile().removed == 0  # reconcile leaves imported sessions alone
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_attribution_is_corrected_once_syncthing_has_scanned(env):
    """Ingest usually beats Syncthing's scan, when Syncthing still names the previous device."""
    cfg, conn, root = env
    cfg.roots[0].machine = "syncthing:demo-folder"
    answers = {"modified_by": None}

    class FakeSyncthing:
        def short_id_names(self, refresh: bool = False):
            return {"AAAAAAA": "laptop", "BBBBBBB": "server"}

        def file_modified_by(self, folder, rel_path):
            return answers["modified_by"]

    mem = root / KEY / "memory" / "fact.md"
    mem.write_text("---\nname: fact\ndescription: d\n---\n\nbody\n")
    write_transcript(root, [rec("user", "u1", "hello", ts=ts(0))])
    idx = Indexer(cfg, conn, syncthing=FakeSyncthing())

    answers["modified_by"] = "AAAAAAA"  # stale: the previous version's device
    idx.reconcile()
    assert conn.execute("SELECT modified_by FROM memories WHERE stem = 'fact'").fetchone()[0] == "laptop"

    answers["modified_by"] = "BBBBBBB"  # Syncthing has scanned the new version
    assert idx.refresh_machine(cfg.roots[0], f"{KEY}/memory/fact.md")
    assert conn.execute("SELECT modified_by FROM memories WHERE stem = 'fact'").fetchone()[0] == "server"
    assert conn.execute("SELECT machine FROM docs WHERE doc_type = 'memory'").fetchone()[0] == "server"
    assert not idx.refresh_machine(cfg.roots[0], f"{KEY}/memory/fact.md")  # nothing left to correct
    # a session keeps the machine it was first attributed to
    idx.refresh_machine(cfg.roots[0], f"{KEY}/{SID}.jsonl")
    assert conn.execute("SELECT machine FROM sessions").fetchone()[0] == "laptop"
