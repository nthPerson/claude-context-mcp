import hashlib
import json
import re
from datetime import timedelta
from pathlib import Path

import pytest
import svc_fixtures as F
from svc_fixtures import NOW

from claude_context.config import Config
from claude_context.service import SEP, HubService, ToolError


@pytest.fixture
def fx(tmp_path: Path) -> F.Fixture:
    return F.build(tmp_path)


@pytest.fixture
def svc(fx: F.Fixture) -> HubService:
    return HubService(fx.cfg, clock=lambda: NOW, auth_failures=lambda: 3, started_at=NOW - timedelta(hours=2))


def body(page: str) -> str:
    """The paged part of a read_session/read_message result (between header and footer)."""
    return page.split(SEP)[1]


def ids(text: str) -> list[str]:
    return re.findall(r"`([0-9a-f]{8}-[0-9a-f-]{27}(?::\w+)?)`", text)


def queued(cfg: Config) -> set[str]:
    return {json.loads(p.read_text())["rel_path"] for p in cfg.queue_dir.glob("*.json")}


# --- general -------------------------------------------------------------------------------


def test_missing_index(tmp_path: Path):
    cfg = F.make_config(tmp_path)
    cfg.roots[0].path.mkdir(parents=True)
    svc = HubService(cfg, clock=lambda: NOW)
    for call in (svc.list_projects, svc.hub_status, lambda: svc.read_session(session_id=F.S1),
                 lambda: svc.log_note(project="x", title="t", summary="s", surface="other")):
        with pytest.raises(ToolError, match="index not built yet"):
            call()


def test_max_chars_cap_and_truncation(svc: HubService):
    with pytest.raises(ToolError, match="exceeds the cap"):
        svc.list_projects(max_chars=100_001)
    with pytest.raises(ToolError, match="at least"):
        svc.read_memory(project="alpha", name="parser-facts", max_chars=10)
    out = svc.project_brief(project="alpha", max_chars=1000)
    assert len(out) <= 1000
    assert out.rstrip().endswith("raise max_chars or lower recent_sessions]")


def test_project_resolution_errors(svc: HubService):
    with pytest.raises(ToolError, match="ambiguous; candidates: alpha, alpha-docs"):
        svc.project_brief(project="alp")
    with pytest.raises(ToolError, match=r"unknown project 'nope'; available: alpha, alpha-docs, beta"):
        svc.list_memories(project="nope")
    assert "Alpha Project" in svc.project_brief(project="ALPHA PROJECT")  # display name, any case
    assert "# Memories of alpha" in svc.list_memories(project=F.KEY_A2)  # raw key


def test_since_until_validation(svc: HubService):
    with pytest.raises(ToolError, match="invalid since"):
        svc.recent_activity(since="last tuesday")
    with pytest.raises(ToolError, match="invalid until"):
        svc.list_sessions(until="2026-13-01")
    # Bare dates are local (PST) days; until is inclusive of the whole day.
    out = svc.list_sessions(since="2026-01-30", until="2026-01-30")
    assert ids(out) == [F.S1]


# --- projects ------------------------------------------------------------------------------


def test_list_projects(svc: HubService):
    out = svc.list_projects()
    assert "| alpha | Alpha Project | -home-user-alpha, -mnt-c-work-alpha | claude.ai, recovered, synced |" in out
    alpha = next(line for line in out.splitlines() if line.startswith("| alpha |"))
    assert "| 4 | 1 | 3 | 1 |" in alpha  # interactive, automated, memories, notes
    assert "desktop, laptop" in alpha
    assert "| gamma |" not in out and "2 project(s) with no activity" in out
    assert "| gamma |" in svc.list_projects(include_inactive=True)


def test_project_brief_sections_and_filters(svc: HubService):
    out = svc.project_brief(project="alpha")
    assert "Hub health" not in out
    order = ["# Alpha Project", "## MEMORY.md (`-home-user-alpha`)", "## MEMORY.md (`-mnt-c-work-alpha`)",
             "## Recently modified memories", "## Last 5 sessions", "## Recent notes",
             "## Recent claude.ai conversations"]
    positions = [out.index(h) for h in order]
    assert positions == sorted(positions)
    sessions = out[out.index("## Last 5"):out.index("## Recent notes")]
    assert ids(sessions) == [F.S2, F.S1, F.S4]  # newest first, no automated, no claude.ai, no subagents
    assert "final reply: The parser is fixed" in sessions
    assert ids(out[out.index("## Recent claude.ai"):]) == [F.S5]
    assert F.NOTE_ID in out and "`parser-facts` Parser facts" in out
    assert "2026-01-30 11:00 PST" in sessions  # rendered in the configured zone
    with_auto = svc.project_brief(project="alpha", include_automated=True, recent_sessions=2)
    assert ids(with_auto[with_auto.index("## Last 2"):with_auto.index("## Recent notes")]) == [F.S2, F.S7]


def test_health_block_prepended(fx: F.Fixture, svc: HubService):
    F.write_status(fx.cfg, written_at=(NOW - timedelta(hours=2)).isoformat())
    conn = fx.rw()
    conn.execute("INSERT INTO conflicts(root, rel_path, project_id, seen_at) VALUES "
                 "('synced', '-home-user-alpha/memory/x.sync-conflict-1.md', 1, 'now')")
    conn.commit()
    conn.close()
    for out in (svc.project_brief(project="alpha"), svc.recent_activity()):
        assert out.startswith("## ⚠ Hub health\n- indexer heartbeat is 2h 0m old")
        assert "conflict file" in out
    assert "conflict file in this project" in svc.project_brief(project="alpha")
    assert "- ⚠ indexer heartbeat" in svc.hub_status()


def test_recent_activity(svc: HubService):
    out = svc.recent_activity()
    assert out.startswith("# Recent activity since 2026-01-24 14:00 PST")
    assert ids(out) == []  # ids are plain (not backticked) in the table
    for expected in (F.S2, F.S5, F.S1, F.S4, F.NOTE_ID, "| memory | laptop | synced | Parser facts | parser-facts |"):
        assert expected in out
    for hidden in (F.S3, F.S7, F.S6, F.SUB):
        assert hidden not in out
    when = [line.split(" | ")[0] for line in out.splitlines() if line.startswith("| 2026")]
    assert len(when) == 8 and when == sorted(when, reverse=True)
    assert F.S7 in svc.recent_activity(include_automated=True)
    desktop = svc.recent_activity(machine="desktop")
    assert F.S2 in desktop and "deploy-steps" in desktop and F.S1 not in desktop
    beta = svc.recent_activity(project="beta", include_automated=True)
    assert F.S3 in beta and F.S1 not in beta
    assert "(2 items)" in svc.recent_activity(limit=2)
    assert "No activity" in svc.recent_activity(since="1h")


# --- search --------------------------------------------------------------------------------


def test_search_renders_hits_with_anchors(svc: HubService):
    out = svc.search(query="parser", mode="keyword")
    assert f'read_message(session_id="{F.S1}", uuid="u-01")' in out
    assert 'read_memory(project="alpha", name="parser-facts")' in out
    assert F.FAKE_SECRET not in out
    assert "machine laptop" in out and "source claude-code" in out
    beta = svc.search(query="pipeline", mode="keyword", include_automated=True, project="beta")
    assert F.S3 in beta
    assert "No hits" in svc.search(query="pipeline", mode="keyword", project="beta")  # automated hidden
    note = svc.search(query="deck", mode="keyword", kinds=["note"])
    assert f'read_note(project="alpha", note_id="{F.NOTE_ID}")' in note


@pytest.mark.parametrize("kwargs, match", [
    ({"query": " "}, "query must not be empty"),
    ({"query": "x", "mode": "fuzzy"}, "mode must be one of"),
    ({"query": "x", "limit": 51}, "limit must be"),
    ({"query": "x", "kinds": ["bogus"]}, "unknown kinds"),
    ({"query": "x", "since": "soon"}, "invalid since"),
])
def test_search_validation(svc: HubService, kwargs, match):
    with pytest.raises(ToolError, match=match):
        svc.search(**kwargs)


# --- sessions ------------------------------------------------------------------------------


def test_list_sessions_filters(svc: HubService):
    assert ids(svc.list_sessions()) == [F.S2, F.S5, F.S1, F.S4, F.S6]
    assert ids(svc.list_sessions(kind="automated")) == [F.S7, F.S3]
    assert ids(svc.list_sessions(kind="all", project="beta")) == [F.S3]
    assert ids(svc.list_sessions(source="recovered")) == [F.S4]
    assert ids(svc.list_sessions(machine="DESKTOP")) == [F.S2]
    assert ids(svc.list_sessions(since="2d")) == [F.S2, F.S5, F.S1]
    out = svc.list_sessions(project="alpha")
    assert ("| 2026-01-30 10:00 PST → 11:00 PST | Fix the parser | alpha | laptop | claude-code | "
            "cli (interactive) | 1/2/3 |") in out
    with pytest.raises(ToolError, match="kind must be one of"):
        svc.list_sessions(kind="subagent")
    with pytest.raises(ToolError, match="invalid cursor"):
        svc.list_sessions(cursor="garbage!")


@pytest.mark.parametrize("kwargs", [{"limit": 2}, {"max_chars": 1000}])
def test_list_sessions_paging_round_trip(svc: HubService, kwargs):
    expected = ids(svc.list_sessions(kind="all", limit=50))
    got, cursor, pages = [], None, 0
    while True:
        out = svc.list_sessions(kind="all", cursor=cursor, **kwargs)
        assert len(out) <= kwargs.get("max_chars", 20_000)
        got += ids(out)
        pages += 1
        m = re.search(r'cursor="([^"]+)"', out)
        if not m:
            break
        cursor = m.group(1)
    assert got == expected and pages > 1


def test_read_session_conversation(svc: HubService):
    out = svc.read_session(session_id=F.S1)
    assert out.startswith("# Fix the parser\n")
    assert "- machine: laptop · source: claude-code · root: synced · entrypoint: cli (interactive)" in out
    assert "cwd: /home/user/alpha · branch: main" in out
    assert f"subagents (1; include_subagents=true appends them): `{F.SUB}` Explore" in out
    text = body(out)
    assert "#### user · 2026-01-30 10:00 PST · seq 0 · `u-01`\nPlease fix the parser bug" in text
    assert "The parser is fixed" in text
    assert "→" not in text and "result" not in text and "Found src/parser.py" not in text
    assert F.THINKING not in out
    assert out.endswith("(end of transcript)\n")


def test_read_session_tools_and_full(svc: HubService):
    tools = body(svc.read_session(session_id=F.S1, detail="tools"))
    assert "- → Bash: Show parser · seq 2 · `toolu_01AAA`" in tools
    assert "- ✓ result · 4.3k chars · seq 3 · `toolu_01AAA`" in tools
    assert "output line 0000" not in tools
    full = body(svc.read_session(session_id=F.S1, detail="full"))
    assert "output line 0000" in full
    assert f'read_message(session_id="{F.S1}", tool_use_id="toolu_01AAA") returns all of it' in full
    assert F.FAKE_SECRET not in full and F.THINKING not in full


def test_read_session_ids_and_prefixes(svc: HubService):
    assert svc.read_session(session_id="11111111").startswith("# Fix the parser")  # parent wins over subagent
    sub = svc.read_session(session_id=F.SUB)
    assert f"subagent (Explore: find parser files) of session `{F.S1}`" in sub
    assert "Find the parser files" in sub
    with pytest.raises(ToolError, match=f"ambiguous; candidates: {F.S2}, {F.S7}"):
        svc.read_session(session_id="22222222")
    assert "Deploy alpha" in svc.read_session(session_id="22222222-b")
    with pytest.raises(ToolError, match="at least 8"):
        svc.read_session(session_id="1111")
    with pytest.raises(ToolError, match="not found"):
        svc.read_session(session_id="99999999")


def test_read_session_flags_and_subagents(svc: HubService):
    assert "flags: missing upstream since 2026-01-31 10:00 PST" in svc.read_session(session_id=F.S2)
    assert "flags: recovered" in svc.read_session(session_id=F.S4)
    out = svc.read_session(session_id=F.S1, include_subagents=True)
    assert f"## Subagent `{F.SUB}` · Explore: find parser files" in out
    assert out.index("The parser is fixed") < out.index("Found src/parser.py")


@pytest.mark.parametrize("detail, max_chars", [("conversation", 1000), ("tools", 1000), ("full", 3500)])
def test_read_session_paging_round_trip(svc: HubService, detail, max_chars):
    full = svc.read_session(session_id=F.S1, detail=detail, include_subagents=True, max_chars=100_000)
    pages, cursor = [], None
    while True:
        page = svc.read_session(session_id=F.S1, detail=detail, include_subagents=True, cursor=cursor,
                                max_chars=max_chars)
        assert len(page) <= max_chars
        pages.append(page)
        m = re.search(r'cursor="([^"]+)"', page)
        if not m:
            break
        cursor = m.group(1)
    assert len(pages) > 1
    assert "".join(body(p) for p in pages) == body(full)
    assert pages[1].startswith("# Fix the parser (continued)")


def test_read_session_oversized_message_is_truncated(svc: HubService):
    page = svc.read_session(session_id=F.S1, detail="full", max_chars=1000)
    pages = [page]
    while m := re.search(r'cursor="([^"]+)"', pages[-1]):
        pages.append(svc.read_session(session_id=F.S1, cursor=m.group(1), max_chars=1000))
    big = next(p for p in pages if "message truncated" in p)
    assert f'read_message(session_id="{F.S1}", tool_use_id="toolu_01AAA") returns all of it' in big
    assert body(big).startswith("- ✓ result")  # alone on its page
    assert "The parser is fixed" in "".join(pages)  # paging continued past it
    with pytest.raises(ToolError, match="invalid cursor"):
        svc.read_session(session_id=F.S2, cursor=re.search(r'cursor="([^"]+)"', page).group(1))


# --- messages ------------------------------------------------------------------------------


def test_read_message_full_tool_result_from_raw_line(svc: HubService):
    out = svc.read_message(session_id=F.S1, tool_use_id="toolu_01AAA", max_chars=100_000)
    assert "content from: raw transcript line" in out
    assert "[seq 2 · tool_call · Bash]\nBash: Show parser" in out
    assert "output line 0249" in out and "end of output" in out
    assert F.FAKE_SECRET not in out and "[REDACTED:anthropic_key]" in out
    assert out.endswith("(end of message)\n")


def test_read_message_never_returns_thinking(svc: HubService):
    out = svc.read_message(session_id=F.S1, uuid="u-02")
    assert "Looking at the parser now." in out and "Bash: Show parser" in out
    assert F.THINKING not in out


def test_read_message_persisted_tool_results(svc: HubService):
    out = svc.read_message(session_id=F.S1, tool_use_id="toolu_01BBB", max_chars=100_000)
    assert "tool-results file bshort01.txt" in out and "log 399" in out
    assert F.FAKE_SECRET not in out and "[REDACTED:anthropic_key]" in out
    out = svc.read_message(session_id=F.S1, tool_use_id="toolu_01CCC")
    assert "tool-results file toolu_01CCC.txt" in out and "TODO full grep output" in out


def test_read_message_offset_paging(svc: HubService):
    whole = body(svc.read_message(session_id=F.S1, tool_use_id="toolu_01AAA", max_chars=100_000))
    parts, offset = [], 0
    while True:
        page = svc.read_message(session_id=F.S1, tool_use_id="toolu_01AAA", offset=offset, max_chars=1000)
        assert len(page) <= 1000
        parts.append(body(page))
        m = re.search(r"offset=(\d+)\)", page)
        if not m:
            break
        offset = int(m.group(1))
    assert len(parts) > 2 and "".join(parts) == whole
    with pytest.raises(ToolError, match="past the end"):
        svc.read_message(session_id=F.S1, tool_use_id="toolu_01AAA", offset=len(whole))


def test_read_message_fallbacks(svc: HubService):
    out = svc.read_message(session_id=F.S2, tool_use_id="toolu_02X")
    assert "truncated indexed copy (2000 of 6000 chars; raw source unavailable)" in out
    out = svc.read_message(session_id=F.S5, uuid="u-52")
    assert "indexed copy (complete)" in out and "Three slides" in out
    out = svc.read_message(session_id=F.S4, uuid="u-42")  # recovered: raw copy only in the archive
    assert "raw transcript line" in out and "Recovered them from backups." in out
    sub = svc.read_message(session_id=F.SUB, uuid="u-s2")
    assert "raw transcript line" in sub and "Found src/parser.py." in sub


def test_read_message_errors(svc: HubService):
    with pytest.raises(ToolError, match="exactly one"):
        svc.read_message(session_id=F.S1)
    with pytest.raises(ToolError, match="exactly one"):
        svc.read_message(session_id=F.S1, uuid="u-01", tool_use_id="toolu_01AAA")
    with pytest.raises(ToolError, match="no message with uuid='nope'"):
        svc.read_message(session_id=F.S1, uuid="nope")


# --- memories and notes --------------------------------------------------------------------


def test_list_memories(fx: F.Fixture, svc: HubService):
    out = svc.list_memories(project="alpha")
    raw = (fx.sync / F.KEY_A / "memory" / "parser-facts.md").read_bytes()
    assert (f"| `parser-facts` | Parser facts | project | how the parser works | 2026-01-31 10:00 PST | laptop | "
            f"synced | {hashlib.sha256(raw).hexdigest()} | -home-user-alpha |") in out
    assert "`win-paths`" in out and "-mnt-c-work-alpha" in out
    assert "old-fact" not in out and "MEMORY" not in out.split("\n\n")[1]
    archived = svc.list_memories(project="alpha", include_archived=True)
    assert "`old-fact--20260101-000000`" in archived and "| yes |" in archived
    assert "`account-memory`" in svc.list_memories(project="claude-ai")


def test_read_memory(fx: F.Fixture, svc: HubService):
    raw = (fx.sync / F.KEY_A / "memory" / "parser-facts.md").read_bytes()
    out = svc.read_memory(project="alpha", name="parser-facts.md")
    assert f"sha256: `{hashlib.sha256(raw).hexdigest()}`" in out
    assert "modified_by: laptop" in out and "key: -home-user-alpha" in out
    assert "name: parser-facts\ndescription: how the parser works" in out
    assert F.FAKE_SECRET not in out and "[REDACTED:anthropic_key]" in out
    assert "key: -mnt-c-work-alpha" in svc.read_memory(project="alpha", name="win-paths")
    index = svc.read_memory(project="alpha", name="MEMORY.md")
    assert index.startswith("# MEMORY.md (alpha)") and "[Deploy steps](deploy-steps.md)" in index
    cai = svc.read_memory(project="claude-ai", name="account-memory")
    assert "claude.ai memory (read-only" in cai and "Prefers concise answers." in cai
    old = svc.read_memory(project="alpha", name="old-fact--20260101-000000")
    assert "archived (read-only" in old and "Obsolete." in old
    with pytest.raises(ToolError, match="no memory 'nope' in project alpha; available: deploy-steps"):
        svc.read_memory(project="alpha", name="nope")
    with pytest.raises(ToolError, match="invalid memory name"):
        svc.read_memory(project="alpha", name="../secrets")


def test_notes(svc: HubService):
    out = svc.list_notes()
    assert f"| 2026-01-31 12:00 PST | alpha | Deck plan | claude.ai | hub | synced | `{F.NOTE_ID}` |" in out
    assert "(none)" in svc.list_notes(project="beta")
    assert "(none)" in svc.list_notes(since="1h")
    note = svc.read_note(project="alpha", note_id=F.NOTE_ID.removesuffix(".md"))
    assert "## Decisions\n\n- three slides" in note and "surface: claude.ai" in note
    assert F.FAKE_SECRET not in note
    with pytest.raises(ToolError, match="no note"):
        svc.read_note(project="alpha", note_id="missing")
    with pytest.raises(ToolError, match="invalid note_id"):
        svc.read_note(project="alpha", note_id="../x")


# --- writes --------------------------------------------------------------------------------


def save(svc: HubService, **kw) -> str:
    args = {"project": "alpha", "name": "new-fact", "title": "New fact", "description": "a new fact",
            "type": "project", "body": "Fact body.", "surface": "claude.ai"}
    return svc.save_memory(**{**args, **kw})


def sha_of(out: str) -> str:
    return re.search(r"sha256: `([0-9a-f]{64})`", out).group(1)


def test_memory_write_cycle(fx: F.Fixture, svc: HubService):
    created = save(svc)
    path = fx.sync / F.KEY_A / "memory" / "new-fact.md"
    assert path.is_file() and "Memory `new-fact` created in project `alpha`" in created
    assert f"- path: {F.KEY_A}/memory/new-fact.md" in created
    assert "- MEMORY.md line: - [New fact](new-fact.md) — a new fact" in created
    sha1 = sha_of(created)
    assert sha1 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert queued(fx.cfg) == {f"{F.KEY_A}/memory/new-fact.md", f"{F.KEY_A}/memory/MEMORY.md"}

    read = svc.read_memory(project="alpha", name="new-fact")  # disk first: works before reindexing
    assert sha_of(read) == sha1 and "Fact body." in read

    with pytest.raises(ToolError, match="already exists"):
        save(svc)
    replaced = save(svc, mode="replace", expected_sha256=sha1, body="Better body.")
    sha2 = sha_of(replaced)
    assert sha2 != sha1 and "replaced" in replaced
    with pytest.raises(ToolError) as e:
        save(svc, mode="replace", expected_sha256=sha1, body="Stale write.")
    assert f"Current sha256: {sha2}" in str(e.value) and "Better body." in str(e.value)
    assert "Better body." in path.read_text()

    with pytest.raises(ToolError, match="changed since it was read"):
        svc.archive_memory(project="alpha", name="new-fact", expected_sha256=sha1, reason="old", surface="claude.ai")
    archived = svc.archive_memory(project="alpha", name="new-fact", expected_sha256=sha2, reason="superseded",
                                  surface="claude.ai")
    assert not path.exists() and "MEMORY.md line removed: yes" in archived
    dest = re.search(r"archived to: (\S+)", archived).group(1)
    assert (fx.sync / dest).is_file() and dest.startswith(f"{F.KEY_A}/memory/.archived/new-fact--")
    assert dest in queued(fx.cfg)
    assert "new-fact" not in (fx.sync / F.KEY_A / "memory" / "MEMORY.md").read_text()


def test_sha_mismatch_excerpt_is_redacted(fx: F.Fixture, svc: HubService):
    with pytest.raises(ToolError) as e:
        save(svc, name="parser-facts", mode="replace", expected_sha256="0" * 64)
    assert "[REDACTED:anthropic_key]" in str(e.value) and F.FAKE_SECRET not in str(e.value)


def test_write_to_existing_memory_in_secondary_key(fx: F.Fixture, svc: HubService):
    sha = sha_of(svc.read_memory(project="alpha", name="win-paths"))
    out = save(svc, name="win-paths", mode="append", title="", description="", type="", body="D: too.",
               expected_sha256=sha)
    assert f"- path: {F.KEY_A2}/memory/win-paths.md" in out
    assert "D: too." in (fx.sync / F.KEY_A2 / "memory" / "win-paths.md").read_text()


def test_create_project(fx: F.Fixture, svc: HubService):
    with pytest.raises(ToolError, match="create_project=true"):
        save(svc, project="deck-work")
    assert not (fx.sync / "-hub-deck-work").exists()
    with pytest.raises(ToolError, match="must match"):
        save(svc, project="Bad Alias!", create_project=True)
    with pytest.raises(ToolError, match="surface must be one of"):  # failed write leaves no folder behind
        save(svc, project="deck-work", create_project=True, surface="fax")
    assert not (fx.sync / "-hub-deck-work").exists()
    out = save(svc, project="deck-work", create_project=True)
    assert "- path: -hub-deck-work/memory/new-fact.md" in out
    # A known project without a folder in the writable root gets a hub folder too.
    assert "- path: -hub-alpha-docs/memory/new-fact.md" in save(svc, project="alpha-docs")


def test_log_note(fx: F.Fixture, svc: HubService):
    out = svc.log_note(project="alpha", title="Sprint wrap", summary="Done.", decisions=["ship it"],
                       next_steps=["tag release"], related_sessions=[F.S1], surface="desktop")
    note_id = re.search(r"note_id: `([^`]+)`", out).group(1)
    assert note_id == "2026-01-31-220000-desktop-sprint-wrap.md"
    assert f"- path: {F.KEY_A}/remote-notes/{note_id}" in out
    assert queued(fx.cfg) == {f"{F.KEY_A}/remote-notes/{note_id}"}
    note = svc.read_note(project="alpha", note_id=note_id)  # not indexed yet: read from disk
    assert "- ship it" in note and "project: alpha" in note
    with pytest.raises(ToolError, match="summary must not be empty"):
        svc.log_note(project="alpha", title="x", summary=" ", surface="desktop")


def test_writes_disabled_without_writable_root(fx: F.Fixture):
    cfg = fx.cfg.model_copy(update={"roots": [r.model_copy(update={"writable": False}) for r in fx.cfg.roots]})
    with pytest.raises(ToolError, match="writes are disabled"):
        save(HubService(cfg, clock=lambda: NOW))


# --- status --------------------------------------------------------------------------------


def test_hub_status(fx: F.Fixture, svc: HubService):
    out = svc.hub_status()
    assert "## Warnings\n- none" in out
    assert "server 0.1.0, up 2h 0m" in out and "heartbeat 2026-01-31 13:59 PST (1m ago)" in out
    assert "| synced | claude-code | yes | 10 | 1 |" in out and "| recovered | stversions | no | 1 | 0 |" in out
    assert "| claude-code | automated | no | 2 |" in out and "| claude-code | interactive | yes | 1 |" in out
    assert "memories: 4 active, 1 archived · notes: 1" in out
    assert "device desktop: disconnected" in out and "completion laptop 100%" in out
    assert "last: export-1.zip" in out and "12 conversations" in out
    assert "2 sessions deleted" in out and "recent auth failures: 3" in out
    conn = fx.rw()
    conn.execute("INSERT INTO drift(category, value, count, last_seen) "
                 "VALUES ('record_type', 'odd', 2, '2026-01-31T00:00:00Z')")
    conn.commit()
    conn.close()
    F.write_status(fx.cfg, errors=[f"boom {F.FAKE_SECRET}"])
    out = svc.hub_status()
    assert "- ⚠ unknown record types seen" in out and "| record_type | odd | 2 |" in out
    assert "## Recent indexer errors\n- boom [REDACTED:anthropic_key]" in out
