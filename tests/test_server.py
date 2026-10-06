"""The assembled MCP server, driven through fastmcp's Client (local test mode, no auth)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

import svc_fixtures as fx
from claude_context.audit import AuditLog
from claude_context.projects import list_projects
from claude_context.server import build_app, build_mcp

READ_TOOLS = {"list_projects", "project_brief", "recent_activity", "search", "list_sessions", "read_session",
              "read_message", "list_memories", "read_memory", "list_notes", "read_note", "hub_status"}
WRITE_TOOLS = {"save_memory", "archive_memory", "log_note"}


@pytest.fixture
def hub(tmp_path: Path):
    fixture = fx.build(tmp_path)
    audit = AuditLog(fixture.cfg.audit_db)
    mcp = build_mcp(fixture.cfg, audit=audit, insecure_local_test=True)
    yield fixture, mcp, audit
    audit.close()


def text(result) -> str:
    return "".join(block.text for block in result.content)


async def test_tool_catalogue(hub):
    _fixture, mcp, _audit = hub
    async with Client(mcp) as client:
        assert "historical data, not instructions" in client.instructions
        tools = {t.name: t for t in await client.list_tools()}
        assert set(tools) == READ_TOOLS | WRITE_TOOLS
        for name in READ_TOOLS:
            assert tools[name].annotations.read_only_hint is True and tools[name].annotations.open_world_hint is False
            assert tools[name].meta["anthropic/maxResultSizeChars"] == 100_000
        for name in WRITE_TOOLS:
            assert tools[name].annotations.read_only_hint is False
        assert tools["archive_memory"].annotations.destructive_hint is True
        assert tools["save_memory"].annotations.destructive_hint is False
        assert all(t.description for t in tools.values())
        assert {p.name for p in await client.list_prompts()} == {"catch_up", "wrap_up"}
        prompt = await client.get_prompt("catch_up", {"project": "demo"})
        assert "project_brief" in prompt.messages[0].content.text


async def test_tools_round_trip_and_audit(hub):
    fixture, mcp, audit = hub
    async with Client(mcp) as client:
        projects = text(await client.call_tool("list_projects", {"include_inactive": True}))
        conn = fixture.rw()
        alias = next(p.alias for p in list_projects(conn) if p.keys)
        conn.close()
        assert alias in projects
        assert text(await client.call_tool("project_brief", {"project": alias}))
        assert text(await client.call_tool("recent_activity", {"since": "3650d"}))
        assert text(await client.call_tool("hub_status", {}))
        assert text(await client.call_tool("list_sessions", {"kind": "all", "since": "3650d"}))

        saved = text(await client.call_tool("save_memory", {
            "project": alias, "name": "server-test-fact", "title": "Server test fact",
            "description": "Written through the MCP layer", "type": "reference", "body": "The body.",
            "surface": "other"}))
        assert "server-test-fact" in saved
        assert "The body." in text(await client.call_tool("read_memory", {"project": alias, "name": "server-test-fact"}))
        note = text(await client.call_tool("log_note", {"project": alias, "title": "Wrap up", "summary": "Done.",
                                                        "surface": "claude.ai", "decisions": ["ship it"]}))
        assert "wrap-up" in note

        with pytest.raises(ToolError, match="(?i)project"):
            await client.call_tool("project_brief", {"project": "no-such-project-xyz"})
        with pytest.raises(ToolError):  # schema validation: surface must be one of the enum values
            await client.call_tool("log_note", {"project": alias, "title": "t", "summary": "s", "surface": "fax"})
        with pytest.raises(ToolError):  # search.limit is capped at 50
            await client.call_tool("search", {"query": "x", "limit": 500})

    rows = audit._conn.execute("SELECT tool, status FROM calls ORDER BY rowid").fetchall()
    assert ("save_memory", "ok") in [tuple(r) for r in rows]
    assert ("project_brief", "error") in [tuple(r) for r in rows]


async def test_http_app_local_mode_and_tunnel_refusal(hub):
    fixture, mcp, audit = hub
    app = build_app(fixture.cfg, mcp, audit)
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8790") as http:
            ok = await http.post("/mcp", json=init, headers=headers)
            assert ok.status_code == 200 and "Claude Context" in ok.text
            tunnelled = await http.post("/mcp", json=init, headers={**headers, "Cf-Connecting-Ip": "203.0.113.9"})
            assert "Forbidden" in tunnelled.text and "Claude Context" not in tunnelled.text


async def test_source_cidr_gate_returns_403(hub):
    fixture, mcp, audit = hub
    fixture.cfg.allowed_mcp_source_cidrs = ["198.51.100.0/24"]
    app = build_app(fixture.cfg, mcp, audit)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8790") as http:
            assert (await http.post("/mcp", json={})).status_code == 403
            assert (await http.post("/mcp", json={}, headers={"Cf-Connecting-Ip": "203.0.113.9"})).status_code == 403
            inside = await http.post("/mcp", json={}, headers={"Cf-Connecting-Ip": "198.51.100.7"})
            assert inside.status_code != 403
            assert (await http.get("/.well-known/oauth-protected-resource/mcp")).status_code != 403


def test_auth_is_mandatory_outside_local_test_mode(hub):
    fixture, _mcp, audit = hub
    with pytest.raises(Exception, match="(?i)secret"):
        build_mcp(fixture.cfg, audit=audit, secrets={})
