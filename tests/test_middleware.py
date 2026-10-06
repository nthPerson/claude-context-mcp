"""GuardMiddleware on its own: a stub provider authenticates every bearer token."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import TokenVerifier
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.middleware import Middleware

from claude_context import middleware as mw
from claude_context.audit import AuditLog
from claude_context.config import Config
from claude_context.middleware import FORBIDDEN, GuardMiddleware

BASE = "https://hub.example.com"
ALLOWED, DENIED = 1001, 2002
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class EveryoneVerifier(TokenVerifier):
    """Accepts any ``gh-<id>`` token, shaped like the real provider's verified token."""

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token.startswith("gh-"):
            return None
        gid = int(token[3:])
        return AccessToken(
            token=token, client_id=str(gid), scopes=[], subject=str(gid),
            claims={"sub": str(gid), "github_user_data": {"id": gid, "login": f"user{gid}"},
                    "mcp_client_id": "client-abc", "mcp_client_name": "Stub Client"},
        )


class Spy(Middleware):
    def __init__(self) -> None:
        self.methods: list[str | None] = []

    async def on_message(self, context, call_next):
        self.methods.append(context.method)
        return await call_next(context)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@asynccontextmanager
async def app_lifespan(app):
    started, stop = asyncio.Event(), asyncio.Event()

    async def run() -> None:
        async with app.lifespan(app):
            started.set()
            await stop.wait()

    task = asyncio.create_task(run())
    await asyncio.wait({task, asyncio.create_task(started.wait())}, return_when=asyncio.FIRST_COMPLETED)
    if task.done():
        task.result()
    try:
        yield
    finally:
        stop.set()
        await task


@asynccontextmanager
async def serve(cfg: Config, audit, *, auth=True, insecure=False, clock=None, extra=()):
    guard = GuardMiddleware(cfg, audit, insecure_local_test=insecure, **({"clock": clock} if clock else {}))
    mcp = FastMCP("guard-test", auth=EveryoneVerifier() if auth else None, middleware=[*extra, guard])

    @mcp.tool
    def echo(text: str) -> str:
        return text

    @mcp.tool
    def fail(reason: str) -> str:
        raise ToolError(f"boom: {reason}")

    @mcp.prompt
    def hello() -> str:
        return "hello"

    app = mcp.http_app(path="/mcp", stateless_http=True)
    async with app_lifespan(app), httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE) as c:
        yield c


def cfg_for(tmp_path: Path, **kw) -> Config:
    return Config(base_url=BASE, allowed_github_ids=[ALLOWED], data_dir=tmp_path, **kw)


async def rpc(c: httpx2.AsyncClient, method: str, params: dict | None = None, *, gid: int | None = ALLOWED,
              ip: str | None = None, notify: bool = False) -> dict | httpx2.Response:
    headers = dict(HEADERS, **({"Authorization": f"Bearer gh-{gid}"} if gid is not None else {}),
                   **({"Cf-Connecting-Ip": ip} if ip else {}), **{"User-Agent": "test-agent/1"})
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}} | ({} if notify else {"id": 7})
    r = await c.post("/mcp", headers=headers, json=body)
    if notify or r.status_code != 200:
        return r
    data = [line[5:] for line in r.text.splitlines() if line.startswith("data:")]
    return json.loads(data[-1]) if data else r.json()


async def echo(c, text="hi", **kw) -> dict:
    return await rpc(c, "tools/call", {"name": "echo", "arguments": {"text": text}}, **kw)


def calls(audit: AuditLog) -> list[tuple]:
    return audit._conn.execute(
        "SELECT github_id, client_id, client_name, cf_ip, user_agent, tool, args_redacted_trunc, result_chars, "
        "duration_ms, status, error FROM calls ORDER BY rowid").fetchall()


def denied_code(resp: dict) -> int | None:
    return resp.get("error", {}).get("code")


@pytest.fixture
def audit(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    yield log
    log.close()


# --- allowlist (layer 3) ---------------------------------------------------------------

async def test_allowlisted_call_succeeds_and_is_audited(tmp_path, audit):
    async with serve(cfg_for(tmp_path), audit) as c:
        resp = await echo(c, "hello", ip="203.0.113.7")
    assert resp["result"]["content"][0]["text"] == "hello"
    (row,) = calls(audit)
    assert row[:7] == (ALLOWED, "client-abc", "Stub Client", "203.0.113.7", "test-agent/1", "echo", '{"text": "hello"}')
    assert row[7] == 5 and row[8] >= 0 and row[9:] == ("ok", None)


async def test_middleware_alone_denies_non_allowlisted_id(tmp_path, audit):
    async with serve(cfg_for(tmp_path), audit) as c:
        resp = await echo(c, gid=DENIED)
    assert denied_code(resp) == FORBIDDEN and "not allowlisted" in resp["error"]["message"]
    (row,) = calls(audit)
    assert row[0] == DENIED and row[5] == "echo" and row[9] == "denied"


@pytest.mark.parametrize("method,params", [
    ("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}}),
    ("tools/list", {}), ("prompts/list", {}), ("prompts/get", {"name": "hello"}), ("resources/list", {}),
    ("resources/templates/list", {}), ("ping", {}),
])
async def test_every_request_method_is_guarded(tmp_path, audit, method, params):
    async with serve(cfg_for(tmp_path), audit) as c:
        assert "error" not in await rpc(c, method, params)  # allowed id passes
        assert denied_code(await rpc(c, method, params, gid=DENIED)) == FORBIDDEN
    assert [r[5] for r in calls(audit) if r[9] == "denied"] == [method]


async def test_notifications_are_guarded(tmp_path, audit, caplog):
    with caplog.at_level(logging.WARNING, logger="claude_context.middleware"):
        async with serve(cfg_for(tmp_path), audit) as c:
            r = await rpc(c, "notifications/initialized", gid=DENIED, notify=True)
            assert r.status_code == 202
            await asyncio.sleep(0.05)
    assert "denied notifications/initialized" in caplog.text


async def test_on_message_fires_once_per_message(tmp_path, audit):
    spy = Spy()
    async with serve(cfg_for(tmp_path), audit, extra=[spy]) as c:
        await rpc(c, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}})
        await rpc(c, "tools/list")
        await echo(c)
        await rpc(c, "notifications/initialized", notify=True)
        await asyncio.sleep(0.05)
    assert spy.methods == ["initialize", "tools/list", "tools/call", "notifications/initialized"]


async def test_unauthenticated_requests_never_reach_middleware(tmp_path, audit):
    spy = Spy()
    async with serve(cfg_for(tmp_path), audit, extra=[spy]) as c:
        for gid in (None,):
            r = await rpc(c, "tools/call", {"name": "echo", "arguments": {"text": "x"}}, gid=gid)
            assert r.status_code == 401
        r = await c.post("/mcp", headers=HEADERS | {"Authorization": "Bearer bogus"}, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert r.status_code == 401
    assert spy.methods == [] and calls(audit) == []


async def test_token_without_github_id_is_denied(tmp_path, audit):
    class NoId(EveryoneVerifier):
        async def verify_token(self, token):
            t = await super().verify_token(token)
            return t.model_copy(update={"claims": {"sub": "1001", "github_user_data": {"login": "x"}}})

    mcp_cfg = cfg_for(tmp_path)
    mcp = FastMCP("x", auth=NoId(), middleware=[GuardMiddleware(mcp_cfg, audit)])
    mcp.tool(lambda: "ok", name="noop")
    app = mcp.http_app(path="/mcp", stateless_http=True)
    async with app_lifespan(app), httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE) as c:
        resp = await rpc(c, "tools/call", {"name": "noop", "arguments": {}})
    assert denied_code(resp) == FORBIDDEN and "no verified GitHub identity" in resp["error"]["message"]


# --- source CIDRs ----------------------------------------------------------------------

@pytest.mark.parametrize("ip,ok", [
    (None, False), ("203.0.113.5", False), ("160.79.104.10", True), ("160.79.111.255", True), ("160.79.112.0", False),
    ("2001:db8::1", True), ("2001:db9::1", False), ("not-an-ip", False),
])
async def test_cidr_check(tmp_path, audit, ip, ok):
    cfg = cfg_for(tmp_path, allowed_mcp_source_cidrs=["160.79.104.0/21", "2001:db8::/32"])
    async with serve(cfg, audit) as c:
        resp = await echo(c, ip=ip)
    if ok:
        assert "result" in resp
    else:
        assert denied_code(resp) == FORBIDDEN
        (row,) = calls(audit)
        assert row[9] == "denied" and row[3] == ip


def test_invalid_cidr_fails_at_startup(tmp_path, audit):
    with pytest.raises(ValueError):
        GuardMiddleware(cfg_for(tmp_path, allowed_mcp_source_cidrs=["300.1.1.0/24"]), audit)


# --- rate limit ------------------------------------------------------------------------

async def test_rate_limit_trips_and_recovers(tmp_path, audit):
    clock = FakeClock()
    async with serve(cfg_for(tmp_path, rate_limit_per_minute=3), audit, clock=clock) as c:
        for _ in range(3):
            assert "result" in await echo(c)
        tripped = await echo(c)
        assert denied_code(tripped) == -32000 and "Rate limit" in tripped["error"]["message"]
        clock.now += 30
        assert denied_code(await echo(c)) == -32000  # still inside the rolling minute
        clock.now += 31
        assert "result" in await echo(c)
    assert [r[9] for r in calls(audit)] == ["ok", "ok", "ok", "denied", "denied", "ok"]
    assert calls(audit)[3][10] == "rate limit exceeded"


async def test_rate_limit_is_per_identity(tmp_path, audit):
    cfg = cfg_for(tmp_path, rate_limit_per_minute=1).model_copy(update={"allowed_github_ids": [ALLOWED, 1002]})
    async with serve(cfg, audit, clock=FakeClock()) as c:
        assert "result" in await echo(c)
        assert "result" in await echo(c, gid=1002)
        assert denied_code(await echo(c)) == -32000


# --- auditing --------------------------------------------------------------------------

async def test_tool_error_is_audited(tmp_path, audit):
    async with serve(cfg_for(tmp_path), audit) as c:
        resp = await rpc(c, "tools/call", {"name": "fail", "arguments": {"reason": "nope"}})
    assert resp["result"]["isError"] is True
    (row,) = calls(audit)
    assert row[5] == "fail" and row[9] == "error" and "boom: nope" in row[10]


async def test_arguments_are_redacted_and_truncated(tmp_path, audit, monkeypatch):
    monkeypatch.setattr(mw, "_redactor", lambda: lambda s: s.replace("hunter2", "[REDACTED:password]"))
    async with serve(cfg_for(tmp_path), audit) as c:
        await echo(c, "pw=hunter2 " + "x" * 2000)
    args = calls(audit)[0][6]
    assert "hunter2" not in args and "[REDACTED:password]" in args and len(args) <= 500


def test_redactor_uses_redact_module_or_identity(monkeypatch):
    mw._redactor.cache_clear()
    try:
        from claude_context.redact import redact
        assert mw._redactor() is redact
        mw._redactor.cache_clear()
        monkeypatch.setitem(sys.modules, "claude_context.redact", None)  # import now fails
        assert mw._redactor()("abc") == "abc"
    finally:
        mw._redactor.cache_clear()


def test_redaction_failure_stores_placeholder(monkeypatch):
    def broken(_text):
        raise RuntimeError("x")

    monkeypatch.setattr(mw, "_redactor", lambda: broken)
    assert mw._redact_args({"a": "secret"}) == "[redaction failed]"


async def test_audit_failure_never_breaks_a_call(tmp_path, audit, monkeypatch):
    def broken(**_kw):
        raise OSError("disk full")

    monkeypatch.setattr(audit, "log_call", broken)
    async with serve(cfg_for(tmp_path), audit) as c:
        assert (await echo(c, "still works"))["result"]["content"][0]["text"] == "still works"
        assert denied_code(await echo(c, gid=DENIED)) == FORBIDDEN


# --- insecure local test mode ----------------------------------------------------------

async def test_insecure_mode_refuses_tunnelled_requests(tmp_path, audit, caplog):
    with caplog.at_level(logging.WARNING, logger="claude_context.middleware"):
        async with serve(cfg_for(tmp_path), audit, auth=False, insecure=True) as c:
            assert "result" in await echo(c, gid=None)
            resp = await echo(c, gid=None, ip="198.51.100.4")
            assert denied_code(resp) == FORBIDDEN and "tunnel" in resp["error"]["message"]
    assert "INSECURE LOCAL TEST MODE" in caplog.text
    assert [r[9] for r in calls(audit)] == ["ok", "denied"]


async def test_without_auth_and_without_insecure_flag_everything_is_denied(tmp_path, audit):
    async with serve(cfg_for(tmp_path), audit, auth=False) as c:
        assert denied_code(await echo(c, gid=None)) == FORBIDDEN


async def test_insecure_flag_does_not_relax_the_allowlist_for_tokens(tmp_path, audit):
    async with serve(cfg_for(tmp_path), audit, insecure=True) as c:  # auth still configured
        assert "result" in await echo(c)
        assert denied_code(await echo(c, gid=DENIED)) == FORBIDDEN
