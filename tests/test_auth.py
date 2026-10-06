"""End-to-end OAuth tests against an in-process fake GitHub (no network)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx2
import pytest
from cryptography.fernet import Fernet
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from claude_context import auth as hub_auth
from claude_context.audit import AuditLog
from claude_context.auth import AuthConfigError, build_auth_provider, local_test_mode_allowed
from claude_context.config import Config
from claude_context.middleware import GuardMiddleware

BASE = "https://hub.example.com"
CLAUDE_CB = "https://claude.ai/api/mcp/auth_callback"
ALLOWED, DENIED = 1001, 2002
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


# --- fake GitHub -----------------------------------------------------------------------

@dataclass
class FakeGitHub:
    """GitHub's token endpoint and /user, keyed by the codes the test hands to the callback."""

    issue_refresh: bool = True
    users: dict[str, dict] = field(default_factory=dict)  # upstream access token -> /user payload
    codes: dict[str, dict] = field(default_factory=dict)  # upstream code -> /user payload
    refresh: dict[str, dict] = field(default_factory=dict)  # upstream refresh token -> /user payload
    user_status: int = 200  # set to simulate a GitHub /user outage

    def code_for(self, user: dict) -> str:
        code = "gh-code-" + secrets.token_hex(8)
        self.codes[code] = dict(user)
        return code

    def _issue(self, user: dict) -> dict:
        access = "gho_" + secrets.token_hex(16)
        self.users[access] = user
        body = {"access_token": access, "token_type": "bearer", "scope": ""}
        if self.issue_refresh:  # GitHub App style expiring user tokens
            ref = "ghr_" + secrets.token_hex(16)
            self.refresh[ref] = user
            body |= {"refresh_token": ref, "expires_in": 28800, "refresh_token_expires_in": 15897600}
        return body

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        if url == "https://github.com/login/oauth/access_token":
            form = parse_qs(request.content.decode())
            if form["grant_type"] == ["authorization_code"]:
                user = self.codes.pop(form["code"][0], None)
            else:
                user = self.refresh.pop(form["refresh_token"][0], None)
            if user is None:
                return httpx2.Response(200, json={"error": "bad_verification_code"})
            return httpx2.Response(200, json=self._issue(user))
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if token not in self.users:
            return httpx2.Response(401, json={"message": "Bad credentials"})
        if url == "https://api.github.com/user":
            return httpx2.Response(self.user_status, json=self.users[token])
        if url == "https://api.github.com/user/repos":
            return httpx2.Response(200, json=[], headers={"X-OAuth-Scopes": ""})
        return httpx2.Response(404)


def user(gid: int | None, login: str = "someone") -> dict:
    return {"login": login} | ({"id": gid} if gid is not None else {})


# --- harness ---------------------------------------------------------------------------

def make_secrets() -> dict[str, str]:
    return {
        "GITHUB_CLIENT_ID": "Iv-test-" + secrets.token_hex(6),
        "GITHUB_CLIENT_SECRET": secrets.token_urlsafe(30),
        "JWT_SIGNING_KEY": secrets.token_urlsafe(48),
        "STORAGE_ENCRYPTION_KEY": Fernet.generate_key().decode(),
    }


def make_cfg(tmp_path: Path, ids: list[int] = [ALLOWED], **kw) -> Config:  # noqa: B006
    return Config(base_url=BASE, allowed_github_ids=ids, data_dir=tmp_path / "data", **kw)


@dataclass
class Hub:
    client: httpx2.AsyncClient
    provider: hub_auth.AllowlistedGitHubProvider
    audit: AuditLog
    github: FakeGitHub


@asynccontextmanager
async def app_lifespan(app):
    """Run the app lifespan in its own task (pytest fixtures set up and tear down in different tasks)."""
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
async def running_hub(cfg: Config, sec: dict[str, str], github: FakeGitHub):
    audit = AuditLog(cfg.audit_db)
    gh_http = httpx2.AsyncClient(transport=httpx2.MockTransport(github.handler))
    provider = build_auth_provider(cfg, sec, audit, http_client=gh_http)
    mcp = FastMCP("hub-test", auth=provider, middleware=[GuardMiddleware(cfg, audit)])

    @mcp.tool
    def echo(text: str) -> str:
        return text

    app = mcp.http_app(path="/mcp", stateless_http=True)
    async with app_lifespan(app), httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE) as client:
        try:
            yield Hub(client, provider, audit, github)
        finally:
            await gh_http.aclose()
            audit.close()


@pytest.fixture
def sec() -> dict[str, str]:
    return make_secrets()


@pytest.fixture
def github() -> FakeGitHub:
    return FakeGitHub()


@pytest.fixture
async def hub(tmp_path, sec, github):
    async with running_hub(make_cfg(tmp_path), sec, github) as h:
        yield h


def pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


async def register(c: httpx2.AsyncClient, redirect_uri: str = CLAUDE_CB) -> httpx2.Response:
    return await c.post("/register", json={
        "redirect_uris": [redirect_uri], "client_name": "Test Connector", "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"], "token_endpoint_auth_method": "none",
    })


async def start_login(h: Hub, *, scope: str | None = None) -> tuple[str, str, str]:
    """Register, authorize, approve consent; returns (client_id, verifier, GitHub authorize URL)."""
    r = await register(h.client)
    assert r.status_code == 201, r.text
    client_id = r.json()["client_id"]
    verifier, challenge = pkce()
    q = {"response_type": "code", "client_id": client_id, "redirect_uri": CLAUDE_CB, "state": "st-123",
         "code_challenge": challenge, "code_challenge_method": "S256", "resource": f"{BASE}/mcp"}
    if scope:
        q["scope"] = scope
    r = await h.client.get("/authorize?" + urlencode(q))
    assert r.status_code == 302, r.text
    consent = r.headers["location"]
    assert consent.startswith(f"{BASE}/consent?txn_id=")
    upstream = await approve(h.client, consent)
    return client_id, verifier, upstream


async def approve(c: httpx2.AsyncClient, consent_url: str) -> str:
    r = await c.get(consent_url)
    assert r.status_code == 200
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', r.text).group(1)
    txn = parse_qs(urlsplit(consent_url).query)["txn_id"][0]
    r = await c.post("/consent", data={"txn_id": txn, "csrf_token": csrf, "action": "approve"})
    assert r.status_code == 302, r.text
    return r.headers["location"]


async def github_callback(h: Hub, upstream_url: str, who: dict) -> httpx2.Response:
    state = parse_qs(urlsplit(upstream_url).query)["state"][0]
    return await h.client.get(f"/auth/callback?code={h.github.code_for(who)}&state={state}")


async def login(h: Hub, who: dict | None = None) -> tuple[str, dict]:
    """Full flow; returns (client_id, token response)."""
    client_id, verifier, upstream = await start_login(h)
    r = await github_callback(h, upstream, who or user(ALLOWED, "alice"))
    assert r.status_code == 302, r.text
    params = parse_qs(urlsplit(r.headers["location"]).query)
    assert r.headers["location"].startswith(CLAUDE_CB) and params["state"] == ["st-123"]
    r = await h.client.post("/token", data={
        "grant_type": "authorization_code", "code": params["code"][0], "redirect_uri": CLAUDE_CB,
        "client_id": client_id, "code_verifier": verifier,
    })
    assert r.status_code == 200, r.text
    return client_id, r.json()


async def call_echo(c: httpx2.AsyncClient, token: str | None, text: str = "hi", **headers: str) -> httpx2.Response:
    h = MCP_HEADERS | headers | ({"Authorization": f"Bearer {token}"} if token else {})
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": text}}}
    return await c.post("/mcp", headers=h, json=body)


def rpc_result(r: httpx2.Response) -> dict:
    assert r.status_code == 200, (r.status_code, r.text)
    data = [line[5:] for line in r.text.splitlines() if line.startswith("data:")]
    return json.loads(data[-1]) if data else r.json()


def auth_events(audit: AuditLog) -> list[tuple]:
    return audit._conn.execute("SELECT event, github_login, github_id, ok, reason FROM auth_events").fetchall()


STORED_SECRETS = ("mcp_authorization_codes", "mcp_upstream_tokens", "mcp_jti_mappings", "mcp_refresh_tokens")


def files_in(cfg: Config, collection: str) -> list[Path]:
    """Entries of one FastMCP storage collection (FileTreeStore: <collection>-<hash>/<key>.json)."""
    return [p for d in cfg.oauth_dir.glob(f"S_{collection}-*") if d.is_dir() for p in d.iterdir()]


# --- discovery / Claude connector requirements ----------------------------------------

async def test_unauthenticated_mcp_gets_401_with_resource_metadata(hub):
    r = await call_echo(hub.client, None)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")
    assert f'resource_metadata="{BASE}/.well-known/oauth-protected-resource/mcp"' in r.headers["www-authenticate"]
    # The guard middleware (which audits every call) was never reached.
    assert hub.audit._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0


async def test_invalid_bearer_gets_401(hub):
    r = await call_echo(hub.client, "not-a-token")
    assert r.status_code == 401 and 'error="invalid_token"' in r.headers["www-authenticate"]


async def test_metadata_documents(hub):
    prm = (await hub.client.get("/.well-known/oauth-protected-resource/mcp")).json()
    assert prm["resource"] == f"{BASE}/mcp"
    assert [s.rstrip("/") for s in prm["authorization_servers"]] == [BASE]
    asm = (await hub.client.get("/.well-known/oauth-authorization-server")).json()
    assert "S256" in asm["code_challenge_methods_supported"]
    assert asm["registration_endpoint"] == f"{BASE}/register"
    assert asm["client_id_metadata_document_supported"] is True
    assert asm["token_endpoint"] == f"{BASE}/token"
    assert asm["authorization_endpoint"] == f"{BASE}/authorize"
    assert "refresh_token" in asm["grant_types_supported"]
    assert not asm.get("scopes_supported")  # no GitHub scope is requested


# --- happy path, refresh, restart ------------------------------------------------------

async def test_full_login_then_tool_call(hub):
    client_id, tok = await login(hub)
    assert tok["token_type"].lower() == "bearer" and tok["refresh_token"]
    result = rpc_result(await call_echo(hub.client, tok["access_token"], "hello", **{"Cf-Connecting-Ip": "203.0.113.9"}))
    assert result["result"]["content"][0]["text"] == "hello"
    assert ("login", "alice", ALLOWED, 1, None) in auth_events(hub.audit)
    row = hub.audit._conn.execute("SELECT github_id, client_id, client_name, cf_ip, tool, status FROM calls").fetchone()
    assert row == (ALLOWED, client_id, "Test Connector", "203.0.113.9", "echo", "ok")


async def test_upstream_authorize_url_requests_no_scope_and_uses_pkce(hub):
    _, _, upstream = await start_login(hub)
    q = parse_qs(urlsplit(upstream).query)
    assert upstream.startswith("https://github.com/login/oauth/authorize?")
    assert "scope" not in q
    assert q["code_challenge_method"] == ["S256"] and q["redirect_uri"] == [f"{BASE}/auth/callback"]


async def test_requested_scopes_are_never_forwarded_upstream(hub):
    # Bypass the SDK's client-scope check (a CIMD document could declare any scope).
    client = OAuthClientInformationFull(client_id="cimd-like", redirect_uris=[AnyUrl(CLAUDE_CB)], scope="repo")
    params = AuthorizationParams(state="s", scopes=["repo", "admin:org"], code_challenge=pkce()[1],
                                 redirect_uri=AnyUrl(CLAUDE_CB), redirect_uri_provided_explicitly=True)
    consent = await hub.provider.authorize(client, params)
    upstream = await approve(hub.client, consent)
    assert "scope" not in parse_qs(urlsplit(upstream).query)


async def test_refresh_rotates_and_bad_refresh_is_invalid_grant(hub):
    client_id, tok = await login(hub)
    r = await hub.client.post("/token", data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"], "client_id": client_id})
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["refresh_token"] != tok["refresh_token"] and new["access_token"] != tok["access_token"]
    rpc_result(await call_echo(hub.client, new["access_token"]))
    for bad in (tok["refresh_token"], "garbage"):  # the rotated-out token is one-time use
        r = await hub.client.post("/token", data={"grant_type": "refresh_token", "refresh_token": bad, "client_id": client_id})
        assert r.status_code == 401 and r.json()["error"] == "invalid_grant"  # FastMCP maps it to 401


async def test_token_endpoint_rejects_json_body(hub):
    client_id, tok = await login(hub)  # login() itself proves form-urlencoded works
    r = await hub.client.post("/token", json={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"], "client_id": client_id})
    assert r.status_code in (400, 401) and "access_token" not in r.text


async def test_tokens_survive_restart(tmp_path, sec, github):
    cfg = make_cfg(tmp_path)
    async with running_hub(cfg, sec, github) as h:
        client_id, tok = await login(h)
    async with running_hub(cfg, sec, github) as h2:  # new provider + app over the same storage
        rpc_result(await call_echo(h2.client, tok["access_token"]))
        r = await h2.client.post("/token", data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"], "client_id": client_id})
        assert r.status_code == 200, r.text


async def test_storage_is_encrypted_at_rest(tmp_path, sec, github):
    cfg = make_cfg(tmp_path)
    async with running_hub(cfg, sec, github) as h:
        await login(h)
    assert all(files_in(cfg, c) for c in STORED_SECRETS if c != "mcp_authorization_codes")  # helper sees entries
    assert files_in(cfg, "mcp_authorization_codes") == []  # one-time code consumed
    upstream_tokens = list(h.github.users)
    blob = b"".join(p.read_bytes() for p in cfg.oauth_dir.rglob("*") if p.is_file())
    assert upstream_tokens and not any(t.encode() in blob for t in upstream_tokens)
    assert b"Test Connector" not in blob
    assert cfg.oauth_dir.stat().st_mode & 0o077 == 0


async def test_rotated_storage_key_invalidates_sessions(tmp_path, sec, github):
    cfg = make_cfg(tmp_path)
    async with running_hub(cfg, sec, github) as h:
        _, tok = await login(h)
    async with running_hub(cfg, sec | {"STORAGE_ENCRYPTION_KEY": Fernet.generate_key().decode()}, github) as h2:
        assert (await call_echo(h2.client, tok["access_token"])).status_code == 401


# --- layer 1: callback -----------------------------------------------------------------

async def test_denied_id_gets_no_code(tmp_path, sec, github):
    cfg = make_cfg(tmp_path)
    async with running_hub(cfg, sec, github) as h:
        _, _, upstream = await start_login(h)
        r = await github_callback(h, upstream, user(DENIED, "mallory"))
        assert r.status_code == 403
        assert "code=" not in r.headers.get("location", "") and "not allowed" in r.text
        assert ("login", "mallory", DENIED, 0, "GitHub id not allowlisted") in auth_events(h.audit)
        assert h.audit.recent_auth_failures() == 1
        # Nothing usable was stored, and the denied user's GitHub token is not a hub token.
        for collection in STORED_SECRETS:
            assert files_in(cfg, collection) == [], collection
        denied_upstream = next(t for t, u in github.users.items() if u.get("id") == DENIED)
        assert (await call_echo(h.client, denied_upstream)).status_code == 401


@pytest.mark.parametrize("who", [user(None, "ghost"), {"id": "1001", "login": "str-id"}, {"id": True, "login": "bool-id"}])
async def test_identity_without_numeric_id_is_denied_at_callback(hub, who):
    _, _, upstream = await start_login(hub)
    r = await github_callback(hub, upstream, who)
    assert r.status_code == 403 and "code=" not in r.headers.get("location", "")


async def test_github_user_lookup_failure_fails_closed(hub):
    _, _, upstream = await start_login(hub)
    hub.github.user_status = 500
    r = await github_callback(hub, upstream, user(ALLOWED))
    assert r.status_code == 403
    assert any(e[3] == 0 and "HTTP 500" in e[4] for e in auth_events(hub.audit))


async def test_callback_refuses_code_if_identity_hook_was_bypassed(hub, monkeypatch):
    """Fail-closed tripwire: if FastMCP stopped using _create_upstream_oauth_client for the
    code exchange, the callback must still not hand out a code."""

    def unchecked():
        raw = GitHubProvider._create_upstream_oauth_client(hub.provider)  # FastMCP's own client
        raw._client = hub.provider._hub_http

        async def keep_open() -> None:
            return None

        raw.aclose = keep_open
        return raw

    monkeypatch.setattr(hub.provider, "_create_upstream_oauth_client", unchecked)
    _, _, upstream = await start_login(hub)
    r = await github_callback(hub, upstream, user(ALLOWED))
    assert r.status_code == 500 and "code=" not in r.headers.get("location", "")
    assert any(e[4] == "allowlist check did not run" for e in auth_events(hub.audit))


async def test_fastmcp_callback_uses_upstream_client_hook(hub):
    """Pins the private hooks layer 1 relies on (fails loudly on a FastMCP upgrade)."""
    from fastmcp.server.auth.oauth_proxy.upstream import AsyncOAuth2Client

    raw = super(hub_auth.AllowlistedGitHubProvider, hub.provider)._create_upstream_oauth_client()
    assert isinstance(raw, AsyncOAuth2Client) and isinstance(raw._client, httpx2.AsyncClient)
    await raw.aclose()
    assert isinstance(hub.provider._create_upstream_oauth_client(), hub_auth._IdentityCheckingClient)
    route = next(r for r in hub.provider.get_routes("/mcp") if getattr(r, "path", None) == "/auth/callback")
    assert route.endpoint.__qualname__.endswith("_guard_callback.<locals>.callback")


# --- layer 2: token verification -------------------------------------------------------

async def test_allowlist_change_revokes_existing_tokens(tmp_path, sec, github):
    cfg = make_cfg(tmp_path)
    async with running_hub(cfg, sec, github) as h:
        _, tok = await login(h)
        rpc_result(await call_echo(h.client, tok["access_token"]))
    async with running_hub(make_cfg(tmp_path, [3003]), sec, github) as h2:
        r = await call_echo(h2.client, tok["access_token"])
        assert r.status_code == 401  # HTTP auth layer, before any middleware
        assert ("token", None, ALLOWED, 0, "GitHub id not allowlisted") in auth_events(h2.audit)


@pytest.mark.parametrize("mutation", [
    lambda u: u.pop("id"),                 # GitHub's verifier itself rejects a missing id
    lambda u: u.update(id=str(u["id"])),   # string id: our layer 2 rejects
    lambda u: u.update(id=1002),           # identity behind the upstream token changed
])
async def test_signed_token_with_bad_upstream_identity_is_rejected(tmp_path, sec, github, mutation):
    async with running_hub(make_cfg(tmp_path, [ALLOWED, 1002]), sec, github) as h:
        _, tok = await login(h)
        for payload in github.users.values():
            mutation(payload)
        assert (await call_echo(h.client, tok["access_token"])).status_code == 401


async def test_layer2_rejects_without_middleware(tmp_path, sec, github):
    """Direct check on the provider: no middleware involved."""
    async with running_hub(make_cfg(tmp_path, [ALLOWED]), sec, github) as h:
        _, tok = await login(h)
        assert await h.provider.load_access_token(tok["access_token"]) is not None
        h.provider.allowed_github_ids = frozenset({3003})
        assert await h.provider.load_access_token(tok["access_token"]) is None


async def test_upstream_refresh_rechecks_identity(hub):
    client_id, tok = await login(hub)
    hub.provider.allowed_github_ids = frozenset({3003})
    r = await hub.client.post("/token", data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"], "client_id": client_id})
    assert r.status_code == 401 and r.json()["error"] == "invalid_grant"
    assert any(e[0] == "refresh" and e[3] == 0 for e in auth_events(hub.audit))


# --- redirect URIs ---------------------------------------------------------------------

async def test_disallowed_redirect_uri_rejected_at_registration(hub):
    r = await register(hub.client, "https://evil.example/cb")
    assert r.status_code == 400 and r.json()["error"] == "invalid_redirect_uri"


@pytest.mark.parametrize("uri", ["http://localhost:8123/callback", "http://127.0.0.1:55001/cb", "https://claude.com/api/mcp/auth_callback"])
async def test_allowed_redirect_uris_register(hub, uri):
    assert (await register(hub.client, uri)).status_code == 201


async def test_disallowed_redirect_uri_rejected_at_authorize(hub):
    client_id = (await register(hub.client)).json()["client_id"]
    q = {"response_type": "code", "client_id": client_id, "redirect_uri": "https://evil.example/cb",
         "state": "s", "code_challenge": pkce()[1], "code_challenge_method": "S256"}
    r = await hub.client.get("/authorize?" + urlencode(q))
    assert r.status_code == 400
    assert "evil.example" not in r.headers.get("location", "")


async def test_upstream_client_id_cannot_smuggle_a_redirect(hub, sec):
    """FastMCP synthesizes a client for the upstream client id; the patterns still apply."""
    q = {"response_type": "code", "client_id": sec["GITHUB_CLIENT_ID"], "redirect_uri": "https://evil.example/cb",
         "state": "s", "code_challenge": pkce()[1], "code_challenge_method": "S256"}
    r = await hub.client.get("/authorize?" + urlencode(q))
    assert r.status_code == 400 and "evil.example" not in r.headers.get("location", "")


async def test_authorize_requires_pkce(hub):
    client_id = (await register(hub.client)).json()["client_id"]
    q = {"response_type": "code", "client_id": client_id, "redirect_uri": CLAUDE_CB, "state": "s"}
    r = await hub.client.get("/authorize?" + urlencode(q))
    assert "code=" not in r.headers.get("location", "") and r.status_code in (302, 400)
    if r.status_code == 302:
        assert "error=" in r.headers["location"]


# --- construction guards ---------------------------------------------------------------

@pytest.mark.parametrize("key", ["GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET", "JWT_SIGNING_KEY", "STORAGE_ENCRYPTION_KEY"])
def test_missing_or_empty_secret_refused(tmp_path, sec, key):
    audit = AuditLog(tmp_path / "a.db")
    for broken in ({k: v for k, v in sec.items() if k != key}, sec | {key: "  "}):
        with pytest.raises(AuthConfigError, match=key):
            build_auth_provider(make_cfg(tmp_path), broken, audit)


@pytest.mark.parametrize("change,match", [
    ({"allowed_github_ids": []}, "allowed_github_ids"),
    ({"allowed_github_ids": [0]}, "positive"),
    ({"base_url": "http://hub.example.com"}, "https"),
    ({"base_url": ""}, "base_url"),
    ({"base_url": "https://hub.example.com/sub"}, "bare origin"),
    ({"base_url": "https://user@hub.example.com"}, "bare origin"),
])
def test_unsafe_config_refused(tmp_path, sec, change, match):
    cfg = make_cfg(tmp_path).model_copy(update=change)
    with pytest.raises(AuthConfigError, match=match):
        build_auth_provider(cfg, sec, AuditLog(tmp_path / "a.db"))


def test_weak_or_malformed_keys_refused(tmp_path, sec):
    audit = AuditLog(tmp_path / "a.db")
    with pytest.raises(AuthConfigError, match="JWT_SIGNING_KEY"):
        build_auth_provider(make_cfg(tmp_path), sec | {"JWT_SIGNING_KEY": "short"}, audit)
    with pytest.raises(AuthConfigError, match="Fernet"):
        build_auth_provider(make_cfg(tmp_path), sec | {"STORAGE_ENCRYPTION_KEY": "not-a-fernet-key"}, audit)


@pytest.mark.parametrize("url", ["http://localhost:8790", "http://127.0.0.1:8790", "https://hub.example.com/"])
def test_local_http_and_https_base_urls_accepted(tmp_path, sec, url):
    provider = build_auth_provider(make_cfg(tmp_path).model_copy(update={"base_url": url.rstrip("/")}), sec, AuditLog(tmp_path / "a.db"))
    assert provider.allowed_github_ids == {ALLOWED}


# --- insecure local test mode ----------------------------------------------------------

TUNNEL = ["/usr/bin/cloudflared", "tunnel", "run", "--token-file", "/etc/cloudflared/claude-context.token"]


@pytest.mark.parametrize("host,procs,base_url,ok", [
    ("127.0.0.1", [], BASE, True),
    ("::1", [], BASE, True),
    ("localhost", [], BASE, True),
    ("0.0.0.0", [], BASE, False),
    ("192.0.2.10", [], "", False),
    ("127.0.0.1", [TUNNEL], BASE, False),
    ("127.0.0.1", [["cloudflared", "tunnel", "run", "other-tunnel"]], BASE, True),
    ("127.0.0.1", [TUNNEL], "", True),  # no base_url: the tunnel cannot route to this instance's auth anyway
])
def test_local_test_mode_guard(tmp_path, host, procs, base_url, ok):
    cfg = make_cfg(tmp_path).model_copy(update={"base_url": base_url})
    allowed, reason = local_test_mode_allowed(cfg, bind_host=host, list_processes=lambda: procs)
    assert allowed is ok, reason


def test_local_test_mode_refused_when_processes_unreadable(tmp_path):
    def boom():
        raise OSError("no /proc")

    allowed, reason = local_test_mode_allowed(make_cfg(tmp_path), bind_host="127.0.0.1", list_processes=boom)
    assert not allowed and "refusing" in reason


def test_default_process_lister_reads_proc():
    assert any("pytest" in " ".join(argv) or "python" in " ".join(argv) for argv in hub_auth._proc_cmdlines())
