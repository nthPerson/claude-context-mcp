"""GitHub OAuth (FastMCP OAuth proxy) restricted to an allowlist of numeric GitHub user ids.

The allowlist is enforced in three independent layers; each one denies on its own:

1. GitHub callback: the upstream token client resolves ``GET /user`` right after the code
   exchange and raises before FastMCP mints an authorization code (``_IdentityCheckingClient``);
   a wrapper around the callback route turns that into a 403 page and refuses any response that
   carries a code without a passed check.
2. Token verification: ``load_access_token`` rejects every token whose live upstream identity is
   not allowlisted, so a correctly signed token stops working when the allowlist changes.
3. Per request: ``middleware.GuardMiddleware`` re-checks the id from the verified token.

See docs/AUTH-NOTES.md for the FastMCP hooks this relies on.
"""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import os
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx2
from cryptography.fernet import Fernet
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.oauth_proxy.ui import create_error_html
from fastmcp.server.auth.providers.github import GitHubProvider
from key_value.aio.stores.filetree import (
    FileTreeStore,
    FileTreeV1CollectionSanitizationStrategy,
    FileTreeV1KeySanitizationStrategy,
)
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route

from claude_context.audit import AuditLog
from claude_context.config import SECRET_KEYS, Config, ConfigError

logger = logging.getLogger(__name__)

REDIRECT_PATH = "/auth/callback"
# GET /user returns the public profile, including the numeric id, for a token with no scope
# at all, so the hub requests none. Any scope a client asks for is dropped before it can be
# forwarded to GitHub (see ``authorize``).
GITHUB_SCOPES: list[str] = []
ALLOWED_CLIENT_REDIRECT_URIS = [
    "https://claude.ai/api/mcp/auth_callback",
    "https://claude.com/api/mcp/auth_callback",
    "http://localhost:*",  # loopback patterns without a path match any port and path (RFC 8252)
    "http://127.0.0.1:*",
]
GITHUB_USER_URL = "https://api.github.com/user"
# Caches the per-request GitHub verification (2 API calls). The allowlist check runs after the
# cache on every request; only a revocation done on github.com takes up to this long to apply.
VERIFY_CACHE_SECONDS = 300
MIN_JWT_KEY_CHARS = 32
_UPSTREAM_ID_KEY = "x_hub_github_id"  # stashed in the (encrypted) upstream token record


class AuthConfigError(ConfigError):
    """The auth configuration is unsafe or incomplete; the server must not start."""


class IdentityDenied(Exception):
    """The GitHub identity behind an upstream token is not allowlisted (or unknown)."""


# --- identity helpers ------------------------------------------------------------------

def _as_id(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None  # rejects bool and strings


def claims_github_id(access_token: AccessToken | None) -> int | None:
    """The verified numeric GitHub id behind ``access_token``, or None if absent or inconsistent.

    Reads the live ``GET /user`` payload that FastMCP's GitHub verifier stores under
    ``github_user_data``; ``sub`` and the id embedded at login (``upstream_claims``) must agree.
    """
    claims = (access_token.claims if access_token else None) or {}
    user = claims.get("github_user_data")
    gid = _as_id(user.get("id")) if isinstance(user, dict) else None
    if gid is None:
        return None
    if "sub" in claims and str(claims["sub"]) != str(gid):
        return None
    upstream = claims.get("upstream_claims")
    if isinstance(upstream, dict) and "github_id" in upstream and upstream["github_id"] != gid:
        return None
    return gid


def request_header(name: str) -> str | None:
    """A header of the HTTP request being served, or None outside a request."""
    from fastmcp.server.dependencies import get_http_request

    try:
        return get_http_request().headers.get(name)
    except RuntimeError:
        return None


# --- layer 1: upstream token client with identity check --------------------------------

@dataclass
class _CallbackState:
    cf_ip: str | None
    allowed_id: int | None = None
    denied: bool = False


_callback_state: ContextVar[_CallbackState | None] = ContextVar("hub_callback_state", default=None)

IdentityCheck = Callable[[dict[str, Any], str], Awaitable[None]]


class _IdentityCheckingClient:
    """Wraps FastMCP's upstream OAuth client; every token GitHub returns is identity-checked.

    Implements the duck-typed surface ``OAuthProxy`` uses (fetch_token, refresh_token,
    client_secret, aclose).
    """

    def __init__(self, inner: Any, check: IdentityCheck, http_client: httpx2.AsyncClient | None) -> None:
        self._inner = inner
        self._check = check
        self._owned = None
        if http_client is not None:
            # Test injection only: AsyncOAuth2Client has no client parameter (private attribute).
            self._owned, inner._client = inner._client, http_client

    @property
    def client_secret(self) -> str | None:
        return self._inner.client_secret

    async def fetch_token(self, url: str, **params: Any) -> dict[str, Any]:
        tokens = await self._inner.fetch_token(url=url, **params)
        await self._check(tokens, "login")
        return tokens

    async def refresh_token(self, url: str, **params: Any) -> dict[str, Any]:
        tokens = await self._inner.refresh_token(url=url, **params)
        await self._check(tokens, "refresh")
        return tokens

    async def aclose(self) -> None:
        if self._owned is not None:
            await self._owned.aclose()  # the injected client belongs to the caller
        else:
            await self._inner.aclose()


def _carries_code(response: Response) -> bool:
    location = response.headers.get("location", "")
    return 300 <= response.status_code < 400 and "code" in parse_qs(urlsplit(location).query)


def _denied_page(status: int, message: str) -> HTMLResponse:
    html = create_error_html(error_title="Access denied", error_message=message)
    return HTMLResponse(html, status_code=status, headers={"Cache-Control": "no-store"})


# --- the provider ----------------------------------------------------------------------

class AllowlistedGitHubProvider(GitHubProvider):
    """FastMCP ``GitHubProvider`` that only ever authenticates allowlisted GitHub ids."""

    def __init__(
        self,
        *,
        allowed_github_ids: Iterable[int],
        audit: AuditLog,
        http_client: httpx2.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(http_client=http_client, **kwargs)
        self.allowed_github_ids = frozenset(allowed_github_ids)
        self._audit = audit
        self._hub_http = http_client

    def _audit_event(self, event: str, **fields: Any) -> None:
        try:
            self._audit.log_auth_event(event, **fields)
        except Exception:
            logger.exception("audit: could not record auth event %s", event)

    # Layer 1 -------------------------------------------------------------------------
    def _create_upstream_oauth_client(self) -> Any:  # FastMCP's documented override point
        return _IdentityCheckingClient(super()._create_upstream_oauth_client(), self._check_upstream, self._hub_http)

    async def _github_user(self, upstream_token: str) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {upstream_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "claude-context-hub",
        }
        async with (
            contextlib.nullcontext(self._hub_http) if self._hub_http else httpx2.AsyncClient(timeout=10)
        ) as client:
            resp = await client.get(GITHUB_USER_URL, headers=headers)
        if resp.status_code != 200:
            raise IdentityDenied(f"GitHub /user returned HTTP {resp.status_code}")
        data = resp.json()
        if not isinstance(data, dict):
            raise IdentityDenied("GitHub /user returned a non-object")
        return data

    async def _check_upstream(self, tokens: dict[str, Any], event: str) -> None:
        """Raise IdentityDenied unless the user behind ``tokens`` is allowlisted."""
        state = _callback_state.get()
        cf_ip = state.cf_ip if state else request_header("cf-connecting-ip")
        user: dict[str, Any] = {}
        try:
            user = await self._github_user(str(tokens.get("access_token") or ""))
            reason = None
        except Exception as e:  # network errors, bad JSON: fail closed
            reason = f"identity lookup failed: {e}"
        gid = _as_id(user.get("id"))
        login = user.get("login") if isinstance(user.get("login"), str) else None
        if reason is None and gid is None:
            reason = "GitHub user has no numeric id"
        elif reason is None and gid not in self.allowed_github_ids:
            reason = "GitHub id not allowlisted"
        if reason is not None:
            if state:
                state.denied = True
            logger.warning("auth: %s denied (github id %s, login %s): %s", event, gid, login, reason)
            self._audit_event(event, ok=False, github_login=login, github_id=gid, cf_ip=cf_ip, reason=reason)
            raise IdentityDenied(reason)
        if state:
            state.allowed_id = gid
        self._audit_event(event, ok=True, github_login=login, github_id=gid, cf_ip=cf_ip)
        tokens[_UPSTREAM_ID_KEY] = gid

    async def _extract_upstream_claims(self, idp_tokens: dict[str, Any]) -> dict[str, Any] | None:
        """Embed the login-time GitHub id in the issued JWT (FastMCP's documented hook)."""
        gid = _as_id(idp_tokens.get(_UPSTREAM_ID_KEY))
        return {"github_id": gid} if gid is not None else None

    def get_routes(self, mcp_path: str | None = None) -> list[Route]:
        """FastMCP's routes, with the GitHub callback wrapped by the allowlist guard."""
        routes = super().get_routes(mcp_path)
        for i, route in enumerate(routes):
            if isinstance(route, Route) and route.path == REDIRECT_PATH:
                routes[i] = Route(REDIRECT_PATH, endpoint=self._guard_callback(route.endpoint), methods=["GET"])
                return routes
        raise RuntimeError(f"FastMCP no longer serves {REDIRECT_PATH}; refusing to start without the allowlist guard")

    def _guard_callback(self, original: Callable[[Request], Awaitable[Response]]) -> Callable[[Request], Awaitable[Response]]:
        async def callback(request: Request) -> Response:
            state = _CallbackState(cf_ip=request.headers.get("cf-connecting-ip"))
            token = _callback_state.set(state)
            try:
                response = await original(request)
            finally:
                _callback_state.reset(token)
            if state.denied:
                return _denied_page(403, "This GitHub account is not allowed to use this server.")
            if state.allowed_id is None and _carries_code(response):
                # The code exchange bypassed _IdentityCheckingClient (FastMCP internals changed?).
                logger.critical("auth: callback issued a code without the allowlist check; withheld")
                self._audit_event("login", ok=False, cf_ip=state.cf_ip, reason="allowlist check did not run")
                return _denied_page(500, "Sign-in is unavailable: the server's allowlist check did not run.")
            return response

        return callback

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        """Never forward a client-requested scope (e.g. ``repo``) to GitHub."""
        if params.scopes:
            kept = [s for s in params.scopes if s in GITHUB_SCOPES]
            if kept != params.scopes:
                logger.info("auth: dropping requested scopes %s", sorted(set(params.scopes) - set(kept)))
            params = params.model_copy(update={"scopes": kept or None})
        return await super().authorize(client, params)

    # Layer 2 -------------------------------------------------------------------------
    async def load_access_token(self, token: str) -> AccessToken | None:  # type: ignore[override]
        """Validate as FastMCP does, then require an allowlisted live GitHub id."""
        result = await super().load_access_token(token)
        if result is None:
            return None
        gid = claims_github_id(result)
        if gid is None or gid not in self.allowed_github_ids:
            reason = "token has no GitHub id" if gid is None else "GitHub id not allowlisted"
            logger.warning("auth: rejected a valid token: %s (github id %s)", reason, gid)
            self._audit_event("token", ok=False, github_id=gid, cf_ip=request_header("cf-connecting-ip"), reason=reason)
            return None
        try:
            client_id = str(self.jwt_issuer.verify_token(token).get("client_id") or "")
        except Exception:  # verified by super a moment ago; can only fail by expiring in between
            return None
        client_name = None
        with contextlib.suppress(Exception):
            client = await self.get_client(client_id)
            client_name = getattr(client, "client_name", None)
        claims = {**(result.claims or {}), "mcp_client_id": client_id, "mcp_client_name": client_name}
        return result.model_copy(update={"claims": claims})


# --- construction ----------------------------------------------------------------------

def _check_base_url(base_url: str) -> None:
    u = urlsplit(base_url)
    if u.username or u.password or u.query or u.fragment or u.path not in ("", "/") or not u.hostname:
        raise AuthConfigError(f"base_url must be a bare origin like https://hub.example.com, got {base_url!r}")
    if u.scheme == "https" or (u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1")):
        return
    raise AuthConfigError("base_url must use https (plain http is allowed only for localhost/127.0.0.1)")


def build_auth_provider(
    cfg: Config,
    secrets: Mapping[str, str],
    audit: AuditLog,
    *,
    http_client: httpx2.AsyncClient | None = None,
) -> AllowlistedGitHubProvider:
    """The GitHub OAuth-proxy provider for ``cfg``; raises AuthConfigError on anything unsafe."""
    missing = [k for k in SECRET_KEYS if not str(secrets.get(k) or "").strip()]
    if missing:
        raise AuthConfigError(f"missing secrets: {', '.join(missing)}")
    if not cfg.allowed_github_ids:
        raise AuthConfigError("allowed_github_ids is empty: nobody could sign in")
    if any(_as_id(i) is None for i in cfg.allowed_github_ids):
        raise AuthConfigError("allowed_github_ids must be positive integers")
    _check_base_url(cfg.base_url)
    jwt_key = secrets["JWT_SIGNING_KEY"].strip()
    if len(jwt_key) < MIN_JWT_KEY_CHARS:
        raise AuthConfigError(f"JWT_SIGNING_KEY must be at least {MIN_JWT_KEY_CHARS} characters")
    try:
        fernet = Fernet(secrets["STORAGE_ENCRYPTION_KEY"].strip())
    except (ValueError, TypeError) as e:
        raise AuthConfigError("STORAGE_ENCRYPTION_KEY must be a Fernet key (32 url-safe base64-encoded bytes)") from e

    oauth_dir = cfg.oauth_dir
    oauth_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(oauth_dir, 0o700)
    # Values are Fernet-encrypted; file names (client ids, token ids) are not.
    storage = FernetEncryptionWrapper(
        key_value=FileTreeStore(
            data_directory=oauth_dir,
            key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(oauth_dir),
            collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(oauth_dir),
        ),
        fernet=fernet,
        raise_on_decryption_error=False,  # after a key rotation old entries read as missing
    )
    return AllowlistedGitHubProvider(
        allowed_github_ids=cfg.allowed_github_ids,
        audit=audit,
        http_client=http_client,
        client_id=secrets["GITHUB_CLIENT_ID"].strip(),
        client_secret=secrets["GITHUB_CLIENT_SECRET"].strip(),
        base_url=cfg.base_url,
        redirect_path=REDIRECT_PATH,
        required_scopes=GITHUB_SCOPES,
        cache_ttl_seconds=VERIFY_CACHE_SECONDS,
        allowed_client_redirect_uris=ALLOWED_CLIENT_REDIRECT_URIS,
        client_storage=storage,
        jwt_signing_key=jwt_key,
        require_authorization_consent=True,
        enable_cimd=True,
    )


# --- --insecure-local-test guard -------------------------------------------------------

def _proc_cmdlines() -> Iterator[list[str]]:
    """Command lines of all visible processes (raises OSError if /proc is unavailable)."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue  # process exited or is not ours to read
        if raw:
            yield [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _is_loopback(host: str) -> bool:
    host = host.strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def local_test_mode_allowed(
    cfg: Config,
    *,
    bind_host: str,
    list_processes: Callable[[], Iterable[list[str]]] = _proc_cmdlines,
) -> tuple[bool, str]:
    """Whether ``--insecure-local-test`` (auth disabled) may start, and why not.

    Requires a loopback bind and, when ``base_url`` is set, no running cloudflared whose command
    line references this hub's tunnel. The per-request half (refusing anything that carries
    ``Cf-Connecting-Ip``) is in ``GuardMiddleware``.
    """
    if not _is_loopback(bind_host):
        return False, f"bind host {bind_host!r} is not a loopback address"
    if cfg.base_url:
        try:
            for argv in list_processes():
                line = " ".join(argv)
                if "cloudflared" in line and "claude-context" in line:
                    return False, "a cloudflared process for this hub's tunnel is running"
        except OSError as e:
            return False, f"cannot inspect running processes ({e}); refusing"
    return True, "ok"
