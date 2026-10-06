"""The MCP server process (``claude-context serve``): FastMCP over Streamable HTTP.

Assembly only: tools live in ``tools.py``, tool behaviour in ``service.py``, authentication in
``auth.py`` and the per-request guard (allowlist re-check, CIDR, rate limit, audit) in
``middleware.py``.
"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Mapping

import uvicorn
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__, tools
from .audit import AuditLog
from .auth import build_auth_provider, local_test_mode_allowed
from .config import Config, ConfigError, load_secrets
from .embed import embedder_from_config
from .middleware import GuardMiddleware
from .service import HubService

log = logging.getLogger(__name__)

SERVER_NAME = "Claude Context"
MCP_PATH = "/mcp"


def build_mcp(cfg: Config, *, audit: AuditLog, secrets: Mapping[str, str] | None = None,
              insecure_local_test: bool = False, embedder=None, http_client=None) -> FastMCP:
    """Build the FastMCP server. Without ``insecure_local_test`` an auth provider is mandatory."""
    auth = None
    if not insecure_local_test:
        auth = build_auth_provider(cfg, secrets or {}, audit, http_client=http_client)
    mcp = FastMCP(SERVER_NAME, instructions=tools.instructions(cfg), version=__version__, auth=auth,
                  middleware=[GuardMiddleware(cfg, audit, insecure_local_test=insecure_local_test)],
                  mask_error_details=True)
    service = HubService(cfg, embedder=embedder, auth_failures=audit.recent_auth_failures)
    tools.register(mcp, service)
    return mcp


class SourceCidrGate:
    """HTTP-level half of the source check: a real 403 for /mcp requests from outside the CIDRs.

    ``GuardMiddleware`` repeats the check per MCP message; this one answers before the request
    reaches token verification. Inactive while ``allowed_mcp_source_cidrs`` is empty.
    """

    def __init__(self, app: ASGIApp, *, cfg: Config, audit: AuditLog) -> None:
        self.app = app
        self.audit = audit
        self.networks = [ipaddress.ip_network(c, strict=False) for c in cfg.allowed_mcp_source_cidrs]

    def _allowed(self, raw: str | None) -> bool:
        try:
            ip = ipaddress.ip_address((raw or "").strip())
        except ValueError:
            return False
        return any(ip in net for net in self.networks)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.networks and scope["type"] == "http" and scope["path"].rstrip("/") == MCP_PATH:
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            cf_ip = headers.get("cf-connecting-ip")
            if not self._allowed(cf_ip):
                log.warning("denied %s from %s: outside allowed_mcp_source_cidrs", MCP_PATH, cf_ip)
                try:
                    self.audit.log_call(tool=None, status="denied", cf_ip=cf_ip, user_agent=headers.get("user-agent"),
                                        error="source address not in allowed_mcp_source_cidrs")
                except Exception:
                    log.exception("audit: could not record denied request")
                await PlainTextResponse("Forbidden", status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(cfg: Config, mcp: FastMCP, audit: AuditLog):
    """The ASGI app. Stateless, so restarts and several connections need no session affinity.

    FastMCP's optional Host/Origin validation stays off: it guards unauthenticated localhost
    servers against DNS rebinding, whereas every request here needs a bearer token, and the
    Host seen behind the tunnel is the public hostname.
    """
    return mcp.http_app(path=MCP_PATH, stateless_http=True,
                        middleware=[Middleware(SourceCidrGate, cfg=cfg, audit=audit)])


def serve(cfg: Config, *, insecure_local_test: bool = False) -> int:
    cfg.ensure_dirs()
    if insecure_local_test:
        ok, reason = local_test_mode_allowed(cfg, bind_host=cfg.host)
        if not ok:
            log.error("refusing --insecure-local-test: %s", reason)
            return 2
        log.warning("=" * 78)
        log.warning("INSECURE LOCAL TEST MODE: AUTHENTICATION IS DISABLED. Never expose this port.")
        log.warning("=" * 78)
        secrets: Mapping[str, str] = {}
    else:
        try:
            secrets = load_secrets()
        except ConfigError as e:
            log.error("%s", e)
            return 2
    audit = AuditLog(cfg.audit_db)
    try:
        mcp = build_mcp(cfg, audit=audit, secrets=secrets, insecure_local_test=insecure_local_test,
                        embedder=embedder_from_config(cfg))
    except Exception as e:  # fail closed with a clear message (missing secrets, empty allowlist, …)
        log.error("cannot start server: %s", e)
        audit.close()
        return 2
    log.info("serving %s on %s (public: %s)", MCP_PATH, cfg.bind, cfg.mcp_url if cfg.base_url else "not set")
    try:
        uvicorn.run(build_app(cfg, mcp, audit), host=cfg.host, port=cfg.port, log_level="info", access_log=False,
                    proxy_headers=False, server_header=False)
    finally:
        audit.close()
    return 0
