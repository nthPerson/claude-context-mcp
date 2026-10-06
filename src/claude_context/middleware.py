"""Per-request guard: allowlist re-check, source CIDR check, rate limit and call auditing.

Runs as FastMCP middleware. ``on_message`` sees every inbound JSON-RPC message (requests and
notifications); ``on_call_tool`` wraps the tool itself. HTTP headers are read from the
Starlette request via ``fastmcp.server.dependencies.get_http_request()`` (``get_http_headers()``
would strip ``authorization``/``cookie``, which we do not need, but is avoided so nothing is
silently filtered).
"""

from __future__ import annotations

import functools
import ipaddress
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastmcp.server.dependencies import get_access_token, get_http_request
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.middleware.rate_limiting import RateLimitError
from mcp import MCPError

from claude_context.audit import MAX_TEXT, AuditLog
from claude_context.auth import claims_github_id
from claude_context.config import Config

logger = logging.getLogger(__name__)

FORBIDDEN = -32003  # JSON-RPC server-defined error code used for policy denials ("403")
WINDOW_SECONDS = 60.0


class Forbidden(MCPError):
    def __init__(self, message: str) -> None:
        super().__init__(code=FORBIDDEN, message=f"Forbidden: {message}")


@functools.cache
def _redactor() -> Callable[[str], str]:
    try:
        from claude_context.redact import redact
    except Exception:
        return lambda text: text
    return redact


def _redact_args(arguments: Any) -> str | None:
    if arguments is None:
        return None
    text = json.dumps(arguments, ensure_ascii=False, default=str, sort_keys=True)
    try:
        text = _redactor()(text)
    except Exception:
        logger.exception("audit: redaction failed; arguments not stored")
        return "[redaction failed]"
    return text[:MAX_TEXT]


def _result_chars(result: Any) -> int:
    total = 0
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        total += len(text) if isinstance(text, str) else len(str(block))
    if not total and getattr(result, "structured_content", None) is not None:
        total = len(json.dumps(result.structured_content, default=str))
    return total


@dataclass
class _Caller:
    github_id: int | None = None
    client_id: str | None = None
    client_name: str | None = None
    cf_ip: str | None = None
    user_agent: str | None = None
    has_token: bool = False


def _current_caller() -> _Caller:
    caller = _Caller()
    try:
        headers = get_http_request().headers
    except RuntimeError:
        headers = None
    if headers is not None:
        caller.cf_ip = headers.get("cf-connecting-ip")
        caller.user_agent = headers.get("user-agent")
    token = get_access_token()
    if token is not None:
        caller.has_token = True
        caller.github_id = claims_github_id(token)
        claims = token.claims or {}
        caller.client_id = claims.get("mcp_client_id") or token.client_id
        caller.client_name = claims.get("mcp_client_name")
    return caller


class GuardMiddleware(Middleware):
    """Deny non-allowlisted callers and out-of-range sources, rate limit, and audit tool calls."""

    def __init__(
        self,
        cfg: Config,
        audit: AuditLog,
        *,
        insecure_local_test: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._allowed = frozenset(cfg.allowed_github_ids)
        # Invalid CIDRs raise here, at startup.
        self._networks = [ipaddress.ip_network(c, strict=False) for c in cfg.allowed_mcp_source_cidrs]
        self._limit = cfg.rate_limit_per_minute  # <= 0 disables rate limiting
        self._audit = audit
        self._insecure = insecure_local_test
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        if insecure_local_test:
            logger.warning(
                "!!! INSECURE LOCAL TEST MODE: authentication is DISABLED. Requests carrying "
                "Cf-Connecting-Ip (anything from the tunnel) are refused. Never expose this server. !!!"
            )

    # --- checks ------------------------------------------------------------------------
    def _deny_reason(self, caller: _Caller) -> str | None:
        if self._insecure:
            if caller.cf_ip is not None:
                return "insecure local test mode refuses tunnelled requests"
            if not caller.has_token:
                return None  # a token, if somehow present, is still held to the allowlist
        if caller.github_id is None:
            return "no verified GitHub identity"
        if caller.github_id not in self._allowed:
            return "GitHub id not allowlisted"
        if self._networks:
            if not caller.cf_ip:
                return "missing Cf-Connecting-Ip"
            try:
                ip = ipaddress.ip_address(caller.cf_ip.strip())
            except ValueError:
                return "unparsable Cf-Connecting-Ip"
            if not any(ip in net for net in self._networks):
                return "source address not in allowed_mcp_source_cidrs"
        return None

    def _rate_ok(self, key: str) -> bool:
        if self._limit <= 0:
            return True
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True

    # --- auditing ----------------------------------------------------------------------
    def _log_call(self, caller: _Caller, tool: str | None, args: Any, status: str, *,
                  error: str | None = None, result_chars: int | None = None, duration_ms: int | None = None) -> None:
        try:
            self._audit.log_call(
                tool=tool, status=status, github_id=caller.github_id, client_id=caller.client_id,
                client_name=caller.client_name, cf_ip=caller.cf_ip, user_agent=caller.user_agent,
                args=_redact_args(args), result_chars=result_chars, duration_ms=duration_ms,
                error=error[:MAX_TEXT] if error else None,
            )
        except Exception:
            logger.exception("audit: could not record call to %s", tool)

    @staticmethod
    def _call_target(context: MiddlewareContext[Any]) -> tuple[str | None, Any]:
        if context.method == "tools/call":
            return getattr(context.message, "name", None), getattr(context.message, "arguments", None)
        return context.method, None

    # --- hooks -------------------------------------------------------------------------
    async def on_message(self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any]) -> Any:
        caller = _current_caller()
        reason = self._deny_reason(caller)
        is_request = context.type == "request"
        if reason is None and is_request and not self._rate_ok(str(caller.github_id or "local")):
            reason = "rate limit exceeded"
        if reason is not None:
            logger.warning("guard: denied %s from github id %s, ip %s: %s",
                           context.method, caller.github_id, caller.cf_ip, reason)
            if is_request:
                tool, args = self._call_target(context)
                self._log_call(caller, tool, args, "denied", error=reason)
            if reason == "rate limit exceeded":
                raise RateLimitError("Rate limit exceeded")
            raise Forbidden(reason)
        return await call_next(context)

    async def on_call_tool(self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any]) -> Any:
        caller = _current_caller()
        tool, args = self._call_target(context)
        start = self._clock()
        status, error, chars = "error", "cancelled", None
        try:
            result = await call_next(context)
            status, error, chars = "ok", None, _result_chars(result)
            return result
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._log_call(caller, tool, args, status, error=error, result_chars=chars,
                           duration_ms=int((self._clock() - start) * 1000))
