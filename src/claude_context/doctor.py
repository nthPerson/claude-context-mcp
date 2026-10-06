"""``claude-context doctor``: one-screen health check of an installation."""

from __future__ import annotations

import socket
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path

import httpx

from . import db
from .config import SECRET_KEYS, Config, default_secrets_path, file_mode, parse_env_file
from .roots import root_dir

UNITS = ("claude-context-server.service", "claude-context-indexer.service", "claude-context-maintenance.timer",
         "cloudflared-claude-context.service")
OK, WARN, FAIL = "ok", "warn", "FAIL"


def _systemctl(*args: str) -> str:
    try:
        return subprocess.run(["systemctl", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def check_secrets(cfg: Config) -> tuple[str, str]:
    path = default_secrets_path()
    if not path.is_file():
        return FAIL, f"{path} missing"
    if file_mode(path) & 0o077:
        return FAIL, f"{path} is mode {file_mode(path):o}; must be 600"
    missing = [k for k in SECRET_KEYS if not parse_env_file(path).get(k)]
    return (FAIL, f"missing keys: {', '.join(missing)}") if missing else (OK, f"{path} (mode 600, all keys present)")


def check_roots(cfg: Config) -> tuple[str, str]:
    if not cfg.roots:
        return FAIL, "no [[roots]] configured"
    bad = [r.label for r in cfg.roots if r.kind == "claude-code" and not root_dir(cfg, r).is_dir()]
    return (FAIL, f"not readable: {', '.join(bad)}") if bad else (OK, ", ".join(r.label for r in cfg.roots))


def check_syncthing(cfg: Config) -> tuple[str, str]:
    from .daemon import make_syncthing
    client = make_syncthing(cfg)
    if client is None or not client.ping():
        return WARN, f"API at {cfg.syncthing.api} unreachable (machine attribution will be 'unknown')"
    return OK, f"API reachable, {len(client.devices())} devices"


def check_vec(cfg: Config) -> tuple[str, str]:
    conn = sqlite3.connect(":memory:")
    try:
        db._load_vec(conn)
        return OK, f"sqlite {sqlite3.sqlite_version}, sqlite-vec {conn.execute('select vec_version()').fetchone()[0]}"
    except Exception as e:
        return FAIL, str(e)
    finally:
        conn.close()


def check_model(cfg: Config) -> tuple[str, str]:
    if not cfg.embedding.enabled:
        return WARN, "embeddings disabled (keyword search only)"
    cached = cfg.models_dir.is_dir() and any(cfg.models_dir.rglob("*.onnx"))
    return (OK, f"{cfg.embedding.model} cached") if cached else (WARN, f"{cfg.embedding.model} not downloaded yet")


def check_index(cfg: Config) -> tuple[str, str]:
    if not cfg.index_db.is_file():
        return WARN, "index.db not built yet (run: claude-context index)"
    conn = db.connect(cfg.index_db, readonly=True)
    try:
        result = conn.execute("PRAGMA quick_check").fetchone()[0]
        n = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()
    return (OK, f"quick_check ok, {n} sessions") if result == "ok" else (FAIL, f"quick_check: {result}")


def check_port(cfg: Config) -> tuple[str, str]:
    with socket.socket() as s:
        s.settimeout(1)
        in_use = s.connect_ex((cfg.host, cfg.port)) == 0
    if not in_use:
        return WARN, f"{cfg.bind} free (server not running)"
    active = _systemctl("is-active", UNITS[0]) == "active"
    return (OK, f"{cfg.bind} bound by {UNITS[0]}") if active else (WARN, f"{cfg.bind} in use, but {UNITS[0]} is not active")


def check_units(cfg: Config) -> tuple[str, str]:
    states = {u: (_systemctl("is-enabled", u) or "missing", _systemctl("is-active", u) or "unknown") for u in UNITS}
    bad = [f"{u}: {e}/{a}" for u, (e, a) in states.items() if e != "enabled" or a != "active"]
    return (WARN, "; ".join(bad)) if bad else (OK, "all units enabled and active")


def check_public(cfg: Config) -> tuple[str, str]:
    if not cfg.base_url:
        return WARN, "base_url not set"
    meta_url = f"{cfg.base_url}/.well-known/oauth-protected-resource/mcp"
    try:
        with httpx.Client(timeout=10) as client:
            meta = client.get(meta_url)
            if meta.status_code != 200 or meta.json().get("resource") != cfg.mcp_url:
                return FAIL, f"GET {meta_url} → {meta.status_code}; resource must be {cfg.mcp_url}"
            unauth = client.post(cfg.mcp_url, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    except (httpx.HTTPError, ValueError) as e:
        return FAIL, f"{meta_url}: {e}"
    challenge = unauth.headers.get("www-authenticate", "")
    if unauth.status_code != 401 or "resource_metadata=" not in challenge:
        return FAIL, f"unauthenticated POST /mcp → {unauth.status_code} (expected 401 with resource_metadata)"
    return OK, "discovery document correct; unauthenticated POST /mcp → 401"


CHECKS: list[tuple[str, Callable[[Config], tuple[str, str]]]] = [
    ("secrets", check_secrets), ("roots", check_roots), ("syncthing", check_syncthing), ("sqlite-vec", check_vec),
    ("embedding model", check_model), ("index.db", check_index), ("port", check_port), ("systemd units", check_units),
    ("public endpoint", check_public),
]


def run(cfg: Config, config_path: Path) -> int:
    """Print every check; exit status 1 if any FAILed."""
    print(f"{OK:>4}  config: {config_path} parses")
    failed = False
    for name, fn in CHECKS:
        try:
            status, detail = fn(cfg)
        except Exception as e:  # a check must never crash the doctor
            status, detail = FAIL, f"{type(e).__name__}: {e}"
        failed |= status == FAIL
        print(f"{status:>4}  {name}: {detail}")
    return 1 if failed else 0
