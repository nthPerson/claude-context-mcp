"""Append-only audit log (SQLite) for MCP calls and authentication events.

One connection guarded by a lock: FastMCP may call in from the event loop and from worker
threads. Timestamps are Unix seconds (``REAL``); use ``datetime(ts, 'unixepoch')`` to read them.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

MAX_TEXT = 500  # args and error text are stored truncated to this many characters

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    ts REAL NOT NULL,
    github_id INTEGER,
    client_id TEXT,
    client_name TEXT,
    cf_ip TEXT,
    user_agent TEXT,
    tool TEXT,
    args_redacted_trunc TEXT,
    result_chars INTEGER,
    duration_ms INTEGER,
    status TEXT NOT NULL CHECK (status IN ('ok', 'error', 'denied')),
    error TEXT
);
CREATE INDEX IF NOT EXISTS calls_ts ON calls(ts);
CREATE TABLE IF NOT EXISTS auth_events (
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    github_login TEXT,
    github_id INTEGER,
    cf_ip TEXT,
    ok INTEGER NOT NULL CHECK (ok IN (0, 1)),
    reason TEXT
);
CREATE INDEX IF NOT EXISTS auth_events_ts ON auth_events(ts);
"""


def _trunc(text: str | None, limit: int = MAX_TEXT) -> str | None:
    return text if text is None or len(text) <= limit else text[: limit - 1] + "…"


class AuditLog:
    """SQLite audit log at ``path`` (created with mode 600, WAL journal)."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Create the file ourselves so it never exists with a umask-derived mode; SQLite
        # gives the -wal/-shm files the same mode as the database file.
        os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))
        os.chmod(self.path, 0o600)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def log_call(
        self,
        *,
        tool: str | None,
        status: str,
        github_id: int | None = None,
        client_id: str | None = None,
        client_name: str | None = None,
        cf_ip: str | None = None,
        user_agent: str | None = None,
        args: str | None = None,
        result_chars: int | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        """Record one MCP call. ``args`` must already be redacted; it is truncated here."""
        self._execute(
            "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (self._clock(), github_id, _trunc(client_id, 200), _trunc(client_name, 200), _trunc(cf_ip, 64),
             _trunc(user_agent, 300), _trunc(tool, 200), _trunc(args), result_chars, duration_ms, status,
             _trunc(error)),
        )

    def log_auth_event(
        self,
        event: str,
        *,
        ok: bool,
        github_login: str | None = None,
        github_id: int | None = None,
        cf_ip: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record a login, refresh or token rejection."""
        self._execute(
            "INSERT INTO auth_events VALUES (?,?,?,?,?,?,?)",
            (self._clock(), event, _trunc(github_login, 100), github_id, _trunc(cf_ip, 64), int(bool(ok)),
             _trunc(reason)),
        )

    def recent_auth_failures(self, hours: int = 24) -> int:
        """Number of failed auth events in the last ``hours``."""
        cutoff = self._clock() - hours * 3600
        return self._execute("SELECT COUNT(*) FROM auth_events WHERE ok = 0 AND ts >= ?", (cutoff,)).fetchone()[0]

    def recent_source_ips(self, hours: int = 24) -> list[tuple[str, int]]:
        """``(cf_ip, call count)`` for calls in the last ``hours``, busiest first."""
        cutoff = self._clock() - hours * 3600
        rows = self._execute(
            "SELECT cf_ip, COUNT(*) AS n FROM calls WHERE ts >= ? AND cf_ip IS NOT NULL "
            "GROUP BY cf_ip ORDER BY n DESC, cf_ip",
            (cutoff,),
        ).fetchall()
        return [(ip, n) for ip, n in rows]

    def prune(self, older_than_days: int = 365) -> int:
        """Delete rows older than ``older_than_days``; returns the number removed."""
        cutoff = self._clock() - older_than_days * 86400
        with self._lock:
            n = self._conn.execute("DELETE FROM calls WHERE ts < ?", (cutoff,)).rowcount
            n += self._conn.execute("DELETE FROM auth_events WHERE ts < ?", (cutoff,)).rowcount
        return n

    def close(self) -> None:
        with self._lock:
            self._conn.close()
