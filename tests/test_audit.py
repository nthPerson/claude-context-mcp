from __future__ import annotations

import stat
import threading

import pytest

from claude_context.audit import AuditLog


class Clock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def audit(tmp_path, clock):
    log = AuditLog(tmp_path / "sub" / "audit.db", clock=clock)
    yield log
    log.close()


def test_file_mode_and_wal(audit):
    assert stat.S_IMODE(audit.path.stat().st_mode) == 0o600
    assert audit._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    audit.log_call(tool="t", status="ok")
    wal = audit.path.with_name(audit.path.name + "-wal")
    assert not wal.exists() or stat.S_IMODE(wal.stat().st_mode) == 0o600


def test_existing_file_is_tightened(tmp_path):
    path = tmp_path / "audit.db"
    path.touch(mode=0o644)
    AuditLog(path).close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_log_call_truncates(audit):
    audit.log_call(tool="search", status="error", github_id=1001, client_id="c", client_name="n", cf_ip="203.0.113.1",
                   user_agent="ua", args="a" * 2000, result_chars=10, duration_ms=3, error="e" * 2000)
    row = audit._conn.execute("SELECT * FROM calls").fetchone()
    assert row[0] == audit._clock() and row[1:6] == (1001, "c", "n", "203.0.113.1", "ua")
    assert len(row[7]) == 500 and len(row[11]) == 500 and row[8:11] == (10, 3, "error")


def test_status_is_constrained(audit):
    with pytest.raises(Exception):
        audit.log_call(tool="t", status="maybe")


def test_auth_failures_window(audit, clock):
    audit.log_auth_event("login", ok=False, github_login="mallory", github_id=2002, reason="not allowlisted")
    audit.log_auth_event("login", ok=True, github_login="alice", github_id=1001)
    clock.now += 23 * 3600
    audit.log_auth_event("token", ok=False, github_id=2002)
    assert audit.recent_auth_failures() == 2
    clock.now += 2 * 3600
    assert audit.recent_auth_failures() == 1
    assert audit.recent_auth_failures(hours=48) == 2


def test_recent_source_ips(audit, clock):
    for ip in ["160.79.104.1", "160.79.104.1", "2001:db8::1", None]:
        audit.log_call(tool="t", status="ok", cf_ip=ip)
    clock.now += 25 * 3600
    audit.log_call(tool="t", status="ok", cf_ip="198.51.100.9")
    assert audit.recent_source_ips() == [("198.51.100.9", 1)]
    assert audit.recent_source_ips(hours=48) == [("160.79.104.1", 2), ("198.51.100.9", 1), ("2001:db8::1", 1)]


def test_prune(audit, clock):
    audit.log_call(tool="old", status="ok")
    audit.log_auth_event("login", ok=True)
    clock.now += 366 * 86400
    audit.log_call(tool="new", status="ok")
    assert audit.prune() == 2
    assert audit._conn.execute("SELECT tool FROM calls").fetchall() == [("new",)]
    assert audit._conn.execute("SELECT COUNT(*) FROM auth_events").fetchone()[0] == 0


def test_thread_safe_writes(audit):
    def work(n: int) -> None:
        for i in range(50):
            audit.log_call(tool=f"t{n}", status="ok", duration_ms=i)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert audit._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 400


def test_reopen_keeps_rows(tmp_path):
    path = tmp_path / "audit.db"
    log = AuditLog(path)
    log.log_auth_event("login", ok=False)
    log.close()
    log = AuditLog(path)
    assert log.recent_auth_failures() == 1
    log.close()
