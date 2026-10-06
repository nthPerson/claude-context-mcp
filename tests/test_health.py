from datetime import timedelta
from pathlib import Path

import pytest
from svc_fixtures import NOW, make_config, write_status

from claude_context import db, health


def iso(delta: timedelta) -> str:
    return (NOW - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def cfg(tmp_path: Path):
    return make_config(tmp_path)


@pytest.fixture
def conn(cfg):
    c = db.connect(cfg.index_db)
    db.init_schema(c)
    yield c
    c.close()


def warn(cfg, conn, **overrides) -> list[str]:
    status = write_status(cfg, **overrides)
    assert health.load_status(cfg) == status
    return health.health_warnings(status, conn, now=NOW)


def test_load_status_tolerates_missing_and_corrupt(cfg):
    assert health.load_status(cfg) == {}
    cfg.status_file.parent.mkdir(parents=True)
    cfg.status_file.write_text("{not json")
    assert health.load_status(cfg) == {}
    cfg.status_file.write_text("[1, 2]")
    assert health.load_status(cfg) == {}


def test_healthy_status_has_no_warnings(cfg, conn):
    assert warn(cfg, conn) == []
    assert health.warnings_block([]) == ""


def test_missing_status_warns(conn):
    (w,) = health.health_warnings({}, conn, now=NOW)
    assert "heartbeat missing" in w


def test_stale_heartbeat(cfg, conn):
    (w,) = warn(cfg, conn, written_at=iso(timedelta(minutes=25)))
    assert "heartbeat is 25m old" in w


def test_stale_reconcile(cfg, conn):
    (w,) = warn(cfg, conn, last_reconcile_at=iso(timedelta(hours=3)))
    assert "reconcile" in w and "3h" in w
    (w,) = warn(cfg, conn, last_reconcile_at=None)  # started a day ago, never reconciled
    assert "no full reconcile" in w
    assert warn(cfg, conn, last_reconcile_at=None, started_at=iso(timedelta(minutes=5))) == []


def test_syncthing_device_and_reachability(cfg, conn):
    devices = {"a": {"connected": False, "last_seen": iso(timedelta(hours=30))},
               "b": {"connected": False, "last_seen": iso(timedelta(hours=2))},
               "c": {"connected": True, "last_seen": iso(timedelta(days=9))},
               "d": {"connected": False, "last_seen": None}}
    ws = warn(cfg, conn, syncthing={"reachable": True, "devices": devices})
    assert ws == ["Syncthing device a disconnected for 1d 6h", "Syncthing device d disconnected (never seen)"]
    (w,) = warn(cfg, conn, syncthing={"reachable": False})
    assert "Syncthing API unreachable" in w


def test_conflicts_come_from_the_table(cfg, conn):
    for i in range(4):
        conn.execute("INSERT INTO conflicts(root, rel_path, seen_at) VALUES ('synced', ?, 'x')",
                     (f"-k/memory/f{i}.sync-conflict-1.md",))
    (w,) = warn(cfg, conn, conflicts=0)
    assert w.startswith("4 Syncthing conflict file(s)") and "(+1 more)" in w


def test_disk_low(cfg, conn):
    (w,) = warn(cfg, conn, disk_free_gb=12.3)
    assert "12.3 GB free" in w


def test_import_age_only_after_first_import(cfg, conn):
    assert warn(cfg, conn) == []
    conn.execute("INSERT INTO imports(name, ingested_at) VALUES ('a.zip', ?)", (iso(timedelta(days=60)),))
    (w,) = warn(cfg, conn)
    assert "claude.ai export" in w and "60d" in w
    conn.execute("INSERT INTO imports(name, ingested_at) VALUES ('b.zip', ?)", (iso(timedelta(days=3)),))
    assert warn(cfg, conn) == []


def test_embedding_backlog(cfg, conn):
    (w,) = warn(cfg, conn, embedding={"pending_chunks": 40, "oldest_pending_ts": iso(timedelta(hours=30))})
    assert "embedding backlog: 40 chunk(s)" in w


def test_drift(cfg, conn):
    db.bump_drift(conn, "record_type", "new-thing", 3)
    db.bump_drift(conn, "entrypoint", "sdk-new", 1)
    db.bump_drift(conn, "malformed", "lines", 9)  # not a drift warning
    ws = warn(cfg, conn)
    assert ws == ["unknown entrypoints seen (transcript format drift?): sdk-new (1)",
                  "unknown record types seen (transcript format drift?): new-thing (3)"]
    assert health.warnings_block(ws).startswith("## ⚠ Hub health\n- unknown entrypoints")
