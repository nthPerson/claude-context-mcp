"""Hub self-reporting health (spec §11.3).

The indexer writes ``status.json`` every minute; the server only reads it and turns stale or
bad values into warnings that are prepended to ``project_brief``/``recent_activity`` output.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from .config import Config
from .render import fmt_duration, parse_iso

HEARTBEAT_MAX = timedelta(minutes=10)
RECONCILE_MAX = timedelta(hours=1)
DEVICE_MAX = timedelta(hours=24)
DISK_MIN_GB = 30.0
IMPORT_MAX = timedelta(days=45)
EMBED_MAX = timedelta(hours=24)
DRIFT_CATEGORIES = ("record_type", "entrypoint")


def load_status(cfg: Config) -> dict[str, Any]:
    """The indexer's last heartbeat; ``{}`` when missing or unreadable."""
    try:
        data = json.loads(cfg.status_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def as_dict(value: Any) -> dict[str, Any]:
    """``value`` if it is a dict, else ``{}`` (status.json is loosely structured)."""
    return value if isinstance(value, dict) else {}


def _older(ts: Any, limit: timedelta, now: datetime) -> timedelta | None:
    """Age of ``ts`` if it is older than ``limit`` (None if fresh or unparseable)."""
    dt = parse_iso(ts)
    return now - dt if dt and now - dt > limit else None


def health_warnings(status: dict[str, Any], conn: sqlite3.Connection, *, now: datetime) -> list[str]:
    """Every §11.3 warning condition that currently holds, as one-line strings."""
    out: list[str] = []
    written = parse_iso(status.get("written_at"))
    if written is None:
        out.append("indexer heartbeat missing (no status.json): the index may be stale")
    elif now - written > HEARTBEAT_MAX:
        out.append(f"indexer heartbeat is {fmt_duration(now - written)} old: the indexer may be down")

    if status:
        last = status.get("last_reconcile_at")
        if age := _older(last, RECONCILE_MAX, now):
            out.append(f"last full reconcile was {fmt_duration(age)} ago")
        elif not last and (age := _older(status.get("started_at"), RECONCILE_MAX, now)):
            out.append(f"no full reconcile completed since the indexer started {fmt_duration(age)} ago")

    st = status.get("syncthing")
    if isinstance(st, dict):
        if st.get("reachable") is False:
            out.append("Syncthing API unreachable: machine attribution and sync state unknown")
        for name, dev in sorted(as_dict(st.get("devices")).items()):
            if not isinstance(dev, dict) or dev.get("connected") is not False:
                continue
            seen = parse_iso(dev.get("last_seen"))
            if seen is None:
                out.append(f"Syncthing device {name} disconnected (never seen)")
            elif now - seen > DEVICE_MAX:
                out.append(f"Syncthing device {name} disconnected for {fmt_duration(now - seen)}")

    conflicts = conn.execute("SELECT root, rel_path FROM conflicts ORDER BY rel_path").fetchall()
    if conflicts:
        shown = ", ".join(f"{r['root']}:{r['rel_path']}" for r in conflicts[:3])
        more = f" (+{len(conflicts) - 3} more)" if len(conflicts) > 3 else ""
        out.append(f"{len(conflicts)} Syncthing conflict file(s) need manual review: {shown}{more}")

    disk = status.get("disk_free_gb")
    if isinstance(disk, (int, float)) and disk < DISK_MIN_GB:
        out.append(f"disk space low: {disk:.1f} GB free (threshold {DISK_MIN_GB:.0f} GB)")

    newest = conn.execute("SELECT MAX(ingested_at) FROM imports").fetchone()[0]
    if age := _older(newest, IMPORT_MAX, now):
        out.append(f"newest claude.ai export was imported {fmt_duration(age)} ago: consider a fresh export")

    emb = as_dict(status.get("embedding"))
    if age := _older(emb.get("oldest_pending_ts"), EMBED_MAX, now):
        out.append(f"embedding backlog: {emb.get('pending_chunks', '?')} chunk(s) pending, "
                   f"oldest waiting {fmt_duration(age)}")

    marks = ",".join("?" * len(DRIFT_CATEGORIES))
    for r in conn.execute(f"SELECT category, GROUP_CONCAT(value || ' (' || count || ')', ', ') AS v "
                          f"FROM drift WHERE category IN ({marks}) GROUP BY category ORDER BY category",
                          DRIFT_CATEGORIES):
        label = "record types" if r["category"] == "record_type" else "entrypoints"
        out.append(f"unknown {label} seen (transcript format drift?): {r['v']}")
    return out


def warnings_block(warnings: list[str]) -> str:
    """The ``⚠ Hub health`` block prepended to brief/activity output ('' when healthy)."""
    if not warnings:
        return ""
    return "## ⚠ Hub health\n" + "".join(f"- {w}\n" for w in warnings) + "\n"
