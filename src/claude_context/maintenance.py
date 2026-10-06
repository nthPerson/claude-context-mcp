"""Daily maintenance: archive, .stversions recovery, retention prune, conflict merge, backups."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import db, writes
from .config import Config, RootConfig, default_config_path
from .indexer import Indexer, classify
from .roots import root_dir

log = logging.getLogger(__name__)

KEEP_BACKUPS = 14
DISK_WARN_GB = 30
# Syncthing version suffix: "<stem>~YYYYMMDD-HHMMSS<.ext>"
_VERSION = re.compile(r"^(?P<stem>.+)~(?P<ts>\d{8}-\d{6})(?P<ext>(?:\.[A-Za-z0-9]+)?)$")


@dataclass
class MaintenanceReport:
    archived_ok: bool = True
    recovered: int = 0
    pruned_sessions: int = 0
    pruned_files: int = 0
    conflicts_merged: int = 0
    backups: int = 0
    disk_free_gb: float = 0.0
    vacuumed: bool = False
    errors: list[str] = field(default_factory=list)


def live_root_of(cfg: Config, sv: RootConfig) -> RootConfig | None:
    """The claude-code root whose Syncthing versions folder ``sv`` is."""
    return next((r for r in cfg.roots if r.kind == "claude-code" and r.path == sv.path.parent), None)


# --- archive -----------------------------------------------------------------------------

def archive_roots(cfg: Config) -> list[str]:
    """Copy every live root into ``archive/<label>/`` (never deleting). Returns error strings."""
    errors = []
    for root in cfg.roots:
        if root.kind != "claude-code" or not root.path.is_dir():
            continue
        dest = cfg.archive_dir / root.label
        dest.mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", "-a", "--exclude=/.stversions", "--exclude=/.stfolder", "--exclude=.syncthing.*",
               "--exclude=~syncthing~*", "--exclude=*.sync-conflict-*", f"{root.path}/", f"{dest}/"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"archive {root.label}: {e}")
            continue
        if proc.returncode not in (0, 24):  # 24 = some files vanished mid-transfer (live sessions)
            errors.append(f"archive {root.label}: rsync exit {proc.returncode}: {proc.stderr.strip()[-300:]}")
    return errors


# --- .stversions recovery ----------------------------------------------------------------

def _recoverable(rel: str) -> bool:
    """Transcripts, subagent transcripts and their side files; never memory or notes."""
    c = classify(rel)
    if c is not None:
        return c.kind in ("transcript", "subagent")
    parts = rel.split("/")
    if len(parts) < 4 or parts[0].startswith("."):
        return False
    return (parts[2] == "subagents" and rel.endswith(".meta.json")) or parts[2] == "tool-results"


def plan_recovery(cfg: Config, sv: RootConfig) -> list[tuple[Path, str]]:
    """(source version file, live-relative path) for the latest version of each lost file."""
    live = live_root_of(cfg, sv)
    if live is None or not sv.path.is_dir():
        return []
    cutoff = time.time() - cfg.retention_days * 86400
    latest: dict[str, tuple[str, Path]] = {}
    for path in sv.path.rglob("*~*"):
        m = _VERSION.match(path.name)
        if m is None or not path.is_file():
            continue
        rel = (path.parent / (m["stem"] + m["ext"])).relative_to(sv.path).as_posix()
        if rel not in latest or m["ts"] > latest[rel][0]:
            latest[rel] = (m["ts"], path)
    dest_root = root_dir(cfg, sv)
    plan = []
    for rel, (_, src) in sorted(latest.items()):
        if not _recoverable(rel) or (live.path / rel).exists() or (cfg.archive_dir / live.label / rel).exists():
            continue
        st = src.stat()
        if st.st_mtime < cutoff:
            continue
        dest = dest_root / rel
        if dest.exists() and dest.stat().st_mtime >= st.st_mtime:
            continue
        plan.append((src, rel))
    return plan


def recover_stversions(cfg: Config, conn: sqlite3.Connection | None, *, dry_run: bool = False) -> list[str]:
    """Copy sessions deleted upstream out of Syncthing's versions folder and index them."""
    recovered = []
    for sv in cfg.roots:
        if sv.kind != "stversions":
            continue
        dest_root = root_dir(cfg, sv)
        for src, rel in plan_recovery(cfg, sv):
            recovered.append(f"{sv.label}/{rel}")
            if dry_run:
                continue
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        if not dry_run and conn is not None:
            Indexer(cfg, conn).reconcile(only_root=sv.label)
    return recovered


# --- retention ---------------------------------------------------------------------------

def prune(cfg: Config, conn: sqlite3.Connection, *, dry_run: bool = False, now: datetime | None = None) -> tuple[int, int]:
    """Delete sessions that ended more than ``retention_days`` ago (rows + archived files).

    Live files under a claude-code root are never touched: Claude Code's own
    ``cleanupPeriodDays`` removes those. Memories and notes are never pruned.
    """
    now = now or datetime.now(UTC)
    cutoff = (now - timedelta(days=cfg.retention_days)).strftime("%Y-%m-%dT%H:%M:%S") + ".000Z"
    rows = conn.execute(
        "SELECT s.id, f.id AS file_id, f.root, f.rel_path FROM sessions s LEFT JOIN files f ON f.id = s.file_id "
        "WHERE COALESCE(s.ended_at, s.started_at, '') < ? AND COALESCE(s.ended_at, s.started_at) IS NOT NULL",
        (cutoff,)).fetchall()
    files_removed = 0
    for r in rows:
        if dry_run:
            continue
        db.delete_session(conn, r["id"])
        if r["rel_path"]:
            for base in {cfg.archive_dir / r["root"]}:
                f = base / r["rel_path"]
                if f.is_file():
                    f.unlink()
                    files_removed += 1
                side = f.with_suffix("")  # <session-id>/ with subagents and tool-results
                if side.is_dir() and ":" not in r["id"]:
                    files_removed += sum(1 for p in side.rglob("*") if p.is_file())
                    shutil.rmtree(side, ignore_errors=True)
            root = cfg.root(r["root"])
            live = root is not None and root.kind == "claude-code" and (root.path / r["rel_path"]).exists()
            if not live:
                conn.execute("DELETE FROM files WHERE id = ?", (r["file_id"],))
    if not dry_run:
        db.set_meta(conn, "last_prune", json.dumps({"at": db.utcnow(), "sessions_deleted": len(rows)}))
        conn.commit()
    return len(rows), files_removed


# --- conflicts ---------------------------------------------------------------------------

def merge_conflicts(cfg: Config, *, dry_run: bool = False) -> list[writes.ConflictMerge]:
    """Auto-merge ``MEMORY.sync-conflict-*.md`` files in every writable root."""
    merges: list[writes.ConflictMerge] = []
    for root in cfg.roots:
        if root.kind != "claude-code" or not root.writable or not root.path.is_dir():
            continue
        for project_dir in sorted(p for p in root.path.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if (project_dir / "memory").is_dir():
                merges.extend(writes.merge_index_conflicts(project_dir, dry_run=dry_run))
    return merges


# --- backups and housekeeping ------------------------------------------------------------

def backup(cfg: Config) -> int:
    """Nightly copies of audit.db and the config file; keeps the newest KEEP_BACKUPS of each."""
    cfg.backups_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    made = 0
    if cfg.audit_db.is_file():
        dest = cfg.backups_dir / f"audit-{stamp}.db"
        src = sqlite3.connect(f"file:{cfg.audit_db}?mode=ro", uri=True)
        try:
            out = sqlite3.connect(dest)
            with out:
                src.backup(out)
            out.close()
            made += 1
        finally:
            src.close()
        dest.chmod(0o600)
    config_path = default_config_path()
    if config_path.is_file():
        dest = cfg.backups_dir / f"config-{stamp}.toml"
        shutil.copy2(config_path, dest)
        dest.chmod(0o600)
        made += 1
    for pattern in ("audit-*.db", "config-*.toml"):
        for old in sorted(cfg.backups_dir.glob(pattern))[:-KEEP_BACKUPS]:
            old.unlink()
    return made


def disk_free_gb(path: Path) -> float:
    return round(shutil.disk_usage(path).free / 1024**3, 1)


def run_all(cfg: Config, conn: sqlite3.Connection, *, now: datetime | None = None) -> MaintenanceReport:
    now = now or datetime.now(UTC)
    report = MaintenanceReport()

    def step(name: str, fn):
        try:
            return fn()
        except Exception as e:
            log.exception("maintenance step failed: %s", name)
            report.errors.append(f"{name}: {type(e).__name__}: {e}")
            return None

    errs = step("archive", lambda: archive_roots(cfg)) or []
    report.errors.extend(errs)
    report.archived_ok = not errs
    report.recovered = len(step("recover", lambda: recover_stversions(cfg, conn)) or [])
    pruned = step("prune", lambda: prune(cfg, conn, now=now))
    if pruned:
        report.pruned_sessions, report.pruned_files = pruned
    merges = step("merge-conflicts", lambda: merge_conflicts(cfg)) or []
    report.conflicts_merged = sum(1 for m in merges if m.merged)
    if step("audit-prune", lambda: _prune_audit(cfg)) is None:
        pass
    report.backups = step("backup", lambda: backup(cfg)) or 0
    step("optimize", lambda: conn.execute("PRAGMA optimize"))
    if now.weekday() == 6:  # Sunday
        report.vacuumed = step("vacuum", lambda: (conn.commit(), conn.execute("VACUUM"))) is not None
    report.disk_free_gb = disk_free_gb(cfg.data_dir)
    if report.disk_free_gb < DISK_WARN_GB:
        log.warning("low disk space: %.1f GB free", report.disk_free_gb)
    db.set_meta(conn, "last_maintenance", json.dumps({"at": db.utcnow(), **asdict(report)}))
    conn.commit()
    return report


def _prune_audit(cfg: Config) -> int:
    if not cfg.audit_db.is_file():
        return 0
    from .audit import AuditLog
    audit = AuditLog(cfg.audit_db)
    try:
        return audit.prune(older_than_days=365)
    finally:
        audit.close()
