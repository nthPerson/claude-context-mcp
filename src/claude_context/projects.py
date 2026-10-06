"""Project registry: maps Claude Code project keys (folder names) to human aliases.

Projects come from three places: the config file (``[projects.<alias>]``), automatic
discovery of unknown keys, and hub-only projects (``-hub-<alias>`` folders created by the
write tools for work that has no Claude Code folder).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .config import Config

WORKTREE_MARK = "--claude-worktrees-"
HUB_PREFIX = "-hub-"
CLAUDEAI_ALIAS = "claude-ai"  # catch-all project for unmapped claude.ai conversations
_HOME_PREFIX = re.compile(r"^-(?:home|Users)-[^-]+")
_SLUG = re.compile(r"[^a-z0-9]+")
ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


@dataclass
class Project:
    id: int
    alias: str
    display: str
    created_by: str
    keys: list[str] = field(default_factory=list)  # primary key first

    @property
    def primary_key(self) -> str | None:
        return self.keys[0] if self.keys else None


def parent_key(key: str) -> str:
    """Worktree keys fold into the project they were created from."""
    return key.split(WORKTREE_MARK, 1)[0]


def slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")


def auto_alias(key: str, cwd: str | None = None) -> str:
    """Derive an alias for a key that is not in the config."""
    key = parent_key(key)
    if key.startswith(HUB_PREFIX):
        return slug(key[len(HUB_PREFIX):]) or "hub"
    m = _HOME_PREFIX.match(key)
    if m:
        return slug(key[m.end():]) or "home"
    if cwd:  # e.g. a Windows drive mount: the last two path components identify it well
        parts = [p for p in PurePosixPath(cwd.replace("\\", "/")).parts if p not in ("/", "")]
        if parts:
            return slug("-".join(parts[-2:])) or "project"
    return slug("-".join(key.strip("-").split("-")[-2:])) or "project"


def sync_config_projects(conn: sqlite3.Connection, cfg: Config) -> None:
    """Make the projects/project_keys tables reflect the config (config wins over auto)."""
    for alias, pc in cfg.projects.items():
        alias = alias.lower()
        display = pc.display or alias
        row = conn.execute("SELECT id FROM projects WHERE alias = ?", (alias,)).fetchone()
        if row:
            pid = row[0]
            conn.execute("UPDATE projects SET display = ?, created_by = 'config' WHERE id = ?", (display, pid))
        else:
            pid = conn.execute("INSERT INTO projects(alias, display, created_by) VALUES (?,?,'config')",
                               (alias, display)).lastrowid
        for i, key in enumerate(pc.keys):
            old = conn.execute("SELECT project_id FROM project_keys WHERE key = ?", (key,)).fetchone()
            conn.execute(
                "INSERT INTO project_keys(key, project_id, is_primary) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET project_id = excluded.project_id, is_primary = excluded.is_primary",
                (key, pid, int(i == 0)),
            )
            if old and old[0] != pid:
                _move_project(conn, key, old[0], pid)
    conn.commit()


def _move_project(conn: sqlite3.Connection, key: str, old_pid: int, new_pid: int) -> None:
    """A key was re-assigned by config: move its rows and drop the old auto project if empty."""
    conn.execute("UPDATE memories SET project_id = ? WHERE project_id = ? AND project_key = ?", (new_pid, old_pid, key))
    conn.execute("UPDATE notes SET project_id = ? WHERE project_id = ? AND project_key = ?", (new_pid, old_pid, key))
    if not conn.execute("SELECT 1 FROM project_keys WHERE project_id = ?", (old_pid,)).fetchone():
        for table in ("sessions", "docs", "chunks", "memories", "notes"):
            conn.execute(f"UPDATE {table} SET project_id = ? WHERE project_id = ?", (new_pid, old_pid))
        conn.execute("UPDATE chunks SET embedded = 0 WHERE project_id = ?", (new_pid,))
        conn.execute("DELETE FROM projects WHERE id = ? AND created_by != 'config'", (old_pid,))


def ensure_project(conn: sqlite3.Connection, alias: str, display: str, created_by: str) -> int:
    row = conn.execute("SELECT id FROM projects WHERE alias = ?", (alias,)).fetchone()
    if row:
        return row[0]
    return conn.execute("INSERT INTO projects(alias, display, created_by) VALUES (?,?,?)",
                        (alias, display, created_by)).lastrowid


def project_for_key(conn: sqlite3.Connection, key: str, cwd: str | None = None) -> int:
    """Project id for a key, creating an auto (or hub) project on first sight."""
    row = conn.execute("SELECT project_id FROM project_keys WHERE key = ?", (key,)).fetchone()
    if row:
        return row[0]
    parent = parent_key(key)
    if parent != key:
        pid = project_for_key(conn, parent, cwd)
        conn.execute("INSERT INTO project_keys(key, project_id, is_primary) VALUES (?,?,0)", (key, pid))
        return pid
    base = auto_alias(key, cwd)
    if key.startswith(HUB_PREFIX):  # a hub folder for an existing project attaches to it
        row = conn.execute("SELECT id FROM projects WHERE alias = ?", (base,)).fetchone()
        if row:
            has_keys = conn.execute("SELECT 1 FROM project_keys WHERE project_id = ?", (row[0],)).fetchone()
            conn.execute("INSERT INTO project_keys(key, project_id, is_primary) VALUES (?,?,?)",
                         (key, row[0], int(not has_keys)))
            return row[0]
    alias, n = base, 1
    while conn.execute("SELECT 1 FROM projects WHERE alias = ?", (alias,)).fetchone():
        n += 1
        alias = f"{base}-{n}"
    created_by = "hub" if key.startswith(HUB_PREFIX) else "auto"
    pid = conn.execute("INSERT INTO projects(alias, display, created_by) VALUES (?,?,?)",
                       (alias, alias, created_by)).lastrowid
    conn.execute("INSERT INTO project_keys(key, project_id, is_primary) VALUES (?,?,1)", (key, pid))
    return pid


def list_projects(conn: sqlite3.Connection) -> list[Project]:
    projects = {r["id"]: Project(r["id"], r["alias"], r["display"], r["created_by"])
                for r in conn.execute("SELECT id, alias, display, created_by FROM projects ORDER BY alias")}
    for r in conn.execute("SELECT key, project_id FROM project_keys ORDER BY is_primary DESC, key"):
        if r["project_id"] in projects:
            projects[r["project_id"]].keys.append(r["key"])
    return list(projects.values())


def resolve(projects: list[Project], query: str) -> tuple[Project | None, list[Project]]:
    """Resolve an alias, display name or raw key (case-insensitive, unique-prefix).

    Returns ``(project, [])`` on success, ``(None, candidates)`` when ambiguous and
    ``(None, [])`` when nothing matches.
    """
    q = query.strip().lower()
    if not q:
        return None, []
    for p in projects:
        if q == p.alias.lower() or q == p.display.lower() or any(q == k.lower() for k in p.keys):
            return p, []
    hits = [p for p in projects if p.alias.lower().startswith(q) or p.display.lower().startswith(q)]
    if len(hits) == 1:
        return hits[0], []
    return None, hits
