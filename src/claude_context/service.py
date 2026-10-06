"""Tool service: one method per MCP tool, each returning Markdown text (spec §8.5).

Framework-free: the MCP layer registers these methods and adds auth, auditing and rate
limiting. Every call opens its own read-only index connection. The only files the service
writes are memories, ``MEMORY.md`` and notes (through ``writes``) plus reindex queue entries.
Raw files are always passed through ``redact`` before they are returned; raw transcript lines
are only ever read through ``parse_line_full``, which never yields thinking blocks.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import __version__, db, health, writes
from . import search as search_mod
from .config import Config
from .health import as_dict
from .models import ROLE_ASSISTANT, ROLE_SUMMARY, ROLE_TOOL_CALL, ROLE_TOOL_RESULT, ROLE_USER
from .parsers.transcript import parse_line_full
from .projects import ALIAS_RE, HUB_PREFIX, Project, list_projects, resolve
from .redact import redact
from .render import (
    cap,
    cut_at_line,
    decode_cursor,
    encode_cursor,
    fmt_duration,
    fmt_span,
    fmt_time,
    iso_utc,
    one_line,
    parse_iso,
    parse_when,
    size,
    table,
)
from .render import row as table_row
from .roots import enqueue_reindex, raw_path, root_dir

DETAIL_ROLES = {
    "conversation": (ROLE_USER, ROLE_ASSISTANT),
    "tools": (ROLE_USER, ROLE_ASSISTANT, ROLE_SUMMARY, ROLE_TOOL_CALL, ROLE_TOOL_RESULT),
    "full": (ROLE_USER, ROLE_ASSISTANT, ROLE_SUMMARY, ROLE_TOOL_CALL, ROLE_TOOL_RESULT),
}
SESSION_KINDS = ("interactive", "automated", "all")
SOURCES = ("claude-code", "claude.ai", "recovered")
SEARCH_MODES = ("hybrid", "keyword", "semantic")
FULL_RESULT_CHARS = 2000
EXCERPT_CHARS = 500
INACTIVE_DAYS = 90
MIN_MAX_CHARS = 1000
FOOTER_RESERVE = 300
MIN_PREFIX = 8
SEP = "\n---\n"  # between a paged tool's header, body and footer

_ACT = "COALESCE(s.ended_at, s.started_at, '')"  # a session's activity time
_SESSION_SQL = (
    "SELECT s.*, p.alias, f.missing_since FROM sessions s JOIN projects p ON p.id = s.project_id "
    "LEFT JOIN files f ON f.id = s.file_id"
)
_NOTES_SQL = "SELECT n.*, p.alias FROM notes n JOIN projects p ON p.id = n.project_id"
_TOOL_REF_RE = re.compile(r"tool-results/([A-Za-z0-9][A-Za-z0-9._-]*\.(?:txt|json))\b")
_TOOL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_NOTE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")


class ToolError(Exception):
    """Error whose message is shown to the model: safe, specific and actionable."""


def _where(base: str, *conds: tuple[str, Any]) -> tuple[str, list[Any]]:
    """``base AND cond…`` for every condition whose value is set (None and '' are skipped)."""
    parts, args = [base], []
    for cond, value in conds:
        if value is not None and value != "":
            parts.append(cond)
            args.append(value)
    return " AND ".join(parts), args


def _remove_empty(project_dir: Path) -> None:
    """Undo a project folder created for a write that then failed (only if still empty)."""
    for d in (project_dir / "memory", project_dir / "remote-notes", project_dir):
        try:
            d.rmdir()
        except OSError:
            pass


class HubService:
    def __init__(self, cfg: Config, *, embedder: Any = None, auth_failures: Callable[[], int] | None = None,
                 started_at: datetime | None = None, clock: Callable[[], datetime] | None = None):
        self.cfg = cfg
        self.embedder = embedder
        self.auth_failures = auth_failures
        self.clock = clock or (lambda: datetime.now(UTC))
        self.started_at = started_at or self._now()
        self.tz = ZoneInfo(cfg.timezone)

    # --- shared helpers ------------------------------------------------------------------

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        if not self.cfg.index_db.is_file():
            raise ToolError("index not built yet: the indexer has not created index.db; "
                            "start it (claude-context index) and retry in a few minutes")
        conn = db.connect(self.cfg.index_db, readonly=True)
        try:
            yield conn
        finally:
            conn.close()

    def _now(self) -> datetime:
        now = self.clock()
        return now.replace(tzinfo=UTC) if now.tzinfo is None else now

    def _t(self, ts: str | None) -> str:
        return fmt_time(ts, self.tz)

    def _max_chars(self, max_chars: int | None) -> int:
        if max_chars is None:
            return self.cfg.default_max_chars
        if max_chars > self.cfg.max_max_chars:
            raise ToolError(f"max_chars={max_chars} exceeds the cap of {self.cfg.max_max_chars}; "
                            "page with cursor/offset or narrow the query instead")
        if max_chars < MIN_MAX_CHARS:
            raise ToolError(f"max_chars must be at least {MIN_MAX_CHARS}")
        return max_chars

    @staticmethod
    def _range(name: str, value: int, lo: int, hi: int) -> int:
        if not isinstance(value, int) or not lo <= value <= hi:
            raise ToolError(f"{name} must be an integer between {lo} and {hi}")
        return value

    @staticmethod
    def _choice(name: str, value: str, options: tuple[str, ...]) -> str:
        if value not in options:
            raise ToolError(f"{name} must be one of: {', '.join(options)}")
        return value

    def _bound(self, value: str | None, name: str, *, end: bool = False) -> str | None:
        """since/until → ISO UTC string comparable with stored timestamps."""
        if value is None or not str(value).strip():
            return None
        try:
            return iso_utc(parse_when(str(value), now=self._now(), tz=self.tz, end=end))
        except ValueError:
            raise ToolError(f"invalid {name}={value!r}: use a relative span (7d, 36h, 2w) "
                            "or an ISO date/datetime") from None

    def _project(self, conn: sqlite3.Connection, query: str | None) -> Project:
        projects = list_projects(conn)
        p, candidates = resolve(projects, query or "")
        if p is not None:
            return p
        if candidates:
            raise ToolError(f"project {query!r} is ambiguous; candidates: "
                            f"{', '.join(c.alias for c in candidates)}")
        raise ToolError(f"unknown project {query!r}; available: "
                        f"{', '.join(x.alias for x in projects) or 'none'}")

    def _project_id(self, conn: sqlite3.Connection, query: str | None) -> int | None:
        return self._project(conn, query).id if query else None

    def _health(self, conn: sqlite3.Connection) -> list[str]:
        return health.health_warnings(health.load_status(self.cfg), conn, now=self._now())

    def _session(self, conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
        """A session by full id or unique prefix (≥ 8 chars)."""
        sid = (session_id or "").strip()
        row = conn.execute(f"{_SESSION_SQL} WHERE s.id = ?", (sid,)).fetchone()
        if row:
            return row
        if len(sid) < MIN_PREFIX:
            raise ToolError(f"session {sid!r} not found (an id prefix needs at least {MIN_PREFIX} characters)")
        like = re.sub(r"([\\%_])", r"\\\1", sid) + "%"
        # Subagent ids extend their parent's id, so a prefix prefers top-level sessions.
        rows = conn.execute(f"{_SESSION_SQL} WHERE s.id LIKE ? ESCAPE '\\' AND s.is_subagent = 0 "
                            "ORDER BY s.id LIMIT 6", (like,)).fetchall()
        rows = rows or conn.execute(f"{_SESSION_SQL} WHERE s.id LIKE ? ESCAPE '\\' ORDER BY s.id LIMIT 6",
                                    (like,)).fetchall()
        if len(rows) == 1:
            return rows[0]
        if rows:
            raise ToolError(f"session prefix {sid!r} is ambiguous; candidates: "
                            + ", ".join(r["id"] for r in rows[:5]) + (" …" if len(rows) > 5 else ""))
        raise ToolError(f"session {sid!r} not found; use list_sessions or search to find ids")

    @staticmethod
    def _session_where(*, project_id: int | None = None, kind: str = "interactive", machine: str | None = None,
                       source: str | None = None, since: str | None = None, until: str | None = None,
                       exclude_source: str | None = None) -> tuple[str, list[Any]]:
        """WHERE clause for top-level session listings (subagents are always excluded)."""
        return _where("s.is_subagent = 0", ("s.project_id = ?", project_id),
                      ("s.kind = ?", None if kind == "all" else kind), ("s.machine = ? COLLATE NOCASE", machine),
                      ("s.source = ?", source), (f"{_ACT} >= ?", since), (f"{_ACT} <= ?", until),
                      ("s.source != ?", exclude_source))

    def _session_item(self, s: sqlite3.Row) -> str:
        """Bullet for a session in project_brief."""
        bits = [self._t(s["ended_at"] or s["started_at"]), f"machine {s['machine']}", f"source {s['source']}"]
        bits += [b for b in (s["entrypoint"], "automated" if s["kind"] == "automated" else None,
                             s["git_branch"] and f"branch {s['git_branch']}") if b]
        title = one_line(s["title"] or s["first_prompt"] or "(untitled)", 120)
        out = f"- **{title}** · {' · '.join(bits)} · `{s['id']}`\n"
        if s["last_prompt"]:
            out += f"  - last prompt: {one_line(s['last_prompt'], 300)}\n"
        if s["final_reply_excerpt"]:
            out += f"  - final reply: {one_line(s['final_reply_excerpt'], EXCERPT_CHARS)}\n"
        return out

    def _memory_dirs(self, p: Project) -> list[tuple[str, str, Path]]:
        """(root label, key, project dir) of every memory folder of a project, primary key first."""
        out = []
        for key in p.keys:
            for root in self.cfg.roots:
                if root.kind == "claude-code" and (root_dir(self.cfg, root) / key / "memory").is_dir():
                    out.append((root.label, key, root_dir(self.cfg, root) / key))
        return out

    def _read_raw(self, root: str, rel_path: str) -> str | None:
        """Redacted text of an indexed file (live copy, else archive); None if unavailable."""
        path = raw_path(self.cfg, root, rel_path)
        if path is None:
            return None
        try:
            return redact(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return None

    # --- projects ------------------------------------------------------------------------

    def list_projects(self, *, include_inactive: bool = False, max_chars: int | None = None) -> str:
        """Table of projects with activity, counts, roots and machines."""
        limit = self._max_chars(max_chars)
        with self._db() as conn:
            projects = list_projects(conn)
            st: dict[int, dict[str, Any]] = {
                p.id: {"interactive": 0, "automated": 0, "memories": 0, "notes": 0, "last": "",
                       "roots": set(), "machines": set()} for p in projects}

            def add(pid: int, field: str | None, n: int, last: str | None, root: str | None,
                    machine: str | None) -> None:
                d = st.get(pid)
                if d is None:
                    return
                if field:
                    d[field] += n
                d["last"] = max(d["last"], last or "")
                d["roots"].update([root] if root else [])
                d["machines"].update([machine] if machine else [])

            for r in conn.execute("SELECT project_id, kind, root, machine, COUNT(*) n, "
                                  "MAX(COALESCE(ended_at, started_at)) last FROM sessions WHERE is_subagent = 0 "
                                  "GROUP BY 1, 2, 3, 4"):
                add(r["project_id"], r["kind"] if r["kind"] in ("interactive", "automated") else None,
                    r["n"], r["last"], r["root"], r["machine"])
            for r in conn.execute("SELECT project_id, root, modified_by, COUNT(*) n, MAX(modified_at) last "
                                  "FROM memories WHERE is_index = 0 AND archived = 0 GROUP BY 1, 2, 3"):
                add(r["project_id"], "memories", r["n"], r["last"], r["root"], r["modified_by"])
            for r in conn.execute("SELECT project_id, root, COUNT(*) n, MAX(created_at) last FROM notes GROUP BY 1, 2"):
                add(r["project_id"], "notes", r["n"], r["last"], r["root"], None)

        cutoff = iso_utc(self._now() - timedelta(days=INACTIVE_DAYS))
        shown = [p for p in projects if include_inactive or st[p.id]["last"] >= cutoff]
        shown.sort(key=lambda p: st[p.id]["last"], reverse=True)
        rows = [[p.alias, p.display, ", ".join(p.keys) or "–", ", ".join(sorted(st[p.id]["roots"])),
                 self._t(st[p.id]["last"] or None), st[p.id]["interactive"], st[p.id]["automated"],
                 st[p.id]["memories"], st[p.id]["notes"], ", ".join(sorted(st[p.id]["machines"]))] for p in shown]
        out = f"# Projects ({len(shown)})\n\n" + table(
            ["alias", "display", "keys", "roots", "last activity", "interactive", "automated", "memories",
             "notes", "machines"], rows)
        hidden = len(projects) - len(shown)
        if hidden:
            out += (f"\n{hidden} project(s) with no activity in {INACTIVE_DAYS} days hidden; "
                    "include_inactive=true shows them.\n")
        return cap(out, limit, "raise max_chars")

    def project_brief(self, *, project: str, recent_sessions: int = 5, include_automated: bool = False,
                      max_chars: int | None = None) -> str:
        """Warnings, MEMORY.md, recent memories, sessions, notes and claude.ai conversations."""
        limit = self._max_chars(max_chars)
        self._range("recent_sessions", recent_sessions, 0, 50)
        with self._db() as conn:
            p = self._project(conn, project)
            warnings = self._health(conn) + [
                f"conflict file in this project: {r['root']}:{r['rel_path']}"
                for r in conn.execute("SELECT root, rel_path FROM conflicts WHERE project_id = ? ORDER BY rel_path",
                                      (p.id,))]
            out = [health.warnings_block(warnings), f"# {p.display} (`{p.alias}`)\n",
                   f"keys: {', '.join(p.keys) or '(none)'}\n\n"]

            dirs = self._memory_dirs(p)
            index_texts = []
            for _label, key, d in dirs:
                try:
                    text, _sha = writes.read_memory_file(d, writes.INDEX_NAME)
                except writes.WriteError:
                    continue
                suffix = f" (`{key}`)" if len(dirs) > 1 else ""
                index_texts.append(f"## MEMORY.md{suffix}\n{redact(text).strip()}\n\n")
            out += index_texts or ["## MEMORY.md\n(none)\n\n"]

            mems = conn.execute("SELECT stem, title, description, modified_at, modified_by, root FROM memories "
                                "WHERE project_id = ? AND is_index = 0 AND archived = 0 "
                                "ORDER BY modified_at DESC LIMIT 5", (p.id,)).fetchall()
            out.append("## Recently modified memories\n")
            out += [f"- `{m['stem']}` {one_line(m['title'] or '', 80)}: {one_line(m['description'] or '', 200)} · "
                    f"{self._t(m['modified_at'])} · by {m['modified_by'] or '?'} · source {m['root']}\n"
                    for m in mems] or ["(none)\n"]

            where, args = self._session_where(project_id=p.id, kind="all" if include_automated else "interactive",
                                              exclude_source="claude.ai")
            sessions = conn.execute(f"{_SESSION_SQL} WHERE {where} ORDER BY {_ACT} DESC LIMIT ?",
                                    (*args, recent_sessions)).fetchall()
            out.append(f"\n## Last {recent_sessions} sessions\n")
            out += [self._session_item(s) for s in sessions] or ["(none)\n"]

            notes = conn.execute("SELECT note_id, title, surface, created_at, machine FROM notes WHERE project_id = ? "
                                 "ORDER BY created_at DESC LIMIT 5", (p.id,)).fetchall()
            out.append("\n## Recent notes\n")
            out += [f"- **{one_line(n['title'] or '', 120)}** · {self._t(n['created_at'])} · surface {n['surface']} · "
                    f"machine {n['machine']} · `{n['note_id']}`\n" for n in notes] or ["(none)\n"]

            where, args = self._session_where(project_id=p.id, kind="all", source="claude.ai")
            convs = conn.execute(f"{_SESSION_SQL} WHERE {where} ORDER BY {_ACT} DESC LIMIT 3", args).fetchall()
            out.append("\n## Recent claude.ai conversations\n")
            out += [self._session_item(s) for s in convs] or ["(none)\n"]
        return cap("".join(out), limit, "raise max_chars or lower recent_sessions")

    def recent_activity(self, *, since: str = "7d", project: str | None = None, machine: str | None = None,
                        include_automated: bool = False, limit: int = 30, max_chars: int | None = None) -> str:
        """Cross-project timeline of sessions, memory changes and notes, newest first."""
        cap_chars = self._max_chars(max_chars)
        self._range("limit", limit, 1, 200)
        start = self._bound(since, "since") or ""
        items: list[tuple[str, list[Any]]] = []
        with self._db() as conn:
            pid = self._project_id(conn, project)
            warnings = self._health(conn)
            where, args = self._session_where(project_id=pid, kind="all" if include_automated else "interactive",
                                              machine=machine, since=start)
            for s in conn.execute(f"{_SESSION_SQL} WHERE {where} ORDER BY {_ACT} DESC LIMIT ?", (*args, limit)):
                title = one_line(s["title"] or s["first_prompt"] or "(untitled)", 100)
                auto = ", automated" if s["kind"] == "automated" else ""
                what = f"**{title}** ({s['entrypoint'] or '?'}{auto}, {s['n_user']} prompts)"
                items.append((s["ended_at"] or s["started_at"] or "",
                              [s["alias"], "session", s["machine"], s["source"], what, s["id"]]))
            where, args = _where("m.is_index = 0", ("m.modified_at >= ?", start), ("m.project_id = ?", pid),
                                 ("m.modified_by = ? COLLATE NOCASE", machine))
            for m in conn.execute("SELECT m.*, p.alias FROM memories m JOIN projects p ON p.id = m.project_id "
                                  f"WHERE {where} ORDER BY m.modified_at DESC LIMIT ?", (*args, limit)):
                what = one_line(m["title"] or m["name"] or "", 100) + (" (archived)" if m["archived"] else "")
                items.append((m["modified_at"] or "", [m["alias"], "memory", m["modified_by"] or "?", m["root"], what,
                                                       m["stem"]]))
            where, args = _where("1 = 1", ("n.created_at >= ?", start), ("n.project_id = ?", pid),
                                 ("n.machine = ? COLLATE NOCASE", machine))
            for n in conn.execute(f"{_NOTES_SQL} WHERE {where} ORDER BY n.created_at DESC LIMIT ?", (*args, limit)):
                items.append((n["created_at"] or "", [n["alias"], "note", n["machine"] or "?", n["root"],
                                                      f"{one_line(n['title'] or '', 100)} ({n['surface']})",
                                                      n["note_id"]]))
        items.sort(key=lambda it: it[0], reverse=True)
        items = items[:limit]
        out = health.warnings_block(warnings) + f"# Recent activity since {self._t(start)} ({len(items)} items)\n\n"
        if items:
            out += table(["when", "project", "type", "machine", "source", "item", "id"],
                         [[self._t(ts), *row] for ts, row in items])
        else:
            out += "No activity in this window; widen `since` or set include_automated=true.\n"
        return cap(out, cap_chars, "lower limit or narrow since/project")

    # --- search --------------------------------------------------------------------------

    def search(self, *, query: str, mode: str = "hybrid", project: str | None = None,
               kinds: list[str] | None = None, machine: str | None = None, since: str | None = None,
               until: str | None = None, include_automated: bool = False, limit: int = 10,
               context_chars: int = 300, max_chars: int | None = None) -> str:
        """Hybrid/keyword/semantic search across transcripts, memories, notes and claude.ai."""
        cap_chars = self._max_chars(max_chars)
        if not (query or "").strip():
            raise ToolError("query must not be empty")
        self._choice("mode", mode, SEARCH_MODES)
        self._range("limit", limit, 1, 50)
        self._range("context_chars", context_chars, 50, 2000)
        kinds = [kinds] if isinstance(kinds, str) else list(kinds or [])
        bad = [k for k in kinds if k not in db.KINDS]
        if bad:
            raise ToolError(f"unknown kinds {bad}; valid kinds: {', '.join(db.KINDS)}")
        lo, hi = self._bound(since, "since"), self._bound(until, "until", end=True)
        with self._db() as conn:
            pid = self._project_id(conn, project)
            params = search_mod.SearchParams(
                query=query, mode=mode, project_ids=[pid] if pid is not None else None, kinds=kinds or None,
                machine=machine or None, since=lo, until=hi, include_automated=include_automated, limit=limit,
                context_chars=context_chars)
            try:
                hits = search_mod.search(conn, params, self.embedder, now=self._now())
            except search_mod.SearchError as e:
                raise ToolError(f"search failed: {e}") from None
            aliases = dict(conn.execute("SELECT id, alias FROM projects").fetchall())
        out = [f"# Search: {one_line(query, 200)} ({len(hits)} hits, {mode})\n"]
        for i, h in enumerate(hits, 1):
            alias = aliases.get(h.project_id, "?")
            meta = [h.kind, alias, self._t(h.ts), f"machine {h.machine or '?'}", f"source {h.source or h.kind}"]
            if h.role:
                meta.append(h.role)
            if h.doc_type == "memory":
                anchor = f'read_memory(project="{alias}", name="{h.ref}")'
            elif h.doc_type == "note":
                anchor = f'read_note(project="{alias}", note_id="{h.ref}")'
            elif h.uuid:
                anchor = f'read_message(session_id="{h.session_id}", uuid="{h.uuid}")'
            else:
                anchor = f'read_session(session_id="{h.session_id}")'
            title = f"**{one_line(h.title, 150)}**\n" if h.title else ""
            out.append(f"\n### {i}. {' · '.join(meta)}\n{title}{h.snippet.strip()}\n→ {anchor}\n")
        if not hits:
            out.append("\nNo hits. Try other words, mode=\"keyword\" for exact terms, or widen the filters.\n")
        return cap(redact("".join(out)), cap_chars, "lower limit or context_chars")

    # --- sessions ------------------------------------------------------------------------

    def list_sessions(self, *, project: str | None = None, since: str | None = None, until: str | None = None,
                      machine: str | None = None, kind: str = "interactive", source: str | None = None,
                      limit: int = 20, cursor: str | None = None, max_chars: int | None = None) -> str:
        """Session rows (newest first) with title, times, machine, source, entrypoint, counts and id."""
        cap_chars = self._max_chars(max_chars)
        self._choice("kind", kind, SESSION_KINDS)
        if source:
            self._choice("source", source, SOURCES)
        self._range("limit", limit, 1, 200)
        lo, hi = self._bound(since, "since"), self._bound(until, "until", end=True)
        with self._db() as conn:
            pid = self._project_id(conn, project)
            where, args = self._session_where(project_id=pid, kind=kind, machine=machine, source=source,
                                              since=lo, until=hi)
            total = conn.execute(f"SELECT COUNT(*) FROM sessions s WHERE {where}", args).fetchone()[0]
            if cursor:
                try:
                    c = decode_cursor(cursor)
                    key, last_id = str(c["k"]), str(c["i"])
                except (ValueError, KeyError):
                    raise ToolError("invalid cursor; call list_sessions without it to start over") from None
                where += f" AND ({_ACT} < ? OR ({_ACT} = ? AND s.id < ?))"
                args += [key, key, last_id]
            rows = conn.execute(f"{_SESSION_SQL} WHERE {where} ORDER BY {_ACT} DESC, s.id DESC LIMIT ?",
                                (*args, limit + 1)).fetchall()
        head = f"# Sessions ({total} match{', continued' if cursor else ''})\n\n" + table(
            ["when", "title", "project", "machine", "source", "entrypoint", "prompts/replies/tools", "id"], [])
        budget = max(cap_chars - len(head) - FOOTER_RESERVE, 200)
        lines: list[str] = []
        last = None
        used = 0
        for r in rows[:limit]:
            line = table_row([fmt_span(r["started_at"], r["ended_at"], self.tz),
                        one_line(r["title"] or r["first_prompt"] or "(untitled)", 100), r["alias"], r["machine"],
                        r["source"], f"{r['entrypoint'] or '?'} ({r['kind']})",
                        f"{r['n_user']}/{r['n_assistant']}/{r['n_tool_calls']}", f"`{r['id']}`"])
            if lines and used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
            last = r
        more = last is not None and len(lines) < len(rows)
        footer = ""
        if more:
            nxt = encode_cursor({"k": last["ended_at"] or last["started_at"] or "", "i": last["id"]})
            footer = f'\n[more sessions: list_sessions(..., cursor="{nxt}") with the same filters]\n'
        elif not rows:
            footer = "\nNo sessions match; try kind=\"all\" or widen since/until.\n"
        return head + "".join(lines) + footer

    def _session_header(self, conn: sqlite3.Connection, s: sqlite3.Row, include_subagents: bool) -> str:
        title = one_line(s["title"] or s["first_prompt"] or "(untitled)", 150)
        lines = [
            f"# {title}",
            f"- session: `{s['id']}` · project: {s['alias']}",
            f"- machine: {s['machine']} · source: {s['source']} · root: {s['root']} · "
            f"entrypoint: {s['entrypoint'] or '?'} ({s['kind']})",
            f"- time: {fmt_span(s['started_at'], s['ended_at'], self.tz)} · {s['n_user']} prompts, "
            f"{s['n_assistant']} replies, {s['n_tool_calls']} tool calls",
        ]
        if s["cwd"] or s["git_branch"]:
            lines.append(f"- cwd: {s['cwd'] or '?'} · branch: {s['git_branch'] or '?'} · "
                         f"Claude Code {s['cc_version'] or '?'}")
        flags = []
        if s["missing_since"]:
            flags.append(f"missing upstream since {self._t(s['missing_since'])} (served from the archive)")
        if s["source"] == "recovered":
            flags.append("recovered (restored from Syncthing versions)")
        if flags:
            lines.append(f"- flags: {'; '.join(flags)}")
        if s["is_subagent"]:
            lines.append(f"- subagent ({s['agent_type'] or '?'}: {one_line(s['agent_description'] or '', 150)}) "
                         f"of session `{s['parent_session_id']}`")
        subs = conn.execute("SELECT id, agent_type, agent_description FROM sessions WHERE parent_session_id = ? "
                            "ORDER BY started_at, id", (s["id"],)).fetchall()
        if subs and not include_subagents:
            shown = "; ".join(f"`{x['id']}` {x['agent_type'] or '?'}: {one_line(x['agent_description'] or '', 80)}"
                              for x in subs[:20])
            lines.append(f"- subagents ({len(subs)}; include_subagents=true appends them): {shown}"
                         + (" …" if len(subs) > 20 else ""))
        if s["summary"]:
            lines.append(f"- summary: {one_line(s['summary'], EXCERPT_CHARS)}")
        return "\n".join(lines) + "\n"

    def _message_block(self, r: sqlite3.Row, detail: str, sid: str) -> str:
        role, text = r["role"], (r["text"] or "").strip()
        if role in (ROLE_USER, ROLE_ASSISTANT, ROLE_SUMMARY):
            if not text:
                return ""
            meta = ["compaction summary" if role == ROLE_SUMMARY else role, self._t(r["ts"]), f"seq {r['seq']}"]
            meta += [f"`{r['uuid']}`"] if r["uuid"] else []
            return f"#### {' · '.join(meta)}\n{text}\n\n"
        ident = f" · `{r['tool_use_id']}`" if r["tool_use_id"] else ""
        if role == ROLE_TOOL_CALL:
            return f"- → {one_line(text)} · seq {r['seq']}{ident}\n\n"
        head = f"- {'✗' if r['is_error'] else '✓'} result · {size(r['text_len'])} · seq {r['seq']}{ident}\n"
        if detail != "full":
            return head + "\n"
        shown = text[:FULL_RESULT_CHARS].rstrip()
        more = ""
        if r["text_len"] > len(shown):
            more = (f"\n[… {size(r['text_len'])} in total: read_message(session_id=\"{sid}\", "
                    f"tool_use_id=\"{r['tool_use_id']}\") returns all of it]")
        return f"{head}{shown}{more}\n\n"

    def _session_blocks(self, conn: sqlite3.Connection, parts: list[sqlite3.Row], detail: str,
                        start: tuple[int, int]) -> Iterator[tuple[tuple[int, int], str, sqlite3.Row | None, str]]:
        """(position, rendered block, message row, session id) from ``start`` on.

        Positions are (part, seq); seq -1 is a subagent part's heading.
        """
        roles = DETAIL_ROLES[detail]
        marks = ",".join("?" * len(roles))
        for pi in range(start[0], len(parts)):
            s = parts[pi]
            q = start[1] if pi == start[0] else -1
            if pi > 0 and q < 0:
                yield (pi, -1), (f"## Subagent `{s['id']}` · {s['agent_type'] or '?'}: "
                                 f"{one_line(s['agent_description'] or '', 150)}\n\n"), None, s["id"]
            for r in conn.execute(
                    "SELECT m.*, t.text FROM messages m JOIN fts_docs t ON t.rowid = m.doc_id "
                    f"WHERE m.session_id = ? AND m.seq >= ? AND m.role IN ({marks}) ORDER BY m.seq",
                    (s["id"], max(q, 0), *roles)):
                if block := self._message_block(r, detail, s["id"]):
                    yield (pi, r["seq"]), block, r, s["id"]

    def read_session(self, *, session_id: str, detail: str = "conversation", include_subagents: bool = False,
                     cursor: str | None = None, max_chars: int | None = None) -> str:
        """Header plus transcript at the requested detail, paged by message seq."""
        cap_chars = self._max_chars(max_chars)
        self._choice("detail", detail, tuple(DETAIL_ROLES))
        with self._db() as conn:
            s = self._session(conn, session_id)
            start = (0, 0)
            if cursor:
                try:
                    c = decode_cursor(cursor)
                    if c["s"] != s["id"]:
                        raise ValueError
                    start, detail, include_subagents = (int(c["p"]), int(c["q"])), c["d"], bool(c["x"])
                    self._choice("detail", detail, tuple(DETAIL_ROLES))
                except (ValueError, KeyError, TypeError):
                    raise ToolError("invalid cursor for this session; "
                                    "call read_session without it to start over") from None
            parts = [s]
            if include_subagents:
                parts += conn.execute(f"{_SESSION_SQL} WHERE s.parent_session_id = ? ORDER BY s.started_at, s.id",
                                      (s["id"],)).fetchall()
            if cursor:
                header = f"# {one_line(s['title'] or '(untitled)', 150)} (continued)\n- session: `{s['id']}`\n"
            else:
                header = self._session_header(conn, s, include_subagents)
            budget = max(cap_chars - len(header) - FOOTER_RESERVE, 200)
            out: list[str] = []
            used, nxt, truncated = 0, None, False
            for pos, block, r, sid in self._session_blocks(conn, parts, detail, start):
                if truncated or (out and used + len(block) > budget):
                    nxt = pos
                    break
                if len(block) > budget:  # a single message larger than the page
                    block = self._truncate_block(block, budget, r, sid)
                    truncated = True
                out.append(block)
                used += len(block)
        if nxt is not None:
            token = encode_cursor({"s": s["id"], "p": nxt[0], "q": nxt[1], "d": detail, "x": include_subagents})
            footer = f'{SEP}[more: read_session(session_id="{s["id"]}", cursor="{token}")]\n'
        else:
            footer = f"{SEP}(end of transcript)\n" if out else f"{SEP}(no messages at detail={detail})\n"
        return header + SEP + "".join(out) + footer

    @staticmethod
    def _truncate_block(block: str, budget: int, r: sqlite3.Row | None, sid: str) -> str:
        if r is not None and r["tool_use_id"]:
            how = f'read_message(session_id="{sid}", tool_use_id="{r["tool_use_id"]}")'
        elif r is not None and r["uuid"]:
            how = f'read_message(session_id="{sid}", uuid="{r["uuid"]}")'
        else:
            how = "a larger max_chars"
        note = f"\n[… message truncated ({size(len(block))}); {how} returns all of it]\n\n"
        return cut_at_line(block, max(budget - len(note), 100)).rstrip() + note

    # --- messages ------------------------------------------------------------------------

    def _persisted_output(self, root: str, rel_path: str, tool_use_id: str, text: str) -> tuple[str, str] | None:
        """Full output Claude Code saved to ``<session dir>/tool-results/`` (redacted), with its name."""
        base = rel_path.split("/subagents/")[0] if "/subagents/" in rel_path else rel_path.removesuffix(".jsonl")
        names = [f"{tool_use_id}{ext}" for ext in (".txt", ".json")] + _TOOL_REF_RE.findall(text)
        for name in dict.fromkeys(names):
            content = self._read_raw(root, f"{base}/tool-results/{name}")
            if content is not None:
                return content, f"tool-results file {name}"
        return None

    def _full_text(self, r: sqlite3.Row, cache: dict[tuple[str, str, int], list[Any]]) -> tuple[str, str]:
        """(complete redacted text, where it came from) for one message row."""
        text, origin = None, None
        if r["froot"] and r["line_offset"] is not None:
            key = (r["froot"], r["rel_path"], r["line_offset"])
            if key not in cache:
                cache[key] = self._raw_messages(*key)
            pm = next((m for m in cache[key] if m.block_index == r["block_index"] and m.role == r["role"]), None)
            if pm is not None:
                text, origin = pm.text, "raw transcript line"
        if r["role"] == ROLE_TOOL_RESULT and r["froot"] and r["tool_use_id"] and _TOOL_ID_RE.match(r["tool_use_id"]):
            found = self._persisted_output(r["froot"], r["rel_path"], r["tool_use_id"], text or r["text"] or "")
            if found:
                return found
        if text is not None:
            return text, origin
        indexed = r["text"] or ""
        if r["text_len"] > len(indexed):
            return indexed, f"truncated indexed copy ({len(indexed)} of {r['text_len']} chars; raw source unavailable)"
        return indexed, "indexed copy (complete)"

    def _raw_messages(self, root: str, rel_path: str, offset: int) -> list[Any]:
        """Parsed messages of one raw JSONL line (redacted, thinking-free); [] if unreadable."""
        path = raw_path(self.cfg, root, rel_path)
        if path is None:
            return []
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                line = f.readline()
        except OSError:
            return []
        return parse_line_full(line)

    def read_message(self, *, session_id: str, uuid: str | None = None, tool_use_id: str | None = None,
                     offset: int = 0, max_chars: int | None = None) -> str:
        """The complete (redacted) content of one record or one tool call + result, paged by ``offset``."""
        cap_chars = self._max_chars(max_chars)
        if bool(uuid) == bool(tool_use_id):
            raise ToolError("pass exactly one of uuid or tool_use_id")
        if not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be a non-negative integer")
        col, val = ("uuid", uuid) if uuid else ("tool_use_id", tool_use_id)
        with self._db() as conn:
            s = self._session(conn, session_id)
            rows = conn.execute(
                "SELECT m.*, t.text, f.root AS froot, f.rel_path FROM messages m "
                "JOIN fts_docs t ON t.rowid = m.doc_id LEFT JOIN files f ON f.id = m.file_id "
                f"WHERE m.session_id = ? AND m.{col} = ? ORDER BY m.seq", (s["id"], str(val).strip())).fetchall()
        if not rows:
            raise ToolError(f"no message with {col}={val!r} in session {s['id']}")
        cache: dict[tuple[str, str, int], list[Any]] = {}
        sections, origins = [], []
        for r in rows:
            text, origin = self._full_text(r, cache)
            origins.append(origin)
            label = [f"seq {r['seq']}", r["role"]] + ([r["tool_name"]] if r["tool_name"] else [])
            label += ["error"] if r["is_error"] else []
            sections.append(f"[{' · '.join(label)}]\n{text}")
        body = "\n\n".join(sections)
        if offset and offset >= len(body):
            raise ToolError(f"offset {offset} is past the end ({len(body)} chars)")
        anchor = f'{col}="{val}"'
        header = (f"# Message in session `{s['id']}`\n"
                  f"- {anchor} · {self._t(rows[0]['ts'])} · project {s['alias']} · machine {s['machine']} · "
                  f"source {s['source']}\n- content from: {'; '.join(dict.fromkeys(origins))}\n")
        budget = max(cap_chars - len(header) - FOOTER_RESERVE, 200)
        piece = cut_at_line(body[offset:], budget)
        end = offset + len(piece)
        header += f"- chars {offset}–{end} of {len(body)}\n"
        if end < len(body):
            footer = f'{SEP}[more: read_message(session_id="{s["id"]}", {anchor}, offset={end})]\n'
        else:
            footer = f"{SEP}(end of message)\n"
        return header + SEP + piece + footer

    # --- memories ------------------------------------------------------------------------

    def list_memories(self, *, project: str, include_archived: bool = False, max_chars: int | None = None) -> str:
        """Memories of a project: name (stem), title, type, description, modified, machine, source, sha256."""
        limit = self._max_chars(max_chars)
        with self._db() as conn:
            p = self._project(conn, project)
            rows = conn.execute(
                "SELECT * FROM memories WHERE project_id = ? AND is_index = 0"
                + ("" if include_archived else " AND archived = 0") + " ORDER BY archived, modified_at DESC",
                (p.id,)).fetchall()
        multi = len({r["project_key"] for r in rows if r["project_key"]}) > 1
        headers = ["name", "title", "type", "description", "modified", "machine", "source", "sha256"]
        headers += ["key"] if multi else []
        headers += ["archived"] if include_archived else []
        data = []
        for r in rows:
            row = [f"`{r['stem']}`", r["title"], r["type"], one_line(r["description"], 200),
                   self._t(r["modified_at"]), r["modified_by"], r["root"], r["sha256"] or "–"]
            row += [r["project_key"] or "–"] if multi else []
            row += ["yes" if r["archived"] else ""] if include_archived else []
            data.append(row)
        out = f"# Memories of {p.alias} ({len(rows)})\n\n"
        out += table(headers, data) if rows else "(none)\n"
        out += "\nread_memory(project, name) shows the file and the sha256 that save_memory/archive_memory expect.\n"
        return cap(out, limit, "raise max_chars")

    def read_memory(self, *, project: str, name: str, max_chars: int | None = None) -> str:
        """A memory file (or MEMORY.md) with frontmatter, modified_by and the raw file's sha256."""
        limit = self._max_chars(max_chars)
        stem = (name or "").strip().removesuffix(".md")
        if not stem:
            raise ToolError("name must not be empty")
        is_index = stem == "MEMORY"
        with self._db() as conn:
            p = self._project(conn, project)
            dirs = self._memory_dirs(p)
            for label, key, d in dirs:
                try:
                    text, sha = writes.read_memory_file(d, writes.INDEX_NAME if is_index else stem)
                except writes.NotFound:
                    continue
                except writes.WriteError as e:
                    raise ToolError(e.message) from None
                meta = conn.execute("SELECT * FROM memories WHERE root = ? AND project_key = ? AND stem = ? "
                                    "AND archived = 0", (label, key, stem)).fetchone()
                head = [f"# {'MEMORY.md' if is_index else f'Memory `{stem}`'} ({p.alias})",
                        f"- source: {label}" + (f" · key: {key}" if len(dirs) > 1 else "")]
                if meta:
                    head.append(f"- modified: {self._t(meta['modified_at'])} · "
                                f"modified_by: {meta['modified_by'] or '?'}")
                head.append(f"- sha256: `{sha}` (pass as expected_sha256 to save_memory/archive_memory)")
                return cap("\n".join(head) + "\n\n" + redact(text), limit, "raise max_chars")

            row = conn.execute(
                "SELECT m.*, f.rel_path FROM memories m LEFT JOIN files f ON f.id = m.file_id "
                "WHERE m.project_id = ? AND m.stem = ? ORDER BY m.archived, m.modified_at DESC LIMIT 1",
                (p.id, stem)).fetchone()
            if row is None:
                names = [r[0] for r in conn.execute("SELECT stem FROM memories WHERE project_id = ? AND is_index = 0 "
                                                    "AND archived = 0 ORDER BY stem LIMIT 40", (p.id,))]
                raise ToolError(f"no memory {stem!r} in project {p.alias}; available: {', '.join(names) or 'none'}")
            text = self._read_raw(row["root"], row["rel_path"]) if row["rel_path"] else None
            if text is None:
                text = db.doc_text(conn, row["doc_id"]) if row["doc_id"] is not None else ""
        if row["type"] == "claudeai-memory":
            note = "claude.ai memory (read-only, from the latest export)"
        elif row["archived"]:
            note = "archived (read-only; restore by moving the file back by hand)"
        else:
            note = "file not found on disk: indexed copy, cannot be edited until it reappears"
        head = (f"# Memory `{stem}` ({p.alias})\n- {note}\n- source: {row['root']} · modified: "
                f"{self._t(row['modified_at'])} · modified_by: {row['modified_by'] or '?'}\n"
                f"- sha256: `{row['sha256'] or '–'}`\n\n")
        return cap(head + text, limit, "raise max_chars")

    # --- notes ---------------------------------------------------------------------------

    def list_notes(self, *, project: str | None = None, since: str | None = None, limit: int = 20,
                   max_chars: int | None = None) -> str:
        """Notes written by log_note, newest first."""
        cap_chars = self._max_chars(max_chars)
        self._range("limit", limit, 1, 200)
        lo = self._bound(since, "since")
        with self._db() as conn:
            pid = self._project_id(conn, project)
            where, args = _where("1 = 1", ("n.project_id = ?", pid), ("n.created_at >= ?", lo))
            rows = conn.execute(f"{_NOTES_SQL} WHERE {where} ORDER BY n.created_at DESC LIMIT ?",
                                (*args, limit)).fetchall()
        out = f"# Notes ({len(rows)})\n\n"
        out += table(["created", "project", "title", "surface", "machine", "source", "note_id"],
                     [[self._t(r["created_at"]), r["alias"], one_line(r["title"], 120), r["surface"], r["machine"],
                       r["root"], f"`{r['note_id']}`"] for r in rows]) if rows else "(none)\n"
        return cap(out, cap_chars, "lower limit or narrow project/since")

    def read_note(self, *, project: str, note_id: str, max_chars: int | None = None) -> str:
        """The full (redacted) note."""
        limit = self._max_chars(max_chars)
        nid = (note_id or "").strip()
        nid = nid if nid.endswith(".md") else f"{nid}.md"
        if not _NOTE_ID_RE.match(nid) or ".." in nid:
            raise ToolError(f"invalid note_id {note_id!r}; use the file name shown by list_notes")
        with self._db() as conn:
            p = self._project(conn, project)
            row = conn.execute("SELECT n.*, f.root AS froot, f.rel_path FROM notes n LEFT JOIN files f "
                               "ON f.id = n.file_id WHERE n.project_id = ? AND n.note_id = ?", (p.id, nid)).fetchone()
            text, meta = None, ""
            if row is not None:
                meta = (f"- created: {self._t(row['created_at'])} · surface: {row['surface']} · "
                        f"machine: {row['machine']} · source: {row['root']}\n")
                if row["rel_path"]:
                    text = self._read_raw(row["froot"], row["rel_path"])
                text = text if text is not None else db.doc_text(conn, row["doc_id"])
            else:  # written moments ago and not indexed yet
                for key in p.keys:
                    for root in self.cfg.roots:
                        if text is None and root.kind == "claude-code":
                            text = self._read_raw(root.label, f"{key}/{writes.NOTES_DIR}/{nid}")
            if text is None:
                recent = [r[0] for r in conn.execute("SELECT note_id FROM notes WHERE project_id = ? "
                                                     "ORDER BY created_at DESC LIMIT 10", (p.id,))]
                raise ToolError(f"no note {nid!r} in project {p.alias}; recent: {', '.join(recent) or 'none'}")
        return cap(f"# Note `{nid}` ({p.alias})\n{meta}\n{text}", limit, "raise max_chars")

    # --- writes --------------------------------------------------------------------------

    @contextmanager
    def _writing(self, project: str, create_project: bool, stem: str | None = None
                 ) -> Iterator[tuple[str, Path, Path]]:
        """Resolve the write target, map write errors to ToolError, undo a folder created for a failed write.

        Yields (alias, writable root dir, project dir). ``stem`` prefers the key folder that
        already holds that memory.
        """
        wr = self.cfg.writable_root
        if wr is None or not root_dir(self.cfg, wr).is_dir():
            raise ToolError("writes are disabled: no writable root is configured or it is not available")
        base = root_dir(self.cfg, wr)
        with self._db() as conn:
            projects = list_projects(conn)
        p, candidates = resolve(projects, project or "")
        if p is None and candidates:
            raise ToolError(f"project {project!r} is ambiguous; candidates: {', '.join(c.alias for c in candidates)}")
        if p is None:
            alias = (project or "").strip().lower()
            if not create_project:
                raise ToolError(f"unknown project {project!r}; available: {', '.join(x.alias for x in projects)}. "
                                "Pass create_project=true to create a hub-only project with this alias.")
            if not ALIAS_RE.match(alias):
                raise ToolError(f"cannot create project {project!r}: a new alias must match {ALIAS_RE.pattern}")
            key = HUB_PREFIX + alias
        else:
            alias = p.alias
            present = [k for k in p.keys if (base / k).is_dir()]
            key = next((k for k in present if stem and (base / k / "memory" / f"{stem}.md").is_file()),
                       present[0] if present else HUB_PREFIX + alias)
        if not key or "/" in key or "\\" in key or key.startswith("."):
            raise ToolError(f"project {alias!r} has no usable folder in the writable root")
        project_dir = base / key
        created = not project_dir.exists()
        if created:
            project_dir.mkdir()
        ok = False
        try:
            yield alias, base, project_dir
            ok = True
        except writes.ShaMismatch as e:
            raise ToolError(f"{e.message}. Current sha256: {e.current_sha256}\n"
                            f"Current content (excerpt):\n{redact(e.excerpt)}") from None
        except writes.WriteError as e:
            raise ToolError(e.message) from None
        except OSError as e:
            raise ToolError(f"write failed: {e.strerror or type(e).__name__}") from None
        finally:
            if created and not ok:
                _remove_empty(project_dir)

    def _enqueue(self, base: Path, *paths: Path) -> list[str]:
        """Queue changed files for reindexing; returns warnings instead of failing the write."""
        label = self.cfg.writable_root.label
        try:
            for path in dict.fromkeys(paths):
                enqueue_reindex(self.cfg, label, path.relative_to(base).as_posix())
        except OSError:
            return ["could not queue a reindex; the indexer will pick the change up on its next scan"]
        return []

    def save_memory(self, *, project: str, name: str, title: str = "", description: str = "", type: str = "",
                    body: str, mode: str = "create", expected_sha256: str | None = None,
                    index_hook: str | None = None, surface: str, create_project: bool = False) -> str:
        """Create, replace or append to a memory file and its MEMORY.md line (spec §9.1)."""
        stem = (name or "").strip().removesuffix(".md")
        with self._writing(project, create_project, stem if mode != "create" else None) as (alias, base, pdir):
            res = writes.save_memory(pdir, name=name, title=title, description=description, type=type, body=body,
                                     mode=mode, expected_sha256=expected_sha256, index_hook=index_hook,
                                     surface=surface, now=self._now())
            warnings = res.warnings + self._enqueue(base, res.path, pdir / "memory" / writes.INDEX_NAME)
        verb = {"create": "created", "replace": "replaced", "append": "appended to"}[mode]
        out = (f"Memory `{res.path.stem}` {verb} in project `{alias}`.\n"
               f"- path: {res.path.relative_to(base).as_posix()}\n"
               f"- sha256: `{res.sha256}` (expected_sha256 for the next replace/append/archive)\n"
               f"- MEMORY.md line: {res.index_line}\n")
        return out + "".join(f"- ⚠ {w}\n" for w in warnings)

    def archive_memory(self, *, project: str, name: str, expected_sha256: str, reason: str, surface: str) -> str:
        """Move a memory to memory/.archived/ and drop its MEMORY.md line (reversible by hand)."""
        stem = (name or "").strip().removesuffix(".md")
        with self._writing(project, False, stem) as (alias, base, pdir):
            res = writes.archive_memory(pdir, name=name, expected_sha256=expected_sha256, reason=reason,
                                        surface=surface, now=self._now())
            memdir = pdir / "memory"
            warnings = self._enqueue(base, memdir / f"{stem}.md", res.archived_path, memdir / writes.INDEX_NAME)
        out = (f"Memory `{stem}` archived in project `{alias}`.\n"
               f"- archived to: {res.archived_path.relative_to(base).as_posix()}\n"
               f"- MEMORY.md line removed: {'yes' if res.index_line_removed else 'no (there was none)'}\n"
               "- to restore: move the file back and re-add its MEMORY.md line by hand\n")
        return out + "".join(f"- ⚠ {w}\n" for w in warnings)

    def log_note(self, *, project: str, title: str, summary: str, decisions: list[str] | None = None,
                 next_steps: list[str] | None = None, open_questions: list[str] | None = None,
                 details: str | None = None, related_sessions: list[str] | None = None, surface: str,
                 create_project: bool = False) -> str:
        """Write an append-only note to remote-notes/ (spec §9.3)."""
        with self._writing(project, create_project) as (alias, base, pdir):
            res = writes.log_note(pdir, project=alias, title=title, summary=summary, decisions=decisions or [],
                                  next_steps=next_steps or [], open_questions=open_questions or [], details=details,
                                  related_sessions=related_sessions or [], surface=surface, now=self._now())
            warnings = self._enqueue(base, res.path)
        out = (f"Note logged in project `{alias}`.\n- note_id: `{res.note_id}`\n"
               f"- path: {res.path.relative_to(base).as_posix()}\n")
        return out + "".join(f"- ⚠ {w}\n" for w in warnings)

    # --- status --------------------------------------------------------------------------

    def hub_status(self) -> str:
        """Versions, freshness, counts, backlog, drift, Syncthing, conflicts, disk, imports, auth failures."""
        now = self._now()
        status = health.load_status(self.cfg)

        def ago(ts: Any) -> str:
            dt = parse_iso(ts)
            return f"{self._t(ts)} ({fmt_duration(now - dt)} ago)" if dt else "never"

        with self._db() as conn:
            warnings = health.health_warnings(status, conn, now=now)

            def q(sql: str, *args: Any) -> list[sqlite3.Row]:
                return conn.execute(sql, args).fetchall()

            schema = db.get_meta(conn, "schema_version", "?")
            file_counts = {r["root"]: (r["n"], r["missing"]) for r in q(
                "SELECT root, COUNT(*) n, SUM(missing_since IS NOT NULL) missing FROM files GROUP BY root")}
            sessions = q("SELECT source, kind, is_subagent, COUNT(*) n FROM sessions GROUP BY 1, 2, 3 ORDER BY 1, 2, 3")
            n_projects = q("SELECT COUNT(*) FROM projects")[0][0]
            mem = q("SELECT SUM(archived = 0), SUM(archived = 1) FROM memories WHERE is_index = 0")[0]
            n_notes = q("SELECT COUNT(*) FROM notes")[0][0]
            chunks = q("SELECT COUNT(*), SUM(embedded = 0) FROM chunks")[0]
            drift = q("SELECT * FROM drift ORDER BY category, count DESC")
            conflicts = q("SELECT * FROM conflicts ORDER BY seen_at DESC LIMIT 20")
            last_import = q("SELECT * FROM imports ORDER BY ingested_at DESC LIMIT 1")
            n_imports = q("SELECT COUNT(*) FROM imports")[0][0]

        out = [f"# Hub status ({self._t(iso_utc(now))})\n\n## Warnings\n"]
        out += [f"- ⚠ {w}\n" for w in warnings] or ["- none\n"]
        out.append(f"\n## Processes\n- server {__version__}, up {fmt_duration(now - self.started_at)} · "
                   f"index schema {schema}\n")
        out.append(f"- indexer {status.get('version', '?')} · heartbeat {ago(status.get('written_at'))} · "
                   f"started {ago(status.get('started_at'))}\n")
        out.append(f"- last full reconcile {ago(status.get('last_reconcile_at'))} · "
                   f"last maintenance {ago(status.get('last_maintenance_at'))}\n")

        st_roots = as_dict(status.get("roots"))
        labels = list(dict.fromkeys([r.label for r in self.cfg.roots] + list(st_roots) + list(file_counts)))
        rows = []
        for label in labels:
            rs = as_dict(st_roots.get(label))
            n, missing = file_counts.get(label, (0, 0))
            rc = self.cfg.root(label)
            rows.append([label, rc.kind if rc else "–", "yes" if rc and rc.writable else "no", n, missing or 0,
                         ago(rs.get("last_event_at")), ago(rs.get("last_reconcile_at"))])
        out.append("\n## Roots\n" + table(["root", "kind", "writable", "files indexed", "missing upstream",
                                            "last event", "last reconcile"], rows))

        out.append(f"\n## Index\n- projects: {n_projects} · memories: {mem[0] or 0} active, {mem[1] or 0} archived · "
                   f"notes: {n_notes}\n\n")
        out.append(table(["source", "kind", "subagent", "sessions"],
                         [[r["source"], r["kind"], "yes" if r["is_subagent"] else "no", r["n"]] for r in sessions]))
        emb = as_dict(status.get("embedding"))
        out.append(f"\n## Embeddings\n- model: {emb.get('model', '?')} · chunks: {chunks[0] or 0}, "
                   f"{chunks[1] or 0} not yet embedded · indexer backlog: {emb.get('pending_chunks', '?')} chunks, "
                   f"{emb.get('pending_sessions', '?')} sessions, oldest pending {ago(emb.get('oldest_pending_ts'))}\n")
        out.append("\n## Format drift (unknown record types / entrypoints)\n")
        out.append(table(["category", "value", "count", "last seen"],
                         [[r["category"], r["value"], r["count"], self._t(r["last_seen"])] for r in drift])
                   if drift else "- none\n")

        sy = status.get("syncthing") if isinstance(status.get("syncthing"), dict) else None
        out.append("\n## Syncthing\n")
        if sy is None:
            out.append("- no data\n")
        else:
            out.append(f"- API reachable: {'yes' if sy.get('reachable') else 'no'}\n")
            for name, d in sorted(as_dict(sy.get("devices")).items()):
                d = as_dict(d)
                out.append(f"- device {name}: {'connected' if d.get('connected') else 'disconnected'}, "
                           f"last seen {ago(d.get('last_seen'))}\n")
            for fid, f in sorted(as_dict(sy.get("folders")).items()):
                f = as_dict(f)
                comp = as_dict(f.get("completion"))
                pct = ", ".join(f"{k} {v:.0f}%" if isinstance(v, (int, float)) else f"{k} ?"
                                for k, v in sorted(comp.items()))
                out.append(f"- folder {fid}: {f.get('state', '?')}, {f.get('need_files', 0)} files needed"
                           + (f"; completion {pct}" if pct else "") + "\n")

        out.append("\n## Sync conflicts\n")
        out += [f"- {r['root']}:{r['rel_path']} (seen {self._t(r['seen_at'])})\n" for r in conflicts] or ["- none\n"]
        disk = status.get("disk_free_gb")
        out.append(f"\n## Storage\n- disk free: {f'{disk:.1f} GB' if isinstance(disk, (int, float)) else '?'}\n")
        out.append("\n## claude.ai imports\n")
        if last_import:
            li = last_import[0]
            result = f" ({li['result']})" if li["result"] else ""
            out.append(f"- last: {li['name']} at {ago(li['ingested_at'])}: {li['conversations']} conversations, "
                       f"{li['updated']} updated{result} · {n_imports} imports total\n")
        else:
            out.append("- none yet\n")
        prune = as_dict(status.get("last_prune"))
        out.append("\n## Retention\n" + (f"- last prune {ago(prune.get('at'))}: {prune.get('sessions_deleted', 0)} "
                                         f"sessions deleted\n" if prune else "- no prune recorded\n"))
        failures = self.auth_failures() if self.auth_failures else None
        out.append(f"\n## Auth\n- recent auth failures: {failures if failures is not None else 'n/a'}\n")
        errors = status.get("errors") if isinstance(status.get("errors"), list) else []
        if errors:
            out.append("\n## Recent indexer errors\n" + "".join(f"- {redact(one_line(str(e), 300))}\n"
                                                                  for e in errors[-10:]))
        return cap("".join(out), self.cfg.default_max_chars, "see the indexer journal for more")
