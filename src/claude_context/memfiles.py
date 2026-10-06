"""Pure parsing/rendering of memory files, MEMORY.md indexes and remote notes.

No filesystem access here; ``writes.py`` does the I/O.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

import yaml

from .models import IndexLine, ParsedMemory, ParsedNote

_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"


class _Loader(yaml.SafeLoader):
    """SafeLoader that keeps timestamps as strings so frontmatter round-trips unchanged."""


_Loader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in resolvers if tag != _TIMESTAMP_TAG]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}

_FM_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)", re.DOTALL | re.MULTILINE)


def sha256_text(text: str) -> str:
    """Hex sha256 of the UTF-8 encoding of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def one_line(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return " ".join(str(text).split())


# --- frontmatter -----------------------------------------------------------------------


def _scalar(value: str) -> Any:
    try:
        parsed = yaml.load(value, Loader=_Loader)
    except yaml.YAMLError:
        return value
    return value if isinstance(parsed, (dict, list)) else parsed


def _lenient_yaml(raw: str) -> dict[str, Any] | None:
    """Fallback for the ``key: value`` frontmatter Claude Code sometimes writes unquoted
    (e.g. a description containing ``": "``). Handles one level of nesting; else None."""
    data: dict[str, Any] = {}
    parent: dict[str, Any] | None = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.strip().partition(":")
        if not sep or not key:
            return None
        value = value.strip()
        if line[0] in " \t":
            if parent is None:
                return None
            parent[key] = _scalar(value) if value else ""
        elif value:
            data[key], parent = _scalar(value), None
        else:
            data[key] = parent = {}
    return data


def split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Return (frontmatter dict or None, body). Unparseable frontmatter counts as none."""
    m = _FM_RE.match(text)
    if not m:
        return None, text
    raw = m.group(1)
    try:
        data = yaml.load(raw, Loader=_Loader) if raw.strip() else {}
    except yaml.YAMLError:
        data = _lenient_yaml(raw)
    if not isinstance(data, dict):
        return None, text
    return data, text[m.end() :].lstrip("\r\n")


def render_frontmatter(frontmatter: dict[str, Any], body: str) -> str:
    """Render ``---`` YAML frontmatter, a blank line and the body (one trailing newline)."""
    dumped = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, width=1_000_000, default_flow_style=False
    )
    body = body.strip("\r\n")
    return f"---\n{dumped}---\n\n{body}\n" if body else f"---\n{dumped}---\n"


# --- memory files ----------------------------------------------------------------------


def parse_memory(text: str, fallback_name: str) -> ParsedMemory:
    """Parse a memory file tolerantly; ``type`` comes from ``metadata.type`` or top-level ``type``."""
    fm, body = split_frontmatter(text)
    if fm is None:
        return ParsedMemory(name=fallback_name, body=text, frontmatter={}, has_frontmatter=False)
    meta = fm.get("metadata")
    mtype = meta.get("type") if isinstance(meta, dict) else None
    if mtype is None:
        mtype = fm.get("type")
    name = fm.get("name")
    return ParsedMemory(
        name=str(name) if name not in (None, "") else fallback_name,
        description=one_line(fm.get("description") or ""),
        type=str(mtype) if mtype is not None else "",
        body=body,
        frontmatter=fm,
        has_frontmatter=True,
    )


def render_memory(
    *,
    name: str,
    description: str,
    type: str,
    body: str,
    surface: str,
    modified: str,
    base_frontmatter: dict[str, Any] | None = None,
) -> str:
    """Render a memory file, preserving unknown keys (and their order) from ``base_frontmatter``."""
    fm = dict(base_frontmatter or {})
    fm["name"] = name
    fm["description"] = one_line(description)
    fm.pop("type", None)  # legacy top-level type; the canonical one lives under metadata
    meta = dict(fm["metadata"]) if isinstance(fm.get("metadata"), dict) else {}
    meta["type"] = type
    meta["source"] = f"claude-context ({surface})"
    meta["modified"] = modified
    fm["metadata"] = meta
    return render_frontmatter(fm, body)


# --- MEMORY.md index -------------------------------------------------------------------

_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+])\s+")
_AFTER_LINK_RE = re.compile(r"^\**\s*(?:(?:—|–|-|:)\s*)?")


def _find_link(line: str) -> tuple[int, int, str, str] | None:
    """Locate the first ``[text](target.md)`` in a list-item line.

    Returns (start, end, raw_text, target); link text may hold balanced or escaped brackets.
    """
    if not _LIST_ITEM_RE.match(line):
        return None
    i = 0
    while (i := line.find("[", i)) != -1:
        depth, j = 0, i
        while j < len(line):
            c = line[j]
            if c == "\\":
                j += 2
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j < len(line) and line.startswith("(", j + 1):
            close = line.find(")", j + 2)
            target = line[j + 2 : close] if close != -1 else ""
            if target.endswith(".md") and not re.search(r"\s|://", target):
                return i, close + 1, line[i + 1 : j], target
        i += 1
    return None


def _unescape(text: str) -> str:
    return re.sub(r"\\([\\\[\]])", r"\1", text)


def _escape_title(title: str) -> str:
    """Escape brackets/backslashes unless the title's brackets are already balanced."""
    title, depth = one_line(title), 0
    for c in title:
        depth += {"[": 1, "]": -1}.get(c, 0)
        if depth < 0:
            break
    if depth == 0 and "\\" not in title:
        return title
    return re.sub(r"([\\\[\]])", r"\\\1", title)


def _parse_line(line: str) -> IndexLine | None:
    raw = line.removesuffix("\r")
    found = _find_link(raw)
    if not found:
        return None
    _start, end, text, target = found
    rest = raw[end:]
    hook = rest[_AFTER_LINK_RE.match(rest).end() :].strip()
    return IndexLine(title=_unescape(text), target=target, hook=hook, raw=line)


def parse_index(text: str) -> list[IndexLine]:
    """Return the link lines of a MEMORY.md (``raw`` keeps any trailing ``\\r``)."""
    return [p for line in text.split("\n") if (p := _parse_line(line))]


def _newline(text: str) -> str:
    """The file's line ending, judged by its first line."""
    first, sep, _ = text.partition("\n")
    return "\r\n" if sep and first.endswith("\r") else "\n"


def format_index_line(
    *, title: str, target: str, hook: str, prefix: str = "", suffix: str = ""
) -> str:
    """Canonical ``- [title](target) — hook`` line (no newline)."""
    line = f"- {prefix}[{_escape_title(title)}]({target}){suffix}"
    hook = one_line(hook)
    return f"{line} — {hook}" if hook else line


def _decoration(line: str) -> tuple[str, str]:
    """Decoration around a line's link worth keeping on rewrite (star ratings, bold wrapping).

    Returns (prefix, suffix); both empty unless the emphasis markers balance.
    """
    raw = line.removesuffix("\r")
    start, end, _, _ = _find_link(raw)
    prefix = raw[_LIST_ITEM_RE.match(raw).end() : start]
    suffix = re.match(r"\**", raw[end:]).group()
    if prefix.count("*") != len(suffix) or re.search(r"[_`\[\]]", prefix):
        return "", ""
    return prefix, suffix


def upsert_index_line(text: str, *, title: str, target: str, hook: str) -> tuple[str, str]:
    """Replace the line linking to ``target`` in place, or append one. Returns (new_text, line)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        parsed = _parse_line(line)
        if parsed and parsed.target == target:
            prefix, suffix = _decoration(line)
            new = format_index_line(
                title=title, target=target, hook=hook, prefix=prefix, suffix=suffix
            )
            lines[i] = new + ("\r" if line.endswith("\r") else "")
            return "\n".join(lines), new
    new = format_index_line(title=title, target=target, hook=hook)
    nl = _newline(text)
    if text and not text.endswith("\n"):
        text += nl
    return text + new + nl, new


def remove_index_line(text: str, target: str) -> tuple[str, bool]:
    """Drop every line linking to ``target``. Returns (new_text, removed_any)."""
    lines = text.split("\n")
    kept = [ln for ln in lines if not ((p := _parse_line(ln)) and p.target == target)]
    return "\n".join(kept), len(kept) != len(lines)


def merge_index_conflict(current: str, conflict: str, *, conflict_is_newer: bool) -> str:
    """Union-merge a Syncthing conflict copy of MEMORY.md into ``current``.

    Shared targets take the newer file's line (at ``current``'s position); conflict-only link
    lines are appended; non-link lines come from ``current``; duplicate targets are dropped.
    """
    theirs: dict[str, str] = {}
    for p in parse_index(conflict):
        theirs.setdefault(p.target, p.raw.removesuffix("\r"))
    out: list[str] = []
    seen: set[str] = set()
    for line in current.split("\n"):
        p = _parse_line(line)
        if p is None:
            out.append(line)
            continue
        if p.target in seen:
            continue
        seen.add(p.target)
        if conflict_is_newer and p.target in theirs:
            line = theirs[p.target] + ("\r" if line.endswith("\r") else "")
        out.append(line)
    merged = "\n".join(out)
    extra = [raw for target, raw in theirs.items() if target not in seen]
    if extra:
        nl = _newline(merged)
        if merged and not merged.endswith("\n"):
            merged += nl
        merged += "".join(raw + nl for raw in extra)
    return merged


# --- remote notes ----------------------------------------------------------------------

NOTE_SECTIONS = ("Summary", "Decisions", "Next steps", "Open questions", "Details")


def parse_note(text: str) -> ParsedNote:
    """Parse a remote note (tolerates missing/invalid frontmatter)."""
    fm, body = split_frontmatter(text)
    fm = fm or {}
    related = fm.get("related_sessions") or []
    title = fm.get("title")
    if not title:
        m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = m.group(1) if m else ""
    return ParsedNote(
        title=one_line(title),
        surface=str(fm.get("surface") or ""),
        created=str(fm.get("created") or ""),
        project=str(fm.get("project") or ""),
        related_sessions=[str(s) for s in related] if isinstance(related, list) else [str(related)],
        body=body,
        frontmatter=fm,
    )


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {one_line(i)}" for i in items if one_line(i))


def render_note(
    *,
    title: str,
    surface: str,
    created: str,
    project: str,
    related_sessions: Iterable[str],
    summary: str,
    decisions: Iterable[str],
    next_steps: Iterable[str],
    open_questions: Iterable[str],
    details: str | None,
) -> str:
    """Render a remote note; empty sections other than Summary are omitted."""
    fm = {
        "title": one_line(title),
        "surface": surface,
        "created": created,
        "project": project,
        "related_sessions": list(related_sessions),
    }
    contents = [
        summary.strip(),
        _bullets(decisions),
        _bullets(next_steps),
        _bullets(open_questions),
        (details or "").strip(),
    ]
    parts = [
        f"## {heading}\n\n{content}"
        for heading, content in zip(NOTE_SECTIONS, contents)
        if content or heading == "Summary"
    ]
    return render_frontmatter(fm, "\n\n".join(parts))


def slugify(text: str, max_len: int = 48) -> str:
    """ASCII kebab-case slug, at most ``max_len`` chars; ``untitled`` if nothing remains."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "untitled"
