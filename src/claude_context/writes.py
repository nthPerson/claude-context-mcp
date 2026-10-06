"""Filesystem side of the write tools: memories, the MEMORY.md index, notes, sync conflicts.

Every write is atomic (temp file + fsync + rename) and the temp files are named
``.syncthing.<name>.<random>.tmp`` so Syncthing never propagates a half-written file.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import memfiles

SURFACES = (
    "claude.ai", "desktop", "cowork", "mobile", "powerpoint", "excel", "word", "claude-code", "other"
)  # fmt: skip
MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_BODY_BYTES = 20_000
INDEX_WARN_LINES = 200
INDEX_WARN_BYTES = 25_000

INDEX_NAME = "MEMORY.md"
ARCHIVE_DIR = ".archived"
NOTES_DIR = "remote-notes"
SAVE_MODES = ("create", "replace", "append")

_NEW_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9:_-]{1,128}$")
_LOCK = threading.RLock()  # serialises read-check-write sequences within this process


class WriteError(Exception):
    """Base error; ``message`` is safe to show to the model."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidInput(WriteError):
    pass


class AlreadyExists(WriteError):
    pass


class NotFound(WriteError):
    pass


class ShaMismatch(WriteError):
    def __init__(self, message: str, *, current_sha256: str, excerpt: str):
        super().__init__(message)
        self.current_sha256 = current_sha256
        self.excerpt = excerpt


@dataclass
class SaveResult:
    path: Path
    sha256: str
    index_line: str
    warnings: list[str] = field(default_factory=list)
    created: bool = False


@dataclass
class ArchiveResult:
    archived_path: Path
    index_line_removed: bool


@dataclass
class NoteResult:
    path: Path
    note_id: str  # the file name


@dataclass
class ConflictMerge:
    project_dir: Path
    conflict_file: Path
    merged: bool
    archived_to: Path | None
    detail: str


# --- low-level file helpers --------------------------------------------------------------


def _sibling_mode(directory: Path) -> int:
    """Permission bits of an existing regular file in ``directory`` (default 0o644)."""
    for p in sorted(directory.iterdir()) if directory.is_dir() else ():
        if not p.name.startswith(".") and p.is_file() and not p.is_symlink():
            return p.stat().st_mode & 0o777
    return 0o644


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_dir(directory: Path) -> None:
    """Create ``directory`` (one level) with its parent's permission bits."""
    if directory.is_dir():
        return
    parent = directory.parent
    if not parent.is_dir():
        raise NotFound(f"project folder does not exist: {parent.name}")
    mode = parent.stat().st_mode & 0o777 or 0o755
    directory.mkdir(exist_ok=True)
    os.chmod(directory, mode)
    _fsync_dir(parent)


def _atomic_write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    """Write ``data`` to ``path`` atomically; ``exclusive`` fails with AlreadyExists if present.

    New files copy a sibling's permission bits; replaced files keep their own.
    """
    directory = path.parent
    mode = path.stat().st_mode & 0o777 if path.exists() else _sibling_mode(directory)
    fd, tmp_name = tempfile.mkstemp(prefix=f".syncthing.{path.name}.", suffix=".tmp", dir=directory)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), mode)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if exclusive:
            try:
                os.link(tmp, path)  # atomic create-if-absent
            except FileExistsError:
                raise AlreadyExists(f"{path.name} already exists") from None
            tmp.unlink()
        else:
            os.replace(tmp, path)
        _fsync_dir(directory)
    finally:
        tmp.unlink(missing_ok=True)


def _move_exclusive(src: Path, dst: Path) -> None:
    """Move without ever overwriting: hard-link to ``dst``, then unlink ``src``."""
    os.link(src, dst)
    src.unlink()
    _fsync_dir(dst.parent)
    if src.parent != dst.parent:
        _fsync_dir(src.parent)


def _read(path: Path) -> tuple[str, str]:
    """(text, sha256 of the raw bytes)."""
    data = path.read_bytes()
    return data.decode("utf-8", errors="replace"), hashlib.sha256(data).hexdigest()


def _utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    return now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)


def _iso(now: datetime) -> str:
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- validation --------------------------------------------------------------------------


def _single_line(value: str | None, field_name: str, *, required: bool = True) -> str:
    text = memfiles.one_line(value or "")
    if required and not text:
        raise InvalidInput(f"{field_name} must be a non-empty single line")
    return text


def _check_surface(surface: str) -> None:
    if surface not in SURFACES:
        raise InvalidInput(f"surface must be one of: {', '.join(SURFACES)}")


def _check_body_size(body: str) -> None:
    size = len(body.encode("utf-8"))
    if size > MAX_BODY_BYTES:
        raise InvalidInput(f"body is {size} bytes; the limit is {MAX_BODY_BYTES}")


def _stem(name: str) -> str:
    """Normalise a user-supplied memory name and reject anything path-like."""
    stem = (name or "").strip().removesuffix(".md")
    if (
        not stem
        or "/" in stem
        or "\\" in stem
        or "\0" in stem
        or ".." in stem
        or stem.startswith(".")
        or stem.lower() == "memory"
        or ".sync-conflict-" in stem
    ):
        raise InvalidInput(f"invalid memory name: {name!r}")
    return stem


def _inside(path: Path, directory: Path) -> Path:
    """Return ``path`` after verifying it resolves to a direct child of ``directory``."""
    if path.resolve().parent != directory.resolve():
        raise InvalidInput("memory name resolves outside the memory folder")
    return path


def _new_memory_path(project_dir: Path, name: str) -> Path:
    stem = _stem(name)
    if not _NEW_NAME_RE.match(stem):
        raise InvalidInput(
            "new memory names must be lowercase slugs matching ^[a-z0-9][a-z0-9_-]{0,63}$"
        )
    memdir = project_dir / "memory"
    return _inside(memdir / f"{stem}.md", memdir)


def _existing_memory_path(project_dir: Path, name: str) -> Path:
    """Path of an existing memory file whose stem is exactly ``name``; raises NotFound."""
    stem = _stem(name)
    memdir = project_dir / "memory"
    path = memdir / f"{stem}.md"
    if not memdir.is_dir() or path.name not in {p.name for p in memdir.iterdir() if p.is_file()}:
        raise NotFound(f"no memory named {stem!r}")
    return _inside(path, memdir)


def _check_sha(path: Path, expected_sha256: str | None) -> str:
    """Return the current text after verifying the caller's sha256 of it."""
    if not expected_sha256:
        raise InvalidInput("expected_sha256 is required; read the memory first")
    text, current = _read(path)
    if expected_sha256.strip().lower() != current:
        raise ShaMismatch(
            f"{path.name} changed since it was read; re-read it and retry",
            current_sha256=current,
            excerpt=text[:500],
        )
    return text


# --- index -------------------------------------------------------------------------------


def _index_warnings(text: str) -> list[str]:
    lines = text.count("\n") + (0 if not text or text.endswith("\n") else 1)
    size = len(text.encode("utf-8"))
    warnings = []
    if lines > INDEX_WARN_LINES:
        warnings.append(
            f"{INDEX_NAME} has {lines} lines; Claude Code only loads the first {INDEX_WARN_LINES}"
        )
    if size > INDEX_WARN_BYTES:
        warnings.append(
            f"{INDEX_NAME} is {size} bytes; Claude Code only loads about {INDEX_WARN_BYTES}"
        )
    return warnings


def _read_index(memdir: Path) -> str:
    path = memdir / INDEX_NAME
    return _read(path)[0] if path.exists() else ""


def _write_index(memdir: Path, old: str, new: str) -> None:
    if new != old:
        _atomic_write(memdir / INDEX_NAME, new.encode("utf-8"))


# --- public API --------------------------------------------------------------------------


def read_memory_file(project_dir: Path, name: str) -> tuple[str, str]:
    """Return (text, sha256) of a memory file or of ``MEMORY.md``."""
    if (name or "").strip() in ("MEMORY", INDEX_NAME):
        path = project_dir / "memory" / INDEX_NAME
        if not path.is_file():
            raise NotFound(f"{INDEX_NAME} does not exist yet")
        return _read(path)
    return _read(_existing_memory_path(project_dir, name))


def save_memory(
    project_dir: Path,
    *,
    name: str,
    title: str,
    description: str,
    type: str,
    body: str,
    mode: str = "create",
    expected_sha256: str | None = None,
    index_hook: str | None = None,
    surface: str,
    now: datetime | None = None,
) -> SaveResult:
    """Create, replace or append to a memory file, then upsert its MEMORY.md line.

    ``append`` adds ``body`` after the existing body; empty ``title``/``description``/``type``
    keep the existing values (and the existing index hook unless ``index_hook`` is given).
    """
    if mode not in SAVE_MODES:
        raise InvalidInput(f"mode must be one of: {', '.join(SAVE_MODES)}")
    _check_surface(surface)
    appending = mode == "append"
    title = _single_line(title, "title", required=not appending)
    description = _single_line(description, "description", required=not appending)
    if (type or not appending) and type not in MEMORY_TYPES:
        raise InvalidInput(f"type must be one of: {', '.join(MEMORY_TYPES)}")
    if not (body or "").strip():
        raise InvalidInput("body must not be empty")
    now = _utc(now)
    memdir = project_dir / "memory"

    with _LOCK:
        index = _read_index(memdir)
        if mode == "create":
            path = _new_memory_path(project_dir, name)
            if path.exists():
                raise AlreadyExists(f"memory {path.stem!r} already exists; use mode='replace'")
            old = None
        else:
            path = _existing_memory_path(project_dir, name)
            old = memfiles.parse_memory(_check_sha(path, expected_sha256), path.stem)
        target = path.name
        existing = next((ln for ln in memfiles.parse_index(index) if ln.target == target), None)
        hook, new_body = index_hook, body
        if appending:
            new_body = (old.body.rstrip() + "\n\n" + body.strip("\n")).lstrip("\n")
            if not hook and not description and existing:
                hook = existing.hook
            description = description or old.description
            type = type or old.type
            if type not in MEMORY_TYPES:
                raise InvalidInput(f"type must be one of: {', '.join(MEMORY_TYPES)}")
        _check_body_size(new_body)
        hook = _single_line(hook or description, "index_hook", required=False)
        if not title:
            title = existing.title if existing else old.name

        text = memfiles.render_memory(
            name=old.name if old else path.stem,
            description=description,
            type=type,
            body=new_body,
            surface=surface,
            modified=_iso(now),
            base_frontmatter=old.frontmatter if old else None,
        )
        data = text.encode("utf-8")
        _ensure_dir(memdir)
        _atomic_write(path, data, exclusive=mode == "create")

        new_index, line = memfiles.upsert_index_line(index, title=title, target=target, hook=hook)
        _write_index(memdir, index, new_index)
        return SaveResult(
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            index_line=line,
            warnings=_index_warnings(new_index),
            created=mode == "create",
        )


def archive_memory(
    project_dir: Path,
    *,
    name: str,
    expected_sha256: str,
    reason: str,
    surface: str,
    now: datetime | None = None,
) -> ArchiveResult:
    """Move a memory to ``memory/.archived/`` (annotated) and drop its MEMORY.md line."""
    _check_surface(surface)
    reason = _single_line(reason, "reason")
    now = _utc(now)
    memdir = project_dir / "memory"
    with _LOCK:
        path = _existing_memory_path(project_dir, name)
        text = _check_sha(path, expected_sha256)
        fm, body = memfiles.split_frontmatter(text)
        if fm is None:
            fm, body = {}, text
        fm.update(
            archived_reason=reason,
            archived_at=_iso(now),
            archived_by=f"claude-context ({surface})",
        )
        archive = memdir / ARCHIVE_DIR
        _ensure_dir(archive)
        dest = _free_path(archive, f"{path.stem}--{now:%Y%m%d-%H%M%S}", ".md")
        _atomic_write(dest, memfiles.render_frontmatter(fm, body).encode("utf-8"), exclusive=True)
        path.unlink()  # the annotated copy is durable before the original goes
        _fsync_dir(memdir)

        index = _read_index(memdir)
        new_index, removed = memfiles.remove_index_line(index, path.name)
        _write_index(memdir, index, new_index)
        return ArchiveResult(archived_path=dest, index_line_removed=removed)


def _free_path(directory: Path, stem: str, suffix: str) -> Path:
    """``stem+suffix`` in ``directory``, or ``stem-2+suffix``, ``-3``… if taken."""
    path, n = directory / f"{stem}{suffix}", 1
    while path.exists():
        n += 1
        path = directory / f"{stem}-{n}{suffix}"
    return path


def log_note(
    project_dir: Path,
    *,
    project: str,
    title: str,
    summary: str,
    decisions: Iterable[str] = (),
    next_steps: Iterable[str] = (),
    open_questions: Iterable[str] = (),
    details: str | None = None,
    related_sessions: Iterable[str] = (),
    surface: str,
    now: datetime | None = None,
) -> NoteResult:
    """Write a new append-only note to ``remote-notes/``; never overwrites."""
    _check_surface(surface)
    title = _single_line(title, "title")
    project = _single_line(project, "project")
    if not (summary or "").strip():
        raise InvalidInput("summary must not be empty")
    lists = {}
    for label, items in (
        ("decisions", decisions),
        ("next_steps", next_steps),
        ("open_questions", open_questions),
    ):
        items = [items] if isinstance(items, str) else list(items)
        if any("\n" in str(i) or "\r" in str(i) for i in items):
            raise InvalidInput(f"each item in {label} must be a single line")
        lists[label] = [memfiles.one_line(i) for i in items if memfiles.one_line(i)]
    sessions = [related_sessions] if isinstance(related_sessions, str) else list(related_sessions)
    bad = [s for s in sessions if not _SESSION_ID_RE.match(str(s))]
    if bad:
        raise InvalidInput(f"invalid related session id(s): {bad[:3]}")
    now = _utc(now)

    text = memfiles.render_note(
        title=title,
        surface=surface,
        created=_iso(now),
        project=project,
        related_sessions=sessions,
        summary=summary,
        details=details,
        **lists,
    )
    _check_body_size(text)
    notes = project_dir / NOTES_DIR
    _ensure_dir(notes)
    stem = f"{now:%Y-%m-%d-%H%M%S}-{surface}-{memfiles.slugify(title)}"
    with _LOCK:
        for n in range(1, 1000):
            path = notes / (f"{stem}.md" if n == 1 else f"{stem}-{n}.md")
            try:
                _atomic_write(path, text.encode("utf-8"), exclusive=True)
            except AlreadyExists:
                continue
            return NoteResult(path=path, note_id=path.name)
    raise WriteError("could not find a free note file name")


def find_conflicts(root: Path) -> list[Path]:
    """All Syncthing conflict copies directly inside ``<root>/<project>/memory/``."""
    return sorted(
        p
        for p in root.glob("*/memory/*.sync-conflict-*")
        if not p.parts[len(root.parts)].startswith(".")
        and not p.name.startswith(".")
        and p.is_file()
    )


def merge_index_conflicts(
    project_dir: Path, *, dry_run: bool = False, now: datetime | None = None
) -> list[ConflictMerge]:
    """Merge each ``MEMORY.sync-conflict-*.md`` into MEMORY.md and archive the conflict copy.

    Conflict-only lines whose target file no longer exists (archived or deleted memories)
    are not resurrected.
    """
    memdir = project_dir / "memory"
    if not memdir.is_dir():
        return []
    now = _utc(now)
    index_path = memdir / INDEX_NAME
    results = []
    with _LOCK:
        current = _read_index(memdir)
        index_mtime = index_path.stat().st_mtime if index_path.exists() else 0.0
        for conflict in sorted(memdir.glob("MEMORY.sync-conflict-*.md")):
            theirs = _read(conflict)[0]
            newer = conflict.stat().st_mtime > index_mtime
            ours_targets = {ln.target for ln in memfiles.parse_index(current)}
            skipped = 0
            for ln in memfiles.parse_index(theirs):
                if ln.target not in ours_targets and not (memdir / ln.target).is_file():
                    theirs = memfiles.remove_index_line(theirs, ln.target)[0]
                    skipped += 1
            merged = memfiles.merge_index_conflict(current, theirs, conflict_is_newer=newer)
            added = len({ln.target for ln in memfiles.parse_index(merged)} - ours_targets)
            detail = (
                f"{'conflict copy' if newer else INDEX_NAME} is newer; "
                f"{added} line(s) added, {skipped} stale line(s) skipped"
                f"{'' if merged != current else '; index unchanged'}"
            )
            archived_to = None
            if dry_run:
                detail = f"dry run: {detail}"
            else:
                _write_index(memdir, current, merged)
                archive = memdir / ARCHIVE_DIR
                _ensure_dir(archive)
                archived_to = archive / conflict.name
                if archived_to.exists():
                    archived_to = _free_path(
                        archive, f"{conflict.stem}--{now:%Y%m%d-%H%M%S}", conflict.suffix
                    )
                _move_exclusive(conflict, archived_to)
            current = merged
            results.append(
                ConflictMerge(
                    project_dir=project_dir,
                    conflict_file=conflict,
                    merged=not dry_run,
                    archived_to=archived_to,
                    detail=detail,
                )
            )
    return results
