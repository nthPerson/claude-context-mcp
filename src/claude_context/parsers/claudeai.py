"""Parse a claude.ai data-export zip into plain dataclasses.

The export schema is not formally documented, so everything here is defensive: unknown keys are
ignored, wrong types are tolerated, and one broken conversation never fails the whole import.
Members are read in memory via ``ZipFile.open`` (never extracted) and located by basename at any
depth. Message order is the order given in the export (no re-sorting by timestamp).
"""

from __future__ import annotations

import json
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claude_context.models import (
    ClaudeAiConversation,
    ClaudeAiExport,
    ClaudeAiMemory,
    ClaudeAiMessage,
)

MAX_MEMBER_BYTES = 2 * 1024**3  # per-member uncompressed cap (zip-bomb guard)
MAX_TOTAL_BYTES = 4 * 1024**3  # cap on the sum of all members we read
MAX_WARNINGS = 50

_WANTED = ("conversations.json", "projects.json", "memories.json")  # users.json is never read
_MEMORY_KEYS = ("conversations_memory", "memory", "content", "text")
_SKIP_BLOCKS = {"thinking", "redacted_thinking"}
_SENDERS = {"human": "human", "user": "human", "assistant": "assistant"}
_READ_ERRORS = (
    zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error, EOFError, OSError,
)  # fmt: skip


_FAILED: Any = object()  # sentinel: member present but unusable (a warning was recorded)


class ExportError(Exception):
    """The file is not a readable zip, or it has no conversations.json."""


class _Reader:
    """Loads JSON members while enforcing the per-member and total size caps."""

    def __init__(self, zf: zipfile.ZipFile) -> None:
        self.zf = zf
        self.total = 0

    def load(self, info: zipfile.ZipInfo) -> Any:
        """Return the parsed JSON of ``info``; raise ValueError with a short reason on failure."""
        name = info.filename.rsplit("/", 1)[-1]
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"{name}: larger than the per-file size cap")
        self.total += info.file_size
        if self.total > MAX_TOTAL_BYTES:
            raise ValueError(f"{name}: total export size cap exceeded")
        try:
            with self.zf.open(info) as fh:
                raw = fh.read(MAX_MEMBER_BYTES + 1)  # bounded even if file_size lies
        except _READ_ERRORS as exc:
            raise ValueError(f"{name}: unreadable ({type(exc).__name__})") from exc
        if len(raw) > MAX_MEMBER_BYTES:
            raise ValueError(f"{name}: larger than the per-file size cap")
        try:
            return json.loads(raw)
        except (ValueError, RecursionError) as exc:  # bad JSON / encoding / nesting
            raise ValueError(f"{name}: not valid JSON") from exc


def parse_export(zip_path: Path) -> ClaudeAiExport:
    """Parse a claude.ai export zip.

    Raises ExportError only if the file is not a readable zip or has no ``conversations.json``.
    A ``conversations.json`` that is unreadable, oversize or not a list yields an empty result
    with a warning instead.
    """
    out = ClaudeAiExport()
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ExportError(f"not a readable zip file: {exc}") from exc
    with zf:
        members = _find_members(zf)
        if "conversations.json" not in members:
            raise ExportError("conversations.json not found in the export")
        reader = _Reader(zf)

        def load(name: str) -> Any:
            if name not in members:
                return None
            try:
                return reader.load(members[name])
            except ValueError as exc:
                out.warnings.append(str(exc))
                return _FAILED

        projects_data = load("projects.json")
        out.projects = _projects(projects_data)
        convs = load("conversations.json")
        memories_data = load("memories.json")

    if convs is not _FAILED:
        _fill_conversations(out, convs)
    out.memories = _memories(memories_data, projects_data, out.projects)
    del out.warnings[MAX_WARNINGS:]
    return out


def _find_members(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Map each wanted basename to its shallowest member, ignoring directories and path parts."""
    found: dict[str, zipfile.ZipInfo] = {}
    for info in zf.infolist():
        path = info.filename.replace("\\", "/")
        base = path.rsplit("/", 1)[-1].lower()
        if info.is_dir() or base not in _WANTED:
            continue
        old = found.get(base)
        if old is None or path.count("/") < old.filename.replace("\\", "/").count("/"):
            found[base] = info
    return found


def _fill_conversations(out: ClaudeAiExport, data: Any) -> None:
    if not isinstance(data, list):
        out.warnings.append("conversations.json: expected a list; no conversations imported")
        return
    empty = 0
    for i, raw in enumerate(data):
        try:
            conv = _conversation(raw, out)
        except Exception as exc:  # noqa: BLE001 - one bad entry must not fail the import
            out.skipped += 1
            out.warnings.append(f"conversation #{i}: skipped ({type(exc).__name__})")
            continue
        if conv is None:
            out.skipped += 1
            out.warnings.append(f"conversation #{i}: skipped (not an object or no uuid)")
        elif conv.messages:
            out.conversations.append(conv)
        else:
            empty += 1
    if empty:
        out.warnings.append(f"{empty} conversation(s) had no usable messages and were omitted")


def _conversation(raw: Any, out: ClaudeAiExport) -> ClaudeAiConversation | None:
    """Build one conversation; None if it is not an object or has no uuid."""
    if not isinstance(raw, dict) or not _str(raw.get("uuid")):
        return None
    uuid = _str(raw["uuid"])
    project = raw.get("project") if isinstance(raw.get("project"), dict) else {}
    project_uuid = _str(raw.get("project_uuid")) or _str(project.get("uuid")) or None
    project_name = _str(project.get("name")) or None
    if project_uuid:
        if project_name:
            out.projects.setdefault(project_uuid, project_name)
        project_name = project_name or out.projects.get(project_uuid)
    msgs = raw.get("chat_messages")
    messages = [
        m
        for i, item in enumerate(msgs if isinstance(msgs, list) else [])
        if (m := _message(item, uuid, i)) is not None
    ]
    return ClaudeAiConversation(
        uuid=uuid,
        name=_str(raw.get("name")),
        created_at=_ts(raw.get("created_at")),
        updated_at=_ts(raw.get("updated_at")),
        project_uuid=project_uuid,
        project_name=project_name,
        summary=_str(raw.get("summary")),
        messages=messages,
    )


def _message(raw: Any, conv_uuid: str, index: int) -> ClaudeAiMessage | None:
    if not isinstance(raw, dict):
        return None
    sender = _SENDERS.get(_str(raw.get("sender")).lower())
    if sender is None:
        return None
    attachments = _file_names(raw.get("attachments")) + _file_names(raw.get("files"))
    attachments = list(dict.fromkeys(attachments))
    text = _message_text(raw)
    if not text and not attachments:
        return None
    return ClaudeAiMessage(
        uuid=_str(raw.get("uuid")) or f"{conv_uuid}:{index}",
        sender=sender,
        text=text,
        created_at=_ts(raw.get("created_at")),
        attachments=attachments,
    )


def _message_text(raw: dict[str, Any]) -> str:
    """Join text blocks (fallback: top-level ``text``); tool calls become ``[tool: name]``."""
    parts: list[str] = []
    has_text = False
    blocks = raw.get("content")
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind in _SKIP_BLOCKS:
            continue
        if kind == "text":
            text = _str(block.get("text"))
            if text:
                parts.append(text)
                has_text = True
        elif kind == "tool_use":
            parts.append(f"[tool: {_str(block.get('name')) or 'unknown'}]")
    if not has_text and (top := _str(raw.get("text"))):
        parts.insert(0, top)
    return "\n\n".join(parts).strip()


def _file_names(items: Any) -> list[str]:
    names = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and (name := _str(item.get("file_name"))):
            names.append(name)
    return names


def _projects(data: Any) -> dict[str, str]:
    """Project uuid -> name from projects.json."""
    result: dict[str, str] = {}
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and (uuid := _str(item.get("uuid"))):
            result[uuid] = _str(item.get("name"))
    return result


def _memories(data: Any, projects_data: Any, projects: dict[str, str]) -> list[ClaudeAiMemory]:
    """Collect account memory, per-project memory mappings and per-project ``memory`` strings."""
    found: dict[tuple[str, str], ClaudeAiMemory] = {}

    def add(scope: str, text: str, updated: Any = None) -> None:
        if text and (scope, text) not in found:
            found[(scope, text)] = ClaudeAiMemory(scope=scope, text=text, updated_at=_ts(updated))

    for item in data if isinstance(data, list) else [data]:
        if not isinstance(item, dict):
            continue
        updated = item.get("updated_at") or item.get("created_at")
        add("account", _memory_text(item), updated)
        by_project = item.get("project_memories")
        for key, value in by_project.items() if isinstance(by_project, dict) else []:
            add(projects.get(key) or key, _memory_text(value), updated)
    for item in projects_data if isinstance(projects_data, list) else []:
        if isinstance(item, dict) and (text := _str(item.get("memory"))):
            name = _str(item.get("name")) or _str(item.get("uuid")) or "unknown project"
            add(name, text, item.get("updated_at"))
    return list(found.values())


def _memory_text(value: Any) -> str:
    """The memory text of a string, or of the first recognised key of an object."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in _MEMORY_KEYS:
            if text := _str(value.get(key)):
                return text
    return ""


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _ts(value: Any) -> str | None:
    """Normalize an ISO-8601 string to UTC ``...Z`` form; None if it does not parse."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return dt.replace(tzinfo=None).isoformat().replace("+00:00", "") + "Z"
