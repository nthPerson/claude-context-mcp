"""Incremental parser for Claude Code JSONL transcripts.

Transcripts are append-only and may be mid-write, so ``parse_chunk`` consumes only complete
lines and reports how many bytes it used. Every emitted text is redacted; thinking blocks are
never read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from claude_context.models import (
    ROLE_ASSISTANT,
    ROLE_SUMMARY,
    ROLE_TOOL_CALL,
    ROLE_TOOL_RESULT,
    ROLE_USER,
    TOOL_RESULT_INDEX_CHARS,
    ParsedMessage,
    SessionFacts,
    TranscriptChunk,
)
from claude_context.redact import redact

# Record types that carry no searchable content. Anything else unrecognised is counted in
# ``TranscriptChunk.unknown_types`` so format drift is visible.
IGNORED_TYPES = frozenset(
    {
        "attachment",
        "file-history-snapshot",
        "file-history-delta",
        "queue-operation",
        "bridge-session",
        "mode",
        "permission-mode",
        "cost-state",
        "atis-latch",
        "system",
        "last-prompt",
        "agent-name",
        "frame-link",
        "artifact-comment-monitor",
        "artifact-autoreact-ledger",
        "continued-in",
        "pr-link",
        "relocated",
        "worktree-state",
    }
)

# Harness-injected blocks removed from prompts together with their content.
STRIPPED_TAGS = (
    "system-reminder",
    "local-command-caveat",
    "task-notification",
    "command-message",
    "ide_opened_file",
    "ide_selection",
)
# Wrapper tags removed while keeping their inner text.
UNWRAPPED_TAGS = ("pasted_content",)
# Command output echoed into the transcript: kept as text, but never a prompt.
OUTPUT_TAGS = ("local-command-stdout", "local-command-stderr", "bash-stdout", "bash-stderr")
# Bracketed markers the harness writes in place of a prompt.
HARNESS_MARKER_RE = re.compile(r"\[Request interrupted by user[^\]\n]*\]")


def _block_re(tags: tuple[str, ...]) -> re.Pattern[str]:
    names = "|".join(map(re.escape, tags))
    return re.compile(rf"<({names})\b[^>]*>(.*?)</\1\s*>", re.DOTALL)


_STRIP_RE = _block_re(STRIPPED_TAGS)
_OUTPUT_RE = _block_re(OUTPUT_TAGS)
_UNWRAP_RE = re.compile(rf"</?(?:{'|'.join(map(re.escape, UNWRAPPED_TAGS))})\b[^>]*>")
_CMD_NAME_RE = re.compile(r"<command-name>(.*?)</command-name>", re.DOTALL)
_CMD_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)
_BASH_INPUT_RE = re.compile(r"<bash-input>(.*?)</bash-input>", re.DOTALL)

_RENDER_MAX = 200


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _blocks(content: object) -> list[Any]:
    """Normalise ``message.content`` to a list of blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return []


def _clean_prompt(text: str) -> tuple[str, bool]:
    """Remove harness markup from user text. Returns (clean text, contains human-typed text)."""
    text = _STRIP_RE.sub("", text)
    args = " ".join(a.strip() for a in _CMD_ARGS_RE.findall(text) if a.strip())
    text = _CMD_ARGS_RE.sub("", text)
    text = _CMD_NAME_RE.sub(
        lambda m: f"/{m[1].strip().lstrip('/')} {args}".rstrip(), text, count=1
    )
    text = _BASH_INPUT_RE.sub(lambda m: f"! {m[1].strip()}", text)
    text = _UNWRAP_RE.sub("", text)
    human = HARNESS_MARKER_RE.sub("", _OUTPUT_RE.sub("", text))
    text = _OUTPUT_RE.sub(lambda m: m[2], text)
    return text.strip(), bool(human.strip())


def _result_text(content: object) -> str:
    """Flatten a tool_result ``content`` (string or list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif kind in ("image", "document"):
            parts.append(f"[{kind}]")
        elif kind == "tool_reference":
            ref = _str(block.get("tool_name")) or _str(block.get("id")) or "?"
            parts.append(f"[tool_reference: {ref}]")
    return "\n".join(parts)


def render_tool_call(name: str, tool_input: object) -> str:
    """One-line, redacted summary of a tool call, e.g. ``Read: /home/user/x.py``."""
    inp = tool_input if isinstance(tool_input, dict) else {}

    def field(key: str) -> str:
        value = inp.get(key)
        return value if isinstance(value, str) else ""

    if name == "Bash":
        body = field("description") or field("command")
    elif name in ("Read", "Edit", "Write", "NotebookEdit"):
        body = field("file_path") or field("notebook_path")
    elif name in ("Grep", "Glob"):
        body = field("pattern") + (f" in {field('path')}" if field("path") else "")
    elif name == "WebFetch":
        body = field("url")
    elif name == "WebSearch":
        body = field("query")
    elif name in ("Agent", "Task"):
        body = field("description")
    elif name == "Skill":
        body = f"{field('skill')} {field('args')}"
    else:
        body = json.dumps(tool_input, ensure_ascii=False, separators=(",", ":"), default=str)
    # Redact before truncating so a secret straddling the cut cannot leak a prefix.
    body = redact(" ".join(body.split()))[:_RENDER_MAX].rstrip()
    return f"{name}: {body}" if body else name


def load_subagent_meta(path: Path) -> dict:
    """Read an ``agent-<id>.meta.json`` sidecar; ``{}`` if missing or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


class _Parser:
    """Accumulates messages and facts from successive JSONL lines."""

    def __init__(self, truncate: bool) -> None:
        self.truncate = truncate
        self.chunk = TranscriptChunk()
        self._compact_seen = False

    def feed(self, line: bytes, offset: int) -> None:
        if not line.strip():
            return
        try:
            rec = json.loads(line.decode("utf-8", errors="replace"))
        except (ValueError, RecursionError):
            rec = None
        if not isinstance(rec, dict):
            self.chunk.malformed_lines += 1
            return
        self._facts(rec)
        rtype = rec.get("type")
        if rtype == "user":
            self._add(self._user(rec, offset))
        elif rtype == "assistant":
            self._add(self._assistant(rec, offset))
        elif rtype == "ai-title":
            if title := _str(rec.get("aiTitle")):
                self.chunk.facts.title = redact(title)
        elif rtype == "summary":
            if (summary := _str(rec.get("summary"))) and not self._compact_seen:
                self.chunk.facts.summary = redact(summary)
        elif rtype not in IGNORED_TYPES:
            self.chunk.unknown_types[rtype if isinstance(rtype, str) else "<missing>"] += 1

    def _add(self, messages: list[ParsedMessage]) -> None:
        self.chunk.messages.extend(sorted(messages, key=lambda m: m.block_index))

    def _facts(self, rec: dict) -> None:
        f: SessionFacts = self.chunk.facts
        f.session_id = f.session_id or _str(rec.get("sessionId"))
        f.agent_id = f.agent_id or _str(rec.get("agentId"))
        for attr, key in (
            ("cwd", "cwd"),
            ("git_branch", "gitBranch"),
            ("cc_version", "version"),
            ("entrypoint", "entrypoint"),
        ):
            if value := _str(rec.get(key)):
                setattr(f, attr, value)
        if rec.get("type") in ("user", "assistant") and (ts := _str(rec.get("timestamp"))):
            f.first_ts = ts if f.first_ts is None else min(f.first_ts, ts)
            f.last_ts = ts if f.last_ts is None else max(f.last_ts, ts)

    def _msg(self, rec: dict, offset: int, index: int, role: str, text: str, **kw: Any) -> ParsedMessage:
        full_len = len(text)
        if self.truncate and role == ROLE_TOOL_RESULT:
            text = text[:TOOL_RESULT_INDEX_CHARS]
        return ParsedMessage(
            role=role,
            text=text,
            text_len=full_len,
            uuid=_str(rec.get("uuid")),
            ts=_str(rec.get("timestamp")),
            line_offset=offset,
            block_index=index,
            **kw,
        )

    def _user(self, rec: dict, offset: int) -> list[ParsedMessage]:
        if rec.get("isMeta"):
            return []
        message = rec.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if rec.get("isCompactSummary"):
            text = redact(_result_text(content)).strip()
            if not text:
                return []
            self.chunk.facts.summary = text
            self._compact_seen = True
            return [self._msg(rec, offset, 0, ROLE_SUMMARY, text)]

        out: list[ParsedMessage] = []
        parts: list[str] = []
        typed = False
        first_index: int | None = None
        for i, block in enumerate(_blocks(content)):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "tool_result":
                text = redact(_result_text(block.get("content")))
                out.append(
                    self._msg(
                        rec, offset, i, ROLE_TOOL_RESULT, text,
                        tool_use_id=_str(block.get("tool_use_id")),
                        is_error=block.get("is_error") is True,
                    )
                )
                continue
            if kind == "text" and isinstance(block.get("text"), str):
                text, human = _clean_prompt(block["text"])
                typed = typed or human
            elif kind in ("image", "document"):
                text = f"[{kind}]"
            else:
                continue
            if text:
                parts.append(text)
                first_index = i if first_index is None else first_index
        if parts:
            # Subagent prompts come from the parent agent; task notifications etc. carry an
            # ``origin`` of another kind. Neither was typed by a human.
            origin = rec.get("origin")
            origin_kind = origin.get("kind") if isinstance(origin, dict) else None
            is_prompt = typed and not rec.get("isSidechain") and origin_kind in (None, "human")
            text = redact("\n\n".join(parts))
            out.append(self._msg(rec, offset, first_index or 0, ROLE_USER, text, is_prompt=is_prompt))
        return out

    def _assistant(self, rec: dict, offset: int) -> list[ParsedMessage]:
        if rec.get("isApiErrorMessage"):
            return []
        message = rec.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        out: list[ParsedMessage] = []
        for i, block in enumerate(_blocks(content)):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")  # thinking / redacted_thinking fall through untouched
            if kind == "text" and isinstance(block.get("text"), str):
                text = redact(block["text"]).strip()
                if text:
                    out.append(self._msg(rec, offset, i, ROLE_ASSISTANT, text))
            elif kind == "tool_use":
                name = _str(block.get("name")) or "unknown"
                out.append(
                    self._msg(
                        rec, offset, i, ROLE_TOOL_CALL, render_tool_call(name, block.get("input")),
                        tool_name=name, tool_use_id=_str(block.get("id")),
                    )
                )
        return out


def parse_chunk(data: bytes, base_offset: int = 0) -> TranscriptChunk:
    """Parse the complete lines in ``data``, a byte range starting at file offset ``base_offset``.

    A trailing partial line is left unconsumed; resume at ``base_offset + chunk.consumed``.
    """
    parser = _Parser(truncate=True)
    pos = 0
    while (nl := data.find(b"\n", pos)) >= 0:
        parser.feed(data[pos:nl], base_offset + pos)
        pos = nl + 1
    parser.chunk.consumed = pos
    return parser.chunk


def parse_line_full(line: bytes) -> list[ParsedMessage]:
    """Parse one JSONL record without truncating tool results (still redacted, no thinking)."""
    parser = _Parser(truncate=False)
    parser.feed(line, 0)
    return parser.chunk.messages
