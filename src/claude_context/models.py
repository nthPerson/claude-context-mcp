"""Shared data contracts between parsers, the indexer and the server.

Parsers return these plain dataclasses; nothing here touches SQLite or the network.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Message roles stored in the index.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL_CALL = "tool_call"
ROLE_TOOL_RESULT = "tool_result"
ROLE_SUMMARY = "summary"  # compaction summary; never counted as a prompt

# Tool results are indexed truncated; the full text is resolved on demand from the raw file.
TOOL_RESULT_INDEX_CHARS = 2000


@dataclass
class ParsedMessage:
    """One searchable unit extracted from a transcript record.

    A single JSONL record can yield several messages (e.g. an assistant record with a text
    block and two tool_use blocks), distinguished by ``block_index``.
    """

    role: str
    text: str  # redacted; tool results truncated to TOOL_RESULT_INDEX_CHARS
    text_len: int  # length of the full redacted text before truncation
    uuid: str | None = None
    ts: str | None = None  # ISO-8601 timestamp exactly as recorded (UTC, "Z" suffix)
    tool_name: str | None = None
    tool_use_id: str | None = None
    is_error: bool = False
    is_prompt: bool = False  # True only for text a human actually typed
    line_offset: int = 0  # absolute byte offset of the source JSONL line in its file
    block_index: int = 0


@dataclass
class SessionFacts:
    """Session-level facts gleaned from a chunk of records. ``None`` means "not seen"."""

    session_id: str | None = None
    agent_id: str | None = None  # set on subagent (sidechain) transcripts
    cwd: str | None = None
    git_branch: str | None = None
    cc_version: str | None = None
    entrypoint: str | None = None
    title: str | None = None  # latest ai-title wins
    summary: str | None = None  # latest compaction summary (redacted)
    first_ts: str | None = None
    last_ts: str | None = None


@dataclass
class TranscriptChunk:
    """Result of parsing a byte range of a transcript file."""

    messages: list[ParsedMessage] = field(default_factory=list)
    facts: SessionFacts = field(default_factory=SessionFacts)
    consumed: int = 0  # bytes consumed (always ends on a newline boundary)
    malformed_lines: int = 0
    unknown_types: Counter[str] = field(default_factory=Counter)


@dataclass
class ParsedMemory:
    """A memory file: YAML frontmatter plus a Markdown body."""

    name: str
    description: str = ""
    type: str = ""  # user | feedback | project | reference (or whatever the file says)
    body: str = ""
    frontmatter: dict[str, Any] = field(default_factory=dict)  # the full parsed frontmatter
    has_frontmatter: bool = True


@dataclass
class IndexLine:
    """One ``- [Title](file.md) — hook`` line of a MEMORY.md index."""

    title: str
    target: str  # link target as written, e.g. "feedback_x.md"
    hook: str = ""
    raw: str = ""  # the original line, verbatim


@dataclass
class ParsedNote:
    """A remote note written by ``log_note``."""

    title: str
    surface: str = ""
    created: str = ""  # ISO-8601 UTC
    project: str = ""
    related_sessions: list[str] = field(default_factory=list)
    body: str = ""
    frontmatter: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaudeAiMessage:
    uuid: str
    sender: str  # "human" | "assistant"
    text: str
    created_at: str | None = None
    attachments: list[str] = field(default_factory=list)  # file names only


@dataclass
class ClaudeAiConversation:
    uuid: str
    name: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    project_uuid: str | None = None
    project_name: str | None = None
    summary: str = ""
    messages: list[ClaudeAiMessage] = field(default_factory=list)


@dataclass
class ClaudeAiMemory:
    """Memory text found in a claude.ai export (indexed read-only)."""

    scope: str  # "account" or a project name
    text: str
    updated_at: str | None = None


@dataclass
class ClaudeAiExport:
    conversations: list[ClaudeAiConversation] = field(default_factory=list)
    projects: dict[str, str] = field(default_factory=dict)  # project uuid -> name
    memories: list[ClaudeAiMemory] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: int = 0  # conversations that could not be parsed


@dataclass
class SearchHit:
    """One ranked search result (see ``search.search``)."""

    kind: str  # db.KIND_*
    doc_type: str  # message | session | memory | note
    project_id: int
    score: float
    snippet: str  # redacted excerpt with **highlighted** matches
    title: str = ""  # session title, memory title or note title
    session_id: str | None = None
    uuid: str | None = None  # message uuid anchor (for read_message)
    ref: str | None = None  # memory stem or note id (filename)
    ts: str | None = None
    machine: str | None = None
    source: str | None = None  # session source: claude-code | claude.ai | recovered
    role: str | None = None  # message role, when doc_type == "message"
    matched: tuple[str, ...] = ()  # which rankers found it: "keyword", "semantic"
