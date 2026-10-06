"""Small Markdown/time helpers shared by the tool service and the health checks."""

from __future__ import annotations

import base64
import json
import re
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

_REL_RE = re.compile(r"^\s*(\d+)\s*([hdw])\s*$", re.IGNORECASE)
_REL_UNITS = {"h": "hours", "d": "days", "w": "weeks"}


# --- time ----------------------------------------------------------------------------------


def parse_iso(value: Any) -> datetime | None:
    """Tolerant ISO-8601 → aware UTC datetime (naive values count as UTC); None if unusable."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def iso_utc(dt: datetime) -> str:
    """Same shape as the timestamps stored in index.db (``…T…:…:….mmmZ``)."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_when(value: str, *, now: datetime, tz: tzinfo, end: bool = False) -> datetime:
    """``7d``/``36h``/``2w`` (before ``now``) or an ISO date/datetime (``tz`` if no offset).

    A bare date used as an upper bound (``end=True``) means the end of that day.
    Raises ValueError on anything else.
    """
    m = _REL_RE.match(value)
    if m:
        return now - timedelta(**{_REL_UNITS[m[2].lower()]: int(m[1])})
    text = value.strip()
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    if end and len(text) == 10:  # a bare YYYY-MM-DD
        dt += timedelta(days=1, milliseconds=-1)
    return dt.astimezone(UTC)


def fmt_time(ts: str | None, tz: tzinfo) -> str:
    """``2026-01-31 14:05 PST`` in ``tz``; ``?`` when unknown."""
    dt = parse_iso(ts)
    return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z") if dt else "?"


def fmt_span(start: str | None, end: str | None, tz: tzinfo) -> str:
    """``start → end``, with the end's date omitted when it is the same day."""
    a, b = fmt_time(start, tz), fmt_time(end, tz)
    if a == b or end is None:
        return a
    if start is None:
        return b
    return f"{a} → {b[11:] if a[:10] == b[:10] else b}"


def fmt_duration(delta: timedelta) -> str:
    """Compact human duration: ``45s``, ``12m``, ``3h 5m``, ``2d 4h``."""
    s = max(int(delta.total_seconds()), 0)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h {s % 3600 // 60}m"
    return f"{s // 86400}d {s % 86400 // 3600}h"


# --- text ----------------------------------------------------------------------------------


def one_line(text: str | None, limit: int | None = None) -> str:
    """Collapse whitespace; cut to ``limit`` chars with an ellipsis."""
    s = " ".join((text or "").split())
    return s if limit is None or len(s) <= limit else s[: limit - 1].rstrip() + "…"


def cell(text: Any, limit: int | None = None) -> str:
    """A safe Markdown table cell."""
    return one_line("" if text is None else str(text), limit).replace("|", "\\|") or "–"


def row(cells: list[Any]) -> str:
    """One Markdown table row, newline-terminated."""
    return "| " + " | ".join(cell(c) for c in cells) + " |\n"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    return head + "".join(row(r) for r in rows)


def size(n: int) -> str:
    return f"{n} chars" if n < 1000 else f"{n / 1000:.1f}k chars"


def cut_at_line(text: str, limit: int) -> str:
    """At most ``limit`` chars, ending at a line break when one is reasonably close."""
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit + 1)
    return text[: cut if cut >= limit // 2 else limit]


def cap(text: str, max_chars: int, hint: str) -> str:
    """Truncate ``text`` to ``max_chars`` at a line boundary with a note saying how to continue."""
    if len(text) <= max_chars:
        return text
    note = f"\n\n[… truncated at max_chars={max_chars}; {hint}]\n"
    return cut_at_line(text, max(max_chars - len(note), 0)).rstrip() + note


# --- cursors -------------------------------------------------------------------------------


def encode_cursor(data: dict[str, Any]) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> dict[str, Any]:
    """Inverse of ``encode_cursor``; raises ValueError on anything malformed."""
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.strip() + "=" * (-len(cursor.strip()) % 4)))
    except (ValueError, TypeError) as e:
        raise ValueError("invalid cursor") from e
    if not isinstance(data, dict):
        raise ValueError("invalid cursor")
    return data
