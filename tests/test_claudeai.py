"""Tests for the claude.ai export parser, using small handwritten synthetic zips."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from claude_context.parsers import claudeai
from claude_context.parsers.claudeai import ExportError, parse_export

THINK = "SENTINEL-PRIVATE-THOUGHT"
PROJ_A = "00000000-0000-4000-8000-00000000000a"
PROJ_B = "00000000-0000-4000-8000-00000000000b"


def make_zip(tmp_path: Path, members: dict[str, Any], name: str = "export.zip") -> Path:
    """Write a zip; dict/list values are JSON-encoded, str/bytes are stored verbatim."""
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for member, data in members.items():
            if not isinstance(data, str | bytes):
                data = json.dumps(data)
            zf.writestr(member, data)
    return path


def msg(sender: str, text: str, uuid: str | None = "m", **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"sender": sender, "text": text, "created_at": "2025-03-01T10:00:00Z"}
    if uuid:
        out["uuid"] = uuid
    return out | extra


def conversations() -> list[Any]:
    return [
        {  # content blocks, thinking, tool_use, attachments
            "uuid": "c1",
            "name": "Blocks",
            "summary": "A summary",
            "created_at": "2025-03-01T10:00:00.000000+00:00",
            "updated_at": "2025-03-01T12:00:00+02:00",
            "project_uuid": PROJ_A,
            "chat_messages": [
                msg("human", "ignored top-level", "m1", content=[
                    {"type": "text", "text": "Hello there"},
                    {"type": "text", "text": "second part"},
                ], attachments=[
                    {"file_name": "notes.txt", "extracted_content": "SECRET-BODY"},
                    {"file_name": "notes.txt"},
                ], files=[{"file_name": "pic.png"}, "junk"]),
                msg("assistant", "top", "m2", content=[
                    {"type": "thinking", "thinking": THINK},
                    {"type": "redacted_thinking", "data": THINK},
                    {"type": "text", "text": "Answer"},
                    {"type": "tool_use", "name": "web_search", "input": {"q": THINK + "no"}},
                    {"type": "tool_result", "content": "HUGE-TOOL-PAYLOAD"},
                    {"type": "mystery", "text": "unknown block"},
                ]),
                msg("system", "dropped sender", "m3"),
                msg("user", "user maps to human", "m4"),
                msg("assistant", "   ", "m5"),  # empty -> dropped
                msg("human", "", "m6", attachments=[{"file_name": "only.pdf"}]),  # kept
                "not a dict",
            ],
        },
        {  # only top-level text, nested project, no message uuids
            "uuid": "c2",
            "name": "",
            "project": {"uuid": PROJ_B, "name": "Nested Name"},
            "chat_messages": [msg("human", "Top-level only", None), msg("assistant", "Reply", None)],
        },
        {"uuid": "c3", "name": "Empty", "chat_messages": []},
        "not a dict",
        {"name": "No uuid", "chat_messages": [msg("human", "x")]},
        {"uuid": "c4", "created_at": "garbage", "chat_messages": [msg("human", "Hi", "z")]},
        {"uuid": "c5", "chat_messages": "wrong type", "project_uuid": 7},
    ]


def full_export(tmp_path: Path) -> Path:
    return make_zip(tmp_path, {
        "data-2025/__MACOSX/._conversations.json": "junk",
        "data-2025/sub/conversations.json": conversations(),
        "data-2025/sub/projects.json": [
            {"uuid": PROJ_A, "name": "Project Alpha", "description": "d", "docs": [{"x": 1}]},
            {"uuid": PROJ_B, "name": "Project Beta", "memory": "Beta likes tests"},
            "bad",
        ],
        "data-2025/users.json": [{"uuid": "u", "full_name": "Someone"}],
        "data-2025/other.txt": "ignored",
    })


def test_full_export(tmp_path: Path) -> None:
    exp = parse_export(full_export(tmp_path))
    by_id = {c.uuid: c for c in exp.conversations}
    assert set(by_id) == {"c1", "c2", "c4"}  # c3, c5 empty; others skipped
    assert exp.skipped == 2  # non-dict + missing uuid
    assert exp.projects == {PROJ_A: "Project Alpha", PROJ_B: "Project Beta"}

    c1 = by_id["c1"]
    assert (c1.name, c1.summary) == ("Blocks", "A summary")
    assert c1.created_at == "2025-03-01T10:00:00Z"
    assert c1.updated_at == "2025-03-01T10:00:00Z"  # +02:00 normalized to UTC
    assert [m.uuid for m in c1.messages] == ["m1", "m2", "m4", "m6"]
    m1, m2, m4, m6 = c1.messages
    assert m1.sender == "human"
    assert m1.text == "Hello there\n\nsecond part"
    assert m1.attachments == ["notes.txt", "pic.png"]
    assert m2.sender == "assistant"
    assert m2.text == "Answer\n\n[tool: web_search]"
    assert (m4.sender, m4.text) == ("human", "user maps to human")
    assert (m6.text, m6.attachments) == ("", ["only.pdf"])

    c2 = by_id["c2"]
    assert [m.uuid for m in c2.messages] == ["c2:0", "c2:1"]
    assert c2.messages[0].text == "Top-level only"

    assert by_id["c4"].created_at is None
    assert any("no usable messages" in w for w in exp.warnings)


def test_thinking_and_payloads_never_appear(tmp_path: Path) -> None:
    exp = parse_export(full_export(tmp_path))
    blob = repr(exp)
    for forbidden in (THINK, "SECRET-BODY", "HUGE-TOOL-PAYLOAD", "unknown block"):
        assert forbidden not in blob


def test_project_name_resolution(tmp_path: Path) -> None:
    by_id = {c.uuid: c for c in parse_export(full_export(tmp_path)).conversations}
    assert (by_id["c1"].project_uuid, by_id["c1"].project_name) == (PROJ_A, "Project Alpha")
    # nested project: uuid known to projects.json wins in the map, nested name is used on the conv
    assert (by_id["c2"].project_uuid, by_id["c2"].project_name) == (PROJ_B, "Nested Name")
    assert by_id["c4"].project_uuid is None and by_id["c4"].project_name is None


def test_project_uuid_without_projects_file(tmp_path: Path) -> None:
    path = make_zip(tmp_path, {"conversations.json": [
        {"uuid": "c", "project_uuid": PROJ_A, "chat_messages": [msg("human", "hi")]},
    ]})
    conv = parse_export(path).conversations[0]
    assert (conv.project_uuid, conv.project_name) == (PROJ_A, None)


@pytest.mark.parametrize(
    ("memories", "expected"),
    [
        ([{"conversations_memory": "About me", "project_memories": {PROJ_A: "Alpha mem"},
           "updated_at": "2025-04-01T00:00:00Z"}],
         {("account", "About me"), ("Project Alpha", "Alpha mem")}),
        ({"memory": "Obj memory", "project_memories": {"Free Name": "By name"}},
         {("account", "Obj memory"), ("Free Name", "By name")}),
        ([{"content": "A"}, {"text": "B"}, {"content": "A"}, "str item", 3],
         {("account", "A"), ("account", "B")}),
        ({"project_memories": {PROJ_B: {"memory": "Nested dict"}}},
         {("Project Beta", "Nested dict")}),
        ([{"conversations_memory": 5, "project_memories": ["x"]}, {}], set()),
        (json.dumps("just a string"), set()),
    ],
)
def test_memory_variants(tmp_path: Path, memories: Any, expected: set[tuple[str, str]]) -> None:
    path = make_zip(tmp_path, {
        "conversations.json": [],
        "projects.json": [{"uuid": PROJ_A, "name": "Project Alpha"},
                          {"uuid": PROJ_B, "name": "Project Beta"}],
        "memories.json": memories,
    })
    exp = parse_export(path)
    assert {(m.scope, m.text) for m in exp.memories} == expected
    assert exp.warnings == []


def test_memory_timestamp_and_project_memory_string(tmp_path: Path) -> None:
    exp = parse_export(full_export(tmp_path))
    assert [(m.scope, m.text) for m in exp.memories] == [("Project Beta", "Beta likes tests")]
    path = make_zip(tmp_path, {
        "conversations.json": [],
        "memories.json": [{"conversations_memory": "M", "updated_at": "2025-04-01T00:00:00Z"}],
    }, "m.zip")
    assert parse_export(path).memories[0].updated_at == "2025-04-01T00:00:00Z"


def test_no_memories_no_warning(tmp_path: Path) -> None:
    exp = parse_export(make_zip(tmp_path, {"conversations.json": []}))
    assert exp.memories == [] and exp.warnings == [] and exp.skipped == 0


def test_not_a_zip(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    bad.write_text("this is not a zip")
    with pytest.raises(ExportError):
        parse_export(bad)
    with pytest.raises(ExportError):
        parse_export(tmp_path / "missing.zip")


def test_zip_without_conversations(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="conversations.json"):
        parse_export(make_zip(tmp_path, {"projects.json": []}))


def test_shallowest_member_wins(tmp_path: Path) -> None:
    other = [{"uuid": "deep", "chat_messages": [msg("human", "x")]}]
    top = [{"uuid": "top", "chat_messages": [msg("human", "x")]}]
    path = make_zip(tmp_path, {"a/b/conversations.json": other, "conversations.json": top})
    assert [c.uuid for c in parse_export(path).conversations] == ["top"]


def test_oversize_member_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_zip(tmp_path, {
        "conversations.json": conversations(),
        "projects.json": [{"uuid": PROJ_A, "name": "P" * 500}],
    })
    monkeypatch.setattr(claudeai, "MAX_MEMBER_BYTES", 300)
    exp = parse_export(path)
    assert exp.conversations == [] and exp.projects == {}
    assert sum("size cap" in w for w in exp.warnings) == 2


def test_total_size_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_zip(tmp_path, {
        "projects.json": [{"uuid": PROJ_A, "name": "P"}],
        "conversations.json": conversations(),
    })
    monkeypatch.setattr(claudeai, "MAX_TOTAL_BYTES", 100)
    exp = parse_export(path)
    assert exp.conversations == []
    assert any("total" in w for w in exp.warnings)


@pytest.mark.parametrize("payload", [{"uuid": "x"}, "text", 5, None, "{not json"])
def test_wrong_top_level_type_gives_empty_result_with_warning(tmp_path: Path, payload: Any) -> None:
    data = payload if payload == "{not json" else json.dumps(payload)
    exp = parse_export(make_zip(tmp_path, {"conversations.json": data}))
    assert exp.conversations == [] and len(exp.warnings) == 1


def test_corrupt_projects_file_does_not_fail_import(tmp_path: Path) -> None:
    path = make_zip(tmp_path, {
        "conversations.json": [{"uuid": "c", "chat_messages": [msg("human", "hi")]}],
        "projects.json": b"\xff\xfe garbage",
    })
    exp = parse_export(path)
    assert len(exp.conversations) == 1
    assert any("projects.json" in w for w in exp.warnings)
