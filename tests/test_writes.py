import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from claude_context import writes
from claude_context.memfiles import parse_index, parse_memory, parse_note
from claude_context.writes import (
    AlreadyExists,
    InvalidInput,
    NotFound,
    ShaMismatch,
    archive_memory,
    find_conflicts,
    log_note,
    merge_index_conflicts,
    read_memory_file,
    save_memory,
)

NOW = datetime(2026, 4, 5, 6, 7, 8, tzinfo=UTC)


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    p = tmp_path / "-home-user-demo-project"
    p.mkdir()
    return p


def _create(proj: Path, name: str = "demo-fact", **kw) -> writes.SaveResult:
    args = {
        "name": name, "title": "Demo fact", "description": "A demo fact", "type": "project",
        "body": "Fact body.", "surface": "claude.ai", "now": NOW,
    }  # fmt: skip
    args.update(kw)
    return save_memory(proj, **args)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _no_temp_files(root: Path) -> bool:
    return not [p for p in root.rglob("*") if p.name.startswith(".syncthing.")]


def test_create_writes_file_and_index(proj):
    r = _create(proj)
    assert r.created and r.path == proj / "memory" / "demo-fact.md"
    data = r.path.read_bytes()
    assert r.sha256 == hashlib.sha256(data).hexdigest()
    m = parse_memory(data.decode(), "demo-fact")
    assert m.name == "demo-fact" and m.type == "project" and m.body == "Fact body.\n"
    assert m.frontmatter["metadata"]["source"] == "claude-context (claude.ai)"
    assert m.frontmatter["metadata"]["modified"] == "2026-04-05T06:07:08Z"
    index = (proj / "memory" / "MEMORY.md").read_text()
    assert index == "- [Demo fact](demo-fact.md) — A demo fact\n"
    assert r.index_line == index.strip() and r.warnings == []
    assert read_memory_file(proj, "demo-fact") == (data.decode(), r.sha256)
    assert read_memory_file(proj, "demo-fact.md")[1] == r.sha256
    assert read_memory_file(proj, "MEMORY.md")[0] == index
    assert _no_temp_files(proj)


def test_create_accepts_md_suffix_and_rejects_existing(proj):
    _create(proj, name="snake_case_name.md")
    assert (proj / "memory" / "snake_case_name.md").exists()
    with pytest.raises(AlreadyExists):
        _create(proj, name="snake_case_name")


@pytest.mark.parametrize(
    "name", ["../x", "a/b", "a\\b", ".hidden", "MEMORY", "memory", "Upper", "", "x" * 65, "-lead"]
)
def test_create_rejects_bad_names(proj, name):
    with pytest.raises(InvalidInput):
        _create(proj, name=name)
    assert not (proj.parent / "x.md").exists()


@pytest.mark.parametrize("name", ["../demo-fact", "memory/demo-fact", ".demo-fact", "MEMORY"])
def test_existing_name_traversal_rejected(proj, name):
    r = _create(proj)
    with pytest.raises((InvalidInput, NotFound)):
        save_memory(
            proj, name=name, title="t", description="d", type="user", body="b",
            mode="replace", expected_sha256=r.sha256, surface="desktop",
        )  # fmt: skip


def test_symlink_escape_rejected(proj, tmp_path):
    _create(proj)
    outside = tmp_path / "outside.md"
    outside.write_text("secret")
    (proj / "memory" / "evil.md").symlink_to(outside)
    with pytest.raises(InvalidInput):
        read_memory_file(proj, "evil")


def test_read_missing(proj):
    with pytest.raises(NotFound):
        read_memory_file(proj, "nope")
    with pytest.raises(NotFound):
        read_memory_file(proj, "MEMORY.md")


@pytest.mark.parametrize(
    "kw",
    [
        {"type": "bogus"},
        {"surface": "fax"},
        {"title": "  "},
        {"description": ""},
        {"body": "   "},
        {"mode": "upsert"},
        {"body": "x" * (writes.MAX_BODY_BYTES + 1)},
    ],
)
def test_create_validation(proj, kw):
    with pytest.raises(InvalidInput):
        _create(proj, **kw)
    assert not (proj / "memory" / "demo-fact.md").exists()


def test_body_limit_counts_utf8_bytes(proj):
    with pytest.raises(InvalidInput):
        _create(proj, body="é" * (writes.MAX_BODY_BYTES // 2 + 1))
    _create(proj, body="é" * (writes.MAX_BODY_BYTES // 2))


def test_replace_preserves_unknown_keys_and_index_position(proj):
    mem = proj / "memory"
    mem.mkdir()
    (mem / "legacy_note.md").write_text(
        "---\nname: legacy-note\ndescription: old\ntype: feedback\n"
        "originSessionId: abc\nnode_type: memory\n---\n\nOld body\n"
    )
    index = "# Index\n\n- [Other](other.md) — x\n- [Legacy](legacy_note.md) — old\n## Tail\n"
    (mem / "MEMORY.md").write_text(index)
    _, sha = read_memory_file(proj, "legacy_note")
    r = save_memory(
        proj, name="legacy_note", title="Legacy v2", description="new desc", type="user",
        body="New body", mode="replace", expected_sha256=sha, surface="desktop", now=NOW,
    )  # fmt: skip
    assert not r.created
    m = parse_memory(r.path.read_text(), "legacy_note")
    assert m.name == "legacy-note"  # frontmatter name kept even though it differs from the stem
    assert m.frontmatter["originSessionId"] == "abc" and m.frontmatter["node_type"] == "memory"
    assert "type" not in m.frontmatter and m.type == "user" and m.body == "New body\n"
    assert (mem / "MEMORY.md").read_text() == index.replace(
        "- [Legacy](legacy_note.md) — old", "- [Legacy v2](legacy_note.md) — new desc"
    )


def test_append_keeps_existing_values(proj):
    r = _create(proj, index_hook="custom hook")
    r2 = save_memory(
        proj, name="demo-fact", title="", description="", type="", body="More.",
        mode="append", expected_sha256=r.sha256, surface="mobile", now=NOW,
    )  # fmt: skip
    m = parse_memory(r2.path.read_text(), "demo-fact")
    assert m.body == "Fact body.\n\nMore.\n"
    assert m.description == "A demo fact" and m.type == "project"
    assert m.frontmatter["metadata"]["source"] == "claude-context (mobile)"
    assert r2.index_line == "- [Demo fact](demo-fact.md) — custom hook"
    # explicit values override
    r3 = save_memory(
        proj, name="demo-fact", title="", description="Updated", type="user", body="Even more.",
        mode="append", expected_sha256=r2.sha256, surface="mobile",
    )  # fmt: skip
    m = parse_memory(r3.path.read_text(), "demo-fact")
    assert m.description == "Updated" and m.type == "user"
    assert r3.index_line == "- [Demo fact](demo-fact.md) — Updated"


def test_replace_and_append_require_sha(proj):
    _create(proj)
    for mode in ("replace", "append"):
        with pytest.raises(InvalidInput):
            _create(proj, mode=mode)
    with pytest.raises(NotFound):
        _create(proj, name="missing", mode="replace", expected_sha256="0" * 64)


def test_sha_mismatch_leaves_disk_untouched(proj):
    _create(proj)
    before = _snapshot(proj)
    with pytest.raises(ShaMismatch) as exc:
        _create(proj, mode="replace", expected_sha256="0" * 64, body="clobber")
    assert exc.value.current_sha256 == read_memory_file(proj, "demo-fact")[1]
    assert exc.value.excerpt.startswith("---\nname: demo-fact")
    with pytest.raises(ShaMismatch):
        archive_memory(proj, name="demo-fact", expected_sha256="0" * 64, reason="r", surface="word")
    assert _snapshot(proj) == before


def test_index_warnings(proj):
    mem = proj / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("".join(f"line {i}\n" for i in range(writes.INDEX_WARN_LINES)))
    r = _create(proj)
    assert len(r.warnings) == 1 and "lines" in r.warnings[0]
    (mem / "MEMORY.md").write_text("x" * writes.INDEX_WARN_BYTES + "\n")
    r = _create(proj, name="second")
    assert len(r.warnings) == 1 and "bytes" in r.warnings[0]


def test_permission_bits_copied_from_sibling(proj):
    mem = proj / "memory"
    mem.mkdir(mode=0o700)
    sibling = mem / "aaa.md"
    sibling.write_text("x")
    os.chmod(sibling, 0o600)
    r = _create(proj)
    assert r.path.stat().st_mode & 0o777 == 0o600
    assert (mem / "MEMORY.md").stat().st_mode & 0o777 == 0o600
    # a replaced file keeps its own mode
    os.chmod(r.path, 0o640)
    _create(proj, mode="replace", expected_sha256=r.sha256)
    assert r.path.stat().st_mode & 0o777 == 0o640


def test_new_dirs_copy_parent_mode(proj):
    os.chmod(proj, 0o750)
    r = _create(proj)
    assert r.path.parent.stat().st_mode & 0o777 == 0o750
    assert r.path.stat().st_mode & 0o777 == 0o644


def test_failed_write_cleans_temp_and_keeps_original(proj, monkeypatch):
    r = _create(proj)
    before = _snapshot(proj)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(writes.os, "replace", boom)
    with pytest.raises(OSError):
        _create(proj, mode="replace", expected_sha256=r.sha256, body="new")
    assert _snapshot(proj) == before and _no_temp_files(proj)


def test_archive(proj):
    _create(proj, name="keep-me")
    r = _create(proj, index_hook="hook")
    archived = archive_memory(
        proj, name="demo-fact", expected_sha256=r.sha256, reason="superseded\nby v2",
        surface="cowork", now=NOW,
    )  # fmt: skip
    assert archived.index_line_removed
    assert archived.archived_path == proj / "memory/.archived/demo-fact--20260405-060708.md"
    assert not r.path.exists()
    m = parse_memory(archived.archived_path.read_text(), "demo-fact")
    assert m.frontmatter["archived_reason"] == "superseded by v2"
    assert m.frontmatter["archived_at"] == "2026-04-05T06:07:08Z"
    assert m.frontmatter["archived_by"] == "claude-context (cowork)"
    assert m.body == "Fact body.\n" and m.type == "project"
    index = (proj / "memory/MEMORY.md").read_text()
    assert [ln.target for ln in parse_index(index)] == ["keep-me.md"]
    # same second again -> no collision
    r = _create(proj)
    again = archive_memory(
        proj, name="demo-fact", expected_sha256=r.sha256, reason="x", surface="cowork", now=NOW
    )
    assert again.archived_path.name == "demo-fact--20260405-060708-2.md"
    # reversible by hand: move back, re-add the line
    os.replace(again.archived_path, r.path)
    assert read_memory_file(proj, "demo-fact")
    with pytest.raises(InvalidInput):
        archive_memory(proj, name="demo-fact", expected_sha256="x", reason=" ", surface="cowork")
    assert _no_temp_files(proj)


def test_log_note(proj):
    kw = {
        "project": "demo-project", "title": "Weekly sync: plans!", "summary": "Talked.\n\nTwo paragraphs.",
        "decisions": ["Ship it"], "open_questions": ["When?"], "related_sessions": ["sess-1", "a:b_c"],
        "surface": "claude.ai", "now": NOW,
    }  # fmt: skip
    r = log_note(proj, **kw)
    assert r.note_id == "2026-04-05-060708-claude.ai-weekly-sync-plans.md"
    assert r.path == proj / "remote-notes" / r.note_id
    n = parse_note(r.path.read_text())
    assert n.title == "Weekly sync: plans!" and n.project == "demo-project"
    assert n.related_sessions == ["sess-1", "a:b_c"] and n.created == "2026-04-05T06:07:08Z"
    assert "## Summary\n\nTalked.\n\nTwo paragraphs." in n.body
    assert "## Decisions\n\n- Ship it" in n.body and "## Next steps" not in n.body
    first = r.path.read_bytes()
    r2 = log_note(proj, **kw)
    r3 = log_note(proj, **kw)
    assert r2.note_id.endswith("-plans-2.md") and r3.note_id.endswith("-plans-3.md")
    assert r.path.read_bytes() == first
    assert _no_temp_files(proj)


@pytest.mark.parametrize(
    "kw",
    [
        {"summary": " "},
        {"title": ""},
        {"surface": "nope"},
        {"decisions": ["two\nlines"]},
        {"related_sessions": ["../etc"]},
        {"related_sessions": ["has space"]},
    ],
)
def test_log_note_validation(proj, kw):
    args = {
        "project": "demo-project",
        "title": "t",
        "summary": "s",
        "surface": "desktop",
        "now": NOW,
    }
    args.update(kw)
    with pytest.raises(InvalidInput):
        log_note(proj, **args)
    assert (
        not list((proj / "remote-notes").glob("*.md")) if (proj / "remote-notes").exists() else True
    )


def _write(path: Path, text: str, mtime: float) -> None:
    path.write_text(text)
    os.utime(path, (mtime, mtime))


def test_merge_index_conflicts(proj):
    mem = proj / "memory"
    mem.mkdir()
    for stem in ("a", "b", "c"):
        (mem / f"{stem}.md").write_text("---\nname: x\n---\n")
    current = "# Index\n- [A](a.md) — ours\n- [B](b.md) — b\n"
    _write(mem / "MEMORY.md", current, 1000)
    conflict = mem / "MEMORY.sync-conflict-20260101-000000-ABCDEFG.md"
    _write(conflict, "- [A](a.md) — theirs\n- [C](c.md) — c\n- [Gone](gone.md) — deleted\n", 2000)
    other = mem / "a.sync-conflict-20260101-000000-ABCDEFG.md"
    other.write_text("not merged")

    before = _snapshot(proj)
    dry = merge_index_conflicts(proj, dry_run=True, now=NOW)
    assert len(dry) == 1 and not dry[0].merged and dry[0].archived_to is None
    assert dry[0].detail.startswith("dry run") and "1 stale" in dry[0].detail
    assert _snapshot(proj) == before

    res = merge_index_conflicts(proj, now=NOW)
    assert len(res) == 1 and res[0].merged
    assert res[0].archived_to == mem / ".archived" / conflict.name
    assert res[0].archived_to.exists() and not conflict.exists()
    assert (mem / "MEMORY.md").read_text() == (
        "# Index\n- [A](a.md) — theirs\n- [B](b.md) — b\n- [C](c.md) — c\n"
    )
    assert other.exists()  # conflict copies of memory files are only reported
    assert _no_temp_files(proj)

    # a second conflict with the same name is archived under a timestamped name
    _write(conflict, "- [B](b.md) — older\n", 500)
    res = merge_index_conflicts(proj, now=NOW)
    assert res[0].archived_to.name == f"{conflict.stem}--20260405-060708.md"
    assert "- [B](b.md) — b\n" in (mem / "MEMORY.md").read_text()


def test_find_conflicts(tmp_path):
    for rel in (
        "proj-a/memory/MEMORY.sync-conflict-1-X.md",
        "proj-a/memory/fact.sync-conflict-1-X.md",
        "proj-a/memory/.archived/old.sync-conflict-1-X.md",
        "proj-a/memory/.syncthing.MEMORY.sync-conflict-1-X.md.tmp",
        "proj-a/other.sync-conflict-1-X.jsonl",
        ".stversions/proj-a/memory/MEMORY.sync-conflict-1-X.md",
        "proj-b/memory/MEMORY.md",
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    assert find_conflicts(tmp_path) == [
        tmp_path / "proj-a/memory/MEMORY.sync-conflict-1-X.md",
        tmp_path / "proj-a/memory/fact.sync-conflict-1-X.md",
    ]
