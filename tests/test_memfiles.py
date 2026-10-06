from claude_context.memfiles import (
    merge_index_conflict,
    parse_index,
    parse_memory,
    parse_note,
    remove_index_line,
    render_memory,
    render_note,
    sha256_text,
    slugify,
    upsert_index_line,
)

MEMORY = """---
name: project-demo-thing
description: "Demo: a thing with # hash and 'quotes'"
metadata:
  node_type: memory
  type: project
  originSessionId: 00000000-1111-2222-3333-444444444444
  modified: 2026-01-02T03:04:05.678Z
  custom_key: kept
---

Body line one.

**Why:** because.
"""


def test_sha256_text():
    assert sha256_text("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_parse_memory_basic():
    m = parse_memory(MEMORY, "project_demo_thing")
    assert m.has_frontmatter
    assert m.name == "project-demo-thing"
    assert m.description == "Demo: a thing with # hash and 'quotes'"
    assert m.type == "project"
    assert m.body.startswith("Body line one.")
    # timestamps stay strings so they round-trip unchanged
    assert m.frontmatter["metadata"]["modified"] == "2026-01-02T03:04:05.678Z"


def test_parse_memory_without_frontmatter():
    m = parse_memory("just text\n", "plain")
    assert not m.has_frontmatter and m.name == "plain" and m.body == "just text\n"
    assert m.type == "" and m.frontmatter == {}


def test_parse_memory_unterminated_frontmatter():
    text = "---\nname: x\nno closing fence\n"
    m = parse_memory(text, "x")
    assert not m.has_frontmatter and m.body == text


def test_parse_memory_legacy_top_level_type():
    text = "---\nname: legacy\ndescription: old style\ntype: feedback\n---\n\nBody\n"
    m = parse_memory(text, "legacy")
    assert m.type == "feedback"
    out = render_memory(
        name=m.name, description=m.description, type="user", body=m.body,
        surface="desktop", modified="2026-01-01T00:00:00Z", base_frontmatter=m.frontmatter,
    )  # fmt: skip
    again = parse_memory(out, "legacy")
    assert "type" not in again.frontmatter
    assert again.frontmatter["metadata"]["type"] == "user" and again.type == "user"


def test_parse_memory_lenient_unquoted_colon():
    # Claude Code occasionally writes descriptions containing ": " unquoted (invalid YAML).
    text = "---\nname: n\ndescription: Status: done, next: ship\nmetadata:\n  type: project\n---\n\nB\n"
    m = parse_memory(text, "n")
    assert m.has_frontmatter
    assert m.description == "Status: done, next: ship"
    assert m.type == "project" and m.body == "B\n"


def test_render_memory_new():
    out = render_memory(
        name="demo", description="line one\nline two", type="feedback", body="Hello\n",
        surface="mobile", modified="2026-05-06T07:08:09Z",
    )  # fmt: skip
    assert out == (
        "---\nname: demo\ndescription: line one line two\nmetadata:\n  type: feedback\n"
        "  source: claude-context (mobile)\n  modified: '2026-05-06T07:08:09Z'\n---\n\nHello\n"
    )


def test_render_memory_preserves_unknown_keys_and_order():
    base = parse_memory(MEMORY, "x").frontmatter
    base["zz_unknown"] = {"nested": [1, 2]}
    tricky = "a: b # c 'd' \"e\" [f] {g} - h: ü"
    out = render_memory(
        name="project-demo-thing", description=tricky, type="reference", body="New body",
        surface="claude.ai", modified="2026-02-02T00:00:00Z", base_frontmatter=base,
    )  # fmt: skip
    m = parse_memory(out, "x")
    assert m.description == tricky
    assert list(m.frontmatter) == ["name", "description", "metadata", "zz_unknown"]
    meta = m.frontmatter["metadata"]
    assert list(meta)[:5] == ["node_type", "type", "originSessionId", "modified", "custom_key"]
    assert meta["originSessionId"] == "00000000-1111-2222-3333-444444444444"
    assert meta["custom_key"] == "kept" and meta["type"] == "reference"
    assert meta["source"] == "claude-context (claude.ai)"
    assert meta["modified"] == "2026-02-02T00:00:00Z"
    assert m.frontmatter["zz_unknown"] == {"nested": [1, 2]}
    assert m.body == "New body\n"


def test_render_memory_long_description_not_folded():
    desc = "word " * 100
    out = render_memory(
        name="n", description=desc, type="user", body="b", surface="other", modified="t"
    )
    assert out.splitlines()[2] == "description: " + desc.strip()


INDEX = (
    "# Memory Index\n"
    "\n"
    "## Feedback\n"
    "- [Prefer short answers](feedback_short.md) — keep replies brief\n"
    "- ⭐⭐ [Starred one](starred.md) — has a rating prefix\n"
    "- **[Bold one](bold-one.md)** — bold wrapped\n"
    "free-form line with a [link](elsewhere.md) that is not a list item\n"
    "- [Nested [id] title](nested.md) - hyphen hook\n"
    "- [En dash](en.md) – en hook\n"
    "- [Colon](colon.md): colon hook\n"
    "- [Bare](bare.md)\n"
    "- [External](https://example.com/x.md) — not a memory\n"
)


def test_parse_index():
    lines = parse_index(INDEX)
    got = [(ln.title, ln.target, ln.hook) for ln in lines]
    assert got == [
        ("Prefer short answers", "feedback_short.md", "keep replies brief"),
        ("Starred one", "starred.md", "has a rating prefix"),
        ("Bold one", "bold-one.md", "bold wrapped"),
        ("Nested [id] title", "nested.md", "hyphen hook"),
        ("En dash", "en.md", "en hook"),
        ("Colon", "colon.md", "colon hook"),
        ("Bare", "bare.md", ""),
    ]
    assert lines[0].raw == "- [Prefer short answers](feedback_short.md) — keep replies brief"


def test_upsert_noop_is_byte_identical():
    for ln in parse_index(INDEX):
        if ln.target in ("en.md", "colon.md", "nested.md"):
            continue  # non-canonical separators are normalised on rewrite
        new, _ = upsert_index_line(INDEX, title=ln.title, target=ln.target, hook=ln.hook)
        assert new == INDEX, ln.target


def test_upsert_replaces_in_place_and_keeps_decoration():
    new, line = upsert_index_line(INDEX, title="Starred v2", target="starred.md", hook="new\nhook")
    assert line == "- ⭐⭐ [Starred v2](starred.md) — new hook"
    old_lines, new_lines = INDEX.split("\n"), new.split("\n")
    assert len(old_lines) == len(new_lines)
    diff = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    assert diff == [4] and new_lines[4] == line

    new, line = upsert_index_line(INDEX, title="Bold", target="bold-one.md", hook="h")
    assert line == "- **[Bold](bold-one.md)** — h"


def test_upsert_appends():
    new, line = upsert_index_line(INDEX, title="New [x", target="new.md", hook="fresh")
    assert line == r"- [New \[x](new.md) — fresh"
    assert new == INDEX + line + "\n"
    assert parse_index(new)[-1].title == "New [x"
    # missing trailing newline / empty text
    assert upsert_index_line("# H", title="T", target="t.md", hook="")[0] == "# H\n- [T](t.md)\n"
    assert upsert_index_line("", title="T", target="t.md", hook="h")[0] == "- [T](t.md) — h\n"


def test_upsert_crlf():
    text = "# H\r\n- [A](a.md) — x\r\n"
    new, _ = upsert_index_line(text, title="A2", target="a.md", hook="y")
    assert new == "# H\r\n- [A2](a.md) — y\r\n"
    new, _ = upsert_index_line(text, title="B", target="b.md", hook="z")
    assert new == text + "- [B](b.md) — z\r\n"


def test_remove_index_line():
    new, removed = remove_index_line(INDEX, "starred.md")
    assert removed and "starred.md" not in new
    assert new == INDEX.replace("- ⭐⭐ [Starred one](starred.md) — has a rating prefix\n", "")
    assert remove_index_line(INDEX, "missing.md") == (INDEX, False)


def test_merge_index_conflict():
    current = "# Index\n- [A](a.md) — a ours\n- [B](b.md) — b\nnote line\n- [A again](a.md) — dup\n"
    conflict = "# Other heading\n- [C](c.md) — c theirs\n- [A](a.md) — a theirs\n"
    newer = merge_index_conflict(current, conflict, conflict_is_newer=True)
    assert newer == (
        "# Index\n- [A](a.md) — a theirs\n- [B](b.md) — b\nnote line\n- [C](c.md) — c theirs\n"
    )
    older = merge_index_conflict(current, conflict, conflict_is_newer=False)
    assert (
        older
        == "# Index\n- [A](a.md) — a ours\n- [B](b.md) — b\nnote line\n- [C](c.md) — c theirs\n"
    )
    assert (
        merge_index_conflict(current.split("note")[0], "", conflict_is_newer=True)
        == (current.split("note")[0])
    )


def test_note_round_trip():
    text = render_note(
        title="Planning: next steps", surface="claude.ai", created="2026-03-04T05:06:07Z",
        project="demo-project", related_sessions=["abc-123"], summary="We met.",
        decisions=["Use X"], next_steps=[], open_questions=["Why Y?"], details=None,
    )  # fmt: skip
    assert "## Summary\n\nWe met." in text
    assert "## Decisions\n\n- Use X" in text and "## Open questions\n\n- Why Y?" in text
    assert "## Next steps" not in text and "## Details" not in text
    n = parse_note(text)
    assert n.title == "Planning: next steps" and n.surface == "claude.ai"
    assert n.created == "2026-03-04T05:06:07Z" and n.project == "demo-project"
    assert n.related_sessions == ["abc-123"] and n.body.startswith("## Summary")


def test_slugify():
    assert slugify("Hello, World! Café") == "hello-world-cafe"
    assert slugify("!!!") == "untitled"
    assert slugify("a" * 30 + " " + "b" * 30, max_len=32) == "a" * 30 + "-b"
    assert not slugify("x " * 40, max_len=10).endswith("-")
