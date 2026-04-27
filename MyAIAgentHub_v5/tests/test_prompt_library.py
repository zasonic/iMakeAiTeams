"""
tests/test_prompt_library.py — Unit tests for the prompt library service.

Covers: protected prompt enforcement, duplicate_prompt(), restore_version(),
export/import round-trip, and name-collision handling on import.
"""

import pytest


@pytest.fixture(autouse=True)
def use_db(in_memory_db):
    """All tests run against an isolated in-memory database."""
    return in_memory_db


def _make_prompt(name="Test Prompt", text="Hello {{user}}", protected=False):
    """Helper: create a prompt, optionally mark it protected, return its id."""
    import db
    from services.prompt_library import create_prompt
    result = create_prompt(
        name=name,
        category="Test",
        description="A test prompt",
        text=text,
    )
    if protected:
        db.execute("UPDATE prompts SET is_protected = 1 WHERE id = ?", (result["id"],))
        db.commit()
    return result["id"]


# ── Protected prompt cannot be edited ────────────────────────────────────────

def test_save_version_raises_on_protected_prompt():
    from services.prompt_library import save_prompt_version
    pid = _make_prompt(protected=True)
    with pytest.raises(ValueError, match="[Pp]rotected"):
        save_prompt_version(pid, text="new text")


def test_save_version_succeeds_on_user_prompt():
    from services.prompt_library import save_prompt_version, get_prompt_versions
    pid = _make_prompt()
    result = save_prompt_version(pid, text="updated text")
    assert "version_id" in result
    versions = get_prompt_versions(pid)
    assert any(v["text"] == "updated text" for v in versions)


# ── duplicate_prompt() creates editable copy ─────────────────────────────────

def test_duplicate_prompt_creates_editable_copy():
    import db
    from services.prompt_library import duplicate_prompt
    pid = _make_prompt(name="Original", text="original text", protected=True)
    result = duplicate_prompt(pid, "Copy of Original")
    new_id = result["id"]
    row = db.fetchone("SELECT is_protected FROM prompts WHERE id = ?", (new_id,))
    assert row is not None
    assert row["is_protected"] == 0


def test_duplicate_prompt_inherits_text():
    from services.prompt_library import duplicate_prompt, get_prompt_versions
    pid = _make_prompt(name="Source", text="source text")
    result = duplicate_prompt(pid, "Source Copy")
    versions = get_prompt_versions(result["id"])
    assert any("source text" in v["text"] for v in versions)


# ── restore_version() ─────────────────────────────────────────────────────────

def test_restore_version_makes_older_version_active():
    import db
    from services.prompt_library import (
        save_prompt_version, get_prompt_versions, restore_version, get_prompt
    )
    pid = _make_prompt(text="v1 text")
    save_prompt_version(pid, text="v2 text")
    versions = get_prompt_versions(pid)
    v1 = next(v for v in versions if "v1 text" in v["text"])
    restore_version(v1["id"])
    # The active version_id should now point to a version containing v1 text
    prompt = get_prompt(pid)
    active_version = db.fetchone(
        "SELECT text FROM prompt_versions WHERE id = ?", (prompt["active_version_id"],)
    )
    assert active_version is not None
    assert "v1 text" in active_version["text"]


def test_restore_version_raises_on_missing_version():
    from services.prompt_library import restore_version
    with pytest.raises(ValueError):
        restore_version("nonexistent-version-id")


# ── export / import round-trip ────────────────────────────────────────────────

def test_export_import_round_trip():
    from services.prompt_library import (
        create_prompt, save_prompt_version, export_prompt, import_prompt, get_prompt
    )
    pid = _make_prompt(name="Export Me", text="export content")
    save_prompt_version(pid, text="export content v2")

    exported = export_prompt(pid)
    assert exported["name"] == "Export Me"
    assert len(exported["versions"]) >= 2

    imported = import_prompt(exported)
    new_pid = imported["id"]
    p = get_prompt(new_pid)
    assert p is not None
    assert "Export Me" in p["name"]


def test_export_preserves_all_fields():
    from services.prompt_library import export_prompt
    pid = _make_prompt(name="Fields Test", text="field content")
    exported = export_prompt(pid)
    assert exported["schema_version"] == "1"
    assert exported["category"] == "Test"
    assert exported["description"] == "A test prompt"
    assert len(exported["versions"]) >= 1
    assert exported["versions"][0]["text"] == "field content"


# ── Name collision on import appends timestamp suffix ────────────────────────

def test_import_name_collision_appends_suffix():
    from services.prompt_library import export_prompt, import_prompt
    pid = _make_prompt(name="Collision Name", text="original")
    exported = export_prompt(pid)
    # Import same data a second time — name conflict should be resolved
    result2 = import_prompt(exported)
    import db
    row = db.fetchone("SELECT name FROM prompts WHERE id = ?", (result2["id"],))
    assert row["name"] != "Collision Name"
    assert "Collision Name" in row["name"]  # original name preserved as prefix
