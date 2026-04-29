"""
tests/test_paths.py — paths.py + migrator tests.

Verifies that:
  * fresh installs do nothing (no legacy files).
  * legacy installs migrate all expected artifacts.
  * re-runs are no-ops (sentinel guards).
  * collisions leave legacy in place.
  * platformdirs returns the expected path.
"""

import json
from pathlib import Path

import pytest

from core import paths


@pytest.fixture
def patched_user_dir(monkeypatch, tmp_path):
    """Redirect paths.user_dir() to a temp directory."""
    target = tmp_path / "user_data"
    target.mkdir()
    monkeypatch.setattr(paths, "user_dir", lambda: target)
    return target


def test_user_dir_returns_existing_directory():
    ud = paths.user_dir()
    assert ud.exists()
    assert ud.is_dir()


def test_path_helpers_resolve_under_user_dir(patched_user_dir):
    assert paths.db_path() == patched_user_dir / "myai.db"
    assert paths.settings_path() == patched_user_dir / "settings.json"
    assert paths.log_path() == patched_user_dir / "app.log"
    assert paths.rag_cache_dir() == patched_user_dir / "rag_cache"
    assert paths.vector_store_dir() == patched_user_dir / "myai_vector_store"
    assert paths.extensions_dir() == patched_user_dir / "extensions"
    # Each helper that returns a directory should create it on demand
    assert paths.rag_cache_dir().is_dir()
    assert paths.vector_store_dir().is_dir()
    assert paths.extensions_dir().is_dir()


def test_migrate_fresh_install_is_noop(tmp_path):
    app_root = tmp_path / "install"
    app_root.mkdir()
    user_dir = tmp_path / "userdata"

    paths.migrate_legacy_install(app_root, user_dir)

    sentinel = user_dir / paths.MIGRATION_SENTINEL
    assert sentinel.exists()
    payload = json.loads(sentinel.read_text())
    assert payload["moved"] == []
    assert payload["skipped"] == []


def test_migrate_moves_all_legacy_artifacts(tmp_path):
    app_root = tmp_path / "install"
    app_root.mkdir()
    user_dir = tmp_path / "userdata"

    # Seed legacy install with one of each artifact
    (app_root / "myai.db").write_bytes(b"sqlite-magic")
    (app_root / "myai.db-wal").write_bytes(b"wal")
    (app_root / "myai.db-shm").write_bytes(b"shm")
    (app_root / "settings.json").write_text(
        json.dumps({"claude_api_key": "abc", "first_run_complete": True})
    )
    (app_root / "app.log").write_text("existing log\n")
    (app_root / "rag_cache").mkdir()
    (app_root / "rag_cache" / "index.npz").write_bytes(b"npz")
    (app_root / "myai_vector_store").mkdir()
    (app_root / "myai_vector_store" / "chroma.sqlite3").write_bytes(b"chroma")

    paths.migrate_legacy_install(app_root, user_dir)

    # Artifacts moved
    assert (user_dir / "myai.db").read_bytes() == b"sqlite-magic"
    assert (user_dir / "myai.db-wal").read_bytes() == b"wal"
    assert (user_dir / "myai.db-shm").read_bytes() == b"shm"
    assert (user_dir / "app.log").read_text() == "existing log\n"
    assert (user_dir / "rag_cache" / "index.npz").read_bytes() == b"npz"
    assert (user_dir / "myai_vector_store" / "chroma.sqlite3").read_bytes() == b"chroma"

    # settings.json carried intact (including first_run_complete flag)
    settings = json.loads((user_dir / "settings.json").read_text())
    assert settings["first_run_complete"] is True
    assert settings["claude_api_key"] == "abc"

    # Originals gone
    assert not (app_root / "myai.db").exists()
    assert not (app_root / "rag_cache").exists()

    # Sentinel lists what moved
    sentinel = user_dir / paths.MIGRATION_SENTINEL
    payload = json.loads(sentinel.read_text())
    assert set(payload["moved"]) == {
        "myai.db", "myai.db-wal", "myai.db-shm",
        "settings.json", "app.log",
        "rag_cache", "myai_vector_store",
    }


def test_migrate_is_idempotent_when_sentinel_present(tmp_path):
    app_root = tmp_path / "install"
    app_root.mkdir()
    user_dir = tmp_path / "userdata"
    user_dir.mkdir()
    (user_dir / paths.MIGRATION_SENTINEL).write_text("{}")

    # Seed legacy files AFTER sentinel — they should be left alone
    (app_root / "myai.db").write_bytes(b"newer")

    paths.migrate_legacy_install(app_root, user_dir)

    assert (app_root / "myai.db").exists()
    assert not (user_dir / "myai.db").exists()


def test_migrate_collision_leaves_legacy_in_place(tmp_path):
    app_root = tmp_path / "install"
    app_root.mkdir()
    user_dir = tmp_path / "userdata"
    user_dir.mkdir()

    # Destination already has a db — migrator must not overwrite
    (user_dir / "myai.db").write_bytes(b"existing-user-data")
    (app_root / "myai.db").write_bytes(b"legacy-would-clobber")

    paths.migrate_legacy_install(app_root, user_dir)

    assert (user_dir / "myai.db").read_bytes() == b"existing-user-data"
    assert (app_root / "myai.db").exists()
    sentinel = user_dir / paths.MIGRATION_SENTINEL
    payload = json.loads(sentinel.read_text())
    assert "myai.db" in payload["skipped"]


def test_migrate_never_raises_on_unexpected_error(monkeypatch, tmp_path):
    """Migrator must swallow exceptions — logging isn't configured yet."""
    def boom(*_a, **_kw):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(paths.shutil, "move", boom)
    monkeypatch.setattr(paths.shutil, "copy2", boom)
    monkeypatch.setattr(paths.shutil, "copytree", boom)

    app_root = tmp_path / "install"
    app_root.mkdir()
    (app_root / "myai.db").write_bytes(b"x")
    user_dir = tmp_path / "userdata"

    paths.migrate_legacy_install(app_root, user_dir)  # must not raise


def test_install_root_in_source_checkout():
    # In a source checkout install_root() should point at the app/ directory.
    root = paths.install_root()
    assert root.exists()
    # The paths.py module lives inside app/core/, so install_root is app/.
    assert (root / "core" / "paths.py").exists()
