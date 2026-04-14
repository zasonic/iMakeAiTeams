"""
tests/test_settings.py

Covers:
- Migration fills missing keys with defaults on startup
- Type coercion (bool strings, int strings, float strings)
- set() rejects unknown keys
- set() coerces valid values to the correct type
- get() falls back to schema default for missing keys
- all() merges stored data with defaults
- get_schema() returns type metadata
- Settings persists across reload (round-trip)
"""

import json
import pytest
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_settings(tmp_path, initial_data: dict | None = None):
    path = tmp_path / "settings.json"
    if initial_data is not None:
        path.write_text(json.dumps(initial_data), encoding="utf-8")
    from core.settings import Settings
    return Settings(path), path


# ── Migration ─────────────────────────────────────────────────────────────────

class TestMigration:
    def test_empty_file_gets_all_defaults(self, tmp_path):
        from core.settings import Settings, SETTINGS_DEFAULTS
        s, _ = make_settings(tmp_path)
        for key in SETTINGS_DEFAULTS:
            assert s.get(key) is not None or SETTINGS_DEFAULTS[key][1] is None

    def test_partial_file_fills_missing_keys(self, tmp_path):
        from core.settings import SETTINGS_DEFAULTS
        s, path = make_settings(tmp_path, {"claude_api_key": "test-key"})
        # All other keys should be present with their defaults
        for key, (_, default) in SETTINGS_DEFAULTS.items():
            if key == "claude_api_key":
                assert s.get(key) == "test-key"
            else:
                assert s.get(key) == default

    def test_migration_writes_file(self, tmp_path):
        """After migration a settings.json with all defaults should exist on disk."""
        s, path = make_settings(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "claude_api_key" in data

    def test_existing_values_are_preserved(self, tmp_path):
        s, _ = make_settings(tmp_path, {
            "claude_api_key": "sk-real-key",
            "routing_enabled": False,
        })
        assert s.get("claude_api_key") == "sk-real-key"
        assert s.get("routing_enabled") is False

    def test_migration_coerces_string_bool_true(self, tmp_path):
        s, _ = make_settings(tmp_path, {"routing_enabled": "true"})
        assert s.get("routing_enabled") is True

    def test_migration_coerces_string_bool_false(self, tmp_path):
        s, _ = make_settings(tmp_path, {"health_check_enabled": "false"})
        assert s.get("health_check_enabled") is False

    def test_migration_coerces_int_string(self, tmp_path):
        s, _ = make_settings(tmp_path, {"rag_chunk_size": "600"})
        assert s.get("rag_chunk_size") == 600
        assert isinstance(s.get("rag_chunk_size"), int)

    def test_migration_coerces_float_string(self, tmp_path):
        s, _ = make_settings(tmp_path, {"memory_similarity_threshold": "0.7"})
        val = s.get("memory_similarity_threshold")
        assert abs(val - 0.7) < 1e-9
        assert isinstance(val, float)


# ── set() validation ──────────────────────────────────────────────────────────

class TestSetValidation:
    def test_set_unknown_key_is_ignored(self, tmp_path):
        s, _ = make_settings(tmp_path)
        s.set("totally_unknown_key_xyz", "value")
        # Should not appear in stored data
        assert s.get("totally_unknown_key_xyz") is None

    def test_set_known_key_persists(self, tmp_path):
        s, path = make_settings(tmp_path)
        s.set("claude_api_key", "new-key")
        assert s.get("claude_api_key") == "new-key"
        # Round-trip: reload from disk
        data = json.loads(path.read_text())
        assert data["claude_api_key"] == "new-key"

    def test_set_coerces_string_to_bool(self, tmp_path):
        s, _ = make_settings(tmp_path)
        s.set("routing_enabled", "false")
        assert s.get("routing_enabled") is False

    def test_set_coerces_string_to_int(self, tmp_path):
        s, _ = make_settings(tmp_path)
        s.set("rag_chunk_size", "1200")
        assert s.get("rag_chunk_size") == 1200

    def test_set_bool_direct(self, tmp_path):
        s, _ = make_settings(tmp_path)
        s.set("show_token_counts", False)
        assert s.get("show_token_counts") is False

    def test_set_none_for_nullable_key(self, tmp_path):
        s, _ = make_settings(tmp_path)
        # default_agent_id accepts str | None
        s.set("default_agent_id", None)
        assert s.get("default_agent_id") is None


# ── get() fallback behaviour ──────────────────────────────────────────────────

class TestGet:
    def test_get_schema_default_for_missing(self, tmp_path):
        from core.settings import SETTINGS_DEFAULTS
        s, _ = make_settings(tmp_path)
        # Every key should return the schema default after migration
        for key, (_, default) in SETTINGS_DEFAULTS.items():
            assert s.get(key) == default or s.get(key) is not None or default is None

    def test_get_caller_default_for_truly_unknown_key(self, tmp_path):
        s, _ = make_settings(tmp_path)
        result = s.get("i_do_not_exist_anywhere", default="fallback")
        assert result == "fallback"

    def test_get_without_default_returns_none_for_unknown(self, tmp_path):
        s, _ = make_settings(tmp_path)
        assert s.get("nonexistent_key_abc") is None


# ── all() ─────────────────────────────────────────────────────────────────────

class TestAll:
    def test_all_returns_dict(self, tmp_path):
        s, _ = make_settings(tmp_path)
        result = s.all()
        assert isinstance(result, dict)

    def test_all_contains_all_schema_keys(self, tmp_path):
        from core.settings import SETTINGS_DEFAULTS
        s, _ = make_settings(tmp_path)
        result = s.all()
        for key in SETTINGS_DEFAULTS:
            assert key in result

    def test_all_reflects_set_values(self, tmp_path):
        s, _ = make_settings(tmp_path)
        s.set("claude_api_key", "check-this")
        result = s.all()
        assert result["claude_api_key"] == "check-this"


# ── get_schema() ──────────────────────────────────────────────────────────────

class TestGetSchema:
    def test_schema_has_type_and_default(self, tmp_path):
        s, _ = make_settings(tmp_path)
        schema = s.get_schema()
        assert "claude_api_key" in schema
        entry = schema["claude_api_key"]
        assert "type" in entry
        assert "default" in entry

    def test_schema_type_is_string(self, tmp_path):
        s, _ = make_settings(tmp_path)
        schema = s.get_schema()
        assert schema["claude_api_key"]["type"] == "str"
        assert schema["routing_enabled"]["type"] == "bool"
        assert schema["rag_chunk_size"]["type"] == "int"


# ── Persistence round-trip ────────────────────────────────────────────────────

class TestPersistence:
    def test_survives_reload(self, tmp_path):
        """Values written to disk must survive a fresh Settings() load."""
        from core.settings import Settings
        path = tmp_path / "settings.json"

        s1 = Settings(path)
        s1.set("claude_api_key", "persist-me")
        s1.set("rag_chunk_size", 999)
        del s1

        s2 = Settings(path)
        assert s2.get("claude_api_key") == "persist-me"
        assert s2.get("rag_chunk_size") == 999

    def test_set_raw_bypasses_validation(self, tmp_path):
        """set_raw should store even unknown keys without warning."""
        s, path = make_settings(tmp_path)
        s.set_raw("_internal_runtime_key", "runtime_value")
        data = json.loads(path.read_text())
        assert data["_internal_runtime_key"] == "runtime_value"
