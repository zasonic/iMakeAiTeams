"""
tests/test_settings_keyring.py — Secret routing through the OS keyring.

Verifies that:
  * claude_api_key is read/written via keyring when a backend is present.
  * a plaintext key in settings.json is migrated into the keyring on load.
  * keyring failures gracefully fall back to plaintext settings.
"""

import json

import pytest

from core import settings as settings_module
from core.settings import Settings


@pytest.fixture
def memory_keyring(monkeypatch):
    """Swap the module-level keyring helpers for an in-memory dict."""
    store: dict[str, str] = {}

    def fake_get(key: str):
        return store.get(key)

    def fake_set(key: str, value: str) -> bool:
        store[key] = value
        return True

    def fake_delete(key: str) -> None:
        store.pop(key, None)

    monkeypatch.setattr(settings_module, "_keyring_get", fake_get)
    monkeypatch.setattr(settings_module, "_keyring_set", fake_set)
    monkeypatch.setattr(settings_module, "_keyring_delete", fake_delete)
    return store


@pytest.fixture
def broken_keyring(monkeypatch):
    """Simulate an environment where keyring is installed but the backend is unusable."""
    monkeypatch.setattr(settings_module, "_keyring_get", lambda _k: None)
    monkeypatch.setattr(settings_module, "_keyring_set", lambda _k, _v: False)
    monkeypatch.setattr(settings_module, "_keyring_delete", lambda _k: None)


def test_set_routes_claude_api_key_through_keyring(tmp_path, memory_keyring):
    s = Settings(tmp_path / "settings.json")
    s.set("claude_api_key", "sk-live-abc")

    assert memory_keyring["claude_api_key"] == "sk-live-abc"
    assert s.get("claude_api_key") == "sk-live-abc"

    # The JSON file on disk must NOT contain the plaintext secret.
    disk = json.loads((tmp_path / "settings.json").read_text())
    assert disk["claude_api_key"] == ""


def test_plaintext_key_is_migrated_to_keyring_on_load(tmp_path, memory_keyring):
    # Simulate a pre-migration settings.json with the secret still in cleartext.
    (tmp_path / "settings.json").write_text(
        json.dumps({"claude_api_key": "sk-legacy-plaintext", "first_run_complete": True})
    )

    s = Settings(tmp_path / "settings.json")

    # Keyring now owns the secret.
    assert memory_keyring["claude_api_key"] == "sk-legacy-plaintext"
    assert s.get("claude_api_key") == "sk-legacy-plaintext"

    # JSON on disk is scrubbed.
    disk = json.loads((tmp_path / "settings.json").read_text())
    assert disk["claude_api_key"] == ""
    # Unrelated keys are untouched.
    assert disk["first_run_complete"] is True


def test_clearing_the_key_removes_it_from_keyring(tmp_path, memory_keyring):
    s = Settings(tmp_path / "settings.json")
    s.set("claude_api_key", "sk-live-abc")
    assert memory_keyring.get("claude_api_key") == "sk-live-abc"

    s.set("claude_api_key", "")
    assert "claude_api_key" not in memory_keyring
    assert s.get("claude_api_key") == ""


def test_falls_back_to_plaintext_when_keyring_unavailable(tmp_path, broken_keyring):
    s = Settings(tmp_path / "settings.json")
    s.set("claude_api_key", "sk-fallback")

    # With a broken backend, set() must still persist to disk — otherwise the
    # app would silently drop the key every time the user enters it.
    assert s.get("claude_api_key") == "sk-fallback"
    disk = json.loads((tmp_path / "settings.json").read_text())
    assert disk["claude_api_key"] == "sk-fallback"


def test_non_secret_keys_never_touch_keyring(tmp_path, memory_keyring):
    s = Settings(tmp_path / "settings.json")
    s.set("claude_model", "claude-opus-4-7")

    assert "claude_model" not in memory_keyring
    disk = json.loads((tmp_path / "settings.json").read_text())
    assert disk["claude_model"] == "claude-opus-4-7"
