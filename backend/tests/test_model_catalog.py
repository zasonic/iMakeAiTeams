"""
tests/test_model_catalog.py — Layer B1: ModelCatalog single source of truth.

Pins three invariants:

  1. Every catalog entry round-trips through ``_estimate_cost`` and
     produces the SAME number that would have come out of the old
     hardcoded ``_DEFAULT_MODEL_PRICES`` table. This is the property
     test the plan called out — Bug 12 (savings calc hardcoding Sonnet
     price) and every future variant of "the price table drifted" is
     blocked.
  2. ``detect_family`` is deterministic over substring order — Bug 12's
     root cause was dict-iteration ambiguity on a name like
     ``claude-haiku-with-opus-fallback``.
  3. ``SETTINGS_DEFAULTS["claude_model"]`` matches the catalog's
     ``default_claude_id``, so the default users see in Settings is
     always priceable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.model_catalog import (
    Catalog, ModelEntry, get_catalog, set_catalog_path_for_testing,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_catalog_cache():
    """Each test starts from a clean cache so set_catalog_path_for_testing
    isn't sticky across runs."""
    set_catalog_path_for_testing(None)
    yield
    set_catalog_path_for_testing(None)


def _shipped_catalog() -> Catalog:
    """Force-load the catalog that actually ships in the installer."""
    set_catalog_path_for_testing(None)
    return get_catalog(force_reload=True)


# ── Schema + parsing ─────────────────────────────────────────────────────────


def test_shipped_catalog_loads_cleanly():
    cat = _shipped_catalog()
    assert isinstance(cat, Catalog)
    assert cat.default_claude_id
    assert cat.models, "shipped catalog must declare at least one model"
    for entry in cat.models:
        assert entry.id.startswith("claude-"), f"non-Claude entry: {entry.id}"
        assert entry.family in ("opus", "sonnet", "haiku")
        assert entry.input_price_per_mtok >= 0
        assert entry.output_price_per_mtok >= 0
        assert entry.context_window_tokens > 0
        assert "claude_api" in entry.available_via


def test_default_claude_id_is_in_catalog():
    cat = _shipped_catalog()
    assert cat.find_by_id(cat.default_claude_id) is not None


def test_settings_default_matches_catalog_default():
    """The hardcoded claude_model default in SETTINGS_DEFAULTS must point
    at a model the catalog actually carries — otherwise a fresh install
    would show a default the price math can't honour without falling
    back to the family substring path."""
    from core.settings import SETTINGS_DEFAULTS
    _, default = SETTINGS_DEFAULTS["claude_model"]
    assert _shipped_catalog().find_by_id(default) is not None, (
        f"SETTINGS_DEFAULTS['claude_model'] = {default!r} is not in "
        f"backend/config/models.json. Either update the default or add "
        f"the entry to the catalog."
    )


# ── Family detection (Bug 12 invariant) ──────────────────────────────────────


def test_detect_family_prefers_opus_over_other_substrings():
    """A name like 'claude-haiku-with-opus-fallback' must resolve to
    'opus' deterministically — the family scan iterates in the fixed
    order opus → sonnet → haiku, so the first substring match wins.

    Pre-catalog code iterated a dict and the result depended on Python's
    dict iteration order. Bug 12 was a downstream symptom.
    """
    cat = _shipped_catalog()
    assert cat.detect_family("claude-haiku-with-opus-fallback") == "opus"


def test_detect_family_returns_none_for_non_claude():
    cat = _shipped_catalog()
    assert cat.detect_family("gpt-4o-mini") is None
    assert cat.detect_family("") is None


# ── prices_for_model: catalog-id ↔ _estimate_cost consistency ────────────────


def test_prices_for_model_returns_exact_match_when_id_in_catalog():
    cat = _shipped_catalog()
    for entry in cat.models:
        price_in, price_out = cat.prices_for_model(entry.id)
        assert price_in == entry.input_price_per_mtok
        assert price_out == entry.output_price_per_mtok


def test_prices_for_model_falls_back_to_family_for_unknown_id():
    cat = _shipped_catalog()
    # Unseen sonnet id should still resolve via the family fallback.
    price_in, price_out = cat.prices_for_model("claude-sonnet-9-99-future")
    assert (price_in, price_out) == cat.family_fallback_prices["sonnet"]


def test_prices_for_model_honours_user_overrides_by_family():
    cat = _shipped_catalog()
    overrides = {"sonnet": (1.0, 5.0)}
    price_in, price_out = cat.prices_for_model("claude-sonnet-4-6", overrides)
    assert (price_in, price_out) == (1.0, 5.0)


def test_prices_for_model_default_when_non_claude():
    """Non-Claude ids fall through to the Sonnet default — kept for
    pre-catalog compatibility with the orchestrator's _estimate_cost
    sentinel."""
    cat = _shipped_catalog()
    assert cat.prices_for_model("gpt-4o-mini") == (3.0, 15.0)


# ── _estimate_cost integration (the actual orchestrator entry point) ─────────


def test_estimate_cost_matches_catalog_prices_for_every_entry():
    """The big property test: every model in the catalog, when fed to
    ``_estimate_cost``, must produce exactly
    ``tokens_in * input + tokens_out * output / 1e6``. Bug 12 is blocked."""
    from services.chat_orchestrator import _estimate_cost

    cat = _shipped_catalog()
    tokens_in, tokens_out = 1_000_000, 500_000
    for entry in cat.models:
        expected = (
            tokens_in * entry.input_price_per_mtok
            + tokens_out * entry.output_price_per_mtok
        ) / 1_000_000
        got = _estimate_cost(entry.id, tokens_in, tokens_out, settings=None)
        assert got == pytest.approx(expected), (
            f"price drift for {entry.id}: got {got}, expected {expected}"
        )


def test_estimate_cost_honours_settings_model_prices_override():
    """User-set ``model_prices`` per family must still win — the catalog
    is the default, not a hard cap."""
    from services.chat_orchestrator import _estimate_cost

    class _Settings:
        def get(self, key, default=None):
            if key == "model_prices":
                return {"sonnet": [2.0, 10.0]}
            return default

    cost = _estimate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, settings=_Settings())
    assert cost == pytest.approx(2.0 + 10.0)


def test_estimate_cost_returns_zero_for_non_claude_model():
    from services.chat_orchestrator import _estimate_cost
    assert _estimate_cost("gpt-4o-mini", 1000, 1000) == 0.0
    assert _estimate_cost("", 1000, 1000) == 0.0


# ── Schema-violation rejection ───────────────────────────────────────────────


def test_catalog_rejects_malformed_models_entry(tmp_path):
    """A models[] row missing a required field must fail loudly at load
    time, not be silently dropped — otherwise a typo in the JSON would
    quietly remove a model from the dropdown."""
    bad = {
        "default_claude_id": "claude-sonnet-4-6",
        "models": [{"id": "claude-sonnet-4-6"}],  # missing every other field
        "family_fallback_prices": {},
    }
    catalog_path = tmp_path / "models.json"
    catalog_path.write_text(json.dumps(bad))
    set_catalog_path_for_testing(catalog_path)
    with pytest.raises(ValueError, match=r"models\[0\] invalid"):
        get_catalog(force_reload=True)


def test_catalog_warns_when_default_id_missing(tmp_path):
    """If default_claude_id points at a model not in the list, fall back
    to the first listed entry rather than crashing — easier to
    diagnose than a startup error during installer rollouts."""
    raw = {
        "default_claude_id": "claude-not-real",
        "models": [
            {
                "id": "claude-sonnet-4-6", "family": "sonnet",
                "display_name": "Sonnet", "input_price_per_mtok": 3.0,
                "output_price_per_mtok": 15.0, "context_window_tokens": 200000,
                "vision": True, "available_via": ["claude_api"],
            },
        ],
        "family_fallback_prices": {},
    }
    p = tmp_path / "models.json"
    p.write_text(json.dumps(raw))
    set_catalog_path_for_testing(p)
    cat = get_catalog(force_reload=True)
    assert cat.default_claude_id == "claude-sonnet-4-6"
