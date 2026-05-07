"""
tests/test_model_canary.py

Phase 5: Local-model behavior-drift canary (arXiv 2511.15992).

Covers:
- capture_baseline writes one row per CANARY_PROMPTS entry on first load
- capture_baseline is idempotent (second call inserts 0 new rows)
- check_drift with identical responses → alert=False, no drifted_prompts
- check_drift with shifted responses → alert=True, drifted_prompts populated
- signal_model_loaded short-circuits when the setting is disabled
- _run_canary on a fresh model_id captures, on an existing model_id checks
- reset_baseline deletes rows
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── Fixtures / helpers ────────────────────────────────────────────────────────

@pytest.fixture
def settings_canary_on(tmp_path):
    from core.settings import Settings
    s = Settings(tmp_path / "settings.json")
    s.set("model_canary_enabled", True)
    return s


@pytest.fixture
def settings_canary_off(tmp_path):
    from core.settings import Settings
    s = Settings(tmp_path / "settings.json")
    s.set("model_canary_enabled", False)
    return s


def _hash_vec(text: str) -> np.ndarray:
    """Deterministic 32-dim float32 vector seeded by MD5 of `text`.

    Identical strings produce identical vectors (same seed); different
    strings produce independent gaussian samples whose cosine similarity
    is ~0 in expectation. That gives the drift test a clean signal —
    "different response" really means "different embedding direction."
    """
    seed = int.from_bytes(
        hashlib.md5(text.encode("utf-8")).digest()[:8], "big",
    )
    rng = np.random.default_rng(seed)
    return rng.standard_normal(32).astype(np.float32)


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace model_canary._embed with a deterministic hash-based stub.

    The real function loads a fastembed model (~200 MB download) which
    is unacceptable in CI. The stub preserves the contract: identical
    text → identical vector; different text → different vector.
    """
    from services import model_canary

    def stub(text, embedder=None):
        return _hash_vec(text)

    monkeypatch.setattr(model_canary, "_embed", stub)
    return stub


def _local_with_responses(responses_by_prompt):
    """Build a mock local_client that returns the canned response per prompt.

    `responses_by_prompt` maps prompt → response. Prompts not in the dict
    fall through to a default response; this lets a test seed only the
    drifted prompts and let the rest stay stable.
    """
    client = MagicMock()
    client.is_available.return_value = True

    def _chat(_system, prompt, max_tokens=200, **_kwargs):
        return responses_by_prompt.get(prompt, f"baseline-response-for::{prompt}")

    client.chat.side_effect = _chat
    return client


# ── capture_baseline ──────────────────────────────────────────────────────────

class TestCaptureBaseline:
    def test_first_load_writes_one_row_per_prompt(
        self, in_memory_db, settings_canary_on, fake_embed,
    ):
        from services import model_canary

        local = _local_with_responses({})
        inserted = model_canary.capture_baseline(local, "model-A")
        assert inserted == len(model_canary.CANARY_PROMPTS)

        rows = in_memory_db.fetchall(
            "SELECT prompt_hash, prompt_text, response_text "
            "FROM canary_baseline WHERE model_id = ?",
            ("model-A",),
        )
        assert len(rows) == len(model_canary.CANARY_PROMPTS)
        prompts_persisted = {r["prompt_text"] for r in rows}
        assert prompts_persisted == set(model_canary.CANARY_PROMPTS)

    def test_capture_is_idempotent(
        self, in_memory_db, settings_canary_on, fake_embed,
    ):
        from services import model_canary

        local = _local_with_responses({})
        first = model_canary.capture_baseline(local, "model-A")
        second = model_canary.capture_baseline(local, "model-A")
        assert first == len(model_canary.CANARY_PROMPTS)
        assert second == 0

        # Local was not asked to regenerate any prompt on the second pass.
        assert local.chat.call_count == len(model_canary.CANARY_PROMPTS)

    def test_baselines_are_per_model_id(
        self, in_memory_db, settings_canary_on, fake_embed,
    ):
        from services import model_canary

        local = _local_with_responses({})
        model_canary.capture_baseline(local, "model-A")
        model_canary.capture_baseline(local, "model-B")

        a_count = in_memory_db.fetchone(
            "SELECT COUNT(*) AS n FROM canary_baseline WHERE model_id = 'model-A'"
        )["n"]
        b_count = in_memory_db.fetchone(
            "SELECT COUNT(*) AS n FROM canary_baseline WHERE model_id = 'model-B'"
        )["n"]
        assert a_count == len(model_canary.CANARY_PROMPTS)
        assert b_count == len(model_canary.CANARY_PROMPTS)

    def test_empty_response_skipped(
        self, in_memory_db, settings_canary_on, fake_embed,
    ):
        from services import model_canary

        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = ""
        inserted = model_canary.capture_baseline(local, "model-A")
        assert inserted == 0

        rows = in_memory_db.fetchall(
            "SELECT id FROM canary_baseline WHERE model_id = 'model-A'"
        )
        assert rows == []


# ── check_drift ───────────────────────────────────────────────────────────────

class TestCheckDrift:
    def test_identical_responses_no_alert(
        self, in_memory_db, settings_canary_on, fake_embed,
    ):
        from services import model_canary

        local = _local_with_responses({})
        model_canary.capture_baseline(local, "model-A")

        # Re-run with the same canned responses → embeddings identical.
        report = model_canary.check_drift(local, "model-A")
        assert report.alert is False
        assert report.drifted_prompts == []
        assert report.mean_drift == pytest.approx(0.0, abs=1e-6)
        assert report.max_cosine_drift == pytest.approx(0.0, abs=1e-6)

    def test_shifted_responses_alert_with_drifted_prompts(
        self, in_memory_db, settings_canary_on, fake_embed,
    ):
        from services import model_canary

        # Capture baseline with default responses.
        baseline_local = _local_with_responses({})
        model_canary.capture_baseline(baseline_local, "model-A")

        # Re-run with a totally different response per prompt — every embedding
        # should diverge, producing high drift across the board.
        def _shifted_chat(_system, prompt, max_tokens=200, **_kwargs):
            return f"DRIFTED-{prompt}-{'X' * 40}"

        shifted_local = MagicMock()
        shifted_local.is_available.return_value = True
        shifted_local.chat.side_effect = _shifted_chat

        report = model_canary.check_drift(shifted_local, "model-A")
        assert report.alert is True
        assert report.mean_drift > model_canary.DRIFT_ALERT_THRESHOLD
        # Hash-based fake embeddings: every prompt's response moved, so the
        # drifted_prompts list should be a subset of CANARY_PROMPTS.
        assert len(report.drifted_prompts) > 0
        assert all(p in model_canary.CANARY_PROMPTS for p in report.drifted_prompts)

    def test_no_baseline_returns_empty_report(
        self, in_memory_db, settings_canary_on, fake_embed,
    ):
        from services import model_canary

        local = _local_with_responses({})
        report = model_canary.check_drift(local, "missing-model")
        assert report.alert is False
        assert report.drifted_prompts == []
        # Without a baseline there is nothing to compare against, so we
        # never call the local model either.
        local.chat.assert_not_called()


# ── signal_model_loaded / _run_canary ────────────────────────────────────────

class TestSignalModelLoaded:
    def test_disabled_setting_skips_capture(
        self, in_memory_db, settings_canary_off, fake_embed,
    ):
        from services import model_canary

        local = _local_with_responses({})
        # Synchronous variant: same gating, no thread fragility.
        model_canary._run_canary(local, "model-A", settings_canary_off)

        rows = in_memory_db.fetchall(
            "SELECT id FROM canary_baseline WHERE model_id = 'model-A'"
        )
        assert rows == []
        local.chat.assert_not_called()

    def test_disabled_setting_skips_drift_check(
        self, in_memory_db, settings_canary_on, settings_canary_off, fake_embed,
    ):
        from services import model_canary

        # Seed a baseline with the gate ON.
        local = _local_with_responses({})
        model_canary.capture_baseline(local, "model-A")
        local.chat.reset_mock()

        # Now disable the canary and call _run_canary — it must not invoke
        # the model or write any new rows.
        model_canary._run_canary(local, "model-A", settings_canary_off)
        local.chat.assert_not_called()

    def test_first_call_captures_then_subsequent_emits_alert(
        self, in_memory_db, settings_canary_on, fake_embed, monkeypatch,
    ):
        from services import model_canary

        # Drop in a captured SSE publisher so we can assert the alert payload.
        published: list[tuple[str, dict]] = []

        class _StubSse:
            @staticmethod
            def publish(event, payload):
                published.append((event, payload))

        monkeypatch.setattr(model_canary, "_sse_events", _StubSse)

        # First call: baseline does not exist, so _run_canary captures and
        # does NOT emit an alert.
        baseline_local = _local_with_responses({})
        model_canary._run_canary(baseline_local, "model-A", settings_canary_on)
        assert published == []
        baseline_count = in_memory_db.fetchone(
            "SELECT COUNT(*) AS n FROM canary_baseline WHERE model_id = 'model-A'"
        )["n"]
        assert baseline_count == len(model_canary.CANARY_PROMPTS)

        # Second call: baseline exists, but a different local client returns
        # different responses → alert fires.
        def _shifted_chat(_system, prompt, max_tokens=200, **_kwargs):
            return f"DRIFT::{prompt}::{'Y' * 40}"

        shifted_local = MagicMock()
        shifted_local.is_available.return_value = True
        shifted_local.chat.side_effect = _shifted_chat

        model_canary._run_canary(shifted_local, "model-A", settings_canary_on)

        assert any(name == "model_canary_alert" for name, _ in published)
        alert = next(p for n, p in published if n == "model_canary_alert")
        assert alert["model_id"] == "model-A"
        assert alert["mean_drift"] > model_canary.DRIFT_ALERT_THRESHOLD
        assert 1 <= len(alert["drifted_prompts"]) <= 3

    def test_signal_model_loaded_with_empty_id_is_noop(
        self, in_memory_db, settings_canary_on, fake_embed,
    ):
        from services import model_canary

        local = _local_with_responses({})
        # Empty model_id must not spawn a thread or write rows.
        model_canary.signal_model_loaded(local, "", settings_canary_on)
        rows = in_memory_db.fetchall("SELECT id FROM canary_baseline")
        assert rows == []


# ── reset_baseline ────────────────────────────────────────────────────────────

class TestResetBaseline:
    def test_reset_deletes_rows_for_model_id(
        self, in_memory_db, settings_canary_on, fake_embed,
    ):
        from services import model_canary

        local = _local_with_responses({})
        model_canary.capture_baseline(local, "model-A")
        model_canary.capture_baseline(local, "model-B")

        deleted = model_canary.reset_baseline("model-A")
        assert deleted == len(model_canary.CANARY_PROMPTS)

        a_count = in_memory_db.fetchone(
            "SELECT COUNT(*) AS n FROM canary_baseline WHERE model_id = 'model-A'"
        )["n"]
        b_count = in_memory_db.fetchone(
            "SELECT COUNT(*) AS n FROM canary_baseline WHERE model_id = 'model-B'"
        )["n"]
        assert a_count == 0
        assert b_count == len(model_canary.CANARY_PROMPTS)

    def test_reset_unknown_model_returns_zero(
        self, in_memory_db, settings_canary_on,
    ):
        from services import model_canary

        deleted = model_canary.reset_baseline("nonexistent")
        assert deleted == 0

    def test_has_baseline(self, in_memory_db, settings_canary_on, fake_embed):
        from services import model_canary

        assert model_canary.has_baseline("model-A") is False
        local = _local_with_responses({})
        model_canary.capture_baseline(local, "model-A")
        assert model_canary.has_baseline("model-A") is True
