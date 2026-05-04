"""
tests/test_research_extensions.py — Tests for the four research-aligned
extensions added on the claude/review-app-architecture branch:

1. Per-complexity adaptive thresholds in the router (MAPPA-inspired).
2. Proof-of-work handoff validation (Symphony-inspired).
3. Toggleable sliding-window risk ledger (DiLoCo-inspired, default off).
4. Post-assembly multi-agent alignment check (Anthropic AI Orgs-inspired).

Each section is self-contained — tests use mocked clients so no real
model inference happens. The goal is to verify the new code paths
behave correctly AND that the default-off / no-op fall-through paths
preserve existing behavior.

Run: pytest tests/test_research_extensions.py -v
"""

import time
import json
from unittest.mock import MagicMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Per-complexity adaptive threshold (router.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveThresholdPerComplexity:
    """The router's _adaptive_threshold should use per-complexity error rates
    when complexity is provided, falling back to the aggregate rate when the
    per-bucket sample is too small."""

    def _make_router(self, settings):
        from services.router import TaskRouter
        local = MagicMock()
        local.is_available.return_value = True
        return TaskRouter(local, settings)

    def _seed_router_log(self, db_module, *, complexity: str, total: int, errors: int):
        """Insert `total` rows for the given complexity, with `errors` of them flagged bad."""
        import uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for i in range(total):
            had_error = 1 if i < errors else 0
            db_module.execute(
                "INSERT INTO router_log (id, conversation_id, message_preview, "
                "route_taken, complexity, reasoning, tokens_out, had_error, "
                "response_empty, model_used, created_at) "
                "VALUES (?, ?, ?, 'local', ?, ?, ?, ?, 0, ?, ?)",
                (str(uuid.uuid4()), "conv-x", "msg", complexity,
                 "test", 100, had_error, "local", now),
            )
        db_module.commit()

    def test_returns_default_when_no_data(self, in_memory_db, settings):
        from services.router import ESCALATION_THRESHOLD
        router = self._make_router(settings)
        assert router._adaptive_threshold(complexity="simple") == ESCALATION_THRESHOLD

    def test_per_complexity_tightens_only_affected_bucket(self, in_memory_db, settings):
        """Medium failing at 40% should tighten medium's threshold but not simple's."""
        from services.router import ESCALATION_THRESHOLD
        router = self._make_router(settings)

        # Simple: 20 rows, 0 errors (healthy)
        self._seed_router_log(in_memory_db, complexity="simple", total=20, errors=0)
        # Medium: 20 rows, 8 errors (40% — well above the 15% floor)
        self._seed_router_log(in_memory_db, complexity="medium", total=20, errors=8)

        simple_threshold = router._adaptive_threshold(complexity="simple")
        medium_threshold = router._adaptive_threshold(complexity="medium")

        assert simple_threshold == ESCALATION_THRESHOLD, "simple should not be tightened"
        assert medium_threshold > ESCALATION_THRESHOLD, "medium should be tightened"
        # And the tightening is proportional to the excess error rate
        assert medium_threshold <= 0.85, "threshold capped at 0.85"

    def test_falls_back_to_aggregate_when_bucket_thin(self, in_memory_db, settings):
        """When a complexity bucket has < 10 rows, fall back to aggregate."""
        router = self._make_router(settings)
        # Only 5 rows for "complex" — below the per-bucket floor
        self._seed_router_log(in_memory_db, complexity="complex", total=5, errors=4)
        # But 30 rows aggregate (mostly clean) so the fall-back rate is low
        self._seed_router_log(in_memory_db, complexity="medium", total=30, errors=0)

        from services.router import ESCALATION_THRESHOLD
        threshold = router._adaptive_threshold(complexity="complex")
        # Aggregate error rate is 4/35 ≈ 11.4%, below the 15% floor →
        # threshold stays at the default rather than tightening based on the
        # noisy 5-row bucket.
        assert threshold == ESCALATION_THRESHOLD

    def test_old_rows_outside_window_ignored(self, in_memory_db, settings):
        """Rows older than 24h should not influence the threshold."""
        import uuid
        from datetime import datetime, timezone, timedelta
        router = self._make_router(settings)

        long_ago = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        for i in range(30):
            in_memory_db.execute(
                "INSERT INTO router_log (id, conversation_id, message_preview, "
                "route_taken, complexity, reasoning, tokens_out, had_error, "
                "response_empty, model_used, created_at) "
                "VALUES (?, 'c', 'm', 'local', 'medium', 't', 0, 1, 0, 'l', ?)",
                (str(uuid.uuid4()), long_ago),
            )
        in_memory_db.commit()

        from services.router import ESCALATION_THRESHOLD
        # All 30 rows are errors but >24h old → should be ignored entirely
        assert router._adaptive_threshold(complexity="medium") == ESCALATION_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Proof-of-work handoff validation (models.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestProofOfWorkHandoff:
    def _make_packet(self, **overrides):
        from models import HandoffPacket
        defaults = dict(
            agent_id="agent-1",
            agent_name="Coder",
            subtask_completed="Wrote a function that adds two numbers",
            artifact="def add(a, b): return a + b",
            assumptions=[],
            uncertainties=["return type for non-numeric inputs"],
            confidence=0.8,
        )
        defaults.update(overrides)
        return HandoffPacket(**defaults)

    def test_no_op_when_local_unavailable(self):
        from models import proof_of_work_validate_handoff
        packet = self._make_packet()
        local = MagicMock()
        local.is_available.return_value = False
        before_notes = list(packet.validation_notes)

        result = proof_of_work_validate_handoff(packet, "Add two numbers", local)
        assert result is packet
        assert packet.validation_notes == before_notes
        local.chat.assert_not_called()

    def test_no_op_when_local_client_none(self):
        from models import proof_of_work_validate_handoff
        packet = self._make_packet()
        before_notes = list(packet.validation_notes)
        result = proof_of_work_validate_handoff(packet, "Task", None)
        assert result is packet
        assert packet.validation_notes == before_notes

    def test_low_score_flags_packet(self):
        from models import proof_of_work_validate_handoff
        packet = self._make_packet(validation_passed=True)
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = (
            '{"score": 2, "reason": "deliverable does not match the request"}'
        )

        result = proof_of_work_validate_handoff(packet, "Translate to French", local)
        assert result is packet
        assert packet.validation_passed is False
        assert any("proof-of-work failed" in n for n in packet.validation_notes)

    def test_high_score_keeps_packet_passing(self):
        from models import proof_of_work_validate_handoff
        packet = self._make_packet(validation_passed=True)
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '{"score": 8, "reason": "deliverable looks correct"}'

        result = proof_of_work_validate_handoff(packet, "Add two numbers", local)
        assert result is packet
        assert packet.validation_passed is True
        assert any("proof-of-work passed" in n for n in packet.validation_notes)

    def test_malformed_json_is_safe_no_op(self):
        from models import proof_of_work_validate_handoff
        packet = self._make_packet(validation_passed=True)
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = "not even close to JSON {{{"

        result = proof_of_work_validate_handoff(packet, "Task", local)
        # Garbled response must not crash and must not flip validation
        assert result is packet
        assert packet.validation_passed is True

    def test_local_chat_exception_is_safe_no_op(self):
        from models import proof_of_work_validate_handoff
        packet = self._make_packet(validation_passed=True)
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.side_effect = RuntimeError("network down")

        result = proof_of_work_validate_handoff(packet, "Task", local)
        assert result is packet
        assert packet.validation_passed is True

    def test_non_numeric_score_is_safe_no_op(self):
        from models import proof_of_work_validate_handoff
        packet = self._make_packet(validation_passed=True)
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '{"score": "low", "reason": "unclear"}'

        result = proof_of_work_validate_handoff(packet, "Task", local)
        assert result is packet
        assert packet.validation_passed is True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sliding-window risk ledger (security_engine.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSlidingWindowRiskLedger:
    def test_default_constructor_preserves_legacy_behavior(self):
        """Without sliding window, entries persist forever (existing behavior)."""
        from services.security_engine import RiskLedger, RiskCategory
        ledger = RiskLedger()
        ledger.record(RiskCategory.DATA_READ, "read 1")
        ledger.record(RiskCategory.DATA_READ, "read 2")
        # Both entries counted
        assert len(ledger.assess().entries) == 2

    def test_sliding_window_prunes_old_entries(self):
        """Entries older than the window should be dropped on assess()."""
        from services.security_engine import RiskLedger, RiskCategory
        ledger = RiskLedger(sliding_window_seconds=1.0)
        ledger.record(RiskCategory.DATA_READ, "old read")
        # Force the entry timestamp to be older than the window
        ledger._entries[0].timestamp = time.time() - 10.0
        ledger.record(RiskCategory.DATA_READ, "fresh read")

        assessment = ledger.assess()
        # Old entry pruned; only "fresh read" remains
        assert len(assessment.entries) == 1
        assert assessment.entries[0].description == "fresh read"

    def test_sliding_window_keeps_recent_entries(self):
        """Entries inside the window should remain after assess()."""
        from services.security_engine import RiskLedger, RiskCategory
        ledger = RiskLedger(sliding_window_seconds=600.0)
        ledger.record(RiskCategory.DATA_READ, "a")
        ledger.record(RiskCategory.DATA_READ, "b")
        ledger.record(RiskCategory.DATA_READ, "c")

        assessment = ledger.assess()
        assert len(assessment.entries) == 3

    def test_negative_window_treated_as_disabled(self):
        """Negative window value should be coerced to 0 (disabled)."""
        from services.security_engine import RiskLedger
        ledger = RiskLedger(sliding_window_seconds=-5.0)
        assert ledger._sliding_window_seconds == 0.0

    def test_window_does_not_break_abort_threshold(self):
        """Even with a sliding window, accumulating fresh high-risk operations
        should still trip the abort threshold."""
        from services.security_engine import (
            RiskLedger, RiskCategory, RISK_ABORT_THRESHOLD,
        )
        ledger = RiskLedger(sliding_window_seconds=600.0)
        ledger.record(RiskCategory.COMMUNICATION, "a")
        ledger.record(RiskCategory.CODE_EXEC, "b")
        ledger.record(RiskCategory.DATA_WRITE, "c")
        ledger.record(RiskCategory.EXTERNAL_API, "d")
        ledger.record(RiskCategory.MULTI_AGENT, "e")

        assessment = ledger.assess()
        assert assessment.cumulative_score >= RISK_ABORT_THRESHOLD
        assert assessment.should_abort

    def test_score_property_does_not_prune(self):
        """The .score property is documented to not mutate the ledger."""
        from services.security_engine import RiskLedger, RiskCategory
        ledger = RiskLedger(sliding_window_seconds=1.0)
        ledger.record(RiskCategory.DATA_READ, "x")
        ledger._entries[0].timestamp = time.time() - 100.0
        # .score should not prune; still sees the old entry
        before = ledger.score
        # But assess() does prune
        ledger.assess()
        after = ledger.score
        assert before > 0
        assert after == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Post-assembly multi-agent alignment check (task_artifacts.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiAgentAlignmentCheck:
    def test_no_op_when_local_unavailable(self):
        from services.task_artifacts import check_multi_agent_alignment
        local = MagicMock()
        local.is_available.return_value = False
        result = check_multi_agent_alignment("request", "output", local)
        assert result["fired"] is False
        assert result["aligned"] is True  # safe default
        assert result["score"] == 1.0

    def test_no_op_when_local_none(self):
        from services.task_artifacts import check_multi_agent_alignment
        result = check_multi_agent_alignment("request", "output", None)
        assert result["fired"] is False
        assert result["aligned"] is True

    def test_no_op_when_inputs_empty(self):
        from services.task_artifacts import check_multi_agent_alignment
        local = MagicMock()
        local.is_available.return_value = True
        result = check_multi_agent_alignment("", "output", local)
        assert result["fired"] is False
        local.chat.assert_not_called()

    def test_high_score_returns_aligned(self):
        from services.task_artifacts import check_multi_agent_alignment
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '{"score": 0.9, "reason": "directly addresses request"}'

        result = check_multi_agent_alignment(
            "Translate this paragraph to French",
            "Voici la traduction française…",
            local,
        )
        assert result["fired"] is True
        assert result["aligned"] is True
        assert result["score"] == 0.9
        assert "directly addresses" in result["reason"]

    def test_low_score_returns_misaligned(self):
        from services.task_artifacts import check_multi_agent_alignment
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '{"score": 0.2, "reason": "drifted into unrelated topic"}'

        result = check_multi_agent_alignment(
            "Translate this paragraph to French",
            "Here is a poem about cats.",
            local,
        )
        assert result["fired"] is True
        assert result["aligned"] is False
        assert result["score"] == 0.2

    def test_score_clamped_to_unit_interval(self):
        from services.task_artifacts import check_multi_agent_alignment
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '{"score": 7.5, "reason": "out of range"}'

        result = check_multi_agent_alignment("req", "out", local)
        assert result["fired"] is True
        assert result["score"] == 1.0  # clamped from 7.5

    def test_malformed_json_safe_default(self):
        from services.task_artifacts import check_multi_agent_alignment
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = "not json at all"

        result = check_multi_agent_alignment("req", "out", local)
        assert result["fired"] is False
        assert result["aligned"] is True

    def test_chat_exception_safe_default(self):
        from services.task_artifacts import check_multi_agent_alignment
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.side_effect = RuntimeError("network gone")

        result = check_multi_agent_alignment("req", "out", local)
        assert result["fired"] is False

    def test_threshold_override(self):
        """Caller-supplied threshold should be respected."""
        from services.task_artifacts import check_multi_agent_alignment
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '{"score": 0.5, "reason": "borderline"}'

        # Default threshold (0.6) → aligned=False
        r1 = check_multi_agent_alignment("req", "out", local)
        # Lower threshold → aligned=True
        r2 = check_multi_agent_alignment("req", "out", local, threshold=0.4)
        assert r1["aligned"] is False
        assert r2["aligned"] is True
