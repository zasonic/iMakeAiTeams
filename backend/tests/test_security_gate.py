"""
tests/test_security_gate.py — Layer 3: SecurityGate module.

Pins the contract Layer 1 hardened (Bug 7 LRU on the per-conversation risk
history dict) plus the structural pieces SecurityGate inherited from the
orchestrator's inline block:

  - quarantine swap of the "Reference documents" header on rag presence
  - context-rule violations counted on assessment
  - risk-ledger sliding window: 5 turns of cumulative scores tripping abort
  - LRU eviction of quiet-but-undeleted conversations
  - forget() removes a single conversation's history immediately
  - abort path emits security_assessment SSE before returning blocked
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from models import ExecutionTarget
from services.memory import MemoryContext
from services.security_gate import (
    DEFAULT_RISK_HISTORY_MAX_CONVERSATIONS,
    RISK_WINDOW_SIZE,
    SecurityGate,
    SecurityResult,
)
from services.turn_context import TurnContext


@dataclass(frozen=True)
class _Target:
    backend: str
    model_name: str = "test-model"


def _ctx(conv_id: str = "conv1") -> TurnContext:
    return TurnContext(conversation_id=conv_id, user_message="hi")


def _mem(rag=None) -> MemoryContext:
    return MemoryContext(
        recent_messages=[], session_facts=[], rag_chunks=rag or [], memories=[],
    )


# ── Happy path ───────────────────────────────────────────────────────────────


def test_evaluate_returns_unblocked_for_safe_turn():
    gate = SecurityGate()
    result = gate.evaluate(
        _ctx(), full_system="BASE", mem=_mem(), target=_Target("local"),
    )
    assert isinstance(result, SecurityResult)
    assert result.blocked is False
    assert result.full_system == "BASE"


def test_evaluate_rewrites_rag_section_for_quarantine():
    gate = SecurityGate()
    sys_prompt = (
        "BASE\n\n## Reference documents the user has provided\nstuff"
    )
    result = gate.evaluate(
        _ctx(), full_system=sys_prompt, mem=_mem(rag=["doc-a", "doc-b"]),
        target=_Target("local"),
    )
    assert "## Retrieved Context (Quarantined)" in result.full_system
    assert "## Reference documents the user has provided" not in result.full_system
    assert result.assessment.quarantined_chunks == 2


def test_evaluate_emits_security_assessment_event():
    events: list = []
    ctx = _ctx()
    ctx.on_event = lambda et, data: events.append((et, data))
    SecurityGate().evaluate(
        ctx, full_system="BASE", mem=_mem(), target=_Target("claude"),
    )
    assert any(et == "security_assessment" for et, _ in events)


# ── Sliding-window abort ─────────────────────────────────────────────────────


def test_window_does_not_trip_below_size():
    gate = SecurityGate()
    # Four high-risk turns aren't enough; window needs 5 entries.
    for _ in range(4):
        result = gate.evaluate(
            _ctx(), full_system="BASE", mem=_mem(rag=["d"]),
            target=_Target("claude"),
        )
        assert result.blocked is False


def test_window_trip_helper_pure_logic():
    """Window logic is exposed as a static helper so we can test the
    arithmetic without driving evaluate() with high-weight events.
    Threshold is RISK_ABORT_THRESHOLD / RISK_WINDOW_SIZE = 0.6 average.
    """
    # Below threshold (0.5 average) → no abort
    assert SecurityGate._window_trips_abort([0.5] * RISK_WINDOW_SIZE) is False
    # Above threshold (0.7 average) → abort
    assert SecurityGate._window_trips_abort([0.7] * RISK_WINDOW_SIZE) is True
    # Window not full → no abort even if individual entries are high
    assert SecurityGate._window_trips_abort([0.9] * (RISK_WINDOW_SIZE - 1)) is False


def test_evaluate_aborts_when_history_already_high():
    """Pre-seed the per-conv history so the next evaluate() trips abort,
    without needing to record high-weight risk categories the chat path
    doesn't normally produce."""
    gate = SecurityGate()
    # Pre-seed RISK_WINDOW_SIZE-1 high scores so this turn fills the
    # window. This turn's own DATA_READ+EXTERNAL_API contribute too.
    gate._risk_history["conv1"] = [0.9] * (RISK_WINDOW_SIZE - 1)
    events: list = []
    ctx = _ctx("conv1")
    ctx.on_event = lambda et, data: events.append((et, data))
    result = gate.evaluate(
        ctx, full_system="BASE", mem=_mem(rag=["d"]),
        target=_Target("claude"),
    )
    assert result.blocked is True
    assert result.assessment.blocked is True
    assert "Cumulative risk score" in result.assessment.block_reason
    assert any(et == "security_assessment" for et, _ in events)


# ── LRU regression (Bug 7) ───────────────────────────────────────────────────


def test_lru_caps_per_conversation_dict():
    """Bug 7: the risk-history dict must not grow without bound."""
    gate = SecurityGate(max_conversations=3)
    for i in range(10):
        gate.evaluate(
            _ctx(f"conv-{i}"), full_system="BASE", mem=_mem(),
            target=_Target("local"),
        )
    # Internal dict capped — only the three most-recently-used survive.
    assert len(gate._risk_history) == 3
    assert "conv-9" in gate._risk_history
    assert "conv-8" in gate._risk_history
    assert "conv-7" in gate._risk_history
    assert "conv-0" not in gate._risk_history


def test_lru_touch_promotes_recent_use():
    gate = SecurityGate(max_conversations=2)
    gate.evaluate(_ctx("a"), full_system="B", mem=_mem(), target=_Target("local"))
    gate.evaluate(_ctx("b"), full_system="B", mem=_mem(), target=_Target("local"))
    # Touch a — should now be most-recently-used.
    gate.evaluate(_ctx("a"), full_system="B", mem=_mem(), target=_Target("local"))
    # Insert c — b is the LRU now and should evict.
    gate.evaluate(_ctx("c"), full_system="B", mem=_mem(), target=_Target("local"))
    assert "a" in gate._risk_history
    assert "c" in gate._risk_history
    assert "b" not in gate._risk_history


def test_default_max_is_256():
    gate = SecurityGate()
    assert gate._max_conversations == DEFAULT_RISK_HISTORY_MAX_CONVERSATIONS == 256


# ── forget() ─────────────────────────────────────────────────────────────────


def test_forget_drops_history_for_a_conversation():
    gate = SecurityGate()
    gate.evaluate(_ctx("x"), full_system="B", mem=_mem(), target=_Target("local"))
    assert "x" in gate._risk_history
    gate.forget("x")
    assert "x" not in gate._risk_history


def test_forget_unknown_conversation_is_a_noop():
    gate = SecurityGate()
    # Must not raise even when the id was never seen.
    gate.forget("never-seen")
