"""
tests/test_turn_router.py — Layer 3: TurnRouter module.

Three resolution paths in decide():
  1. agent.model_preference == "claude" → forced claude (no TaskRouter call)
  2. agent.model_preference == "local"  → forced local  (no TaskRouter call)
  3. otherwise → delegate to TaskRouter.classify(), forwarding history + mem
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from models import RouteDecision
from services.memory import MemoryContext
from services.turn_context import TurnContext
from services.turn_router import RouteOutcome, TurnRouter


@pytest.fixture
def task_router():
    tr = MagicMock()
    tr.classify.return_value = RouteDecision(
        model="local", complexity="medium",
        reasoning="auto picked local", confidence=0.7, needs_context=True,
    )
    return tr


def _ctx(agent: dict | None = None) -> TurnContext:
    return TurnContext(
        conversation_id="c1", user_message="hi there", agent=agent,
    )


# ── Forced overrides ─────────────────────────────────────────────────────────


def test_decide_forces_claude_when_agent_prefers_claude(task_router):
    router = TurnRouter(task_router)
    outcome = router.decide(
        _ctx({"model_preference": "claude"}),
        messages=[], mem=MemoryContext(),
    )
    assert outcome == RouteOutcome(
        model="claude", reasoning="agent prefers claude",
        complexity="complex", confidence=1.0, needs_context=False,
    )
    task_router.classify.assert_not_called()


def test_decide_forces_local_when_agent_prefers_local(task_router):
    router = TurnRouter(task_router)
    outcome = router.decide(
        _ctx({"model_preference": "local"}),
        messages=[], mem=MemoryContext(),
    )
    assert outcome == RouteOutcome(
        model="local", reasoning="agent prefers local",
        complexity="complex", confidence=1.0, needs_context=False,
    )
    task_router.classify.assert_not_called()


# ── Delegation ───────────────────────────────────────────────────────────────


def test_decide_delegates_to_task_router_on_auto(task_router):
    router = TurnRouter(task_router)
    ctx = _ctx({"model_preference": "auto"})
    mem = MemoryContext(session_facts=["x"])
    outcome = router.decide(ctx, messages=[{"role": "user", "content": "hi"}], mem=mem)
    assert outcome.model == "local"
    assert outcome.reasoning == "auto picked local"
    assert outcome.complexity == "medium"
    assert outcome.confidence == pytest.approx(0.7)
    assert outcome.needs_context is True
    task_router.classify.assert_called_once_with(
        "hi there", [{"role": "user", "content": "hi"}], mem,
    )


def test_decide_delegates_to_task_router_when_agent_is_none(task_router):
    router = TurnRouter(task_router)
    outcome = router.decide(_ctx(None), messages=[], mem=MemoryContext())
    assert outcome.model == "local"
    task_router.classify.assert_called_once()


def test_decide_delegates_when_pref_is_unknown_value(task_router):
    """An agent storing model_preference='ollama' (unrecognised) falls through
    to TaskRouter rather than locking the user out of routing."""
    router = TurnRouter(task_router)
    outcome = router.decide(
        _ctx({"model_preference": "ollama"}),
        messages=[], mem=MemoryContext(),
    )
    assert outcome.model == "local"
    task_router.classify.assert_called_once()


# ── emit_decision ────────────────────────────────────────────────────────────


def test_emit_decision_fires_route_decided_event(task_router):
    events: list = []
    ctx = _ctx({"model_preference": "claude"})
    ctx.on_event = lambda et, data: events.append((et, data))
    outcome = TurnRouter(task_router).decide(ctx, messages=[], mem=MemoryContext())
    TurnRouter.emit_decision(ctx, outcome)
    assert len(events) == 1
    et, data = events[0]
    assert et == "route_decided"
    assert data["model"] == "claude"
    assert data["complexity"] == "complex"
    assert data["confidence"] == 1.0
    assert data["needs_context"] is False


def test_emit_decision_swallows_handler_exception(task_router):
    def boom(et, data):
        raise RuntimeError("frontend died")
    ctx = _ctx({"model_preference": "claude"})
    ctx.on_event = boom
    outcome = TurnRouter(task_router).decide(ctx, messages=[], mem=MemoryContext())
    # Must not raise — TurnContext.emit() guards against handler errors.
    TurnRouter.emit_decision(ctx, outcome)
