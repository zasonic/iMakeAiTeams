"""
tests/test_worker_dispatch.py — Layer 3.6: WorkerDispatch module.

Three things WorkerDispatch owns:

  build_turn_decision()  — main per-turn RoutingDecision; routes through
                           ``hub_router.route_for_agent`` when ``agent_id``
                           is set (AuthorizationError must propagate) and
                           synthesizes a hub-direct decision driven by
                           the TurnRouter's RouteOutcome otherwise.
  build_phase_decision() — Reader/Actor phase RoutingDecision; tries
                           ``route_for_agent`` and silently falls back to
                           a hub-direct claude decision on any failure.
  dispatch()             — thin pass-through to ``hub_router.invoke``;
                           must forward all six keyword arguments and
                           return the WorkerResult unmodified.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from models import RoutingDecision, TaskDescriptor, WorkerResult
from services.hub_router import AuthorizationError
from services.turn_router import RouteOutcome
from services.worker_dispatch import WorkerDispatch


# ── Helpers ──────────────────────────────────────────────────────────────────


def _outcome(model: str = "claude", reasoning: str = "auto picked claude") -> RouteOutcome:
    return RouteOutcome(
        model=model, reasoning=reasoning,
        complexity="medium", confidence=0.7, needs_context=False,
    )


def _task(text: str = "hello world") -> TaskDescriptor:
    return TaskDescriptor(text=text, preferred_agent_id=None, backend_hint="claude")


def _fake_agent_decision() -> RoutingDecision:
    return RoutingDecision(
        agent_id="agent-1", backend="claude", score=0.9,
        reasoning="caller-selected agent agent-1",
        used_fallback=False, skill_matched="writing",
    )


# ── build_turn_decision ──────────────────────────────────────────────────────


def test_build_turn_decision_routes_through_hub_when_agent_id_set():
    hub = MagicMock()
    hub.route_for_agent.return_value = _fake_agent_decision()

    dispatch = WorkerDispatch(hub)
    task = _task()
    decision = dispatch.build_turn_decision("agent-1", task, _outcome())

    assert decision.agent_id == "agent-1"
    assert decision.backend == "claude"
    assert decision.skill_matched == "writing"
    hub.route_for_agent.assert_called_once_with("agent-1", task)


def test_build_turn_decision_propagates_authorization_error():
    """AuthorizationError must propagate so a misconfigured agent surfaces a
    clear error rather than silently downgrading to the hub-direct path."""
    hub = MagicMock()
    hub.route_for_agent.side_effect = AuthorizationError("nope")

    dispatch = WorkerDispatch(hub)
    with pytest.raises(AuthorizationError):
        dispatch.build_turn_decision("agent-1", _task(), _outcome())


def test_build_turn_decision_synthesizes_hub_direct_when_no_agent():
    hub = MagicMock()
    dispatch = WorkerDispatch(hub)
    decision = dispatch.build_turn_decision(
        None, _task(), _outcome(model="local", reasoning="auto picked local"),
    )

    assert decision.agent_id == ""
    assert decision.backend == "local"
    assert decision.reasoning == "auto picked local"
    assert decision.score == 1.0
    assert decision.used_fallback is False
    assert decision.skill_matched == ""
    hub.route_for_agent.assert_not_called()


def test_build_turn_decision_uses_route_outcome_backend():
    """The hub-direct backend comes from RouteOutcome.model, not from the
    task's backend_hint or any default."""
    hub = MagicMock()
    dispatch = WorkerDispatch(hub)
    decision = dispatch.build_turn_decision(
        None, _task(), _outcome(model="local", reasoning="local pref"),
    )
    assert decision.backend == "local"


# ── build_phase_decision ─────────────────────────────────────────────────────


def test_build_phase_decision_routes_through_hub_when_agent_id_set():
    hub = MagicMock()
    hub.route_for_agent.return_value = _fake_agent_decision()

    dispatch = WorkerDispatch(hub)
    decision = dispatch.build_phase_decision("agent-1", "do the thing")

    assert decision.agent_id == "agent-1"
    hub.route_for_agent.assert_called_once()
    # Verify the synthesized TaskDescriptor: text + preferred_agent_id
    call_args = hub.route_for_agent.call_args
    assert call_args[0][0] == "agent-1"
    task = call_args[0][1]
    assert task.text == "do the thing"
    assert task.preferred_agent_id == "agent-1"


def test_build_phase_decision_swallows_route_for_agent_failure():
    """Reader/Actor phases must always produce a working decision — any
    route_for_agent failure falls back to a hub-direct claude decision."""
    hub = MagicMock()
    hub.route_for_agent.side_effect = AuthorizationError("nope")

    dispatch = WorkerDispatch(hub)
    decision = dispatch.build_phase_decision("agent-1", "do the thing")

    assert decision.agent_id == "agent-1"
    assert decision.backend == "claude"
    assert decision.reasoning == "reader_actor phase"


def test_build_phase_decision_synthesizes_hub_direct_when_no_agent():
    hub = MagicMock()
    dispatch = WorkerDispatch(hub)
    decision = dispatch.build_phase_decision(None, "do the thing")

    assert decision.agent_id == ""
    assert decision.backend == "claude"
    assert decision.reasoning == "reader_actor phase"
    hub.route_for_agent.assert_not_called()


def test_build_phase_decision_swallows_generic_exception():
    """Any exception type from route_for_agent — not just
    AuthorizationError — must fall back, since legacy code paths might
    raise RuntimeError or KeyError on misconfigured agents."""
    hub = MagicMock()
    hub.route_for_agent.side_effect = RuntimeError("db gone")

    dispatch = WorkerDispatch(hub)
    decision = dispatch.build_phase_decision("agent-1", "x")
    assert decision.backend == "claude"


# ── dispatch ─────────────────────────────────────────────────────────────────


def test_dispatch_forwards_all_args_to_hub_invoke():
    hub = MagicMock()
    expected = WorkerResult(
        text="hi", backend="claude", model_name="claude-sonnet",
        input_tokens=10, output_tokens=20,
    )
    hub.invoke.return_value = expected

    dispatch = WorkerDispatch(hub)
    decision = _fake_agent_decision()
    on_token = lambda chunk: None
    result = dispatch.dispatch(
        decision, "system prompt",
        [{"role": "user", "content": "hi"}],
        max_tokens=2048, on_token=on_token, agent_role="actor",
    )

    assert result is expected
    hub.invoke.assert_called_once_with(
        decision, "system prompt",
        [{"role": "user", "content": "hi"}],
        max_tokens=2048, on_token=on_token, agent_role="actor",
    )


def test_dispatch_uses_default_agent_role_monolithic():
    hub = MagicMock()
    hub.invoke.return_value = WorkerResult(
        text="x", backend="claude", model_name="m",
    )

    dispatch = WorkerDispatch(hub)
    dispatch.dispatch(_fake_agent_decision(), "S", [])

    kwargs = hub.invoke.call_args.kwargs
    assert kwargs["agent_role"] == "monolithic"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["on_token"] is None


def test_dispatch_returns_hub_invoke_result_unmodified():
    """No translation, no wrapping — dispatch is a 1:1 pass-through so
    callers can read .had_error / .input_tokens / .output_tokens directly."""
    hub = MagicMock()
    expected = WorkerResult(
        text="[Error: boom]", backend="claude", model_name="m",
        input_tokens=0, output_tokens=0, had_error=True,
    )
    hub.invoke.return_value = expected

    dispatch = WorkerDispatch(hub)
    result = dispatch.dispatch(_fake_agent_decision(), "S", [])
    assert result is expected
    assert result.had_error is True
