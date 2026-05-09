"""
tests/test_escalation_ladder.py — Layer 3: EscalationLadder module.

Pins the two-rung ladder behaviour and the Bug-4 invariant (response_empty
must reflect the post-escalation state, not the pre-escalation reading).

Eligibility gates each have a test:
  - had_error short-circuits
  - split_enabled short-circuits (Reader/Actor owns its own escalation)
  - target.backend != local short-circuits
  - local unavailable short-circuits
  - sub-MIN_WORDS messages short-circuit

Behavioural tests:
  - Empty response → escalate via empty rung
  - Quality score < threshold → escalate via quality rung
  - Quality score >= threshold → no escalation
  - Bug 4: post-escalation response_empty reflects rescued response
  - Escalation failure leaves the original response intact
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from models import RoutingDecision
from services.escalation_ladder import (
    EMPTY_RESPONSE_CHAR_LIMIT,
    EscalationLadder,
    EscalationOutcome,
    QUALITY_ESCALATION_THRESHOLD,
)
from services.turn_context import TurnContext


@dataclass(frozen=True)
class _Target:
    backend: str = "local"
    max_tokens: int = 1024


def _decision(agent_id: str = "a", backend: str = "local") -> RoutingDecision:
    return RoutingDecision(
        agent_id=agent_id, backend=backend, score=1.0,
        reasoning="initial", used_fallback=False, skill_matched="",
    )


def _ctx(user_message: str = "tell me about elephants and their habits"):
    return TurnContext(conversation_id="c1", user_message=user_message)


def _local_available():
    local = MagicMock()
    local.is_available.return_value = True
    return local


# ── Eligibility gates ────────────────────────────────────────────────────────


def test_had_error_skips_escalation():
    hub = MagicMock()
    ladder = EscalationLadder(hub, _local_available())
    out = ladder.maybe_escalate(
        ctx=_ctx(), decision=_decision(), target=_Target(),
        full_system="S", messages=[],
        response_text="", tokens_in=0, tokens_out=0,
        route_model="local", model_name="m", had_error=True, split_enabled=False,
    )
    assert out.escalated is False
    hub.invoke.assert_not_called()


def test_split_enabled_skips_escalation():
    hub = MagicMock()
    ladder = EscalationLadder(hub, _local_available())
    out = ladder.maybe_escalate(
        ctx=_ctx(), decision=_decision(), target=_Target(),
        full_system="S", messages=[],
        response_text="", tokens_in=0, tokens_out=0,
        route_model="local", model_name="m", had_error=False, split_enabled=True,
    )
    assert out.escalated is False
    hub.invoke.assert_not_called()


def test_claude_target_skips_escalation():
    hub = MagicMock()
    ladder = EscalationLadder(hub, _local_available())
    out = ladder.maybe_escalate(
        ctx=_ctx(), decision=_decision(), target=_Target(backend="claude"),
        full_system="S", messages=[],
        response_text="", tokens_in=0, tokens_out=0,
        route_model="claude", model_name="m",
        had_error=False, split_enabled=False,
    )
    assert out.escalated is False
    hub.invoke.assert_not_called()


def test_local_unavailable_skips_escalation():
    hub = MagicMock()
    local = MagicMock()
    local.is_available.return_value = False
    ladder = EscalationLadder(hub, local)
    out = ladder.maybe_escalate(
        ctx=_ctx(), decision=_decision(), target=_Target(),
        full_system="S", messages=[],
        response_text="", tokens_in=0, tokens_out=0,
        route_model="local", model_name="m",
        had_error=False, split_enabled=False,
    )
    assert out.escalated is False
    hub.invoke.assert_not_called()


def test_no_local_client_skips_escalation():
    hub = MagicMock()
    ladder = EscalationLadder(hub, None)
    out = ladder.maybe_escalate(
        ctx=_ctx(), decision=_decision(), target=_Target(),
        full_system="S", messages=[],
        response_text="", tokens_in=0, tokens_out=0,
        route_model="local", model_name="m",
        had_error=False, split_enabled=False,
    )
    assert out.escalated is False


def test_short_message_skips_escalation():
    hub = MagicMock()
    ladder = EscalationLadder(hub, _local_available())
    out = ladder.maybe_escalate(
        ctx=_ctx(user_message="hi"),
        decision=_decision(), target=_Target(),
        full_system="S", messages=[],
        response_text="", tokens_in=0, tokens_out=0,
        route_model="local", model_name="m",
        had_error=False, split_enabled=False,
    )
    assert out.escalated is False
    hub.invoke.assert_not_called()


# ── Empty rung ───────────────────────────────────────────────────────────────


def test_empty_response_escalates_to_claude():
    hub = MagicMock()
    hub.invoke.return_value = MagicMock(
        text="full claude rescue answer", input_tokens=33, output_tokens=44,
        model_name="claude-sonnet-4",
    )
    ladder = EscalationLadder(hub, _local_available())
    out = ladder.maybe_escalate(
        ctx=_ctx(), decision=_decision(), target=_Target(),
        full_system="S", messages=[],
        response_text="", tokens_in=1, tokens_out=2,
        route_model="local", model_name="local-m",
        had_error=False, split_enabled=False,
    )
    assert out.escalated is True
    assert out.response_text == "full claude rescue answer"
    assert out.tokens_in == 33
    assert out.tokens_out == 44
    assert out.route_model == "claude"
    assert out.model_name == "claude-sonnet-4"
    assert out.escalation_reason == "local response empty; escalated"
    # Bug 4 regression: response_empty reflects the POST-escalation text.
    assert out.response_empty is False


def test_below_char_limit_treated_as_empty():
    """A response shorter than EMPTY_RESPONSE_CHAR_LIMIT triggers the empty
    rung, not the quality scorer."""
    hub = MagicMock()
    hub.invoke.return_value = MagicMock(
        text="x" * 200, input_tokens=1, output_tokens=1, model_name="claude",
    )
    ladder = EscalationLadder(hub, _local_available())
    short = "x" * (EMPTY_RESPONSE_CHAR_LIMIT - 1)
    out = ladder.maybe_escalate(
        ctx=_ctx(), decision=_decision(), target=_Target(),
        full_system="S", messages=[],
        response_text=short, tokens_in=0, tokens_out=0,
        route_model="local", model_name="m",
        had_error=False, split_enabled=False,
    )
    assert out.escalated is True


# ── Quality rung ─────────────────────────────────────────────────────────────


def test_low_quality_score_escalates():
    hub = MagicMock()
    hub.invoke.return_value = MagicMock(
        text="claude rescue", input_tokens=10, output_tokens=20, model_name="claude",
    )
    ladder = EscalationLadder(hub, _local_available())
    with patch("services.task_artifacts.local_first_call",
               return_value='{"score": 2, "reason": "off-topic"}'):
        out = ladder.maybe_escalate(
            ctx=_ctx(), decision=_decision(), target=_Target(),
            full_system="S", messages=[],
            response_text="local answer that is plausibly long enough",
            tokens_in=0, tokens_out=0, route_model="local", model_name="m",
            had_error=False, split_enabled=False,
        )
    assert out.escalated is True
    assert out.escalation_reason == "local response failed quality gate; escalated"


def test_high_quality_score_does_not_escalate():
    hub = MagicMock()
    ladder = EscalationLadder(hub, _local_available())
    with patch(
        "services.task_artifacts.local_first_call",
        return_value='{"score": 8, "reason": "good"}',
    ):
        out = ladder.maybe_escalate(
            ctx=_ctx(), decision=_decision(), target=_Target(),
            full_system="S", messages=[],
            response_text="solid local answer that meets the bar",
            tokens_in=0, tokens_out=0, route_model="local", model_name="m",
            had_error=False, split_enabled=False,
        )
    assert out.escalated is False
    hub.invoke.assert_not_called()


def test_unparseable_quality_score_does_not_escalate():
    hub = MagicMock()
    ladder = EscalationLadder(hub, _local_available())
    with patch("services.task_artifacts.local_first_call",
               return_value="not json"):
        out = ladder.maybe_escalate(
            ctx=_ctx(), decision=_decision(), target=_Target(),
            full_system="S", messages=[],
            response_text="solid local answer that meets the bar",
            tokens_in=0, tokens_out=0, route_model="local", model_name="m",
            had_error=False, split_enabled=False,
        )
    assert out.escalated is False


def test_quality_threshold_constant_is_4():
    """Pin the threshold publicly so changes can't slip silently."""
    assert QUALITY_ESCALATION_THRESHOLD == 4.0


# ── Failure handling ─────────────────────────────────────────────────────────


def test_escalation_invoke_failure_keeps_local_response():
    hub = MagicMock()
    hub.invoke.side_effect = RuntimeError("claude offline")
    ladder = EscalationLadder(hub, _local_available())
    out = ladder.maybe_escalate(
        ctx=_ctx(), decision=_decision(), target=_Target(),
        full_system="S", messages=[],
        response_text="", tokens_in=1, tokens_out=2,
        route_model="local", model_name="local-m",
        had_error=False, split_enabled=False,
    )
    # Failed escalation must not regress to crash. The original local
    # response stays — even though it's empty.
    assert out.escalated is False
    assert out.response_text == ""
    assert out.tokens_in == 1
    assert out.tokens_out == 2
    assert out.route_model == "local"
    assert out.model_name == "local-m"
    # response_empty is recomputed from the (still empty) text.
    assert out.response_empty is True
