"""
services/worker_dispatch.py — Worker selection + invocation in one place.

Sixth and final extraction in the Layer 3 decomposition. Owns the two
pieces of the worker-dispatch shell that the orchestrator used to inline
at every call site:

  1. ``build_turn_decision(agent_id, task, route_outcome)`` — main per-turn
     decision: routes through ``hub_router.route_for_agent`` when the
     caller specified an agent (so AuthorizationError still propagates
     to the caller), or synthesizes a hub-direct decision driven by the
     TurnRouter's RouteOutcome when no agent is selected.
  2. ``build_phase_decision(agent_id, text)`` — Reader/Actor phase
     decision: tries route_for_agent and silently falls back to a
     hub-direct claude decision on any failure. Replaces the
     ``_build_decision_for_role`` helper that used to live inline on
     ``ChatOrchestrator``.
  3. ``dispatch(decision, full_system, messages, ...)`` — thin wrapper
     around ``hub_router.invoke`` that returns a WorkerResult.

The high-stakes-consensus path (3x parallel invokes with weighted-vote
math) stays in the orchestrator because the voting logic is
orchestrator-specific; it still calls ``hub_router.invoke`` directly via
this module's ``dispatch`` helper if desired, but parallelism is owned
by the consensus call site.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from models import RoutingDecision, TaskDescriptor, WorkerResult
from services.hub_router import HubRouter
from services.turn_router import RouteOutcome

log = logging.getLogger("iMakeAiTeams.worker_dispatch")


class WorkerDispatch:
    """Owns the per-turn worker selection + invocation shell."""

    def __init__(self, hub_router: HubRouter):
        self._hub = hub_router

    # ── Public API ──────────────────────────────────────────────────────

    def build_turn_decision(
        self,
        agent_id: str | None,
        task: TaskDescriptor,
        route_outcome: RouteOutcome,
    ) -> RoutingDecision:
        """Build the main per-turn RoutingDecision.

        When ``agent_id`` is set we go through ``hub_router.route_for_agent``;
        any AuthorizationError it raises propagates to the caller so a
        misconfigured agent surfaces a clear error rather than silently
        downgrading to the hub-direct path. When ``agent_id`` is None we
        synthesize a hub-direct decision whose ``backend`` and ``reasoning``
        come from the TurnRouter's RouteOutcome.
        """
        if agent_id:
            return self._hub.route_for_agent(agent_id, task)
        return RoutingDecision(
            agent_id="",
            backend=route_outcome.model,
            score=1.0,
            reasoning=route_outcome.reasoning,
            used_fallback=False,
            skill_matched="",
        )

    def build_phase_decision(
        self,
        agent_id: str | None,
        text: str,
    ) -> RoutingDecision:
        """Build a Reader/Actor phase RoutingDecision.

        Reader/Actor phases must always produce a working decision —
        falling back to the hub-direct claude path when ``route_for_agent``
        raises is the existing behaviour and is preserved here. The text
        is forwarded to ``TaskDescriptor.text`` so the hub can match
        skills if the agent declares any.
        """
        if agent_id:
            try:
                return self._hub.route_for_agent(
                    agent_id,
                    TaskDescriptor(text=text, preferred_agent_id=agent_id),
                )
            except Exception as exc:
                log.debug("route_for_agent failed in phase decision: %s", exc)
        return RoutingDecision(
            agent_id=agent_id or "",
            backend="claude",
            score=1.0,
            reasoning="reader_actor phase",
            used_fallback=False,
            skill_matched="",
        )

    def dispatch(
        self,
        decision: RoutingDecision,
        full_system: str,
        messages: list,
        *,
        max_tokens: int = 4096,
        on_token: Optional[Callable[[str], None]] = None,
        agent_role: str = "monolithic",
    ) -> WorkerResult:
        """Invoke the routed worker through hub_router.invoke."""
        return self._hub.invoke(
            decision, full_system, messages,
            max_tokens=max_tokens,
            on_token=on_token,
            agent_role=agent_role,
        )
