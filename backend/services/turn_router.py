"""
services/turn_router.py — Per-turn model-backend routing decision.

Third extraction in the Layer 3 decomposition. Wraps the existing
``TaskRouter.classify`` (which decides Claude vs local from message
content) with the agent-level ``model_preference`` override the
orchestrator used to inline in ~25 lines of branchy if/elif.

Naming note: the existing ``services/router.py`` already exposes
``TaskRouter``. To avoid a name collision and to reflect the broader
contract (it owns the *turn-level* routing decision, not just message
classification), this module's class is ``TurnRouter``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from services.turn_context import TurnContext

log = logging.getLogger("iMakeAiTeams.turn_router")


@dataclass(frozen=True)
class RouteOutcome:
    """The five fields the orchestrator extracts from a routing decision."""
    model:         str       # "claude" | "local"
    reasoning:     str       # human-readable selection reason
    complexity:    str       # "simple" | "medium" | "complex"
    confidence:    float     # 0.0–1.0
    needs_context: bool      # whether the router thinks RAG is required


class TurnRouter:
    """Owns the model-backend routing decision for one chat turn."""

    def __init__(self, task_router):
        # task_router is the existing services.router.TaskRouter; we
        # delegate to its classify() when no agent-level preference exists.
        self._task_router = task_router

    def decide(
        self,
        ctx: TurnContext,
        messages: list,
        mem,
    ) -> RouteOutcome:
        """Decide which model backend handles this turn.

        Resolution order:
          1. ``agent.model_preference == "claude"`` → forced claude
          2. ``agent.model_preference == "local"``  → forced local
          3. otherwise → delegate to TaskRouter.classify()

        ``messages`` is the trimmed history; ``mem`` is the recalled
        MemoryContext. Both are forwarded verbatim to the task router.
        """
        agent = ctx.agent
        model_pref = agent.get("model_preference", "auto") if agent else "auto"

        if model_pref == "claude":
            return RouteOutcome(
                model="claude",
                reasoning="agent prefers claude",
                complexity="complex",
                confidence=1.0,
                needs_context=False,
            )
        if model_pref == "local":
            return RouteOutcome(
                model="local",
                reasoning="agent prefers local",
                complexity="complex",
                confidence=1.0,
                needs_context=False,
            )

        route = self._task_router.classify(ctx.user_message, messages, mem)
        return RouteOutcome(
            model=route.model,
            reasoning=route.reasoning,
            complexity=route.complexity,
            confidence=route.confidence,
            needs_context=route.needs_context,
        )

    @staticmethod
    def emit_decision(ctx: TurnContext, outcome: RouteOutcome) -> None:
        """Forward a ``route_decided`` SSE so the frontend can show the
        chosen backend, complexity, and confidence as a turn badge.
        """
        ctx.emit("route_decided", {
            "model":         outcome.model,
            "complexity":    outcome.complexity,
            "reasoning":     outcome.reasoning,
            "confidence":    outcome.confidence,
            "needs_context": outcome.needs_context,
        })
