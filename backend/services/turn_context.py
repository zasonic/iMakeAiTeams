"""
services/turn_context.py — Per-turn shared state for ChatOrchestrator.

Layer 3 of the architectural plan introduces a ``TurnContext`` dataclass
that flows through the six extracted modules (TurnLifecycle, MemoryRecall,
Router, SecurityGate, WorkerDispatch, EscalationLadder), so each can read
and annotate the turn without ChatOrchestrator.send() needing to plumb 30+
parameters by hand.

This file is intentionally small. Fields are added only when a second
consumer demands them — adding speculative fields is the same anti-pattern
that produced the 2351-line orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


@dataclass
class TurnContext:
    """Shared per-turn state.

    Constructed once at the top of ChatOrchestrator.send() and passed by
    reference into the extracted modules. Mutability is intentional: each
    module annotates fields that downstream modules need to read (e.g. the
    Router writes ``route_model``, WorkerDispatch reads it).
    """

    conversation_id: str
    user_message:    str

    # Optional inputs ────────────────────────────────────────────────────
    agent_id:        Optional[str] = None
    agent:           Optional[dict] = None
    on_event:        Optional[Callable[[str, dict], None]] = None
    on_token:        Optional[Callable[[str], None]] = None

    # Lifecycle bookkeeping ──────────────────────────────────────────────
    user_msg_id:     str = ""                                  # set by TurnLifecycle on user-msg INSERT
    # Layer C1: per-turn correlation id. Generated in TurnLifecycle.open()
    # and echoed onto every SSE payload (via .emit()) + audit_log row +
    # token_usage row so a single grep across logs reconstructs the full
    # turn timeline. Stays empty before open() runs.
    turn_id:         str = ""
    started_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Budget bookkeeping (set by TurnLifecycle.open) ──────────────────────
    budget:          float = 0.0      # max_conversation_budget_usd at turn start
    warn_pct:        float = 0.0      # budget_warning_threshold_pct at turn start
    spent:           float = 0.0      # cumulative spend BEFORE this turn (snapshot)
    budget_exceeded: bool  = False    # True iff open() saw spent >= budget

    def emit(self, event_type: str, data: dict) -> None:
        """Forward a structured event to ``on_event`` if one was provided.

        Mirrors the inline ``_emit_event`` closure that send() uses today —
        having it on TurnContext means the extracted modules don't each
        need to define their own copy.

        Layer C1: when ``self.turn_id`` is set, it's auto-stamped onto the
        outgoing payload so every SSE event the renderer receives carries
        the same correlation id without each call site having to remember
        to thread it through. Caller-supplied ``turn_id`` keys win — we
        only fill the field when it's absent.
        """
        if self.on_event is None:
            return
        if self.turn_id and isinstance(data, dict) and "turn_id" not in data:
            data = {**data, "turn_id": self.turn_id}
        try:
            self.on_event(event_type, data)
        except Exception:
            pass
