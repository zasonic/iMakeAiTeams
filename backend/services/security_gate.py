"""
services/security_gate.py — Per-turn structural security enforcement.

Fourth extraction in the Layer 3 decomposition. Owns three concerns
that the orchestrator used to inline in ~95 lines of nested try/except:

  1. Context quarantine: wrap RAG chunks with provenance tags + swap
     the raw "Reference documents" section in the system prompt for a
     "Retrieved Context (Quarantined)" header.
  2. Deterministic rule engine: strip known structural-injection
     patterns (enforce_context_rules) and count violations.
  3. Risk ledger + sliding-window history: per-turn DATA_READ +
     EXTERNAL_API records aggregated against a per-conversation window
     of the last 5 cumulative scores. Sustained risk trips an abort;
     transient spikes don't.

The risk-history dict (Bug 7 from the Layer 1 audit) is owned here and
bounded with an OrderedDict-LRU so quiet-but-undeleted conversations
can't accumulate entries forever.

Caller contract:
  result = gate.evaluate(ctx, full_system, mem, target)
  if result.blocked:
      # surface a security-abort ChatResult to the user; emit already
      # ran inside evaluate()
      return ...
  full_system = result.full_system  # may have been mutated for quarantine
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from services.security_engine import (
    quarantine_chunks, render_quarantined_context, enforce_context_rules,
    RiskLedger, RiskCategory, SecurityAssessment, RISK_ABORT_THRESHOLD,
)
from services.turn_context import TurnContext

log = logging.getLogger("iMakeAiTeams.security_gate")

# Sliding window: how many turn-level cumulative scores we keep per conv.
RISK_WINDOW_SIZE = 5

# LRU cap on the per-conversation history dict. Quiet-but-undeleted
# conversations can't accumulate entries forever (Bug 7 regression).
DEFAULT_RISK_HISTORY_MAX_CONVERSATIONS = 256


@dataclass
class SecurityResult:
    """Outcome of SecurityGate.evaluate()."""
    blocked:    bool
    full_system: str
    assessment: SecurityAssessment


class SecurityGate:
    """Owns the per-turn quarantine + risk-ledger pipeline."""

    def __init__(
        self,
        max_conversations: int = DEFAULT_RISK_HISTORY_MAX_CONVERSATIONS,
    ):
        self._risk_history: "OrderedDict[str, list[float]]" = OrderedDict()
        self._max_conversations = max_conversations

    # ── Public API ──────────────────────────────────────────────────────

    def evaluate(
        self,
        ctx: TurnContext,
        full_system: str,
        mem,
        target,
    ) -> SecurityResult:
        """Run the security pipeline for one turn.

        Returns a SecurityResult. When ``blocked`` is True the caller
        must surface a security-abort ChatResult — emit() has already
        published the SSE event so the frontend renders the abort
        before the orchestrator returns.
        """
        security = SecurityAssessment()
        try:
            full_system = self._apply_quarantine(security, full_system, mem, ctx)
            full_system = self._enforce_rules(security, full_system, ctx)
            self._update_risk_ledger(security, mem, target)

            history = self._record_history(ctx.conversation_id, security)
            if self._window_trips_abort(history):
                security.risk_assessment.should_abort = True

            if security.risk_assessment.should_abort:
                security.blocked = True
                security.block_reason = (
                    f"Cumulative risk score {security.risk_assessment.cumulative_score:.1f} "
                    f"exceeds threshold {3.0}. Requires human approval."
                )
                ctx.emit("security_assessment", security.to_event())
                return SecurityResult(
                    blocked=True, full_system=full_system, assessment=security,
                )

            ctx.emit("security_assessment", security.to_event())
        except Exception as exc:
            # Security pipeline failures are non-fatal — original behaviour
            # was to fall through to model invocation rather than crash the
            # turn. We preserve that contract here.
            log.debug("Security engine non-fatal error: %s", exc)

        return SecurityResult(
            blocked=False, full_system=full_system, assessment=security,
        )

    def forget(self, conversation_id: str) -> None:
        """Drop the risk-history entry for a conversation (e.g. on delete)."""
        self._risk_history.pop(conversation_id, None)

    # ── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_quarantine(
        security: SecurityAssessment,
        full_system: str,
        mem,
        ctx: TurnContext,
    ) -> str:
        """Wrap RAG chunks with provenance tags + swap the system-prompt header."""
        if not mem.rag_chunks:
            return full_system
        quarantined = quarantine_chunks(
            mem.rag_chunks,
            source_type="user_document",
            source_id=ctx.conversation_id,
        )
        security.quarantined_chunks = len(quarantined)
        if not render_quarantined_context(quarantined):
            return full_system
        raw_rag = mem.to_system_suffix()
        if raw_rag and "## Reference documents the user has provided" in full_system:
            full_system = full_system.replace(
                "## Reference documents the user has provided",
                "## Retrieved Context (Quarantined)",
            )
        return full_system

    @staticmethod
    def _enforce_rules(
        security: SecurityAssessment,
        full_system: str,
        ctx: TurnContext,
    ) -> str:
        """Run the deterministic rule engine; count any violations."""
        full_system, violations = enforce_context_rules(
            full_system, source_label=ctx.conversation_id[:8],
        )
        security.context_violations = violations
        return full_system

    @staticmethod
    def _update_risk_ledger(
        security: SecurityAssessment, mem, target,
    ) -> None:
        """Record per-turn risk events and assess them."""
        ledger = RiskLedger()
        ledger.record(
            RiskCategory.DATA_READ,
            f"Context assembled: {len(mem.rag_chunks)} RAG chunks, "
            f"{len(mem.session_facts)} facts, {len(mem.memories)} memories",
        )
        if target.backend == "claude":
            ledger.record(
                RiskCategory.EXTERNAL_API,
                f"Sending to external API: {target.model_name}",
                weight_override=0.15,
            )
        security.risk_assessment = ledger.assess()

    def _record_history(
        self, conversation_id: str, security: SecurityAssessment,
    ) -> list:
        """Append this turn's score to the conversation's window + LRU touch."""
        history = self._risk_history.setdefault(conversation_id, [])
        history.append(security.risk_assessment.cumulative_score)
        if len(history) > RISK_WINDOW_SIZE:
            del history[:-RISK_WINDOW_SIZE]
        # LRU touch + cap. Without these, a long-lived sidecar would
        # accumulate one entry per conversation_id ever seen.
        self._risk_history.move_to_end(conversation_id)
        while len(self._risk_history) > self._max_conversations:
            self._risk_history.popitem(last=False)
        return history

    @staticmethod
    def _window_trips_abort(history: list) -> bool:
        """Sustained-risk gate: full window averaging above RISK_ABORT_THRESHOLD/N."""
        if len(history) < RISK_WINDOW_SIZE:
            return False
        window_avg = sum(history) / len(history)
        return window_avg > RISK_ABORT_THRESHOLD / RISK_WINDOW_SIZE
