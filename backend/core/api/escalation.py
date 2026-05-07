"""
core/api/escalation.py — Phase 5: JS-API surface for the Wiser-Human
escalation channel.

The frontend resolves a pending escalation via ``approve_escalation(id)``
or ``deny_escalation(id)``; ``list_pending_escalations()`` returns every
row whose ``decision`` is still "pending" so a fresh UI session can paint
the queue without waiting for an SSE event.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import db as _db

from ._base import BaseAPI

log = logging.getLogger("iMakeAiTeams.api.escalation")


class EscalationAPI(BaseAPI):

    def list_pending(self) -> list[dict]:
        rows = _db.fetchall(
            "SELECT id, conversation_id, triggered_at, trigger_type, "
            "trigger_detail, model_input, proposed_action, decision, decided_at "
            "FROM escalations WHERE decision = 'pending' "
            "ORDER BY triggered_at ASC"
        )
        return [dict(r) for r in rows]

    def approve(self, escalation_id: str) -> dict:
        return self._resolve(escalation_id, "approved")

    def deny(self, escalation_id: str) -> dict:
        return self._resolve(escalation_id, "denied")

    def _resolve(self, escalation_id: str, decision: str) -> dict:
        if not isinstance(escalation_id, str) or not escalation_id:
            return {"ok": False, "error": "escalation_id is required"}
        row = _db.fetchone(
            "SELECT id, conversation_id, decision FROM escalations WHERE id = ?",
            (escalation_id,),
        )
        if row is None:
            return {"ok": False, "error": "escalation not found"}
        if row["decision"] not in ("pending", None):
            return {
                "ok": False,
                "error": f"escalation already {row['decision']}",
                "decision": row["decision"],
            }
        decided_at = datetime.now(timezone.utc).isoformat()
        _db.execute(
            "UPDATE escalations SET decision = ?, decided_at = ? WHERE id = ?",
            (decision, decided_at, escalation_id),
        )
        _db.commit()
        if decision == "approved":
            self._emit("escalation_resolved", {
                "escalation_id": escalation_id,
                "conversation_id": row["conversation_id"],
                "decision": decision,
            })
        return {
            "ok": True,
            "escalation_id": escalation_id,
            "decision": decision,
            "decided_at": decided_at,
        }
