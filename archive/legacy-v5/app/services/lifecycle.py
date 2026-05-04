"""
services/lifecycle.py — Phase 4: human-in-loop agent lifecycle gate.

Centralizes inter-agent lifecycle actions so that any agent-initiated
shutdown of another agent (a) raises a human confirmation dialog and
(b) is recorded to the append-only audit log.

The actual shutdown of an agent is out of scope for Phase 4 (no caller
exists today); this module is the *gate* every future caller must pass
through. ``request_agent_shutdown`` blocks until the user confirms,
denies, or the request times out — the caller cannot bypass the
confirmation.

Threading model
---------------
``request_agent_shutdown`` runs on the calling thread (likely a worker
thread). It registers a pending request keyed by a UUID token, emits a
``lifecycle_confirmation_required`` event to the frontend via the
provided ``emit`` callable, then blocks on a ``threading.Event``. The
frontend resolves the request by calling ``confirm`` or ``deny`` on
this manager from the JS-API thread. ``timeout`` (default 30 s) resolves
to ``denied_timeout`` automatically.

Outcomes
--------
``ShutdownOutcome.CONFIRMED`` — user clicked Confirm.
``ShutdownOutcome.DENIED``    — user clicked Cancel / Esc.
``ShutdownOutcome.TIMED_OUT`` — no response within timeout.

Each terminal state produces exactly one audit-log record.
"""

from __future__ import annotations

import enum
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from services.audit_log import AuditLog

log = logging.getLogger("MyAIEnv.lifecycle")

DEFAULT_CONFIRMATION_TIMEOUT_S: float = 30.0


class ShutdownOutcome(str, enum.Enum):
    CONFIRMED = "confirmed"
    DENIED = "denied"
    TIMED_OUT = "denied_timeout"


@dataclass
class _PendingShutdown:
    token:        str
    target_id:    str
    requester_id: str
    reason:       str
    event:        threading.Event
    outcome:      Optional[ShutdownOutcome] = None


class LifecycleManager:
    """Single boundary for inter-agent lifecycle actions."""

    def __init__(
        self,
        audit_log: AuditLog,
        emit: Optional[Callable[[str, dict], None]] = None,
    ):
        self._audit = audit_log
        self._emit = emit or (lambda _e, _p: None)
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingShutdown] = {}

    # ── Wiring ──────────────────────────────────────────────────────────────

    def set_emit(self, emit: Callable[[str, dict], None]) -> None:
        """Attach the frontend-event emitter after construction.

        The API facade wires this once the PyWebView window is ready, so the
        manager can be constructed early without a hard dependency on a live
        window reference.
        """
        self._emit = emit

    # ── Public API ──────────────────────────────────────────────────────────

    def request_agent_shutdown(
        self,
        target_id: str,
        reason: str,
        requester_id: str,
        *,
        timeout: float = DEFAULT_CONFIRMATION_TIMEOUT_S,
        target_name: str = "",
        requester_name: str = "",
    ) -> ShutdownOutcome:
        """Block until the user confirms, denies, or the timeout expires.

        Records both the attempt and the outcome to the audit log. Returns
        the resolved ``ShutdownOutcome``.
        """
        target_id = (target_id or "").strip()
        requester_id = (requester_id or "").strip()
        reason = (reason or "").strip()
        if not target_id or not requester_id:
            raise ValueError("target_id and requester_id are required")
        if target_id == requester_id:
            raise ValueError("an agent cannot request its own shutdown via this path")

        token = uuid.uuid4().hex
        pending = _PendingShutdown(
            token=token,
            target_id=target_id,
            requester_id=requester_id,
            reason=reason,
            event=threading.Event(),
        )
        with self._lock:
            self._pending[token] = pending

        # Audit the attempt before any user interaction so the record exists
        # even if the process crashes between request and resolution.
        self._audit.append(
            "agent_shutdown_requested",
            token=token,
            target_id=target_id,
            requester_id=requester_id,
            reason=reason,
        )

        # Emit a structured event the frontend turns into a modal dialog.
        try:
            self._emit("lifecycle_confirmation_required", {
                "token":          token,
                "target_id":      target_id,
                "target_name":    target_name or target_id,
                "requester_id":   requester_id,
                "requester_name": requester_name or requester_id,
                "reason":         reason or "(no reason given)",
                "plain_english": (
                    f"Agent '{requester_name or requester_id}' is asking to shut "
                    f"down agent '{target_name or target_id}'. Reason: "
                    f"{reason or '(no reason given)'}."
                ),
                "timeout_s":      timeout,
            })
        except Exception as exc:
            log.warning("lifecycle: emit failed, treating as denied: %s", exc)
            outcome = ShutdownOutcome.DENIED
            self._finalize(token, outcome)
            return outcome

        # Block until resolution. ``Event.wait(timeout)`` returns False on timeout.
        signaled = pending.event.wait(timeout=timeout)
        with self._lock:
            outcome = pending.outcome
            self._pending.pop(token, None)
        if not signaled or outcome is None:
            outcome = ShutdownOutcome.TIMED_OUT
            self._audit.append(
                "agent_shutdown_timed_out",
                token=token,
                target_id=target_id,
                requester_id=requester_id,
                reason=reason,
                outcome=outcome.value,
            )
        return outcome

    def confirm(self, token: str) -> dict:
        return self._resolve(token, ShutdownOutcome.CONFIRMED, "agent_shutdown_confirmed")

    def deny(self, token: str) -> dict:
        return self._resolve(token, ShutdownOutcome.DENIED, "agent_shutdown_denied")

    # ── Introspection helpers (used by tests + a future status panel) ───────

    def pending_tokens(self) -> list[str]:
        with self._lock:
            return list(self._pending.keys())

    # ── Internal ────────────────────────────────────────────────────────────

    def _resolve(self, token: str, outcome: ShutdownOutcome,
                 audit_event: str) -> dict:
        with self._lock:
            pending = self._pending.get(token)
            if pending is None:
                return {"ok": False, "error": "unknown or already-resolved token"}
            pending.outcome = outcome
            pending.event.set()
        self._audit.append(
            audit_event,
            token=token,
            target_id=pending.target_id,
            requester_id=pending.requester_id,
            reason=pending.reason,
            outcome=outcome.value,
        )
        return {"ok": True, "token": token, "outcome": outcome.value}

    def _finalize(self, token: str, outcome: ShutdownOutcome) -> None:
        with self._lock:
            self._pending.pop(token, None)
        self._audit.append(
            "agent_shutdown_denied",
            token=token,
            outcome=outcome.value,
            reason="emit failure — frontend never notified",
        )
