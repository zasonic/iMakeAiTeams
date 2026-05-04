"""
core/api/lifecycle.py — Phase 4: JS-API surface for agent lifecycle gate.

The frontend resolves a pending shutdown via ``confirm_shutdown(token)``
or ``deny_shutdown(token)``. The actual shutdown logic does not exist yet
(no caller initiates it today); this module is the safety gate that any
future caller must pass through.

``request_agent_shutdown_demo`` is a small JS-API method intended to let
manual a11y / UX testers raise the dialog without writing Python — useful
for the manual a11y audit step in Phase 4 verification.
"""

from __future__ import annotations

import logging
import threading

from ._base import BaseAPI

log = logging.getLogger("MyAIEnv.api.lifecycle")


class LifecycleAPI(BaseAPI):

    # ── Confirmation resolution ──────────────────────────────────────────────

    def confirm_shutdown(self, token: str) -> dict:
        if not isinstance(token, str) or not token:
            return {"ok": False, "error": "token is required"}
        return self._lifecycle.confirm(token)

    def deny_shutdown(self, token: str) -> dict:
        if not isinstance(token, str) or not token:
            return {"ok": False, "error": "token is required"}
        return self._lifecycle.deny(token)

    # ── Audit log (read-only surfaces for the UI) ────────────────────────────

    def list_lifecycle_audit(self, limit: int = 100) -> dict:
        try:
            limit = max(0, min(int(limit), 1000))
        except (TypeError, ValueError):
            limit = 100
        return {
            "events": self._audit_log.tail(limit),
            "path":   str(self._audit_log.path),
        }

    # ── Manual a11y testing helper ───────────────────────────────────────────

    def request_agent_shutdown_demo(self, target_id: str = "agent-b",
                                     requester_id: str = "agent-a",
                                     reason: str = "demo") -> dict:
        """Spawn a non-blocking shutdown request so the UI can render.

        Returns immediately with the issued token; the dialog lifecycle plays
        out in a background thread. Intended for QA — never hooks into a real
        shutdown path.
        """

        def _run():
            try:
                self._lifecycle.request_agent_shutdown(
                    target_id=target_id,
                    reason=reason,
                    requester_id=requester_id,
                    target_name=target_id,
                    requester_name=requester_id,
                )
            except Exception as exc:
                log.warning("lifecycle demo request failed: %s", exc)

        threading.Thread(target=_run, daemon=True, name="lifecycle-demo").start()
        return {"ok": True}
