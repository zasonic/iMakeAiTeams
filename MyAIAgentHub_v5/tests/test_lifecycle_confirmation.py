"""
tests/test_lifecycle_confirmation.py — Phase 4: human-in-loop shutdown gate.

Covers the spec's "no agent can be stopped without a recorded confirmation"
property and the three terminal states (confirmed, denied, timed out).
"""

from __future__ import annotations

import threading
import time

import pytest

from services.audit_log import AuditLog
from services.lifecycle import LifecycleManager, ShutdownOutcome


@pytest.fixture
def manager(tmp_path):
    audit = AuditLog(tmp_path / "lifecycle_audit.jsonl")
    events: list[tuple[str, dict]] = []
    mgr = LifecycleManager(audit, emit=lambda e, p: events.append((e, p)))
    return mgr, audit, events


def _request_in_thread(mgr, **kwargs):
    """Run request_agent_shutdown on a worker thread; return the thread + outcome holder."""
    holder: dict = {}

    def _run():
        holder["outcome"] = mgr.request_agent_shutdown(**kwargs)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, holder


# ── Confirmation flow ──────────────────────────────────────────────────────


class TestConfirm:
    def test_blocks_until_confirm(self, manager):
        mgr, audit, _ = manager
        t, holder = _request_in_thread(
            mgr, target_id="t1", reason="why", requester_id="r1", timeout=5,
        )
        # Wait briefly for the worker to register the pending request.
        for _ in range(50):
            if mgr.pending_tokens():
                break
            time.sleep(0.01)
        assert mgr.pending_tokens(), "pending request was never registered"
        assert "outcome" not in holder, "request unblocked without confirmation"
        token = mgr.pending_tokens()[0]
        result = mgr.confirm(token)
        assert result["ok"] is True
        t.join(timeout=2)
        assert holder["outcome"] is ShutdownOutcome.CONFIRMED

        events = [e["event"] for e in audit.read_all()]
        assert "agent_shutdown_requested" in events
        assert "agent_shutdown_confirmed" in events


class TestDeny:
    def test_deny_records_outcome(self, manager):
        mgr, audit, _ = manager
        t, holder = _request_in_thread(
            mgr, target_id="t1", reason="x", requester_id="r1", timeout=5,
        )
        for _ in range(50):
            if mgr.pending_tokens():
                break
            time.sleep(0.01)
        token = mgr.pending_tokens()[0]
        mgr.deny(token)
        t.join(timeout=2)
        assert holder["outcome"] is ShutdownOutcome.DENIED
        events = [e["event"] for e in audit.read_all()]
        assert "agent_shutdown_denied" in events


class TestTimeout:
    def test_timeout_records_denied_timeout(self, manager):
        mgr, audit, _ = manager
        # Synchronous call with short timeout — no resolver thread needed.
        outcome = mgr.request_agent_shutdown(
            target_id="t1", reason="hangs", requester_id="r1", timeout=0.1,
        )
        assert outcome is ShutdownOutcome.TIMED_OUT
        events = [e["event"] for e in audit.read_all()]
        assert "agent_shutdown_timed_out" in events


# ── Validation ─────────────────────────────────────────────────────────────


class TestValidation:
    def test_self_shutdown_rejected(self, manager):
        mgr, _, _ = manager
        with pytest.raises(ValueError):
            mgr.request_agent_shutdown(
                target_id="same", reason="x", requester_id="same",
            )

    def test_missing_ids_rejected(self, manager):
        mgr, _, _ = manager
        with pytest.raises(ValueError):
            mgr.request_agent_shutdown(
                target_id="", reason="x", requester_id="r",
            )
        with pytest.raises(ValueError):
            mgr.request_agent_shutdown(
                target_id="t", reason="x", requester_id="",
            )

    def test_unknown_token_resolution_is_safe(self, manager):
        mgr, _, _ = manager
        out = mgr.confirm("nonexistent")
        assert out["ok"] is False
        out = mgr.deny("nonexistent")
        assert out["ok"] is False


# ── Property: no unconfirmed shutdown ──────────────────────────────────────


class TestNoUnconfirmedShutdown:
    def test_no_unconfirmed_shutdown_outcome(self, manager):
        """
        Spec: "assert no agent can be stopped without a recorded confirmation".
        We assert that the *outcome* of every request is recorded, and that
        an outcome of CONFIRMED is only ever produced by an explicit confirm().
        """
        mgr, audit, _ = manager
        # Confirmed path
        t1, h1 = _request_in_thread(
            mgr, target_id="t1", reason="a", requester_id="r1", timeout=5,
        )
        for _ in range(50):
            if mgr.pending_tokens():
                break
            time.sleep(0.01)
        mgr.confirm(mgr.pending_tokens()[0])
        t1.join(timeout=2)
        # Denied path
        t2, h2 = _request_in_thread(
            mgr, target_id="t2", reason="b", requester_id="r2", timeout=5,
        )
        for _ in range(50):
            if mgr.pending_tokens():
                break
            time.sleep(0.01)
        mgr.deny(mgr.pending_tokens()[0])
        t2.join(timeout=2)
        # Timed-out path
        mgr.request_agent_shutdown(
            target_id="t3", reason="c", requester_id="r3", timeout=0.1,
        )
        events = audit.read_all()
        # For every requested attempt there is exactly one terminal record.
        requested = [e for e in events if e["event"] == "agent_shutdown_requested"]
        terminals = [e for e in events if e["event"] in (
            "agent_shutdown_confirmed",
            "agent_shutdown_denied",
            "agent_shutdown_timed_out",
        )]
        assert len(requested) == 3
        assert len(terminals) == 3
        # CONFIRMED only ever appears once and corresponds to t1.
        confirmed = [e for e in terminals if e["event"] == "agent_shutdown_confirmed"]
        assert len(confirmed) == 1
        assert confirmed[0]["target_id"] == "t1"

    def test_emit_failure_records_denied(self, tmp_path):
        """If the frontend can never be notified, we must not silently confirm."""
        audit = AuditLog(tmp_path / "audit.jsonl")

        def boom(_e, _p):
            raise RuntimeError("frontend gone")

        mgr = LifecycleManager(audit, emit=boom)
        outcome = mgr.request_agent_shutdown(
            target_id="t", reason="x", requester_id="r", timeout=5,
        )
        assert outcome is ShutdownOutcome.DENIED
        events = [e["event"] for e in audit.read_all()]
        assert "agent_shutdown_denied" in events
