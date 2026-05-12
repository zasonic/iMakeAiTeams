"""
services/memory/write_gate.py — MINJA-defense + memory-write review.

Owns four concerns that all gate the path from "extracted/proposed
memory content" to "row in session_facts / memory_entries":

  1. ``_trust_scan(content)`` — PromptGuard pass via input_sanitizer.
     Returns a verdict dict; failure modes never block (fail-open).
  2. ``_write_to_pending_review(...)`` — writes a ``pending_review``
     row when a trust scan flags content.
  3. ``MemoryWriteGate`` — MINJA-style shadow consistency check. A
     fact that contradicts an existing fact is routed to ``pending_writes``
     for explicit user approval instead of silently overwriting.
  4. Pending-review + pending-write CRUD used by the
     Memory Review panel routes.

Plus ``save_explicit_memory(content, category)`` — the user-initiated
explicit-memory write path. Goes through the trust scan and lands
either in ``memory_entries`` or ``pending_review``.

MINJA reference: https://arxiv.org/abs/2503.03704 — 95%+ poisoning
success without the gate; the consistency check brings that to near 0
for the documented attack patterns.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import db as _db
from services.redact import redact

from ._context import _scrub_deflections

try:
    import sse_events as _sse_events
except ImportError:
    _sse_events = None

log = logging.getLogger("iMakeAiTeams.memory.write_gate")


# ── Priority 7: Trust scanning ────────────────────────────────────────────────


def _trust_scan(content: str) -> dict:
    """Run PromptGuard on memory content before writing it.

    Returns the scan result dict from input_sanitizer. On any error,
    returns a safe "pass" result so memory writes are never blocked by
    scanner failures.
    """
    try:
        from services import input_sanitizer  # noqa: PLC0415
        if not input_sanitizer.is_firewall_enabled():
            return {"verdict": "pass", "blocked": False, "degraded": True}
        return input_sanitizer.scan_document(content, filename="memory_write")
    except Exception as exc:
        log.debug("Trust scan failed (non-fatal): %s", exc)
        return {"verdict": "pass", "blocked": False, "degraded": True}


def _write_to_pending_review(
    content:     str,
    source_type: str,   # "session_fact" | "memory_entry"
    context_id:  str,   # conversation_id or empty string
    scan_result: dict,
) -> str:
    """Route flagged memory content to the pending_review table instead of
    committing it to session_facts or memory_entries.

    Returns the pending_review row ID.
    """
    review_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    try:
        _db.execute(
            """
            INSERT INTO pending_review
                (id, content, source_type, context_id,
                 scan_verdict, scan_score, scan_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id, content, source_type, context_id,
                scan_result.get("verdict", "warn"),
                scan_result.get("score"),
                scan_result.get("reason", "")[:500],
                now,
            ),
        )
        _db.commit()
        log.warning(
            "Memory trust: flagged %s content routed to pending_review (id=%s, score=%s)",
            source_type, review_id[:8], scan_result.get("score"),
        )
    except Exception as exc:
        log.warning("_write_to_pending_review failed: %s", exc)
    return review_id


# ── Pending review CRUD (called from core/api/memory.py) ─────────────────────


def get_pending_review(limit: int = 50) -> list[dict]:
    """Return unresolved flagged memory items for the Settings review panel."""
    try:
        rows = _db.fetchall(
            "SELECT * FROM pending_review WHERE status = 'pending' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning("get_pending_review failed: %s", exc)
        return []


def approve_pending(review_id: str) -> bool:
    """Approve a pending review item: commit the content to the appropriate
    store then mark it approved."""
    try:
        row = _db.fetchone(
            "SELECT * FROM pending_review WHERE id = ?", (review_id,)
        )
        if not row:
            return False

        content     = row["content"]
        source_type = row["source_type"]
        context_id  = row["context_id"] or ""
        now         = datetime.now(timezone.utc).isoformat()

        if source_type == "session_fact":
            _db.execute(
                "INSERT INTO session_facts (id, conversation_id, fact, source, created_at) "
                "VALUES (?, ?, ?, 'approved', ?)",
                (str(uuid.uuid4()), context_id, content, now),
            )
        else:  # memory_entry
            mem_id = str(uuid.uuid4())
            _db.execute(
                "INSERT INTO memory_entries "
                "(id, content, category, source, embedding_status, created_at, last_accessed) "
                "VALUES (?, ?, 'fact', 'approved', 'dirty', ?, ?)",
                (mem_id, content, now, now),
            )

        _db.execute(
            "UPDATE pending_review SET status='approved', resolved_at=? WHERE id=?",
            (now, review_id),
        )
        _db.commit()
        log.info("Approved pending_review %s", review_id[:8])
        return True
    except Exception as exc:
        log.warning("approve_pending failed: %s", exc)
        return False


def reject_pending(review_id: str) -> bool:
    """Mark a pending review item as rejected (discards the content)."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        _db.execute(
            "UPDATE pending_review SET status='rejected', resolved_at=? WHERE id=?",
            (now, review_id),
        )
        _db.commit()
        log.info("Rejected pending_review %s", review_id[:8])
        return True
    except Exception as exc:
        log.warning("reject_pending failed: %s", exc)
        return False


def get_pending_count() -> int:
    """Return the count of unresolved pending review items (for badge display)."""
    try:
        row = _db.fetchone(
            "SELECT COUNT(*) as n FROM pending_review WHERE status='pending'"
        )
        return row["n"] if row else 0
    except Exception:
        return 0


# ── MemoryWriteGate (MINJA defense) ──────────────────────────────────────────


_GATE_SYSTEM_PROMPT = (
    "Does the new fact contradict any of these existing facts? "
    "Reply ONLY with JSON {contradicts: bool, id: str or null, reason: str}"
)


class MemoryWriteGate:
    """Shadow-consistency gate for auto-extracted facts.

    Fail-open by design: any error in the local-model consistency check is
    treated as "consistent". The gate's purpose is detecting the injection
    pattern, not enforcing strict logical consistency.
    """

    def __init__(self, local_client, settings=None):
        self.local_client = local_client
        self._settings = settings

    def is_enabled(self) -> bool:
        if self._settings is None:
            return True
        try:
            return bool(self._settings.get("memory_write_gate_enabled", True))
        except Exception:
            return True

    def shadow_consistency_check(
        self, new_fact: str, existing_facts: list[dict]
    ) -> tuple[bool, str | None, str]:
        """Best-effort consistency check via the local model.

        On any failure or when the local model is unavailable, treat as
        consistent (fail-open). Returns ``(is_consistent, contradicting_id_or_None, reason)``.
        """
        if not new_fact or not existing_facts:
            return (True, None, "")
        if not self.local_client or not self.local_client.is_available():
            return (True, None, "")
        try:
            existing_payload = json.dumps([
                {"id": str(f.get("id", "")), "fact": str(f.get("fact", ""))}
                for f in existing_facts
            ])
            user_prompt = (
                f"New fact: {new_fact}\n\nExisting facts: {existing_payload}"
            )
            raw = self.local_client.chat(
                _GATE_SYSTEM_PROMPT, user_prompt, max_tokens=200,
            )
            text = (raw or "").strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
            contradicts = bool(parsed.get("contradicts"))
            if not contradicts:
                return (True, None, "")
            cid_raw = parsed.get("id")
            cid = str(cid_raw) if cid_raw else None
            reason = str(parsed.get("reason", ""))
            return (False, cid, reason)
        except Exception as exc:
            log.debug("shadow_consistency_check failed (fail-open): %s", exc)
            return (True, None, "")

    def gate_fact_write(self, conversation_id: str, fact: str) -> str:
        """Route a fact through the consistency check.

        Returns "accepted" when the gate is bypassed or the fact is
        consistent; returns "pending_review" after writing a
        pending_writes row and emitting the memory_review_required SSE
        event.
        """
        if not self.is_enabled():
            return "accepted"
        try:
            rows = _db.fetchall(
                "SELECT id, fact FROM session_facts WHERE conversation_id = ? "
                "AND (status = 'confirmed' OR status IS NULL OR status = 'pending')",
                (conversation_id,),
            )
        except Exception as exc:
            log.debug("gate_fact_write: existing-fact lookup failed: %s", exc)
            return "accepted"
        if not rows:
            return "accepted"
        existing = [{"id": r["id"], "fact": r["fact"]} for r in rows]
        is_consistent, contradicts_id, reason = self.shadow_consistency_check(
            fact, existing,
        )
        if is_consistent:
            return "accepted"

        contradicts_content = None
        if contradicts_id:
            for e in existing:
                if e["id"] == contradicts_id:
                    contradicts_content = e["fact"]
                    break

        pending_id = str(uuid.uuid4())
        proposed_at = datetime.now(timezone.utc).isoformat()
        try:
            _db.execute(
                "INSERT INTO pending_writes "
                "(id, conversation_id, write_type, content, "
                "contradicts_id, contradicts_content, proposed_at) "
                "VALUES (?, ?, 'fact', ?, ?, ?, ?)",
                (pending_id, conversation_id, fact,
                 contradicts_id, contradicts_content, proposed_at),
            )
            _db.commit()
        except Exception as exc:
            log.warning("pending_writes insert failed: %s", exc)
            return "accepted"

        if _sse_events is not None:
            try:
                _sse_events.publish("memory_review_required", {
                    "id": pending_id,
                    "conversation_id": conversation_id,
                    "write_type": "fact",
                    "content": fact,
                    "contradicts_id": contradicts_id,
                    "contradicts_content": contradicts_content,
                    "reason": reason,
                })
            except Exception as exc:
                log.debug("memory_review_required emit failed: %s", exc)

        log.info(
            "MemoryWriteGate: contradiction detected — fact routed to "
            "pending_writes (id=%s, contradicts=%s)",
            pending_id[:8], (contradicts_id or "")[:8],
        )
        return "pending_review"


# ── Pending-writes CRUD (called from core/api/memory.py) ─────────────────────


def list_pending_writes(limit: int = 100) -> list[dict]:
    """Return undecided pending_writes rows for the Memory Review panel."""
    try:
        rows = _db.fetchall(
            "SELECT id, conversation_id, write_type, content, "
            "contradicts_id, contradicts_content, proposed_at, "
            "decision, decided_at "
            "FROM pending_writes WHERE decision IS NULL "
            "ORDER BY proposed_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning("list_pending_writes failed: %s", exc)
        return []


def approve_pending_write(pending_id: str) -> dict:
    """Accept a pending fact: INSERT into session_facts, mark approved."""
    row = _db.fetchone(
        "SELECT id, conversation_id, write_type, content, decision "
        "FROM pending_writes WHERE id = ?",
        (pending_id,),
    )
    if row is None:
        return {"ok": False, "error": "pending_write not found"}
    if row["decision"] is not None:
        return {
            "ok": False,
            "error": f"already {row['decision']}",
            "decision": row["decision"],
        }
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _db.transaction() as conn:
            if row["write_type"] == "fact":
                conn.execute(
                    "INSERT INTO session_facts "
                    "(id, conversation_id, fact, source, status, created_at) "
                    "VALUES (?, ?, ?, 'auto', 'confirmed', ?)",
                    (str(uuid.uuid4()), row["conversation_id"], row["content"], now),
                )
            elif row["write_type"] == "memory":
                conn.execute(
                    "INSERT INTO memory_entries "
                    "(id, content, category, source, embedding_status, "
                    "created_at, last_accessed) "
                    "VALUES (?, ?, 'fact', 'auto', 'dirty', ?, ?)",
                    (str(uuid.uuid4()), row["content"], now, now),
                )
            conn.execute(
                "UPDATE pending_writes SET decision='approved', decided_at=? "
                "WHERE id=?",
                (now, pending_id),
            )
    except Exception as exc:
        log.warning("approve_pending_write failed: %s", exc)
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "id": pending_id, "decision": "approved"}


def deny_pending_write(pending_id: str) -> dict:
    """Reject a pending fact: mark denied, do NOT insert into session_facts."""
    row = _db.fetchone(
        "SELECT decision FROM pending_writes WHERE id = ?", (pending_id,),
    )
    if row is None:
        return {"ok": False, "error": "pending_write not found"}
    if row["decision"] is not None:
        return {
            "ok": False,
            "error": f"already {row['decision']}",
            "decision": row["decision"],
        }
    now = datetime.now(timezone.utc).isoformat()
    try:
        _db.execute(
            "UPDATE pending_writes SET decision='denied', decided_at=? WHERE id=?",
            (now, pending_id),
        )
        _db.commit()
    except Exception as exc:
        log.warning("deny_pending_write failed: %s", exc)
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "id": pending_id, "decision": "denied"}


# ── Explicit-memory write path ───────────────────────────────────────────────


def save_explicit_memory(content: str, category: str = "fact") -> str:
    """Let the user or agent store an explicit long-term memory.

    Goes through ``_scrub_deflections`` → redact → trust scan. Flagged
    content (``block`` or ``warn`` verdict) is routed to pending_review
    instead of writing directly to ``memory_entries``. Returns either the
    new memory_entries row id, or ``"pending:<review_id>"`` when the
    write was deferred for human approval.
    """
    # Strip assistant-deflection sentences and redact credentials before
    # any persistence path (trust scan or DB insert) sees the content.
    content = _scrub_deflections(content)
    if not content:
        return "Nothing substantive to remember (deflection scrubbed)"
    content = redact(content)

    scan = _trust_scan(content)
    if scan.get("blocked") or scan.get("verdict") in ("block", "warn"):
        review_id = _write_to_pending_review(content, "memory_entry", "", scan)
        log.info("Trust scan: memory routed to pending_review (verdict=%s)", scan.get("verdict"))
        return f"pending:{review_id}"

    now    = datetime.now(timezone.utc).isoformat()
    mem_id = str(uuid.uuid4())
    _db.execute(
        "INSERT INTO memory_entries "
        "(id, content, category, source, embedding_status, created_at, last_accessed) "
        "VALUES (?, ?, ?, 'user', 'dirty', ?, ?)",
        (mem_id, content, category, now, now),
    )
    _db.commit()
    return mem_id
