"""
services/memory.py

Three-tier memory system.

- Short-term:  conversation message buffer (in-memory deque, per session)
- Working:     session facts extracted by local model (SQLite)
- Long-term:   RAG index + memory_entries (ChromaDB + SQLite)

Stage 2 changes:
  - Similarity score gating: RAG chunks and semantic memories below
    SIMILARITY_THRESHOLD (0.5) are excluded from context.

Stage 3 changes:
  - Defensive fact extraction (retry on JSONDecodeError, deduplication)
  - Smarter conversation summarizer (topic-boundary shift detection)

Stage 5 changes:
  - SessionHistory tracking (Improvement 5)
  - Hard-trim fallback (Improvement 7)

Priority 7 additions (Memory Trust Scoring):
  - _trust_scan(content)             — scans content with PromptGuard before write
  - _write_to_pending_review()       — routes flagged content to pending_review table
  - _extract_facts() gated by trust scan
  - save_explicit_memory() gated by trust scan
  - get_pending_review() / approve_pending() / reject_pending() — review workflow
"""

import json
import logging
import re as _re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import db as _db
from models import SessionHistory
from services.prompt_library import get_active_prompt
from services.redact import redact
from services.security_engine import validate_fact_for_storage, MAX_FACTS_PER_CONVERSATION

try:
    import sse_events as _sse_events
except ImportError:
    _sse_events = None

log = logging.getLogger("iMakeAiTeams.memory")

SIMILARITY_THRESHOLD = 0.5

# Patterns that match sentences whose PURPOSE is to record an assistant
# failure/limitation. Deliberately narrow — false positives erase real
# content, which is worse than letting a few deflections through.
_DEFLECTION_PATTERNS: tuple[_re.Pattern[str], ...] = tuple(
    _re.compile(p, _re.IGNORECASE) for p in (
        # "the assistant <failure verb>"
        r"the assistant\s+(was unable|could not|did not have|does not have|"
        r"offered to (search|help|look)|suggested checking|"
        r"recommended consulting|clarified that[^.]{0,40}(could not|did not have|cannot|limit)|"
        r"explained (it|that it) (could not|cannot|does not|did not)|"
        r"stated[^.]{0,40}(could not|did not have)|"
        r"indicated[^.]{0,40}(could not|did not have))",
        # Capability-denial framing
        r"the assistant\s+(lacks|cannot|can'?t)\s+(access|the ability|information|specific information|details)",
        # "the AI" variants
        r"the AI\s+(was unable|could not|did not have|does not have|cannot|doesn'?t have)",
        # Self-referential limitations
        r"(I|the model)\s+(don'?t|do not|cannot|can'?t)\s+have\s+(access|real-time|current|specific)",
    )
)

# Sentence splitter: split on . ! ? followed by whitespace or end-of-string.
_SENTENCE_SPLIT = _re.compile(r"(?<=[.!?])\s+")


def _scrub_deflections(text: str) -> str:
    """Remove sentences that narrate assistant failures/limitations.

    Returns the text with deflection sentences removed. If every sentence
    is a deflection, returns empty string (caller should discard the fact).
    """
    if not text:
        return text
    sentences = _SENTENCE_SPLIT.split(text)
    kept = []
    for sentence in sentences:
        is_deflection = any(p.search(sentence) for p in _DEFLECTION_PATTERNS)
        if not is_deflection:
            kept.append(sentence)
    return " ".join(kept).strip()

_extract_attempts = 0
_extract_failures  = 0

_SUMMARIZE_LENGTH_TRIGGER = 30
_TOPIC_SHIFT_WINDOW       = 3

_SUMMARY_PROMPT = (
    "Summarize this conversation segment in 3–5 sentences. "
    "Focus on: decisions made, open questions, and any preferences or commitments "
    "the user expressed. Be specific — preserve names, numbers, and dates."
)

_FACT_RETRY_PROMPT = (
    "Reply with ONLY a JSON array of strings, nothing else. "
    "No markdown, no explanation, no backticks. Example: [\"fact one\", \"fact two\"]\n\n"
)

_TRIPLE_PROMPT = (
    "Extract (subject, predicate, object) triples from these facts.\n"
    "Return ONLY a JSON array. Each element: "
    '{"subject": "...", "predicate": "...", "object": "..."}\n'
    "If a fact cannot be decomposed into a triple, skip it.\n\n"
    "Facts:\n{facts}"
)

_CONTRADICTION_SIGNALS = _re.compile(
    r"\b(no[,.]?\s|actually|that'?s wrong|that'?s not|incorrect|"
    r"I meant|not right|correction|I said|wrong)\b",
    _re.IGNORECASE,
)


@dataclass
class MemoryContext:
    recent_messages: list = field(default_factory=list)
    session_facts:   list = field(default_factory=list)
    rag_chunks:      list = field(default_factory=list)
    memories:        list = field(default_factory=list)

    def to_system_suffix(self) -> str:
        # Section headers are deliberately phrased to mirror the trigger
        # conditions of canonical denial templates ("I don't have personal
        # information about you"). Pre-filling those exact slots with real
        # data dampens the denial reflex on small local models.
        parts = []
        if self.session_facts:
            parts.append(
                "## Personal information about the user\n"
                "(These are facts the user told you. Reference them naturally.)\n\n" +
                "\n".join(f"- {f}" for f in self.session_facts)
            )
        if self.rag_chunks:
            parts.append(
                "## Reference documents the user has provided\n"
                "(The user uploaded these. Use them to answer their question.)\n\n" +
                "\n---\n".join(self.rag_chunks)
            )
        if self.memories:
            parts.append(
                "## Information the user has shared in prior conversations\n"
                "(You have access to this — it was stored from previous sessions.)\n\n" +
                "\n".join(f"- {m}" for m in self.memories)
            )
        return "\n\n".join(parts) if parts else ""


# ── Priority 7: Trust scanning ────────────────────────────────────────────────

def _trust_scan(content: str) -> dict:
    """
    Run PromptGuard on memory content before writing it.
    Returns the scan result dict from input_sanitizer.
    On any error, returns a safe "pass" result so memory writes are never blocked
    by scanner failures.
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
    """
    Route flagged memory content to the pending_review table instead of
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


# ── Pending review CRUD (called from api.py) ─────────────────────────────────

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
    """
    Approve a pending review item: commit the content to the appropriate store
    then mark it approved.
    """
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
#
# MINJA (https://arxiv.org/abs/2503.03704) demonstrates a 95%+ success rate at
# poisoning agent memory through query-only conversations: each turn nudges the
# model into recording a fact that contradicts what's already there. The gate
# runs a shadow consistency check on every newly extracted fact and routes
# contradictions to a user-approval queue. Auto-extracted facts that don't
# contradict existing memory go through unchanged, so the gate adds zero
# friction for benign conversations.
#
# Fail-open by design: any error in the local-model consistency check is
# treated as "consistent". The gate's purpose is detecting the injection
# pattern, not enforcing strict logical consistency.

_GATE_SYSTEM_PROMPT = (
    "Does the new fact contradict any of these existing facts? "
    "Reply ONLY with JSON {contradicts: bool, id: str or null, reason: str}"
)


class MemoryWriteGate:
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
        """
        Best-effort consistency check via the local model. On any failure or
        when the local model is unavailable, treat as consistent (fail-open).
        Returns (is_consistent, contradicting_id_or_None, reason).
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
        """
        Route a fact through the consistency check.
        Returns "accepted" when the gate is bypassed or the fact is consistent;
        returns "pending_review" after writing a pending_writes row and emitting
        the memory_review_required SSE event.
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


# ── MemoryManager ─────────────────────────────────────────────────────────────

class MemoryManager:
    def __init__(self, rag_index, semantic_search_mod, local_client, settings=None):
        self.rag      = rag_index
        self.semantic = semantic_search_mod
        self.local    = local_client
        self.write_gate = MemoryWriteGate(local_client, settings)
        self._buffers:   dict[str, deque]         = {}
        self._histories: dict[str, SessionHistory] = {}

    def _get_history(self, conversation_id: str) -> SessionHistory:
        if conversation_id not in self._histories:
            self._histories[conversation_id] = SessionHistory()
        return self._histories[conversation_id]

    def get_session_history(self, conversation_id: str) -> list[dict]:
        history = self._get_history(conversation_id)
        return [
            {
                "event_type": e.event_type,
                "detail":     e.detail,
                "timestamp":  e.timestamp,
            }
            for e in history.recent(50)
        ]

    def _get_buffer(self, conversation_id: str) -> deque:
        if conversation_id not in self._buffers:
            self._buffers[conversation_id] = deque(maxlen=60)
        return self._buffers[conversation_id]

    def add_to_buffer(self, conversation_id: str, role: str, content: str) -> None:
        buf = self._get_buffer(conversation_id)
        buf.append({"role": role, "content": content})

    def get_buffer(self, conversation_id: str) -> list:
        return list(self._get_buffer(conversation_id))

    def should_summarize(self, conversation_id: str) -> bool:
        buf = self._get_buffer(conversation_id)
        if len(buf) >= _SUMMARIZE_LENGTH_TRIGGER:
            return True
        if len(buf) >= _TOPIC_SHIFT_WINDOW + 2:
            recent = list(buf)[-_TOPIC_SHIFT_WINDOW:]
            earlier = list(buf)[: len(buf) - _TOPIC_SHIFT_WINDOW]
            recent_words = set(
                w.lower() for m in recent
                for w in m["content"].split() if len(w) > 4
            )
            earlier_words = set(
                w.lower() for m in earlier
                for w in m["content"].split() if len(w) > 4
            )
            overlap = recent_words & earlier_words
            if earlier_words and len(overlap) / max(len(recent_words), 1) < 0.1:
                return True
        return False

    def summarize_buffer(self, conversation_id: str) -> str | None:
        buf = self._get_buffer(conversation_id)
        if len(buf) < 4:
            return None
        if not self.local or not self.local.is_available():
            if len(buf) >= 50:
                overflow = len(buf) - 30
                original_len = len(buf)
                for _ in range(overflow):
                    buf.popleft()
                log.info("Hard-trimmed conversation buffer from %d to %d messages",
                         original_len, len(buf))
                hist = self._get_history(conversation_id)
                hist.add("hard_trim",
                         f"Hard-trimmed buffer from {original_len} to {len(buf)} messages")
            return None
        messages_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in list(buf)[-20:]
        )
        try:
            summary = self.local.chat(
                _SUMMARY_PROMPT,
                f"Conversation to summarize:\n\n{messages_text}",
                max_tokens=300,
            )
            original_count = len(list(buf))
            buf.clear()
            buf.append({"role": "system", "content": f"[Earlier conversation summary: {summary}]"})
            hist = self._get_history(conversation_id)
            hist.add("summarized", f"Summarized {original_count} messages into compact form")
            return summary
        except Exception as exc:
            log.debug("Buffer summarization failed: %s", exc)
            return None

    def get_context(
        self,
        conversation_id: str,
        user_message:    str,
        agent_id:        str | None = None,
    ) -> MemoryContext:
        ctx = MemoryContext()
        ctx.recent_messages = self.get_buffer(conversation_id)

        facts = _db.fetchall(
            "SELECT id, fact FROM session_facts WHERE conversation_id = ? "
            "AND (status = 'confirmed' OR status IS NULL) "
            "ORDER BY COALESCE(last_accessed, created_at) DESC LIMIT 10",
            (conversation_id,),
        )
        ctx.session_facts = [r["fact"] for r in facts]
        if facts:
            try:
                now = datetime.now(timezone.utc).isoformat()
                ids = [r["id"] for r in facts]
                placeholders = ",".join("?" * len(ids))
                _db.execute(
                    f"UPDATE session_facts SET last_accessed = ? WHERE id IN ({placeholders})",
                    tuple([now] + ids),
                )
                _db.commit()
            except Exception as exc:
                log.debug("session_facts last_accessed update failed: %s", exc)

        try:
            rag_results = self.rag.search(user_message, top_k=3)
            ctx.rag_chunks = [
                r[0] if isinstance(r, (list, tuple)) else r
                for r in rag_results
            ]
        except Exception:
            pass

        try:
            mem_results = self.semantic.search_memories(user_message, top_k=3)
            ctx.memories = [
                m["content"] for m in mem_results
                if m.get("score", 0) >= SIMILARITY_THRESHOLD
            ]
        except Exception:
            pass

        hist = self._get_history(conversation_id)
        hist.add("memory_recall",
                 f"RAG: {len(ctx.rag_chunks)} chunks, Memories: {len(ctx.memories)}, "
                 f"Facts: {len(ctx.session_facts)}")

        return ctx

    def _resolve_pending_facts(self, conversation_id: str, user_message: str) -> None:
        """Promote or discard pending facts based on user's follow-up."""
        pending = _db.fetchall(
            "SELECT id, fact FROM session_facts "
            "WHERE conversation_id = ? AND status = 'pending'",
            (conversation_id,),
        )
        if not pending:
            return
        has_contradiction = bool(_CONTRADICTION_SIGNALS.search(user_message))
        new_status = "discarded" if has_contradiction else "confirmed"
        for row in pending:
            _db.execute(
                "UPDATE session_facts SET status = ? WHERE id = ?",
                (new_status, row["id"]),
            )
        _db.commit()
        if has_contradiction and pending:
            log.info("Discarded %d pending facts (contradiction detected)", len(pending))

    def extract_facts(self, conversation_id: str, user_msg: str,
                      assistant_msg: str) -> None:
        """
        Extract facts from an exchange via local model.
        Priority 7: scans each fact before writing. Flagged facts go to pending_review.
        Fix 5: grounding check — only stores facts with keyword overlap to source text.
        """
        global _extract_attempts, _extract_failures
        if not self.local or not self.local.is_available():
            return
        _extract_attempts += 1
        self._resolve_pending_facts(conversation_id, user_msg)
        try:
            system = get_active_prompt("fact_extractor")
            prompt = (
                f"User said: {user_msg[:500]}\n"
                f"Assistant said: {assistant_msg[:500]}\n"
            )
            result = self.local.chat(system, prompt, max_tokens=300)

            facts = self._parse_facts_json(result, allow_retry=True,
                                           user_msg=user_msg,
                                           assistant_msg=assistant_msg)
            if facts is None:
                return

            # ── Fix 5: Grounding check ───────────────────────────────────────
            # Each fact must have meaningful keyword overlap with the source
            # messages. This prevents local model hallucinations from poisoning
            # long-term memory.
            _stopwords = {
                "the","a","an","is","are","was","were","it","in","on","to","for",
                "of","and","or","that","this","with","has","have","had","be","been",
                "not","but","they","their","them","he","she","his","her","we","our",
                "you","your","i","my","me","so","at","by","from","up","no","yes",
                "do","does","did","will","would","can","could","should","may","might",
                "about","just","also","very","much","more","some","any","all","each",
            }
            source_text = (user_msg + " " + assistant_msg).lower()
            source_words = set(source_text.split())

            grounded_facts = []
            for fact in facts[:3]:
                if not isinstance(fact, str) or not fact.strip():
                    continue
                # Strip deflection sentences before grounding so we score
                # only the substantive content of the fact.
                fact = _scrub_deflections(fact)
                if not fact:
                    continue
                fact_words = set(fact.lower().split())
                meaningful = fact_words - _stopwords
                if not meaningful:
                    continue
                overlap = meaningful & source_words
                ratio = len(overlap) / len(meaningful) if meaningful else 0
                if ratio >= 0.4:
                    # Redact credentials AFTER grounding so [REDACTED_*] tokens
                    # don't deflate the meaningful-word ratio.
                    grounded_facts.append(redact(fact.strip()))
                else:
                    log.debug("Discarded ungrounded fact (%.0f%% overlap): %s",
                              ratio * 100, fact[:80])

            # Guard: the local model call above can take several seconds.
            # If the conversation was deleted in the meantime, skip to avoid
            # inserting orphaned session_facts rows.
            if not _db.fetchone(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ):
                log.debug(
                    "extract_facts: conversation %s was deleted during extraction; discarding facts",
                    conversation_id[:8],
                )
                return

            existing_rows = _db.fetchall(
                "SELECT fact FROM session_facts WHERE conversation_id = ?",
                (conversation_id,),
            )
            existing_lower = {r["fact"].lower().strip() for r in existing_rows}

            now      = datetime.now(timezone.utc).isoformat()
            inserted = 0
            inserted_facts: list[str] = []

            for fact_clean in grounded_facts:
                if fact_clean.lower() in existing_lower:
                    log.debug("memory: skipping duplicate fact: %r", fact_clean)
                    continue

                # ── Memory Firewall: structural validation before storage ─────
                # Enforces length caps, pattern blocklist, special-char density,
                # and conversation-level fact limits. Based on MINJA (98.2% ASR)
                # and SpAIware findings. Constrains what CAN be stored.
                if len(existing_lower) >= MAX_FACTS_PER_CONVERSATION:
                    log.info("Memory firewall: fact cap reached (%d) for %s",
                             MAX_FACTS_PER_CONVERSATION, conversation_id[:8])
                    break

                fw_valid, fw_reason, fw_attestation = validate_fact_for_storage(
                    fact_clean, conversation_id, extraction_method="local_model"
                )
                if not fw_valid:
                    log.info("Memory firewall rejected fact: %s — %r",
                             fw_reason, fact_clean[:60])
                    continue

                # ── Priority 7: trust scan before write ───────────────────────
                scan = _trust_scan(fact_clean)
                if scan.get("blocked") or scan.get("verdict") == "block":
                    _write_to_pending_review(fact_clean, "session_fact", conversation_id, scan)
                    log.info("Trust scan: fact routed to pending_review: %r", fact_clean[:60])
                    continue
                if scan.get("verdict") == "warn":
                    _write_to_pending_review(fact_clean, "session_fact", conversation_id, scan)
                    log.info("Trust scan: warn verdict — fact routed to pending_review: %r", fact_clean[:60])
                    continue

                # ── Phase 5: MINJA-style write gate (consistency check) ──────
                if self.write_gate.gate_fact_write(conversation_id, fact_clean) == "pending_review":
                    continue

                _db.execute(
                    "INSERT INTO session_facts "
                    "(id, conversation_id, fact, source, status, created_at) "
                    "VALUES (?, ?, ?, 'auto', 'pending', ?)",
                    (str(uuid.uuid4()), conversation_id, fact_clean, now),
                )
                existing_lower.add(fact_clean.lower())
                inserted_facts.append(fact_clean)
                inserted += 1

            if inserted:
                _db.commit()

            if inserted_facts:
                hist = self._get_history(conversation_id)
                hist.add("fact_extracted",
                         f"Extracted {len(inserted_facts)} facts: {inserted_facts}")

            self._extract_triples(grounded_facts, conversation_id)

        except Exception as exc:
            _extract_failures += 1
            if _extract_attempts >= 20 and _extract_failures / _extract_attempts > 0.5:
                log.warning("Memory fact extraction failing frequently.")
            log.debug(f"Fact extraction failed: {exc}")

    def _extract_triples(self, facts: list, conversation_id: str) -> None:
        """Decompose grounded facts into (subject, predicate, object) triples."""
        if not facts or not self.local or not self.local.is_available():
            return
        try:
            prompt = _TRIPLE_PROMPT.format(facts="\n".join(f"- {f}" for f in facts))
            raw = self.local.chat("", prompt, max_tokens=500)
            if not raw:
                return
            text = raw.strip()
            if "```" in text:
                match = _re.search(r"```(?:json)?\s*\n?(.*?)```", text, _re.DOTALL)
                if match:
                    text = match.group(1).strip()
            items = json.loads(text)
            if not isinstance(items, list):
                return
            now = datetime.now(timezone.utc).isoformat()
            for item in items[:20]:
                if not isinstance(item, dict):
                    continue
                subj = str(item.get("subject", "")).strip()
                pred = str(item.get("predicate", "")).strip()
                obj = str(item.get("object", "")).strip()
                if subj and pred and obj:
                    _db.execute(
                        "INSERT INTO knowledge_triples "
                        "(id, subject, predicate, object, confidence, "
                        "source_conversation_id, created_at, last_accessed_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (str(uuid.uuid4()), redact(subj), redact(pred),
                         redact(obj), 0.8, conversation_id, now, now),
                    )
            _db.commit()
        except Exception as exc:
            log.debug("Triple extraction failed (non-fatal): %s", exc)

    def _parse_facts_json(
        self,
        raw:           str,
        allow_retry:   bool = False,
        user_msg:      str  = "",
        assistant_msg: str  = "",
    ) -> list | None:
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        return v
            return None
        except json.JSONDecodeError:
            if not allow_retry or not self.local or not self.local.is_available():
                return None
            try:
                system = get_active_prompt("fact_extractor")
                retry_result = self.local.chat(
                    system,
                    _FACT_RETRY_PROMPT +
                    f"User: {user_msg[:300]}\nAssistant: {assistant_msg[:300]}",
                    max_tokens=200,
                )
                retry_raw = retry_result.strip()
                if retry_raw.startswith("```"):
                    parts = retry_raw.split("```")
                    retry_raw = parts[1] if len(parts) > 1 else retry_raw
                    if retry_raw.startswith("json"):
                        retry_raw = retry_raw[4:]
                parsed = json.loads(retry_raw)
                return parsed if isinstance(parsed, list) else None
            except Exception:
                return None

    def save_explicit_memory(self, content: str, category: str = "fact") -> str:
        """
        Let the user or agent store an explicit long-term memory.
        Priority 7: scans content before writing. Flagged → pending_review.
        """
        # Strip assistant-deflection sentences and redact credentials before
        # any persistence path (trust scan or DB insert) sees the content.
        content = _scrub_deflections(content)
        if not content:
            return "Nothing substantive to remember (deflection scrubbed)"
        content = redact(content)

        # ── Priority 7: trust scan ────────────────────────────────────────────
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
