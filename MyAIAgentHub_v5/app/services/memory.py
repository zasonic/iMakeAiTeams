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
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import db as _db
from models import SessionHistory
from services.prompt_library import get_active_prompt
from services.security_engine import validate_fact_for_storage, MAX_FACTS_PER_CONVERSATION

log = logging.getLogger("MyAIEnv.memory")

SIMILARITY_THRESHOLD = 0.5

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


@dataclass
class MemoryContext:
    recent_messages: list = field(default_factory=list)
    session_facts:   list = field(default_factory=list)
    rag_chunks:      list = field(default_factory=list)
    memories:        list = field(default_factory=list)

    def to_system_suffix(self) -> str:
        parts = []
        if self.session_facts:
            parts.append(
                "## Known facts about this session\n" +
                "\n".join(f"- {f}" for f in self.session_facts)
            )
        if self.rag_chunks:
            parts.append(
                "## Relevant documents\n" +
                "\n---\n".join(self.rag_chunks)
            )
        if self.memories:
            parts.append(
                "## Long-term memory\n" +
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


# ── MemoryManager ─────────────────────────────────────────────────────────────

class MemoryManager:
    def __init__(self, rag_index, semantic_search_mod, local_client):
        self.rag      = rag_index
        self.semantic = semantic_search_mod
        self.local    = local_client
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
            "SELECT fact FROM session_facts WHERE conversation_id = ? "
            "ORDER BY created_at DESC LIMIT 10",
            (conversation_id,),
        )
        ctx.session_facts = [r["fact"] for r in facts]

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
                fact_words = set(fact.lower().split())
                meaningful = fact_words - _stopwords
                if not meaningful:
                    continue
                overlap = meaningful & source_words
                ratio = len(overlap) / len(meaningful) if meaningful else 0
                if ratio >= 0.4:
                    grounded_facts.append(fact.strip())
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

                _db.execute(
                    "INSERT INTO session_facts "
                    "(id, conversation_id, fact, source, created_at) "
                    "VALUES (?, ?, ?, 'auto', ?)",
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

        except Exception as exc:
            _extract_failures += 1
            if _extract_attempts >= 20 and _extract_failures / _extract_attempts > 0.5:
                log.warning("Memory fact extraction failing frequently.")
            log.debug(f"Fact extraction failed: {exc}")

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
