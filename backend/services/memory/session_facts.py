"""
services/memory/session_facts.py — Per-conversation fact extraction.

Working-memory tier. Pulls (subject, predicate, object) facts out of a
user/assistant exchange via the local model, runs each candidate
through:

  - sentence-level deflection scrubbing (``_scrub_deflections``)
  - keyword-overlap grounding check (≥40% to source text) — Fix 5
  - structural firewall (``validate_fact_for_storage``)
  - PromptGuard trust scan
  - MINJA shadow-consistency gate (``MemoryWriteGate``)

…before inserting into ``session_facts``. Pending facts can be
contradicted by the user's next message (``_resolve_pending_facts``).

Module-level counters ``_extract_attempts`` and ``_extract_failures``
track the success rate so ``services.health_monitor`` can surface an
"extraction failing frequently" warning. The counters live on this
module by design — ``services.memory.__init__`` re-exports them via
PEP 562 ``__getattr__`` so legacy ``from services.memory import
_extract_attempts`` reads still get a live view, not a stale snapshot.
"""

from __future__ import annotations

import json
import logging
import re as _re
import uuid
from datetime import datetime, timezone

import db as _db
from models import SessionHistory
from services.prompt_library import get_active_prompt
from services.redact import redact
from services.security_engine import validate_fact_for_storage, MAX_FACTS_PER_CONVERSATION

from ._context import _scrub_deflections
from .write_gate import (
    MemoryWriteGate, _trust_scan, _write_to_pending_review,
)

log = logging.getLogger("iMakeAiTeams.memory.session_facts")

# ── Module-level counters (read via __getattr__ from services.memory) ────────
# health_monitor.py reaches in for these via ``from services.memory import
# _extract_attempts, _extract_failures``. The names live here, not on the
# façade, because they are MUTATED on every extract call — re-binding via
# a `from import` in __init__.py would freeze the values at import time.
_extract_attempts = 0
_extract_failures = 0


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


# Stopwords for the grounding check. Module-level constant so the set
# isn't reallocated on every fact.
_GROUNDING_STOPWORDS: frozenset[str] = frozenset({
    "the","a","an","is","are","was","were","it","in","on","to","for",
    "of","and","or","that","this","with","has","have","had","be","been",
    "not","but","they","their","them","he","she","his","her","we","our",
    "you","your","i","my","me","so","at","by","from","up","no","yes",
    "do","does","did","will","would","can","could","should","may","might",
    "about","just","also","very","much","more","some","any","all","each",
})


class _FactExtractor:
    """Owns the local-model fact-extraction pipeline.

    Instantiated by MemoryManager with the local client and write-gate
    instance. Public surface mirrors the legacy MemoryManager methods
    that the façade delegates to (extract_facts, _resolve_pending_facts).
    """

    def __init__(self, local_client, write_gate: MemoryWriteGate):
        self._local = local_client
        self._write_gate = write_gate

    def resolve_pending(self, conversation_id: str, user_message: str) -> None:
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

    def extract(
        self,
        conversation_id: str,
        user_msg: str,
        assistant_msg: str,
        history: SessionHistory,
    ) -> None:
        """Extract facts from an exchange via local model.

        Each fact goes through the grounding check, structural firewall,
        trust scan, and MINJA write gate before reaching session_facts.
        Failures are non-fatal — extraction is best-effort.
        """
        global _extract_attempts, _extract_failures
        if not self._local or not self._local.is_available():
            return
        _extract_attempts += 1
        self.resolve_pending(conversation_id, user_msg)
        try:
            system = get_active_prompt("fact_extractor")
            prompt = (
                f"User said: {user_msg[:500]}\n"
                f"Assistant said: {assistant_msg[:500]}\n"
            )
            result = self._local.chat(system, prompt, max_tokens=300)

            facts = self._parse_facts_json(
                result, allow_retry=True,
                user_msg=user_msg, assistant_msg=assistant_msg,
            )
            if facts is None:
                return

            # ── Fix 5: Grounding check ───────────────────────────────────────
            # Each fact must have meaningful keyword overlap with the source
            # messages. Prevents local model hallucinations from poisoning
            # long-term memory.
            source_text = (user_msg + " " + assistant_msg).lower()
            source_words = set(source_text.split())

            grounded_facts: list[str] = []
            for fact in facts[:3]:
                if not isinstance(fact, str) or not fact.strip():
                    continue
                # Strip deflection sentences before grounding so we score
                # only the substantive content of the fact.
                fact = _scrub_deflections(fact)
                if not fact:
                    continue
                fact_words = set(fact.lower().split())
                meaningful = fact_words - _GROUNDING_STOPWORDS
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
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,),
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

                fw_valid, fw_reason, _fw_attestation = validate_fact_for_storage(
                    fact_clean, conversation_id, extraction_method="local_model",
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
                if self._write_gate.gate_fact_write(conversation_id, fact_clean) == "pending_review":
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
                history.add(
                    "fact_extracted",
                    f"Extracted {len(inserted_facts)} facts: {inserted_facts}",
                )

            self._extract_triples(grounded_facts, conversation_id)

        except Exception as exc:
            _extract_failures += 1
            if _extract_attempts >= 20 and _extract_failures / _extract_attempts > 0.5:
                log.warning("Memory fact extraction failing frequently.")
            log.debug(f"Fact extraction failed: {exc}")

    def _extract_triples(self, facts: list, conversation_id: str) -> None:
        """Decompose grounded facts into (subject, predicate, object) triples."""
        if not facts or not self._local or not self._local.is_available():
            return
        try:
            prompt = _TRIPLE_PROMPT.format(facts="\n".join(f"- {f}" for f in facts))
            raw = self._local.chat("", prompt, max_tokens=500)
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
            if not allow_retry or not self._local or not self._local.is_available():
                return None
            try:
                system = get_active_prompt("fact_extractor")
                retry_result = self._local.chat(
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
