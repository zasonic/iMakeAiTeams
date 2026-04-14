"""
services/context_compressor.py

Three-tier context compression to keep conversations within token limits.

Tier 1 — Micro-compact (every turn, zero API cost):
    Truncate old assistant tool-result blocks and long code fences
    in the message history to short placeholders.

Tier 2 — Session-memory compact (uses local model):
    Summarize the oldest N messages into a compact paragraph,
    replace them with a single system note.

Tier 3 — Full compact (uses Claude or local model):
    When token usage > 80%, generate a high-quality summary of the
    entire conversation and restart with just that summary.

Circuit breaker: if compression fails 3× in a row, stop trying
until manually reset or a new conversation starts.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("MyAIEnv.compressor")


# ── Result types ──────────────────────────────────────────────────────────────

class CompactLayer(str, Enum):
    MICRO   = "micro"
    SESSION = "session"
    FULL    = "full"


@dataclass
class CompactResult:
    layer:          CompactLayer
    success:        bool
    messages_before: int = 0
    messages_after:  int = 0
    chars_saved:     int = 0
    detail:          str = ""


# ── Configuration ─────────────────────────────────────────────────────────────

# Micro-compact
KEEP_RECENT_MESSAGES      = 10   # never truncate the last N messages
TRUNCATE_CONTENT_ABOVE    = 300  # chars — only truncate blocks longer than this
TRUNCATED_PLACEHOLDER     = "[Truncated — {n} chars omitted]"

# Patterns that indicate tool/code output safe to truncate
_TRUNCATABLE_PATTERNS = [
    re.compile(r"```[\s\S]{300,}?```", re.DOTALL),          # long code fences
    re.compile(r"\[Tool result:[\s\S]{200,}?\]", re.DOTALL), # tool results
    re.compile(r"<tool_result>[\s\S]{200,}?</tool_result>", re.DOTALL),
]

# Session-memory compact
SESSION_COMPACT_TRIGGER   = 30   # compress when history exceeds this many messages
SESSION_KEEP_RECENT       = 10   # keep the last N messages verbatim
SESSION_SUMMARY_MAX_CHARS = 1500 # cap input to summariser

# Full compact
FULL_COMPACT_TOKEN_PCT    = 0.80 # trigger at 80% of estimated context budget
ESTIMATED_CONTEXT_BUDGET  = 180_000  # rough char budget (~45k tokens)

# Circuit breaker
MAX_CONSECUTIVE_FAILURES  = 3


# ── Micro-compact ─────────────────────────────────────────────────────────────

def micro_compact(messages: list[dict]) -> CompactResult:
    """
    In-place truncation of old, long content blocks.
    Returns a CompactResult describing what changed.
    """
    if not messages:
        return CompactResult(CompactLayer.MICRO, True, 0, 0, detail="empty")

    total      = len(messages)
    safe_zone  = max(0, total - KEEP_RECENT_MESSAGES)
    chars_saved = 0
    truncated   = 0

    for i in range(safe_zone):
        msg = messages[i]
        content = msg.get("content", "")
        if not content or len(content) <= TRUNCATE_CONTENT_ABOVE:
            continue

        new_content = content
        for pattern in _TRUNCATABLE_PATTERNS:
            def _replacer(m):
                nonlocal chars_saved
                original = m.group(0)
                if len(original) > TRUNCATE_CONTENT_ABOVE:
                    chars_saved += len(original)
                    return TRUNCATED_PLACEHOLDER.format(n=len(original))
                return original
            new_content = pattern.sub(_replacer, new_content)

        # Also truncate very long plain-text assistant messages
        if msg.get("role") == "assistant" and len(new_content) > 2000:
            excess = len(new_content) - 800
            chars_saved += excess
            new_content = (
                new_content[:400]
                + f"\n\n{TRUNCATED_PLACEHOLDER.format(n=excess)}\n\n"
                + new_content[-400:]
            )

        if new_content != content:
            messages[i] = {**msg, "content": new_content}
            truncated += 1

    return CompactResult(
        layer=CompactLayer.MICRO,
        success=truncated > 0,
        messages_before=total,
        messages_after=total,
        chars_saved=chars_saved,
        detail=f"truncated {truncated} messages, saved {chars_saved} chars",
    )


# ── Session-memory compact ───────────────────────────────────────────────────

_SESSION_SUMMARY_PROMPT = (
    "Summarize this conversation history in 3–5 concise sentences. "
    "Preserve: decisions made, key facts, user preferences, open questions, "
    "specific names/numbers/dates. Do NOT include pleasantries or filler."
)


def session_compact(
    messages:     list[dict],
    local_client,
) -> CompactResult:
    """
    Summarize older messages using the local model and replace them with
    a single summary note.  Returns a new list (does not mutate in-place).
    """
    total = len(messages)
    if total <= SESSION_COMPACT_TRIGGER:
        return CompactResult(CompactLayer.SESSION, False, total, total,
                             detail="below trigger threshold")

    if not local_client or not local_client.is_available():
        return CompactResult(CompactLayer.SESSION, False, total, total,
                             detail="no local model available")

    # Split into old (to summarize) and recent (to keep)
    split = max(0, total - SESSION_KEEP_RECENT)
    old_msgs   = messages[:split]
    recent     = messages[split:]

    # Build text for summariser — cap total chars
    lines = []
    char_count = 0
    for m in old_msgs:
        snippet = f"{m.get('role', '?').upper()}: {(m.get('content') or '')[:300]}"
        if char_count + len(snippet) > SESSION_SUMMARY_MAX_CHARS:
            break
        lines.append(snippet)
        char_count += len(snippet)

    try:
        summary = local_client.chat(
            _SESSION_SUMMARY_PROMPT,
            "\n".join(lines),
            max_tokens=400,
        )
        summary_msg = {
            "role": "system",
            "content": f"[Earlier conversation summary: {summary.strip()}]",
        }
        new_history = [summary_msg] + recent
        return CompactResult(
            layer=CompactLayer.SESSION,
            success=True,
            messages_before=total,
            messages_after=len(new_history),
            chars_saved=sum(len(m.get("content", "")) for m in old_msgs),
            detail=f"summarized {len(old_msgs)} messages → 1 summary",
        )
    except Exception as exc:
        log.warning("Session compact failed: %s", exc)
        return CompactResult(CompactLayer.SESSION, False, total, total,
                             detail=str(exc))


# ── Full compact ──────────────────────────────────────────────────────────────

_FULL_SUMMARY_PROMPT = (
    "You are compressing a long conversation to save context space. "
    "Write a detailed summary that captures ALL important information:\n"
    "- Every decision, conclusion, or commitment made\n"
    "- All specific facts, names, numbers, dates, URLs mentioned\n"
    "- The user's current goal and where they left off\n"
    "- Any open questions or pending items\n"
    "Be thorough — this summary REPLACES the original conversation."
)


def full_compact(
    messages:      list[dict],
    claude_client  = None,
    local_client   = None,
) -> CompactResult:
    """
    Generate a thorough summary using the best available model and
    replace the entire history with it.
    """
    total = len(messages)
    if total <= SESSION_KEEP_RECENT:
        return CompactResult(CompactLayer.FULL, False, total, total,
                             detail="too few messages")

    # Build conversation text (capped to avoid context overflow in the
    # summary call itself)
    lines = []
    char_count = 0
    for m in messages:
        snippet = f"{m.get('role', '?').upper()}: {(m.get('content') or '')[:500]}"
        if char_count + len(snippet) > 12_000:
            lines.append("… [earlier messages omitted for brevity] …")
            break
        lines.append(snippet)
        char_count += len(snippet)

    conversation_text = "\n".join(lines)

    # Try Claude first, fall back to local
    summary = None
    try:
        if claude_client:
            result = claude_client.chat_multi_turn(
                _FULL_SUMMARY_PROMPT,
                [{"role": "user", "content": conversation_text}],
                max_tokens=1000,
            )
            summary = result.get("text", "")
    except Exception as exc:
        log.warning("Full compact (Claude) failed: %s", exc)

    if not summary and local_client and local_client.is_available():
        try:
            summary = local_client.chat(
                _FULL_SUMMARY_PROMPT,
                conversation_text,
                max_tokens=800,
            )
        except Exception as exc:
            log.warning("Full compact (local) failed: %s", exc)

    if not summary:
        return CompactResult(CompactLayer.FULL, False, total, total,
                             detail="all models failed")

    # Keep only the last 2 messages + the summary
    keep_last = messages[-2:] if len(messages) >= 2 else messages[-1:]
    summary_msg = {
        "role": "system",
        "content": f"[Full conversation summary: {summary.strip()}]",
    }
    new_history = [summary_msg] + keep_last

    return CompactResult(
        layer=CompactLayer.FULL,
        success=True,
        messages_before=total,
        messages_after=len(new_history),
        chars_saved=sum(len(m.get("content", "")) for m in messages),
        detail=f"full-compacted {total} messages → {len(new_history)}",
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ContextCompressor:
    """
    Wired into ChatOrchestrator. Call `auto_compact()` after each exchange.
    Runs micro-compact every time, then escalates if needed.
    """

    def __init__(self, local_client=None, claude_client=None):
        self.local_client  = local_client
        self.claude_client = claude_client
        self._consecutive_failures = 0
        self._circuit_broken       = False
        self._last_compact_time    = 0.0
        self._compact_cooldown     = 30.0  # seconds between session/full compacts

    def reset_circuit_breaker(self):
        self._consecutive_failures = 0
        self._circuit_broken       = False
        log.info("Context compressor circuit breaker reset")

    def auto_compact(self, messages: list[dict]) -> tuple[list[dict], CompactResult | None]:
        """
        Run compression on a message list.  Returns (possibly_new_list, result).
        The returned list may be the same object (micro) or a new one (session/full).
        """
        if not messages:
            return messages, None

        # Always run micro-compact (zero cost)
        micro_result = micro_compact(messages)
        if micro_result.chars_saved > 0:
            log.info("Micro-compact: %s", micro_result.detail)

        # Check if deeper compaction is needed
        total_chars = sum(len(m.get("content", "")) for m in messages)
        msg_count   = len(messages)

        # Circuit breaker
        if self._circuit_broken:
            return messages, micro_result

        # Cooldown — don't spam compaction
        now = time.monotonic()
        if now - self._last_compact_time < self._compact_cooldown:
            return messages, micro_result

        # Tier 2: session compact
        if msg_count > SESSION_COMPACT_TRIGGER:
            # Pre-compaction memory flush (OpenClaw pattern):
            # Extract durable facts BEFORE summarization destroys them
            if self.local_client and hasattr(self.local_client, "is_available"):
                try:
                    if self.local_client.is_available():
                        old_text = "\n".join(
                            f"{m.get('role','?')}: {(m.get('content') or '')[:200]}"
                            for m in messages[:SESSION_COMPACT_TRIGGER]
                        )
                        flush_result = self.local_client.chat(
                            "Extract 1-3 important facts worth remembering from this conversation. "
                            "Return ONLY a JSON array of short strings. If nothing notable, return [].",
                            old_text[:3000],
                            max_tokens=200,
                        )
                        if flush_result:
                            log.debug("Pre-compaction flush: %s", flush_result[:100])
                            # Facts will be picked up by memory.extract_facts on next turn
                except Exception:
                    pass  # best-effort, never block compaction

            split = max(0, msg_count - SESSION_KEEP_RECENT)
            old_msgs = messages[:split]
            recent   = messages[split:]
            lines = []
            for m in old_msgs:
                lines.append(f"{m.get('role','?').upper()}: {(m.get('content') or '')[:300]}")
            result = session_compact(messages, self.local_client)
            if result.success:
                self._consecutive_failures = 0
                self._last_compact_time = now
                log.info("Session compact: %s", result.detail)
                try:
                    summary = self.local_client.chat(
                        _SESSION_SUMMARY_PROMPT,
                        "\n".join(lines[:20]),
                        max_tokens=400,
                    )
                    summary_msg = {"role": "system",
                                   "content": f"[Earlier conversation summary: {summary.strip()}]"}
                    return [summary_msg] + recent, result
                except Exception:
                    pass
            else:
                self._consecutive_failures += 1

        # Tier 3: full compact
        if total_chars > ESTIMATED_CONTEXT_BUDGET:
            result = full_compact(messages, self.claude_client, self.local_client)
            if result.success:
                self._consecutive_failures = 0
                self._last_compact_time = now
                log.info("Full compact: %s", result.detail)
                keep_last = messages[-2:] if len(messages) >= 2 else messages[-1:]
                # Re-generate summary (result doesn't carry the text)
                try:
                    lines = []
                    for m in messages:
                        lines.append(f"{m.get('role','?').upper()}: {(m.get('content') or '')[:500]}")
                    conv_text = "\n".join(lines[:30])
                    if self.claude_client:
                        r = self.claude_client.chat_multi_turn(
                            _FULL_SUMMARY_PROMPT,
                            [{"role": "user", "content": conv_text}],
                            max_tokens=1000,
                        )
                        summary = r.get("text", "")
                    else:
                        summary = self.local_client.chat(
                            _FULL_SUMMARY_PROMPT, conv_text, max_tokens=800)
                    summary_msg = {"role": "system",
                                   "content": f"[Full conversation summary: {summary.strip()}]"}
                    return [summary_msg] + keep_last, result
                except Exception:
                    pass
            else:
                self._consecutive_failures += 1

        # Check circuit breaker
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self._circuit_broken = True
            log.warning("Context compressor circuit breaker tripped after %d failures",
                        self._consecutive_failures)

        return messages, micro_result
