"""
services/redact.py — Deterministic credential redaction.

Scrubs API keys, tokens, emails, card numbers, and JWTs from text
before disk writes. Pattern ordering matters: vendor-specific patterns
match before generic catches so the more informative label wins
(e.g. "[REDACTED_AWS_KEY]" beats "[REDACTED_HEX]").

Design principle: structural regex, no LLM. Runs in <1ms on typical
message lengths. Cannot be prompt-injected because it doesn't use a model.
"""

import re

_REDACTION_RULES: list[tuple[re.Pattern[str], str]] = [
    # Email addresses
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE), "[REDACTED_EMAIL]"),
    # Credit card numbers (13-19 digits, possibly separated by spaces/dashes)
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD]"),
    # AWS access keys
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    # Stripe keys
    (re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), "[REDACTED_STRIPE_KEY]"),
    # GitHub tokens
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "[REDACTED_GH_TOKEN]"),
    # Anthropic keys (must come before generic OpenAI sk- pattern)
    (re.compile(r"\bsk-ant-[A-Za-z0-9\-]{32,}\b"), "[REDACTED_ANTHROPIC_KEY]"),
    # OpenAI keys
    (re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"), "[REDACTED_OPENAI_KEY]"),
    # Google API keys
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOG_KEY]"),
    # Authorization headers
    (re.compile(r"Authorization:\s*Bearer\s+\S+", re.IGNORECASE), "Authorization: Bearer [REDACTED]"),
    (re.compile(r"Authorization:\s*Basic\s+[A-Za-z0-9+/=]+", re.IGNORECASE), "Authorization: Basic [REDACTED]"),
    # Generic vendor-prefix tokens (after vendor-specific rules)
    (re.compile(r"\b(AWS|GH|GCP|AZURE|xox[abpcr]-)[A-Za-z0-9_\-]{10,}\b", re.IGNORECASE), "[REDACTED_TOKEN]"),
    # JWTs (eyJ...)
    (re.compile(r"\b(?:eyJ[0-9A-Za-z._\-]+)\b"), "[REDACTED_JWT]"),
    # Keyword-anchored credentials (password=, secret=, token=, etc.)
    (re.compile(
        r"\b(pass(?:word)?|secret|token|apikey|api_key|"
        r"(?:refresh|access|id|oauth)_?token|session(?:_?id)?)\s*[:=]\s*\S+\b",
        re.IGNORECASE,
    ), r"\1=[REDACTED]"),
    # Long hex strings (32+ chars) — likely tokens/hashes
    (re.compile(r"\b[0-9A-Fa-f]{32,}\b"), "[REDACTED_HEX]"),
]


def redact(text: str) -> str:
    """Scrub credentials from text. Stateless, <1ms, no LLM."""
    if not text:
        return text
    result = text
    for pattern, replacement in _REDACTION_RULES:
        result = pattern.sub(replacement, result)
    return result
