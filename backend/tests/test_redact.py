"""
tests/test_redact.py — Upgrade 1: structural credential redaction.

Covers the success criteria from the spec:
  - Vendor-specific patterns produce informative labels (Anthropic, OpenAI, AWS)
  - Authorization headers redact the bearer/basic token while keeping the header
  - Plain text passes through unchanged
  - End-to-end: a credential pasted into chat is redacted in the persisted
    SQLite row, not in the in-flight ChatResult.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from services.redact import redact


class TestRedactPatterns:
    def test_anthropic_key(self):
        out = redact("my key is sk-ant-abc123def456ghi789jkl012mno345pqr678")
        assert out == "my key is [REDACTED_ANTHROPIC_KEY]"

    def test_openai_key(self):
        out = redact("sk-abcdef0123456789abcdef0123456789abcdef test")
        assert "[REDACTED_OPENAI_KEY]" in out
        assert "sk-abcdef" not in out

    def test_aws_access_key(self):
        out = redact("AKIAIOSFODNN7EXAMPLE token live")
        assert "[REDACTED_AWS_KEY]" in out
        assert "AKIA" not in out

    def test_stripe_key(self):
        # Restricted-key test prefix — same regex shape as a Stripe live key
        # but won't trip GitHub's secret scanner on commit.
        out = redact("rk_test_" + "a" * 24 + " ok")
        assert "[REDACTED_STRIPE_KEY]" in out

    def test_google_api_key(self):
        out = redact("AIza" + "B" * 35 + " key")
        assert "[REDACTED_GOOG_KEY]" in out

    def test_authorization_bearer(self):
        out = redact("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")
        assert "Bearer [REDACTED]" in out
        assert "eyJ" not in out

    def test_authorization_basic(self):
        out = redact("Authorization: Basic dXNlcjpwYXNzd29yZA==")
        assert "Basic [REDACTED]" in out

    def test_email(self):
        out = redact("ping me at alice@example.com today")
        assert "[REDACTED_EMAIL]" in out
        assert "alice@" not in out

    def test_jwt(self):
        out = redact("token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signature")
        # Either the JWT pattern or the keyword-anchored token rule wins
        assert ("[REDACTED_JWT]" in out) or ("token=[REDACTED]" in out)

    def test_keyword_anchored_password(self):
        out = redact("password=hunter2 in the config")
        assert "[REDACTED]" in out
        assert "hunter2" not in out

    def test_long_hex_string(self):
        out = redact("hash deadbeef0123456789cafebabe9876543210feedface")
        assert "[REDACTED_HEX]" in out

    def test_plain_text_unchanged(self):
        assert redact("hello world") == "hello world"
        assert redact("the quick brown fox jumps over the lazy dog") == \
            "the quick brown fox jumps over the lazy dog"

    def test_empty_input(self):
        assert redact("") == ""
        assert redact(None) is None

    def test_vendor_specific_beats_generic(self):
        # The Anthropic pattern must match before the long-hex / generic rules
        out = redact("sk-ant-" + "x" * 40)
        assert "[REDACTED_ANTHROPIC_KEY]" in out
        assert "[REDACTED_HEX]" not in out


class TestPersistedAssistantReplyIsRedacted:
    """
    End-to-end: a credential in the assistant's reply must be scrubbed when
    written to the messages table (and only there — the streamed copy and the
    returned ChatResult.text stay un-redacted).
    """

    def _seed_conv(self, in_memory_db, cid: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, 'test', ?, ?)", (cid, now, now),
        )
        in_memory_db.commit()

    def test_assistant_reply_redacted_in_db(self, in_memory_db, settings):
        from services.chat_orchestrator import ChatOrchestrator
        from services.memory import MemoryManager
        from models import RouteDecision

        secret = "sk-ant-abc123def456ghi789jkl012mno345pqr678"
        reply = f"Sure — your key is {secret} (please rotate it)"

        claude = MagicMock(_model="claude-test")
        claude.chat_multi_turn.return_value = {
            "text": reply, "input_tokens": 5, "output_tokens": 10,
        }
        claude.chat_unified.return_value = {
            "text": reply, "input_tokens": 5, "output_tokens": 10,
        }
        claude.client_name.return_value = "claude-test"
        local = MagicMock()
        local.is_available.return_value = False

        task_router = MagicMock()
        task_router.classify.return_value = RouteDecision(
            model="claude", reasoning="r", complexity="simple",
        )

        memory = MemoryManager(rag_index=None, semantic_search_mod=None,
                                local_client=local)
        orch = ChatOrchestrator(claude, local, task_router, memory, settings)

        cid = str(uuid.uuid4())
        self._seed_conv(in_memory_db, cid)
        result = orch.send(cid, "what's my anthropic key?")

        # Returned to caller un-redacted (so the streaming UI shows the truth).
        assert secret in result.text

        # But persisted copy is redacted.
        row = in_memory_db.fetchone(
            "SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant'",
            (cid,),
        )
        assert row is not None
        assert secret not in row["content"]
        assert "[REDACTED_ANTHROPIC_KEY]" in row["content"]
