"""
tests/test_router.py — Tests for the task complexity router.

Run: pytest tests/test_router.py -v
"""

import pytest
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.router import TaskRouter


@pytest.fixture
def mock_local():
    local = MagicMock()
    local.is_available.return_value = True
    return local


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.get.return_value = ""
    return settings


@pytest.fixture
def router(mock_local, mock_settings):
    return TaskRouter(mock_local, mock_settings)


class TestExplicitOverrides:
    def test_at_claude(self, router):
        result = router.classify("@claude explain this")
        assert result.model == "claude"

    def test_use_claude(self, router):
        result = router.classify("use claude for this analysis")
        assert result.model == "claude"

    def test_at_local(self, router):
        result = router.classify("@local what time is it")
        assert result.model == "local"

    def test_use_local(self, router):
        result = router.classify("use local for formatting")
        assert result.model == "local"


class TestFallbacks:
    def test_local_unavailable_defaults_to_claude(self, router):
        router.local.is_available.return_value = False
        result = router.classify("hello")
        assert result.model == "claude"

    def test_routing_disabled_defaults_to_claude(self, router):
        router.set_enabled(False)
        result = router.classify("simple greeting")
        assert result.model == "claude"

    def test_parse_failure_defaults_to_claude(self, router):
        router.local.chat.return_value = "I don't understand the format you want"
        result = router.classify("analyze this complex dataset")
        assert result.model == "claude"

    def test_exception_defaults_to_claude(self, router):
        router.local.chat.side_effect = ConnectionError("timeout")
        result = router.classify("hello")
        assert result.model == "claude"


class TestLocalClassification:
    def test_valid_json_response(self, router):
        router.local.chat.return_value = '{"model":"local","complexity":"simple","reasoning":"greeting"}'
        result = router.classify("hi there, how are you?")
        assert result.model == "local"
        assert result.complexity == "simple"

    def test_claude_classification(self, router):
        router.local.chat.return_value = '{"model":"claude","complexity":"complex","reasoning":"needs deep analysis"}'
        result = router.classify("Compare the economic policies of three countries")
        assert result.model == "claude"
        assert result.complexity == "complex"

    def test_json_with_backticks(self, router):
        router.local.chat.return_value = '```json\n{"model":"local","complexity":"simple","reasoning":"basic"}\n```'
        result = router.classify("format this text")
        assert result.model == "local"


class TestConfidenceScoring:
    """v4.1 — Uncertainty-Aware Routing tests."""

    def test_confidence_parsed_from_response(self, router):
        router.local.chat.return_value = (
            '{"model":"local","complexity":"simple","reasoning":"clear",'
            '"confidence":0.9,"needs_context":false}'
        )
        result = router.classify("what is 2+2?")
        assert result.model == "local"
        assert result.confidence == 0.9
        assert result.needs_context is False

    def test_low_confidence_escalates_to_claude(self, router):
        router.local.chat.return_value = (
            '{"model":"local","complexity":"medium","reasoning":"not sure",'
            '"confidence":0.3,"needs_context":false}'
        )
        result = router.classify("explain the implications of this contract")
        assert result.model == "claude"
        assert result.confidence == 0.3
        assert "escalated" in result.reasoning.lower() or "low confidence" in result.reasoning.lower()

    def test_needs_context_propagated(self, router):
        router.local.chat.return_value = (
            '{"model":"claude","complexity":"complex","reasoning":"domain-specific",'
            '"confidence":0.6,"needs_context":true}'
        )
        result = router.classify("what did the Q3 report say about margins?")
        assert result.needs_context is True

    def test_missing_confidence_defaults_to_0_8(self, router):
        router.local.chat.return_value = (
            '{"model":"local","complexity":"simple","reasoning":"greeting"}'
        )
        result = router.classify("hello")
        assert result.confidence == 0.8  # default from from_json

    def test_parse_failure_sets_confidence_zero(self, router):
        router.local.chat.return_value = "not json at all"
        result = router.classify("something")
        assert result.model == "claude"
        assert result.confidence == 0.0

    def test_low_confidence_sets_needs_context(self, router):
        """Confidence below CONTEXT_EXPANSION_THRESHOLD should auto-set needs_context."""
        router.local.chat.return_value = (
            '{"model":"claude","complexity":"complex","reasoning":"unsure",'
            '"confidence":0.4,"needs_context":false}'
        )
        result = router.classify("what were our previous findings on this topic?")
        assert result.needs_context is True

    def test_high_confidence_no_escalation(self, router):
        router.local.chat.return_value = (
            '{"model":"local","complexity":"simple","reasoning":"clear greeting",'
            '"confidence":0.95,"needs_context":false}'
        )
        result = router.classify("hey there!")
        assert result.model == "local"
        assert result.confidence == 0.95
        assert result.needs_context is False

    def test_confidence_clamped_to_valid_range(self, router):
        router.local.chat.return_value = (
            '{"model":"local","complexity":"simple","reasoning":"test",'
            '"confidence":1.5}'
        )
        result = router.classify("hello")
        assert result.confidence <= 1.0

        router.local.chat.return_value = (
            '{"model":"local","complexity":"simple","reasoning":"test",'
            '"confidence":-0.5}'
        )
        result = router.classify("hello")
        assert result.confidence >= 0.0

    def test_explicit_override_has_full_confidence(self, router):
        result = router.classify("@claude analyze this deeply")
        assert result.confidence == 1.0

        result = router.classify("@local format this")
        assert result.confidence == 1.0
