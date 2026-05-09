"""tests/camel/test_pipeline.py — End-to-end privileged + quarantined LLM split.

Covers the orchestration layer above the interpreter:
  - The privileged client gets P_LLM_SYSTEM_PROMPT + the user message.
  - The quarantined client gets Q_LLM_SYSTEM_PROMPT + the chunks payload.
  - When the privileged plan is wrapped in ```python fences, they're
    stripped before parsing.
  - The interpreter runs to completion and the dict's ``output_text``
    matches the plan's terminal value.
  - When the privileged client returns malformed Python, the result
    surfaces a ``plan parse error: ...`` reason instead of crashing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services.camel.capabilities import CapabilityTaggedResult, capabilities_for_tool
from services.camel.pipeline import camel_plan_and_execute
from services.camel.prompts import P_LLM_SYSTEM_PROMPT, Q_LLM_SYSTEM_PROMPT


def _client(text: str):
    """Build a MagicMock that exposes chat_unified the way the production
    clients do (returns dict with ``text`` key)."""
    c = MagicMock()
    c.chat_unified.return_value = {"text": text, "input_tokens": 5, "output_tokens": 5}
    return c


def _stub_tool_exec(canned: dict | None = None):
    canned = dict(canned or {})

    def _exec(name, args, kwargs):
        value = canned.get(name, f"<tool {name}>")
        return CapabilityTaggedResult(
            value=value, capabilities=capabilities_for_tool(name),
        )
    return _exec


class TestPrivilegedSurface:
    def test_privileged_client_called_with_system_prompt_and_message(self):
        priv = _client("output = 'hi'\n")
        quar = _client("noop")
        camel_plan_and_execute(
            user_message="What's the weather?",
            retrieved_chunks=[],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        assert priv.chat_unified.called
        # First positional arg is the system prompt.
        args, kwargs = priv.chat_unified.call_args
        assert args[0] == P_LLM_SYSTEM_PROMPT
        # Second arg is the messages list with the user message.
        messages = args[1]
        assert any("weather" in m["content"] for m in messages)

    def test_quarantined_only_called_when_plan_invokes_it(self):
        priv = _client("output = 'no quarantined'\n")
        quar = _client("noop")
        camel_plan_and_execute(
            user_message="hi",
            retrieved_chunks=["chunk-A"],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        # Plan didn't call quarantined_llm() so the quarantined client
        # was never invoked.
        assert not quar.chat_unified.called


class TestQuarantinedSurface:
    def test_quarantined_called_when_plan_uses_q_llm(self):
        priv = _client(
            "summary = quarantined_llm('summarise', retrieved)\n"
            "output = summary\n"
        )
        quar = _client("CHUNKS_SUMMARY")
        # The plan refers to ``retrieved`` but doesn't bind it; the
        # interpreter will raise the unknown-name error. Instead pass the
        # source as a literal string so we can test the quarantined call.
        priv = _client(
            "summary = quarantined_llm('summarise', 'use retrieved chunks')\n"
            "output = summary\n"
        )
        result = camel_plan_and_execute(
            user_message="ask",
            retrieved_chunks=["doc1", "doc2"],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        assert quar.chat_unified.called
        # System prompt is Q_LLM_SYSTEM_PROMPT.
        args, _ = quar.chat_unified.call_args
        assert args[0] == Q_LLM_SYSTEM_PROMPT
        # Quarantined output flowed through to the plan output.
        assert result["output_text"] == "CHUNKS_SUMMARY"

    def test_default_source_falls_back_to_retrieved_chunks(self):
        # When the plan calls quarantined_llm with a source that resolves
        # to None, the pipeline closure substitutes the configured
        # retrieved_chunks. We verify by checking the outgoing user message
        # for both chunk strings.
        priv = _client(
            "summary = quarantined_llm('q', None)\n"
            "output = summary\n"
        )
        quar = _client("ok")
        camel_plan_and_execute(
            user_message="ask",
            retrieved_chunks=["alpha", "beta"],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        args, _ = quar.chat_unified.call_args
        user_text = args[1][0]["content"]
        assert "alpha" in user_text
        assert "beta" in user_text


class TestFenceStripping:
    def test_python_code_fence_stripped(self):
        priv = _client("```python\noutput = 'fenced'\n```")
        quar = _client("noop")
        result = camel_plan_and_execute(
            user_message="hi",
            retrieved_chunks=[],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        assert result["output_text"] == "fenced"
        assert "```" not in result["plan_source"]

    def test_bare_fence_also_stripped(self):
        priv = _client("```\noutput = 'bare'\n```")
        quar = _client("noop")
        result = camel_plan_and_execute(
            user_message="hi",
            retrieved_chunks=[],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        assert result["output_text"] == "bare"


class TestEndToEnd:
    def test_combined_tool_and_quarantined_plan(self):
        priv = _client(
            "facts = rag_search('aurora deadlines')\n"
            "summary = quarantined_llm('list deadlines', facts)\n"
            "output = f'Aurora deadlines:\\n{summary}'\n"
        )
        quar = _client("- 2026-06-01: launch")

        rag_result = ["doc-aurora-1", "doc-aurora-2"]

        def _exec(name, args, kwargs):
            if name == "rag_search":
                return CapabilityTaggedResult(
                    value=rag_result, capabilities=capabilities_for_tool(name),
                )
            return CapabilityTaggedResult(
                value=None, capabilities=capabilities_for_tool(name),
            )

        result = camel_plan_and_execute(
            user_message="When is Aurora?",
            retrieved_chunks=["other_doc"],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_exec,
        )
        assert result["error"] == ""
        assert "Aurora deadlines" in result["output_text"]
        assert "2026-06-01" in result["output_text"]
        assert result["capability_violations"] == 0


class TestMalformedPlan:
    def test_syntax_error_returns_error_message(self):
        priv = _client("output = (")  # unclosed paren
        quar = _client("noop")
        result = camel_plan_and_execute(
            user_message="hi",
            retrieved_chunks=[],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        assert "plan parse error" in result["error"]
        assert result["output_text"] == ""

    def test_empty_plan_returns_error(self):
        priv = _client("")
        quar = _client("noop")
        result = camel_plan_and_execute(
            user_message="hi",
            retrieved_chunks=[],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        assert "empty plan" in result["error"]

    def test_disallowed_construct_returns_error(self):
        priv = _client("import os\noutput = 'pwned'\n")
        quar = _client("noop")
        result = camel_plan_and_execute(
            user_message="hi",
            retrieved_chunks=[],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        assert "plan parse error" in result["error"]

    def test_capability_violation_recorded_in_result(self):
        priv = _client(
            "fn = quarantined_llm('extract', 'data')\n"
            "output = fn('arg')\n"
        )
        quar = _client("ANY_NAME")
        result = camel_plan_and_execute(
            user_message="hi",
            retrieved_chunks=[],
            privileged_client=priv,
            quarantined_client=quar,
            tool_executor=_stub_tool_exec(),
        )
        assert result["capability_violations"] >= 1
        assert "capability" in result["error"].lower()
