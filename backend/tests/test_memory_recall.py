"""
tests/test_memory_recall.py — Layer 3: MemoryRecall module.

Covers the extraction's contract:
  1. recall() returns a MemoryRecallResult whose mem/mem_suffix/full_system
     are stitched the same way the inline orchestrator code used to stitch them.
  2. trim_for_complexity() caps RAG chunks per complexity tier and rebuilds
     the system prompt — the bug-1 regression case from Layer 1.
  3. Tool-restriction and MCP-tool blocks are present iff their inputs are.
  4. memory_recalled_event() shape matches the SSE the orchestrator emits.
  5. maybe_summarize() respects the manager's should_summarize verdict.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services.memory import MemoryContext
from services.memory_recall import (
    DEFAULT_MAX_CONTEXT_ITEMS,
    MAX_CONTEXT_ITEMS_BY_COMPLEXITY,
    MemoryRecall,
    MemoryRecallResult,
)


def _ctx(facts=None, rag=None, memories=None) -> MemoryContext:
    return MemoryContext(
        recent_messages=[],
        session_facts=facts or [],
        rag_chunks=rag or [],
        memories=memories or [],
    )


@pytest.fixture
def mem_manager():
    m = MagicMock()
    m.get_context.return_value = _ctx()
    m.should_summarize.return_value = False
    return m


@pytest.fixture
def recall(mem_manager, settings):
    return MemoryRecall(mem_manager, settings)


# ── recall() ─────────────────────────────────────────────────────────────────


def test_recall_returns_full_system_with_no_extras(recall, mem_manager):
    mem_manager.get_context.return_value = _ctx()
    result = recall.recall("conv1", "hi", "BASE PROMPT")
    assert isinstance(result, MemoryRecallResult)
    assert result.mem_suffix == ""
    assert result.full_system == "BASE PROMPT"


def test_recall_appends_mem_suffix(recall, mem_manager):
    mem_manager.get_context.return_value = _ctx(facts=["user is named Alex"])
    result = recall.recall("conv1", "hi", "BASE")
    assert "Alex" in result.mem_suffix
    assert result.full_system.startswith("BASE\n\n")
    assert "Alex" in result.full_system


def test_recall_injects_tool_restriction_when_allowed_tools_present(recall):
    result = recall.recall(
        "conv1", "hi", "BASE", allowed_tools=["read_file", "search"],
    )
    assert "## Tool Restrictions" in result.full_system
    assert "read_file, search" in result.full_system


def test_recall_omits_tool_restriction_when_allowed_tools_empty(recall):
    result = recall.recall("conv1", "hi", "BASE", allowed_tools=[])
    assert "Tool Restrictions" not in result.full_system


def test_recall_injects_mcp_block_when_registry_returns_tools(mem_manager, settings):
    registry = MagicMock()
    registry.get_tools_for_tags.return_value = [
        {"name": "fetch", "description": "fetch a URL"},
        {"name": "shell", "description": "run a shell command"},
    ]
    rec = MemoryRecall(mem_manager, settings, registry)
    agent = {"skills": '[{"name": "web"}]'}
    result = rec.recall("conv1", "hi", "BASE", agent=agent)
    assert "Available External Tools" in result.full_system
    assert "**fetch**" in result.full_system
    registry.get_tools_for_tags.assert_called_once_with(["web"])


def test_recall_skips_mcp_block_when_registry_is_none(recall):
    agent = {"skills": '[{"name": "web"}]'}
    result = recall.recall("conv1", "hi", "BASE", agent=agent)
    assert "Available External Tools" not in result.full_system


def test_recall_omits_mcp_block_when_registry_returns_no_tools(mem_manager, settings):
    """No tools matched → registry is consulted but the MCP block is not injected."""
    registry = MagicMock()
    registry.get_tools_for_tags.return_value = []
    rec = MemoryRecall(mem_manager, settings, registry)
    agent = {"skills": '[{"name": "web"}]'}
    result = rec.recall("conv1", "hi", "BASE", agent=agent)
    assert "Available External Tools" not in result.full_system


# ── trim_for_complexity() ────────────────────────────────────────────────────


def test_trim_caps_rag_chunks_for_simple(recall, mem_manager):
    mem_manager.get_context.return_value = _ctx(rag=[f"c{i}" for i in range(20)])
    result = recall.recall("conv1", "hi", "BASE")
    trimmed = recall.trim_for_complexity(result, "simple", "BASE")
    assert len(trimmed.mem.rag_chunks) == MAX_CONTEXT_ITEMS_BY_COMPLEXITY["simple"]


def test_trim_uses_default_for_unknown_complexity(recall, mem_manager):
    mem_manager.get_context.return_value = _ctx(rag=[f"c{i}" for i in range(20)])
    result = recall.recall("conv1", "hi", "BASE")
    trimmed = recall.trim_for_complexity(result, "weird-tier", "BASE")
    assert len(trimmed.mem.rag_chunks) == DEFAULT_MAX_CONTEXT_ITEMS


def test_trim_is_noop_when_under_cap(recall, mem_manager):
    mem_manager.get_context.return_value = _ctx(rag=["only one"])
    result = recall.recall("conv1", "hi", "BASE")
    before = result.full_system
    trimmed = recall.trim_for_complexity(result, "simple", "BASE")
    assert trimmed.full_system == before
    assert len(trimmed.mem.rag_chunks) == 1


def test_trim_rebuilds_system_prompt_with_tool_restriction(recall, mem_manager):
    """Bug 1 regression: post-trim system prompt must still carry tool restrictions."""
    mem_manager.get_context.return_value = _ctx(rag=[f"c{i}" for i in range(20)])
    result = recall.recall("conv1", "hi", "BASE", allowed_tools=["read"])
    trimmed = recall.trim_for_complexity(
        result, "simple", "BASE", allowed_tools=["read"],
    )
    assert "Tool Restrictions" in trimmed.full_system
    assert "read" in trimmed.full_system


# ── memory_recalled_event ────────────────────────────────────────────────────


def test_memory_recalled_event_shape(recall):
    mem = _ctx(facts=["a", "b"], rag=["x"], memories=["m1", "m2", "m3"])
    payload = MemoryRecall.memory_recalled_event(mem)
    assert payload == {"facts_count": 2, "rag_chunks": 1, "memories": 3}


# ── maybe_summarize ──────────────────────────────────────────────────────────


def test_maybe_summarize_fires_when_manager_says_yes(recall, mem_manager):
    mem_manager.should_summarize.return_value = True
    recall.maybe_summarize("conv1")
    mem_manager.summarize_buffer.assert_called_once_with("conv1")


def test_maybe_summarize_skips_when_manager_says_no(recall, mem_manager):
    mem_manager.should_summarize.return_value = False
    recall.maybe_summarize("conv1")
    mem_manager.summarize_buffer.assert_not_called()


def test_maybe_summarize_swallows_exceptions(recall, mem_manager):
    mem_manager.should_summarize.side_effect = RuntimeError("boom")
    # Must not raise — summarisation is best-effort.
    recall.maybe_summarize("conv1")
