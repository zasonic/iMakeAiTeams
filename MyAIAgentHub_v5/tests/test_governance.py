"""
tests/test_governance.py — Unit tests for GovernanceEngine.

Covers: tool allowlist, forbidden tools, policy precedence,
token budget enforcement, and TTL-based counter eviction.
"""

import time
import pytest

from services.governance import GovernanceEngine, AgentPolicy


@pytest.fixture
def engine():
    return GovernanceEngine()


# ── Forbidden tools ───────────────────────────────────────────────────────────

def test_forbidden_tool_denied(engine):
    verdict = engine.check_tool_call("rm_recursive", task_key="t1")
    assert not verdict.allowed
    assert "forbidden" in verdict.reason.lower()


def test_allowed_tool_passes(engine):
    verdict = engine.check_tool_call("file_read", task_key="t1")
    assert verdict.allowed


# ── Allowlist enforcement ─────────────────────────────────────────────────────

def test_allowlist_blocks_unlisted_tool():
    e = GovernanceEngine()
    e._policies["strict"] = AgentPolicy(
        agent_role="strict_role",
        allowed_tools=["file_read", "file_grep"],
    )
    e._rebuild_indexes()
    verdict = e.check_tool_call("shell_exec", agent_role="strict_role", task_key="t1")
    assert not verdict.allowed
    assert "allowed list" in verdict.reason.lower()


def test_allowlist_permits_listed_tool():
    e = GovernanceEngine()
    e._policies["strict"] = AgentPolicy(
        agent_role="strict_role",
        allowed_tools=["file_read", "file_grep"],
    )
    e._rebuild_indexes()
    verdict = e.check_tool_call("file_read", agent_role="strict_role", task_key="t1")
    assert verdict.allowed


# ── Policy precedence ─────────────────────────────────────────────────────────

def test_agent_id_policy_overrides_role():
    e = GovernanceEngine()
    e._policies["by_id"] = AgentPolicy(
        agent_id="agent-001",
        max_tool_calls=1,
    )
    e._rebuild_indexes()
    # First call allowed
    v1 = e.check_tool_call("file_read", agent_id="agent-001", task_key="tid")
    assert v1.allowed
    # Second call exceeds budget of 1
    v2 = e.check_tool_call("file_read", agent_id="agent-001", task_key="tid")
    assert not v2.allowed


def test_role_policy_applied_when_no_id_match():
    e = GovernanceEngine()
    # worker policy has max_tool_calls=50, but forbidden_tools includes git_checkout
    verdict = e.check_tool_call("git_checkout", agent_role="worker", task_key="t1")
    assert not verdict.allowed


# ── Tool call budget ──────────────────────────────────────────────────────────

def test_tool_budget_exhausted():
    e = GovernanceEngine()
    e._policies["tiny"] = AgentPolicy(agent_role="tiny_role", max_tool_calls=2)
    e._rebuild_indexes()
    e.check_tool_call("file_read", agent_role="tiny_role", task_key="budget-task")
    e.check_tool_call("file_read", agent_role="tiny_role", task_key="budget-task")
    verdict = e.check_tool_call("file_read", agent_role="tiny_role", task_key="budget-task")
    assert not verdict.allowed
    assert "exhausted" in verdict.reason.lower()


def test_tool_budget_resets_after_reset_call():
    e = GovernanceEngine()
    e._policies["tiny"] = AgentPolicy(agent_role="tiny_role", max_tool_calls=1)
    e._rebuild_indexes()
    e.check_tool_call("file_read", agent_role="tiny_role", task_key="budget-task")
    e.reset_task_counters("budget-task")
    verdict = e.check_tool_call("file_read", agent_role="tiny_role", task_key="budget-task")
    assert verdict.allowed


# ── Token budget ──────────────────────────────────────────────────────────────

def test_token_budget_within_limit(engine):
    engine._policies["default"].max_tokens = 1000
    verdict = engine.check_token_budget(500, task_key="tok-task")
    assert verdict.allowed


def test_token_budget_exceeded(engine):
    engine._policies["default"].max_tokens = 100
    verdict = engine.check_token_budget(200, task_key="tok-task")
    assert not verdict.allowed
    assert "exceeded" in verdict.reason.lower()


def test_zero_token_budget_means_unlimited(engine):
    engine._policies["default"].max_tokens = 0
    verdict = engine.check_token_budget(10_000_000, task_key="tok-task")
    assert verdict.allowed


# ── TTL eviction ──────────────────────────────────────────────────────────────

def test_stale_counters_evicted_on_reset():
    e = GovernanceEngine()
    task_key = "stale-task"
    e._tool_counts[task_key] = 5
    e._token_counts[task_key] = 999
    # Backdate timestamp beyond 24-hour TTL
    e._counter_timestamps[task_key] = time.monotonic() - 90_000
    # Any reset call should evict stale entry
    e.reset_task_counters("other-task")
    assert task_key not in e._tool_counts
    assert task_key not in e._token_counts
    assert task_key not in e._counter_timestamps


def test_fresh_counters_not_evicted():
    e = GovernanceEngine()
    task_key = "fresh-task"
    e._tool_counts[task_key] = 3
    e._counter_timestamps[task_key] = time.monotonic()
    e.reset_task_counters("other-task")
    assert task_key in e._tool_counts


# ── Governance disabled ───────────────────────────────────────────────────────

def test_disabled_engine_allows_everything(engine):
    engine.set_enabled(False)
    verdict = engine.check_tool_call("rm_recursive", task_key="t1")
    assert verdict.allowed
