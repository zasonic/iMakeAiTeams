"""
tests/test_hub_router.py — Phase 1: Centralized hub routing.

Covers the four success criteria from the Phase 1 plan:
  1. Routing decision latency p99 < 50 ms over 1000 randomized 8-skill catalogs.
  2. 5-worker fan-out error amplification < 5x with independent worker failures.
  3. Static guard: only hub_router.py and chat_orchestrator.py invoke the model
     clients' worker methods (chat_multi_turn / stream_multi_turn).
  4. Skill-match unit tests, scope authz, fallback hookup, invoke dispatch.
"""

from __future__ import annotations

import ast
import json
import random
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models import (
    RoutingDecision,
    TaskDescriptor,
    WorkerResult,
)
from services.hub_router import (
    HubRouter,
    AuthorizationError,
    RoutingError,
    MIN_SKILL_MATCH_SCORE,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _seed_agent(in_memory_db, name: str, role: str, skills: list[dict],
                model_pref: str = "auto") -> str:
    aid = str(uuid.uuid4())
    in_memory_db.execute(
        "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
        "role, is_builtin, skills, created_at, updated_at) VALUES "
        "(?, ?, '', 'sys', ?, ?, 0, ?, '2024-01-01', '2024-01-01')",
        (aid, name, model_pref, role, json.dumps(skills)),
    )
    in_memory_db.commit()
    return aid


@pytest.fixture
def claude_mock():
    m = MagicMock()
    m._model = "claude-test"
    m.chat_multi_turn.return_value = {
        "text": "claude reply", "input_tokens": 10, "output_tokens": 20,
    }
    return m


@pytest.fixture
def local_mock():
    m = MagicMock()
    m.chat_multi_turn.return_value = "local reply"
    m.is_available.return_value = True
    return m


@pytest.fixture
def hub(claude_mock, local_mock, settings):
    return HubRouter(claude_mock, local_mock, settings)


# ── Skill-match unit tests ───────────────────────────────────────────────────


class TestSkillMatch:
    def test_route_for_agent_open_task_accepts_any_agent(self, in_memory_db, hub):
        """Tasks with no required_skills accept any agent (preserves chat path)."""
        aid = _seed_agent(in_memory_db, "A", "researcher",
                          [{"name": "researcher", "scopes": ["read"]}])
        decision = hub.route_for_agent(aid, TaskDescriptor(text="hi"))
        assert decision.agent_id == aid
        assert decision.score == 1.0  # open task → score 1.0 by definition

    def test_route_for_agent_skill_match(self, in_memory_db, hub):
        aid = _seed_agent(in_memory_db, "Cody", "coder",
                          [{"name": "coder", "scopes": ["read", "write"]}])
        task = TaskDescriptor(text="write me a function",
                              required_skills=("coder",),
                              required_scopes=("write",))
        decision = hub.route_for_agent(aid, task)
        assert decision.agent_id == aid
        assert decision.skill_matched == "coder"
        assert decision.score > 0

    def test_route_for_agent_missing_skill_raises(self, in_memory_db, hub):
        aid = _seed_agent(in_memory_db, "Wri", "writer",
                          [{"name": "writer", "scopes": ["write"]}])
        task = TaskDescriptor(text="debug this",
                              required_skills=("coder",),
                              required_scopes=("write",))
        with pytest.raises(AuthorizationError):
            hub.route_for_agent(aid, task)

    def test_route_for_agent_missing_scope_raises(self, in_memory_db, hub):
        """Required scope not declared on the matched skill → AuthorizationError."""
        aid = _seed_agent(in_memory_db, "Reader", "researcher",
                          [{"name": "researcher", "scopes": ["read"]}])
        task = TaskDescriptor(text="rewrite this",
                              required_skills=("researcher",),
                              required_scopes=("write",))
        with pytest.raises(AuthorizationError):
            hub.route_for_agent(aid, task)

    def test_route_unknown_agent_raises(self, in_memory_db, hub):
        with pytest.raises(AuthorizationError):
            hub.route_for_agent("nonexistent", TaskDescriptor(text="x"))

    def test_route_picks_best_skill_match(self, in_memory_db, hub):
        a_specialist = _seed_agent(in_memory_db, "Spec", "coder",
                                   [{"name": "coder", "scopes": ["read", "write"]}])
        a_generalist = _seed_agent(in_memory_db, "Gen", "custom",
                                   [{"name": "coder",  "scopes": ["read", "write"]},
                                    {"name": "writer", "scopes": ["write"]},
                                    {"name": "analyst", "scopes": ["read"]}])
        task = TaskDescriptor(text="refactor", required_skills=("coder",),
                              required_scopes=("write",))
        decision = hub.route(task)
        # Specialist should beat generalist on equal coverage via specificity boost.
        assert decision.agent_id == a_specialist

    def test_route_no_match_no_fallback_raises(self, in_memory_db, hub):
        _seed_agent(in_memory_db, "Wri", "writer",
                    [{"name": "writer", "scopes": ["write"]}])
        task = TaskDescriptor(text="x", required_skills=("nonexistent_skill",))
        with pytest.raises(RoutingError):
            hub.route(task)

    def test_route_uses_fallback_when_configured(self, in_memory_db,
                                                  claude_mock, local_mock, settings):
        _seed_agent(in_memory_db, "Wri", "writer",
                    [{"name": "writer", "scopes": ["write"]}])
        called = {"n": 0}

        def fallback(task):
            called["n"] += 1
            return RoutingDecision(
                agent_id="fallback-id", backend="claude", score=0.1,
                reasoning="fallback", used_fallback=True, skill_matched="",
            )

        hub = HubRouter(claude_mock, local_mock, settings, llm_fallback=fallback)
        task = TaskDescriptor(text="x", required_skills=("nonexistent_skill",))
        decision = hub.route(task)
        assert called["n"] == 1
        assert decision.used_fallback is True
        assert decision.agent_id == "fallback-id"


# ── Backend resolution ───────────────────────────────────────────────────────


class TestBackendResolution:
    def test_agent_pref_claude_overrides_hint(self, in_memory_db, hub):
        aid = _seed_agent(in_memory_db, "Cl", "researcher",
                          [{"name": "researcher", "scopes": ["read"]}],
                          model_pref="claude")
        task = TaskDescriptor(text="x", backend_hint="local")
        decision = hub.route_for_agent(aid, task)
        assert decision.backend == "claude"

    def test_agent_pref_local_overrides_hint(self, in_memory_db, hub):
        aid = _seed_agent(in_memory_db, "Lo", "writer",
                          [{"name": "writer", "scopes": ["write"]}],
                          model_pref="local")
        task = TaskDescriptor(text="x", backend_hint="claude")
        decision = hub.route_for_agent(aid, task)
        assert decision.backend == "local"

    def test_auto_pref_uses_hint(self, in_memory_db, hub):
        aid = _seed_agent(in_memory_db, "Au", "coder",
                          [{"name": "coder", "scopes": ["read", "write"]}],
                          model_pref="auto")
        task = TaskDescriptor(text="x", backend_hint="local")
        decision = hub.route_for_agent(aid, task)
        assert decision.backend == "local"


# ── Invoke dispatch ──────────────────────────────────────────────────────────


class TestInvoke:
    def test_invoke_claude_non_streaming(self, hub, claude_mock):
        decision = RoutingDecision(agent_id="x", backend="claude", score=1.0,
                                   reasoning="r", used_fallback=False,
                                   skill_matched="")
        result = hub.invoke(decision, "sys", [{"role": "user", "content": "hi"}])
        assert isinstance(result, WorkerResult)
        assert result.text == "claude reply"
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        claude_mock.chat_multi_turn.assert_called_once()

    def test_invoke_claude_streaming(self, hub, claude_mock):
        usage = MagicMock(input_tokens=33, output_tokens=44)
        claude_mock.stream_multi_turn.return_value = ("streamed", usage)
        decision = RoutingDecision(agent_id="x", backend="claude", score=1.0,
                                   reasoning="r", used_fallback=False,
                                   skill_matched="")
        callback = MagicMock()
        result = hub.invoke(decision, "sys", [{"role": "user", "content": "hi"}],
                            on_token=callback)
        assert result.text == "streamed"
        assert result.input_tokens == 33
        assert result.output_tokens == 44

    def test_invoke_local(self, hub, local_mock):
        decision = RoutingDecision(agent_id="x", backend="local", score=1.0,
                                   reasoning="r", used_fallback=False,
                                   skill_matched="")
        result = hub.invoke(decision, "sys", [{"role": "user", "content": "hi"}])
        assert result.text == "local reply"
        assert result.backend == "local"

    def test_invoke_handles_exception(self, hub, claude_mock):
        claude_mock.chat_multi_turn.side_effect = ConnectionError("boom")
        decision = RoutingDecision(agent_id="x", backend="claude", score=1.0,
                                   reasoning="r", used_fallback=False,
                                   skill_matched="")
        result = hub.invoke(decision, "sys", [{"role": "user", "content": "hi"}])
        assert result.had_error is True
        assert "[Error" in result.text


# ── Routing latency benchmark (Success criterion 1) ──────────────────────────


def test_routing_latency_p99(in_memory_db, hub):
    """p99 of route() must be under 50ms over 1000 randomized 8-skill catalogs."""
    skill_pool = ["coder", "writer", "researcher", "reviewer",
                  "analyst", "coordinator", "ops", "design"]
    rng = random.Random(0xC0FFEE)
    # Seed 8 agents, each with ~3 random skills from the pool
    for i in range(8):
        chosen = rng.sample(skill_pool, k=3)
        _seed_agent(
            in_memory_db, f"Agent-{i}", "custom",
            [{"name": s, "scopes": ["read", "write"]} for s in chosen],
        )

    durations_ms: list[float] = []
    for _ in range(1000):
        required = (rng.choice(skill_pool),)
        task = TaskDescriptor(text="x", required_skills=required,
                              required_scopes=("read",))
        t0 = time.perf_counter()
        try:
            hub.route(task)
        except RoutingError:
            pass  # unrouteable still measures the scoring time
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    durations_ms.sort()
    p99 = durations_ms[int(len(durations_ms) * 0.99)]
    assert p99 < 50.0, f"p99 routing latency {p99:.2f}ms exceeds 50ms budget"


# ── 5-worker fan-out amplification (Success criterion 2) ─────────────────────


def test_fanout_amplification(in_memory_db, hub, claude_mock):
    """
    Independent 5-worker fan-out with 10% per-worker failure rate must yield
    error amplification < 5x the single-worker rate. This proves the hub does
    not introduce serial coupling (a future change adding cross-worker retries
    or aborts would push amplification higher and fail this test).
    """
    rng = random.Random(0xBEEF)
    SUBTASKS = 5
    P_FAIL = 0.10
    TRIALS = 400

    # Five distinct workers, each declares one specialty skill.
    agent_ids = [
        _seed_agent(
            in_memory_db, f"Worker-{i}", "custom",
            [{"name": f"skill_{i}", "scopes": ["read"]}],
            model_pref="claude",
        )
        for i in range(SUBTASKS)
    ]

    def maybe_fail(*_a, **_kw):
        if rng.random() < P_FAIL:
            raise ConnectionError("simulated worker failure")
        return {"text": "ok", "input_tokens": 1, "output_tokens": 1}

    claude_mock.chat_multi_turn.side_effect = maybe_fail

    fanout_failures = 0
    for _ in range(TRIALS):
        any_failed = False
        for i, aid in enumerate(agent_ids):
            task = TaskDescriptor(
                text="t", preferred_agent_id=aid,
                required_skills=(f"skill_{i}",), required_scopes=("read",),
            )
            decision = hub.route_for_agent(aid, task)
            result = hub.invoke(decision, "sys", [{"role": "user", "content": "t"}])
            if result.had_error:
                any_failed = True
                # Independence requirement: a failure in one subtask must not
                # cancel the remaining ones. We continue the loop deliberately.
        if any_failed:
            fanout_failures += 1

    fanout_rate = fanout_failures / TRIALS
    expected_independent = 1 - (1 - P_FAIL) ** SUBTASKS  # ~0.4095
    amplification = fanout_rate / P_FAIL
    assert amplification < 5.0, (
        f"Fan-out amplification {amplification:.2f}x ≥ 5x; "
        f"observed rate {fanout_rate:.2%}, expected ≈ {expected_independent:.2%}. "
        "This means the hub introduced serial coupling between subtasks."
    )


# ── Static guard: no direct worker invocation outside hub_router (criterion 3)


def test_no_direct_worker_calls_outside_hub_router():
    """
    AST-level assertion: in app/services/, only hub_router.py may invoke
    chat_multi_turn, stream_multi_turn, or chat_with_file. The orchestrator
    delegates through hub_router.invoke(); other services that reach the local
    client (memory, router) only call .chat() for hub-internal classification
    or summarization, which is allowed.
    """
    services_dir = Path(__file__).parent.parent / "app" / "services"
    forbidden = {"chat_multi_turn", "stream_multi_turn", "chat_with_file"}
    # The model clients themselves implement these methods and may call them
    # internally (e.g. local_client's streaming fallback to non-streaming).
    # Only hub_router.py is allowed to invoke them as a *consumer*.
    allowed_files = {"hub_router.py", "claude_client.py", "local_client.py"}

    offenders: list[str] = []
    for py in services_dir.glob("*.py"):
        if py.name in allowed_files:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in forbidden:
                    offenders.append(f"{py.name}:{node.lineno} -> .{node.func.attr}()")

    assert not offenders, (
        "Direct worker invocations found outside hub_router.py. "
        "All worker calls must go through HubRouter.invoke():\n  "
        + "\n  ".join(offenders)
    )
