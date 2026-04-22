"""
tests/test_qwen_thinking.py — Phase 3: Qwen3 hybrid thinking integration.

Covers the four Phase 3 success criteria:
  1. One GGUF serves both hub and workers (same model id in both calls).
  2. Cold-start under 3s from detect_qwen3_30b_a3b() to first stub token.
  3. Settings UI exposes a single model dropdown (no second knob) — verified
     by asserting only ``default_local_model`` is consulted in the request.
  4. Fallback to next-best model with a plain-English notice when Qwen3 is
     absent.

Plus prompt-shape tests for /think and /no_think directives, budget cap,
LLM-fallback parsing, and round-trip through HubRouter.invoke().
"""

from __future__ import annotations

import json
import time
import uuid
from unittest.mock import MagicMock

import pytest

from models import RoutingDecision, TaskDescriptor
from services import qwen_thinking
from services.hub_router import HubRouter
from services.local_client import LocalClient


# ── Prompt-shape tests ──────────────────────────────────────────────────────


class TestDirectives:
    def test_think_prepended(self):
        out = qwen_thinking.with_think_directive("System rules.", think=True)
        assert out.startswith("/think\n")
        assert "System rules." in out

    def test_no_think_prepended(self):
        out = qwen_thinking.with_think_directive("System rules.", think=False)
        assert out.startswith("/no_think\n")

    def test_empty_system_still_carries_directive(self):
        out = qwen_thinking.with_think_directive("", think=True)
        assert out.startswith("/think")

    def test_directive_survives_extra_whitespace(self):
        out = qwen_thinking.with_think_directive("\n\n  hi", think=False)
        assert out.startswith("/no_think\n")


class TestStripThinkBlock:
    def test_extracts_reasoning_and_answer(self):
        text = "<think>I should reply politely.</think>\nHello there."
        answer, reasoning = qwen_thinking.strip_think_block(text)
        assert answer == "Hello there."
        assert "politely" in reasoning

    def test_no_block_returns_original(self):
        answer, reasoning = qwen_thinking.strip_think_block("Plain answer.")
        assert answer == "Plain answer."
        assert reasoning == ""

    def test_empty_input(self):
        assert qwen_thinking.strip_think_block("") == ("", "")


# ── worker_think dispatch ──────────────────────────────────────────────────


class TestWorkerThink:
    def test_zero_budget_uses_no_think(self):
        local = MagicMock()
        local.chat_multi_turn.return_value = "ok"
        qwen_thinking.worker_think(local, "sys", [{"role": "user", "content": "hi"}],
                                    budget_tokens=0)
        sent_system = local.chat_multi_turn.call_args[0][0]
        assert sent_system.startswith("/no_think\n")

    def test_positive_budget_uses_think(self):
        local = MagicMock()
        local.chat_multi_turn.return_value = "ok"
        qwen_thinking.worker_think(local, "sys", [{"role": "user", "content": "hi"}],
                                    budget_tokens=2048)
        sent_system = local.chat_multi_turn.call_args[0][0]
        assert sent_system.startswith("/think\n")

    def test_budget_passed_as_max_tokens(self):
        local = MagicMock()
        local.chat_multi_turn.return_value = "ok"
        qwen_thinking.worker_think(local, "sys", [], budget_tokens=1024)
        kwargs = local.chat_multi_turn.call_args.kwargs
        assert kwargs.get("max_tokens") == 1024

    def test_streaming_path(self):
        local = MagicMock()
        local.stream_multi_turn.return_value = "streamed"
        cb = MagicMock()
        out = qwen_thinking.worker_think(local, "sys", [], budget_tokens=512,
                                          on_token=cb)
        assert out == "streamed"
        local.stream_multi_turn.assert_called_once()
        # Ensure on_token was passed through
        args = local.stream_multi_turn.call_args[0]
        assert args[2] is cb


# ── LLM fallback (no_think_route) ──────────────────────────────────────────


def _local_returning(text: str) -> MagicMock:
    local = MagicMock()
    local.chat.return_value = text
    return local


class TestNoThinkRoute:
    def test_picks_chosen_agent(self):
        local = _local_returning('{"agent_id": "ag-2", "reason": "best skills"}')
        candidates = [
            {"id": "ag-1", "name": "A", "role": "writer", "skills": "[]",
             "model_preference": "claude"},
            {"id": "ag-2", "name": "B", "role": "coder", "skills": "[]",
             "model_preference": "local"},
        ]
        decision = qwen_thinking.no_think_route(
            local, candidates, TaskDescriptor(text="refactor this"),
        )
        assert decision.agent_id == "ag-2"
        assert decision.used_fallback is True
        assert decision.backend == "local"  # follows agent's model preference

    def test_falls_back_on_unknown_agent_id(self):
        local = _local_returning('{"agent_id": "nonexistent"}')
        candidates = [
            {"id": "ag-1", "name": "A", "role": "writer", "skills": "[]",
             "model_preference": "claude"},
        ]
        decision = qwen_thinking.no_think_route(
            local, candidates, TaskDescriptor(text="x"),
        )
        assert decision.agent_id == "ag-1"  # defaulted to first candidate
        assert "unknown agent_id" in decision.reasoning.lower()

    def test_falls_back_on_parse_failure(self):
        local = _local_returning("not even close to JSON")
        candidates = [
            {"id": "ag-1", "name": "A", "role": "writer", "skills": "[]",
             "model_preference": "auto"},
        ]
        decision = qwen_thinking.no_think_route(
            local, candidates, TaskDescriptor(text="x"),
        )
        assert decision.agent_id == "ag-1"

    def test_strips_think_block_before_parsing(self):
        # Even if Qwen accidentally emits a <think> block in /no_think mode,
        # we strip it before parsing the JSON answer.
        local = _local_returning(
            '<think>routing logic</think>\n{"agent_id":"ag-1","reason":"only one"}'
        )
        candidates = [
            {"id": "ag-1", "name": "A", "role": "writer", "skills": "[]",
             "model_preference": "claude"},
        ]
        decision = qwen_thinking.no_think_route(
            local, candidates, TaskDescriptor(text="x"),
        )
        assert decision.agent_id == "ag-1"

    def test_uses_no_think_directive(self):
        local = _local_returning('{"agent_id": "ag-1"}')
        candidates = [
            {"id": "ag-1", "name": "A", "role": "writer", "skills": "[]",
             "model_preference": "claude"},
        ]
        qwen_thinking.no_think_route(
            local, candidates, TaskDescriptor(text="x"),
        )
        sent_system = local.chat.call_args[0][0]
        assert sent_system.startswith("/no_think\n")

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            qwen_thinking.no_think_route(MagicMock(), [], TaskDescriptor(text="x"))


# ── make_no_think_router (HubRouter integration) ────────────────────────────


def test_make_no_think_router_reflects_provider_changes():
    local = _local_returning('{"agent_id": "ag-2"}')
    snapshots = [
        [{"id": "ag-1", "name": "A", "role": "w", "skills": "[]", "model_preference": "auto"}],
        [
            {"id": "ag-1", "name": "A", "role": "w", "skills": "[]", "model_preference": "auto"},
            {"id": "ag-2", "name": "B", "role": "c", "skills": "[]", "model_preference": "claude"},
        ],
    ]
    state = {"i": 0}

    def provider():
        out = snapshots[state["i"]]
        state["i"] = min(state["i"] + 1, len(snapshots) - 1)
        return out

    router_fn = qwen_thinking.make_no_think_router(local, provider)
    # First call sees only ag-1, so even if Qwen says ag-2 we fall back.
    d1 = router_fn(TaskDescriptor(text="x"))
    assert d1.agent_id == "ag-1"  # fallback to first candidate
    # Second call sees both, so the picked agent is honored.
    d2 = router_fn(TaskDescriptor(text="x"))
    assert d2.agent_id == "ag-2"


# ── Single-GGUF criterion (Success criterion 1) ────────────────────────────


def test_single_gguf_serves_both_hub_and_workers(monkeypatch, settings):
    """The same ``model`` value is sent in routing and worker requests."""
    settings.set("default_local_model", "qwen3-30b-a3b-q4")
    captured: list[dict] = []

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": '{"agent_id":"ag-1"}'}}]}

    def fake_post(url, json=None, **kwargs):
        captured.append(json)
        return _Resp()

    def fake_get(url, **kwargs):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"data": [{"id": "qwen3-30b-a3b-q4"}]}
        return R()

    monkeypatch.setattr("services.local_client.requests.post", fake_post)
    monkeypatch.setattr("services.local_client.requests.get", fake_get)

    local = LocalClient(settings)
    settings.set("default_local_backend", "lmstudio")

    # Routing call (no_think)
    qwen_thinking.no_think_route(
        local,
        [{"id": "ag-1", "name": "A", "role": "w", "skills": "[]", "model_preference": "local"}],
        TaskDescriptor(text="x"),
    )
    # Worker call (think)
    qwen_thinking.worker_think(local, "sys", [{"role": "user", "content": "hi"}],
                                budget_tokens=1024)

    assert len(captured) >= 2
    models = {p.get("model") for p in captured}
    assert models == {"qwen3-30b-a3b-q4"}, (
        f"Expected one model id across hub+worker calls; got {models}"
    )


# ── LM Studio detection + fallback (Success criterion 4) ────────────────────


class TestDetection:
    def test_detects_qwen3_30b_a3b_id(self, monkeypatch, settings):
        models = ["qwen3-30b-a3b-q4_k_m", "llama3:8b"]
        _patch_models(monkeypatch, models)
        local = LocalClient(settings)
        settings.set("default_local_backend", "lmstudio")
        result = local.detect_qwen3_30b_a3b()
        assert result["detected"] is True
        assert "30b" in result["model_id"].lower()
        assert result["fallback_reason"] == ""

    def test_falls_back_with_plain_english(self, monkeypatch, settings):
        _patch_models(monkeypatch, ["llama3:8b", "phi3:mini"])
        local = LocalClient(settings)
        settings.set("default_local_backend", "lmstudio")
        result = local.detect_qwen3_30b_a3b()
        assert result["detected"] is False
        assert result["model_id"] == "llama3:8b"
        assert "Qwen3-30B-A3B not detected" in result["fallback_reason"]
        # No tracebacks, no JSON — plain English suitable for the UI
        assert "Traceback" not in result["fallback_reason"]

    def test_no_models_reachable(self, monkeypatch, settings):
        def fake_get(url, **kwargs):
            raise OSError("connection refused")
        monkeypatch.setattr("services.local_client.requests.get", fake_get)
        local = LocalClient(settings)
        settings.set("default_local_backend", "lmstudio")
        result = local.detect_qwen3_30b_a3b()
        assert result["detected"] is False
        assert result["model_id"] == ""
        assert "reachable" in result["fallback_reason"].lower()


def _patch_models(monkeypatch, ids: list[str]) -> None:
    class R:
        status_code = 200
        def __init__(self, data): self._data = data
        def raise_for_status(self): pass
        def json(self): return self._data
    def fake_get(url, **kwargs):
        if "/v1/models" in url:
            return R({"data": [{"id": m} for m in ids]})
        if "/api/tags" in url:
            return R({"models": [{"name": m} for m in ids]})
        return R({})
    monkeypatch.setattr("services.local_client.requests.get", fake_get)


# ── Cold-start (Success criterion 2) ───────────────────────────────────────


def test_cold_start_under_3s(monkeypatch, settings):
    """
    Time from first ``detect_qwen3_30b_a3b()`` call to first emitted token from
    a stub LM Studio is under 3 seconds. Real LM Studio + GGUF load times are
    out of scope; this asserts the framework overhead introduced by Phase 3 is
    negligible.
    """
    _patch_models(monkeypatch, ["qwen3-30b-a3b"])
    settings.set("default_local_model", "qwen3-30b-a3b")
    settings.set("default_local_backend", "lmstudio")

    # Stub /v1/chat/completions streaming response: yield three tokens.
    class _StreamResp:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def raise_for_status(self): pass
        def iter_lines(self):
            chunks = [
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {"content": ", there"}}]},
                {"choices": [{"delta": {"content": "."}}]},
            ]
            for c in chunks:
                yield ("data: " + json.dumps(c)).encode()
            yield b"data: [DONE]"
    def fake_post_stream(url, json=None, stream=True, **kwargs):
        return _StreamResp()
    monkeypatch.setattr("services.local_client.requests.post", fake_post_stream)

    local = LocalClient(settings)
    first_token_ts: list[float] = []
    def on_token(t):
        if not first_token_ts:
            first_token_ts.append(time.perf_counter())

    t0 = time.perf_counter()
    local.detect_qwen3_30b_a3b()
    qwen_thinking.worker_think(local, "sys", [{"role": "user", "content": "hi"}],
                                budget_tokens=512, on_token=on_token)
    assert first_token_ts, "stub never emitted a token"
    elapsed = first_token_ts[0] - t0
    assert elapsed < 3.0, f"Cold-start path took {elapsed:.2f}s (budget 3s)"


# ── HubRouter integration: thinking_budget flows through ────────────────────


def _seed_agent_with_budget(in_memory_db, name: str, budget: int,
                             skills: list[dict], model_pref: str = "local") -> str:
    aid = str(uuid.uuid4())
    in_memory_db.execute(
        "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
        "role, is_builtin, skills, thinking_budget, created_at, updated_at) VALUES "
        "(?, ?, '', 'sys', ?, 'custom', 0, ?, ?, '2024-01-01', '2024-01-01')",
        (aid, name, model_pref, json.dumps(skills), budget),
    )
    in_memory_db.commit()
    return aid


def test_hub_router_stamps_budget_and_invoke_dispatches_through_qwen(
    in_memory_db, settings,
):
    """
    End-to-end: route_for_agent reads agent.thinking_budget, stamps it on the
    decision, and HubRouter.invoke routes through qwen_thinking.worker_think
    so the /think directive is applied.
    """
    settings.set("default_local_model", "qwen3-30b-a3b")
    aid = _seed_agent_with_budget(
        in_memory_db, "Worker", 1024,
        [{"name": "researcher", "scopes": ["read"]}],
    )
    local = MagicMock()
    local.chat_multi_turn.return_value = "/think echoes back here"
    claude = MagicMock(_model="claude-test")

    hub = HubRouter(claude, local, settings)
    decision = hub.route_for_agent(aid, TaskDescriptor(
        text="x", required_skills=("researcher",), required_scopes=("read",),
    ))
    assert decision.thinking_budget == 1024
    hub.invoke(decision, "sys", [{"role": "user", "content": "hi"}])
    sent_system = local.chat_multi_turn.call_args[0][0]
    assert sent_system.startswith("/think\n")


def test_hub_router_caps_budget_at_global_setting(in_memory_db, settings):
    settings.set("qwen_thinking_global_budget_cap", 512)
    aid = _seed_agent_with_budget(
        in_memory_db, "Worker", 8192,
        [{"name": "researcher", "scopes": ["read"]}],
    )
    hub = HubRouter(MagicMock(), MagicMock(), settings)
    decision = hub.route_for_agent(aid, TaskDescriptor(
        text="x", required_skills=("researcher",), required_scopes=("read",),
    ))
    assert decision.thinking_budget == 512
