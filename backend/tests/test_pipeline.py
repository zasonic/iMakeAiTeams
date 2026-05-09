"""
tests/test_pipeline.py — Team pipeline executor.

Covers the spec's success criteria:
  1. Coordinator + specialists pipeline runs end-to-end and synthesises.
  2. Team with no specialists falls back to coordinator-only.
  3. Coordinator emitting non-JSON falls back to coordinator-only.
  4. handoff_log table is populated per specialist invocation.
  5. SSE events fire at each stage.
  6. MAX_SUBTASKS caps decomposition.
  7. Coordinator referencing an unknown agent_id is filtered out.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from models import WorkerResult
from services.hub_router import HubRouter
from services.pipeline import (
    MAX_SUBTASKS,
    MAX_UPSTREAM_CONTEXT_CHARS,
    PipelineExecutor,
    SubTask,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _seed_agent(in_memory_db, name: str, role: str,
                system_prompt: str = "You are a specialist.",
                model_pref: str = "auto") -> str:
    aid = str(uuid.uuid4())
    in_memory_db.execute(
        "INSERT INTO agents (id, name, description, system_prompt, "
        "model_preference, role, is_builtin, skills, created_at, updated_at) "
        "VALUES (?, ?, '', ?, ?, ?, 0, '[]', '2024-01-01', '2024-01-01')",
        (aid, name, system_prompt, model_pref, role),
    )
    in_memory_db.commit()
    return aid


def _seed_team(in_memory_db, coordinator_id: str, member_ids: list,
               name: str = "T") -> str:
    tid = str(uuid.uuid4())
    in_memory_db.execute(
        "INSERT INTO agent_teams (id, name, description, coordinator_id, "
        "created_at, updated_at) VALUES (?, ?, '', ?, '2024-01-01', "
        "'2024-01-01')",
        (tid, name, coordinator_id),
    )
    for i, mid in enumerate(member_ids):
        in_memory_db.execute(
            "INSERT INTO agent_team_members (team_id, agent_id, role, "
            "sort_order) VALUES (?, ?, 'worker', ?)",
            (tid, mid, i),
        )
    in_memory_db.commit()
    return tid


@pytest.fixture
def hub_mock():
    """A HubRouter mock that returns scriptable WorkerResults."""
    hub = MagicMock(spec=HubRouter)
    # route_for_agent always succeeds with a Claude-backed RoutingDecision
    from models import RoutingDecision
    hub.route_for_agent.side_effect = lambda aid, task: RoutingDecision(
        agent_id=aid, backend="claude", score=1.0,
        reasoning="test", used_fallback=False, skill_matched="",
    )
    return hub


@pytest.fixture
def executor(hub_mock, settings):
    return PipelineExecutor(hub_mock, settings)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDecomposition:
    def test_pipeline_runs_end_to_end(self, in_memory_db, executor, hub_mock):
        coord_id = _seed_agent(in_memory_db, "Coordinator", "coordinator",
                               "You coordinate.")
        researcher_id = _seed_agent(in_memory_db, "Researcher", "researcher",
                                    "You research.")
        writer_id = _seed_agent(in_memory_db, "Writer", "writer",
                                "You write.")
        team_id = _seed_team(in_memory_db, coord_id, [researcher_id, writer_id])

        decomp_json = json.dumps([
            {"agent_id": researcher_id, "agent_name": "Researcher",
             "description": "Research the topic"},
            {"agent_id": writer_id, "agent_name": "Writer",
             "description": "Write a summary"},
        ])
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp_json, backend="claude",
                         model_name="claude-test", input_tokens=10,
                         output_tokens=20),
            WorkerResult(text="Found facts A, B, C.", backend="claude",
                         model_name="claude-test", input_tokens=5,
                         output_tokens=15),
            WorkerResult(text="Drafted summary.", backend="claude",
                         model_name="claude-test", input_tokens=8,
                         output_tokens=18),
            WorkerResult(text="Final synthesised reply.", backend="claude",
                         model_name="claude-test", input_tokens=20,
                         output_tokens=30),
        ]

        events = []
        result = executor.run(
            team_id=team_id, user_message="Research and summarise X.",
            conversation_id="cid", history=[],
            on_event=lambda et, data: events.append((et, data)),
        )

        assert result.synthesis == "Final synthesised reply."
        assert len(result.steps) == 2
        assert result.steps[0]["agent"] == "Researcher"
        assert result.steps[1]["agent"] == "Writer"
        assert result.synthesis_model == "claude-test"
        # tokens_in includes decomp(10) + researcher(5) + writer(8) + synth(20)
        assert result.total_tokens_in == 43
        assert result.total_tokens_out == 83

        event_types = [e[0] for e in events]
        assert "pipeline_started" in event_types
        assert "pipeline_decomposing" in event_types
        assert "pipeline_plan" in event_types
        assert "pipeline_step_started" in event_types
        assert "pipeline_step_complete" in event_types
        assert "pipeline_synthesising" in event_types
        assert "pipeline_complete" in event_types

    def test_team_with_no_specialists_falls_back(self, in_memory_db,
                                                  executor, hub_mock):
        coord_id = _seed_agent(in_memory_db, "Lone", "coordinator",
                               "You work alone.")
        team_id = _seed_team(in_memory_db, coord_id, [])

        hub_mock.invoke.return_value = WorkerResult(
            text="Direct reply.", backend="claude", model_name="claude-test",
            input_tokens=5, output_tokens=10,
        )
        result = executor.run(
            team_id=team_id, user_message="Hi",
            conversation_id="cid", history=[],
        )
        assert result.synthesis == "Direct reply."
        assert result.steps == []
        assert result.handoffs == []
        # Single invocation (the coordinator), no decomposition
        assert hub_mock.invoke.call_count == 1

    def test_coordinator_invalid_json_falls_back(self, in_memory_db,
                                                  executor, hub_mock):
        coord_id = _seed_agent(in_memory_db, "C", "coordinator")
        spec_id = _seed_agent(in_memory_db, "S", "researcher")
        team_id = _seed_team(in_memory_db, coord_id, [spec_id])

        hub_mock.invoke.side_effect = [
            WorkerResult(text="Not valid JSON at all.", backend="claude",
                         model_name="claude-test"),
            # The fallback path then invokes the coordinator alone
            WorkerResult(text="Fallback reply.", backend="claude",
                         model_name="claude-test", input_tokens=5,
                         output_tokens=10),
        ]
        result = executor.run(
            team_id=team_id, user_message="Hi",
            conversation_id="cid", history=[],
        )
        assert result.synthesis == "Fallback reply."
        assert result.steps == []

    def test_unknown_agent_id_in_decomposition_is_skipped(
        self, in_memory_db, executor, hub_mock,
    ):
        coord_id = _seed_agent(in_memory_db, "C", "coordinator")
        spec_id = _seed_agent(in_memory_db, "S", "researcher")
        team_id = _seed_team(in_memory_db, coord_id, [spec_id])

        decomp = json.dumps([
            {"agent_id": "unknown-agent", "agent_name": "Ghost",
             "description": "Should be filtered"},
            {"agent_id": spec_id, "agent_name": "S",
             "description": "Real subtask"},
        ])
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp, backend="claude", model_name="claude-test"),
            WorkerResult(text="Specialist work.", backend="claude",
                         model_name="claude-test"),
            WorkerResult(text="Synthesis.", backend="claude",
                         model_name="claude-test"),
        ]
        result = executor.run(
            team_id=team_id, user_message="Hi",
            conversation_id="cid", history=[],
        )
        # Only the real specialist runs (not the ghost agent)
        assert len(result.steps) == 1
        assert result.steps[0]["agent"] == "S"

    def test_max_subtasks_caps_decomposition(self, in_memory_db, executor,
                                              hub_mock):
        coord_id = _seed_agent(in_memory_db, "C", "coordinator")
        spec_id = _seed_agent(in_memory_db, "S", "researcher")
        team_id = _seed_team(in_memory_db, coord_id, [spec_id])

        # Coordinator tries to decompose into 20 steps
        decomp = json.dumps([
            {"agent_id": spec_id, "agent_name": "S",
             "description": f"task {i}"}
            for i in range(20)
        ])
        invokes = [WorkerResult(text=decomp, backend="claude",
                                model_name="claude-test")]
        # Specialist invocations (capped at MAX_SUBTASKS) + 1 synthesis
        for _ in range(MAX_SUBTASKS):
            invokes.append(WorkerResult(text="step done", backend="claude",
                                         model_name="claude-test"))
        invokes.append(WorkerResult(text="Final.", backend="claude",
                                     model_name="claude-test"))
        hub_mock.invoke.side_effect = invokes

        result = executor.run(
            team_id=team_id, user_message="Hi",
            conversation_id="cid", history=[],
        )
        assert len(result.steps) == MAX_SUBTASKS

    def test_handoffs_logged_to_db(self, in_memory_db, executor, hub_mock):
        coord_id = _seed_agent(in_memory_db, "C", "coordinator")
        spec_id = _seed_agent(in_memory_db, "S", "researcher")
        team_id = _seed_team(in_memory_db, coord_id, [spec_id])

        decomp = json.dumps([{"agent_id": spec_id, "agent_name": "S",
                              "description": "do thing"}])
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp, backend="claude", model_name="claude-test"),
            WorkerResult(text="Result", backend="claude", model_name="claude-test"),
            WorkerResult(text="Done.", backend="claude", model_name="claude-test"),
        ]

        result = executor.run(
            team_id=team_id, user_message="Hi",
            conversation_id="cid", history=[],
        )
        rows = in_memory_db.fetchall(
            "SELECT * FROM handoff_log WHERE workflow_id = ?",
            (result.pipeline_id,),
        )
        assert len(rows) == 1
        assert rows[0]["agent_id"] == spec_id
        assert rows[0]["subtask_completed"] == "do thing"

    def test_unknown_team_raises(self, in_memory_db, executor):
        with pytest.raises(ValueError):
            executor.run(
                team_id="bogus", user_message="Hi",
                conversation_id="cid", history=[],
            )


class TestUpstreamContextCap:
    def test_upstream_context_capped(self, executor):
        from models import HandoffPacket
        # Build many handoffs whose combined size exceeds the cap
        big_text = "x" * 5000
        handoffs = [
            HandoffPacket(
                agent_id=f"a{i}", agent_name=f"Agent{i}",
                subtask_completed=f"task {i}", artifact=big_text,
                confidence=0.8,
            )
            for i in range(10)
        ]
        ctx = executor._build_upstream_context(handoffs)
        # Cap is 12_000 chars; with 5KB blocks we only fit 1-2 of them. We
        # tolerate ~1KB of formatting overhead beyond MAX_UPSTREAM_CONTEXT_CHARS
        # (the heading and "## Upstream result..." block markers).
        assert len(ctx) <= MAX_UPSTREAM_CONTEXT_CHARS + 1500
        assert "## Results from earlier pipeline steps" in ctx

    def test_empty_handoffs_yields_empty_context(self, executor):
        assert executor._build_upstream_context([]) == ""


class TestSubtaskParsing:
    def test_parses_json_array(self, executor):
        members = [{"id": "a1", "name": "A"}]
        coord = {"id": "c1"}
        raw = json.dumps([
            {"agent_id": "a1", "agent_name": "A", "description": "task"},
        ])
        subs = executor._parse_subtasks(raw, members, coord)
        assert len(subs) == 1
        assert subs[0].agent_id == "a1"
        assert subs[0].description == "task"

    def test_parses_fenced_json(self, executor):
        members = [{"id": "a1", "name": "A"}]
        coord = {"id": "c1"}
        raw = (
            "Here's the plan:\n```json\n"
            + json.dumps([{"agent_id": "a1", "agent_name": "A",
                           "description": "task"}])
            + "\n```\n"
        )
        subs = executor._parse_subtasks(raw, members, coord)
        assert len(subs) == 1

    def test_invalid_json_returns_empty(self, executor):
        assert executor._parse_subtasks("nonsense", [], {"id": "c1"}) == []

    def test_non_array_returns_empty(self, executor):
        members = [{"id": "a1"}]
        assert executor._parse_subtasks(
            '{"not": "an array"}', members, {"id": "c1"}
        ) == []

    def test_empty_description_skipped(self, executor):
        members = [{"id": "a1"}]
        coord = {"id": "c1"}
        raw = json.dumps([
            {"agent_id": "a1", "agent_name": "A", "description": ""},
            {"agent_id": "a1", "agent_name": "A", "description": "real"},
        ])
        subs = executor._parse_subtasks(raw, members, coord)
        assert len(subs) == 1
        assert subs[0].description == "real"


# ── Layer 2: workflow_checkpoints saga ───────────────────────────────────────


class TestWorkflowCheckpoints:
    """Verify the provisional → committed/rolled_back/abandoned transitions."""

    def _run_one_step(self, in_memory_db, hub_mock, executor):
        coord_id = _seed_agent(in_memory_db, "C", "coordinator")
        spec_id = _seed_agent(in_memory_db, "S", "researcher")
        team_id = _seed_team(in_memory_db, coord_id, [spec_id])

        decomp = json.dumps([{
            "agent_id": spec_id, "agent_name": "S",
            "description": "do thing",
        }])
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp, backend="claude", model_name="t"),
            WorkerResult(text="solid output", backend="claude", model_name="t"),
            WorkerResult(text="Done.", backend="claude", model_name="t"),
        ]
        return executor.run(
            team_id=team_id, user_message="x",
            conversation_id="cid", history=[],
        )

    def test_committed_row_written_on_pass(self, in_memory_db, executor, hub_mock):
        result = self._run_one_step(in_memory_db, hub_mock, executor)
        rows = in_memory_db.fetchall(
            "SELECT * FROM workflow_checkpoints WHERE workflow_id = ?",
            (result.pipeline_id,),
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["state"] == "committed"
        assert row["validation_passed"] == 1
        assert row["committed_at"] is not None
        assert row["rolled_back_at"] is None
        assert row["failure_reason"] is None
        assert row["success_criteria"] == "do thing"

    def test_rolled_back_then_committed_on_retry(
        self, in_memory_db, executor, hub_mock,
    ):
        coord_id = _seed_agent(in_memory_db, "C", "coordinator")
        spec_id = _seed_agent(in_memory_db, "S", "researcher")
        team_id = _seed_team(in_memory_db, coord_id, [spec_id])

        decomp = json.dumps([{
            "agent_id": spec_id, "agent_name": "S",
            "description": "do thing",
        }])
        # First specialist call returns empty artifact → structural validation
        # fails (artifact-empty). Retry returns a real answer → commit.
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp, backend="claude", model_name="t"),
            WorkerResult(text="", backend="claude", model_name="t"),
            WorkerResult(text="actual content here", backend="claude", model_name="t"),
            WorkerResult(text="Done.", backend="claude", model_name="t"),
        ]
        result = executor.run(
            team_id=team_id, user_message="x",
            conversation_id="cid", history=[],
        )
        rows = in_memory_db.fetchall(
            "SELECT state, retry_count, failure_reason FROM workflow_checkpoints "
            "WHERE workflow_id = ? ORDER BY created_at",
            (result.pipeline_id,),
        )
        # One checkpoint per step. Final state should be 'committed' because
        # the retry succeeded; the row records the prior failure.
        assert len(rows) == 1
        assert rows[0]["state"] == "committed"

    def test_provisional_marked_abandoned_on_startup(self, in_memory_db):
        from datetime import datetime, timezone
        from services.pipeline import (
            CHECKPOINT_PROVISIONAL,
            mark_abandoned_provisional_checkpoints,
        )
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO workflow_checkpoints "
            "(checkpoint_id, workflow_id, step_index, task_id, agent_id, "
            " agent_name, state, success_criteria, retry_count, max_retries, created_at) "
            "VALUES ('cid1', 'wf1', 0, 'a', 'a', 'A', ?, '', 0, 3, ?)",
            (CHECKPOINT_PROVISIONAL, now),
        )
        in_memory_db.execute(
            "INSERT INTO workflow_checkpoints "
            "(checkpoint_id, workflow_id, step_index, task_id, agent_id, "
            " agent_name, state, success_criteria, retry_count, max_retries, created_at) "
            "VALUES ('cid2', 'wf1', 1, 'a', 'a', 'A', 'committed', '', 0, 3, ?)",
            (now,),
        )
        in_memory_db.commit()

        n = mark_abandoned_provisional_checkpoints()
        assert n == 1
        rows = in_memory_db.fetchall(
            "SELECT checkpoint_id, state FROM workflow_checkpoints ORDER BY checkpoint_id"
        )
        states = {r["checkpoint_id"]: r["state"] for r in rows}
        assert states["cid1"] == "abandoned"
        assert states["cid2"] == "committed"  # untouched

    def test_checkpoint_state_event_emitted(self, in_memory_db, executor, hub_mock):
        events = []
        coord_id = _seed_agent(in_memory_db, "C", "coordinator")
        spec_id = _seed_agent(in_memory_db, "S", "researcher")
        team_id = _seed_team(in_memory_db, coord_id, [spec_id])
        decomp = json.dumps([{
            "agent_id": spec_id, "agent_name": "S", "description": "x",
        }])
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp, backend="claude", model_name="t"),
            WorkerResult(text="something", backend="claude", model_name="t"),
            WorkerResult(text="ok", backend="claude", model_name="t"),
        ]
        executor.run(
            team_id=team_id, user_message="x",
            conversation_id="cid", history=[],
            on_event=lambda et, data: events.append((et, data)),
        )
        states = [
            d.get("state") for et, d in events if et == "checkpoint_state"
        ]
        # Provisional opens, committed closes — both must fire for one step.
        assert "provisional" in states
        assert "committed" in states


# ── Layer 2: debate_log challenger gating ────────────────────────────────────


class TestDebateGating:
    """Ensure the challenger only fires when both gates are open."""

    def _build_executor_with_clients(self, hub_mock, settings):
        """Create an executor with stub claude/local clients so debate can run."""
        from unittest.mock import MagicMock
        local = MagicMock()
        local.is_available.return_value = True
        local.client_name.return_value = "local"
        # Even with a local client, debate routes through hub.invoke, not the
        # client directly — so the challenger's WorkerResult is scripted via
        # hub_mock.invoke.side_effect in each test.
        return PipelineExecutor(hub_mock, settings, local_client=local)

    def _seed_minimal_team(self, in_memory_db):
        coord_id = _seed_agent(in_memory_db, "C", "coordinator")
        spec_id = _seed_agent(in_memory_db, "S", "researcher")
        team_id = _seed_team(in_memory_db, coord_id, [spec_id])
        return coord_id, spec_id, team_id

    def test_debate_disabled_skips_challenger(
        self, in_memory_db, hub_mock, settings,
    ):
        settings.set("debate_enabled", "0")
        executor = self._build_executor_with_clients(hub_mock, settings)
        coord_id, spec_id, team_id = self._seed_minimal_team(in_memory_db)
        decomp = json.dumps([{
            "agent_id": spec_id, "agent_name": "S", "description": "x",
        }])
        # Only 3 invokes: decomp, specialist, synthesis. No challenger.
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp, backend="claude", model_name="t"),
            WorkerResult(text="answer", backend="claude", model_name="t"),
            WorkerResult(text="ok", backend="claude", model_name="t"),
        ]
        result = executor.run(
            team_id=team_id, user_message="trivial",
            conversation_id="cid", history=[],
        )
        rows = in_memory_db.fetchall("SELECT * FROM debate_log")
        assert rows == []
        assert hub_mock.invoke.call_count == 3

    def test_debate_high_stakes_only_skips_low_stakes(
        self, in_memory_db, hub_mock, settings,
    ):
        settings.set("debate_enabled", "1")
        settings.set("debate_only_high_stakes", "1")
        executor = self._build_executor_with_clients(hub_mock, settings)
        coord_id, spec_id, team_id = self._seed_minimal_team(in_memory_db)
        decomp = json.dumps([{
            "agent_id": spec_id, "agent_name": "S", "description": "x",
        }])
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp, backend="claude", model_name="t"),
            WorkerResult(text="answer", backend="claude", model_name="t"),
            WorkerResult(text="ok", backend="claude", model_name="t"),
        ]
        result = executor.run(
            team_id=team_id, user_message="hello there",  # not high-stakes
            conversation_id="cid", history=[],
        )
        assert in_memory_db.fetchall("SELECT * FROM debate_log") == []

    def test_debate_fires_on_high_stakes(
        self, in_memory_db, hub_mock, settings,
    ):
        settings.set("debate_enabled", "1")
        settings.set("debate_only_high_stakes", "1")
        executor = self._build_executor_with_clients(hub_mock, settings)
        coord_id, spec_id, team_id = self._seed_minimal_team(in_memory_db)
        decomp = json.dumps([{
            "agent_id": spec_id, "agent_name": "S", "description": "x",
        }])
        challenger_json = json.dumps({
            "assumption_diffs": ["assumed cost is fixed"],
            "fact_conflicts": [],
            "missing_analysis": ["did not consider rollback"],
            "changed_position": False,
            "revised_conclusion": "",
            "overall_assessment": "Solid but incomplete.",
        })
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp, backend="claude", model_name="t"),
            WorkerResult(text="answer", backend="claude", model_name="t"),
            WorkerResult(text=challenger_json, backend="claude", model_name="t"),
            WorkerResult(text="ok", backend="claude", model_name="t"),
        ]
        # "delete the production database" is high-stakes per governance regex
        result = executor.run(
            team_id=team_id,
            user_message="please delete the production database for me",
            conversation_id="cid", history=[],
        )
        rows = in_memory_db.fetchall(
            "SELECT * FROM debate_log WHERE workflow_id = ?",
            (result.pipeline_id,),
        )
        assert len(rows) == 1
        assert rows[0]["overall_assessment"] == "Solid but incomplete."
        assert rows[0]["parse_failed"] == 0
        # changed_position stored as int
        assert rows[0]["changed_position"] == 0

    def test_debate_unparseable_marked_parse_failed(
        self, in_memory_db, hub_mock, settings,
    ):
        settings.set("debate_enabled", "1")
        settings.set("debate_only_high_stakes", "0")
        executor = self._build_executor_with_clients(hub_mock, settings)
        coord_id, spec_id, team_id = self._seed_minimal_team(in_memory_db)
        decomp = json.dumps([{
            "agent_id": spec_id, "agent_name": "S", "description": "x",
        }])
        hub_mock.invoke.side_effect = [
            WorkerResult(text=decomp, backend="claude", model_name="t"),
            WorkerResult(text="answer", backend="claude", model_name="t"),
            WorkerResult(text="not json at all", backend="claude", model_name="t"),
            WorkerResult(text="ok", backend="claude", model_name="t"),
        ]
        result = executor.run(
            team_id=team_id, user_message="anything",
            conversation_id="cid", history=[],
        )
        rows = in_memory_db.fetchall(
            "SELECT * FROM debate_log WHERE workflow_id = ?",
            (result.pipeline_id,),
        )
        assert len(rows) == 1
        assert rows[0]["parse_failed"] == 1
