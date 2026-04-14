"""
tests/test_task_scheduler.py

Covers:
- _validate_task_defs: valid plan, empty list, duplicate names,
  unknown depends_on reference, dependency cycle
- create_workflow / add_task: DB round-trip
- _get_ready_tasks: dependency gating
- run_workflow: successful execution, failed task handling
- plan_workflow: happy path (mocked Claude response)
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_claude(response_text: str):
    """Return a minimal mock claude_client whose chat returns response_text."""
    cc = MagicMock()
    cc.chat.return_value = response_text
    cc.chat_multi_turn.return_value = {
        "text": '{"result": "done"}',
        "input_tokens": 10,
        "output_tokens": 5,
    }
    cc._model = "claude-sonnet-4-20250514"
    return cc


# ── _validate_task_defs ───────────────────────────────────────────────────────

class TestValidateTaskDefs:
    def test_valid_plan(self, in_memory_db):
        from services.task_scheduler import _validate_task_defs
        tasks = [
            {"name": "fetch", "agent_role": "Researcher", "depends_on": []},
            {"name": "write", "agent_role": "Writer", "depends_on": ["fetch"]},
        ]
        # Should not raise
        _validate_task_defs(tasks, available_roles=["Researcher", "Writer"])

    def test_empty_list_raises(self, in_memory_db):
        from services.task_scheduler import _validate_task_defs
        with pytest.raises(ValueError, match="non-empty"):
            _validate_task_defs([], available_roles=["Researcher"])

    def test_not_a_list_raises(self, in_memory_db):
        from services.task_scheduler import _validate_task_defs
        with pytest.raises(ValueError):
            _validate_task_defs({"name": "task"}, available_roles=[])

    def test_duplicate_name_raises(self, in_memory_db):
        from services.task_scheduler import _validate_task_defs
        tasks = [
            {"name": "fetch", "agent_role": "Researcher", "depends_on": []},
            {"name": "fetch", "agent_role": "Writer", "depends_on": []},
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            _validate_task_defs(tasks, available_roles=["Researcher", "Writer"])

    def test_missing_name_raises(self, in_memory_db):
        from services.task_scheduler import _validate_task_defs
        with pytest.raises(ValueError, match="missing"):
            _validate_task_defs([{"agent_role": "Researcher"}], available_roles=["Researcher"])

    def test_unknown_depends_on_raises(self, in_memory_db):
        from services.task_scheduler import _validate_task_defs
        tasks = [
            {"name": "write", "agent_role": "Writer", "depends_on": ["nonexistent"]},
        ]
        with pytest.raises(ValueError, match="nonexistent"):
            _validate_task_defs(tasks, available_roles=["Writer"])

    def test_cycle_detection(self, in_memory_db):
        from services.task_scheduler import _validate_task_defs
        # A → B → A
        tasks = [
            {"name": "A", "agent_role": "Researcher", "depends_on": ["B"]},
            {"name": "B", "agent_role": "Writer",     "depends_on": ["A"]},
        ]
        with pytest.raises(ValueError, match="cycle"):
            _validate_task_defs(tasks, available_roles=["Researcher", "Writer"])

    def test_unknown_role_is_warned_not_raised(self, in_memory_db):
        """Unknown role falls back to General Assistant; no exception."""
        from services.task_scheduler import _validate_task_defs
        tasks = [{"name": "task1", "agent_role": "MysteryBot", "depends_on": []}]
        _validate_task_defs(tasks, available_roles=["General Assistant"])
        assert tasks[0]["agent_role"] == "General Assistant"

    def test_long_valid_chain(self, in_memory_db):
        from services.task_scheduler import _validate_task_defs
        tasks = [{"name": str(i), "agent_role": "General Assistant",
                  "depends_on": [str(i - 1)] if i > 0 else []}
                 for i in range(10)]
        _validate_task_defs(tasks, available_roles=["General Assistant"])


# ── create_workflow / add_task ────────────────────────────────────────────────

class TestWorkflowCRUD:
    def test_create_workflow_returns_id(self, in_memory_db):
        from services.task_scheduler import create_workflow
        wf_id = create_workflow("My Workflow")
        assert len(wf_id) == 36  # UUID format

        row = in_memory_db.fetchone("SELECT * FROM workflows WHERE id = ?", (wf_id,))
        assert row is not None
        assert row["name"] == "My Workflow"
        assert row["status"] == "pending"

    def test_add_task_stores_uuid_deps(self, in_memory_db):
        from services.task_scheduler import create_workflow, add_task
        wf_id = create_workflow("wf")
        t1 = add_task(wf_id, "task1", "Researcher", {"prompt": "hello"})
        t2 = add_task(wf_id, "task2", "Writer", {"prompt": "write"}, depends_on=[t1])

        row = in_memory_db.fetchone("SELECT depends_on FROM tasks WHERE id = ?", (t2,))
        deps = json.loads(row["depends_on"])
        assert deps == [t1], "depends_on must store the UUID of the upstream task"

    def test_list_workflows(self, in_memory_db):
        from services.task_scheduler import create_workflow, list_workflows
        create_workflow("wf1")
        create_workflow("wf2")
        result = list_workflows()
        assert len(result) >= 2


# ── _get_ready_tasks ──────────────────────────────────────────────────────────

class TestGetReadyTasks:
    def test_no_deps_ready_immediately(self, in_memory_db):
        from services.task_scheduler import create_workflow, add_task, _get_ready_tasks
        wf_id = create_workflow("wf")
        add_task(wf_id, "task1", "Researcher", {})
        ready = _get_ready_tasks(wf_id)
        assert len(ready) == 1
        assert ready[0]["name"] == "task1"

    def test_dep_not_met_blocks_task(self, in_memory_db):
        from services.task_scheduler import create_workflow, add_task, _get_ready_tasks
        wf_id = create_workflow("wf")
        t1 = add_task(wf_id, "task1", "Researcher", {})
        add_task(wf_id, "task2", "Writer", {}, depends_on=[t1])
        ready = _get_ready_tasks(wf_id)
        # task2 depends on task1 (still pending), so only task1 is ready
        assert len(ready) == 1
        assert ready[0]["name"] == "task1"

    def test_dep_met_unblocks_task(self, in_memory_db):
        from services.task_scheduler import create_workflow, add_task, _get_ready_tasks
        wf_id = create_workflow("wf")
        t1 = add_task(wf_id, "task1", "Researcher", {})
        t2 = add_task(wf_id, "task2", "Writer", {}, depends_on=[t1])
        # Mark t1 as succeeded
        in_memory_db.execute(
            "UPDATE tasks SET status = 'succeeded' WHERE id = ?", (t1,)
        )
        in_memory_db.commit()
        ready = _get_ready_tasks(wf_id)
        assert len(ready) == 1
        assert ready[0]["id"] == t2


# ── run_workflow ──────────────────────────────────────────────────────────────

class TestRunWorkflow:
    def test_successful_single_task(self, in_memory_db):
        from services.task_scheduler import (
            create_workflow, add_task, run_workflow, get_workflow_status
        )
        # Seed an agent so get_role_prompt can find it
        in_memory_db.execute(
            "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
            "is_builtin, created_at, updated_at) VALUES "
            "('a1', 'Researcher', 'desc', 'You are a researcher.', 'claude', 1, '2024-01-01', '2024-01-01')"
        )
        in_memory_db.commit()

        wf_id = create_workflow("single-task-wf")
        add_task(wf_id, "do_thing", "Researcher", {"prompt": "summarize"})

        cc = _make_claude('{"result": "summary text"}')
        result = run_workflow(wf_id, cc, local_client=None)

        assert result["status"] == "succeeded"
        tasks = result["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["status"] == "succeeded"

    def test_failed_task_marks_workflow_failed(self, in_memory_db):
        from services.task_scheduler import (
            create_workflow, add_task, run_workflow
        )
        in_memory_db.execute(
            "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
            "is_builtin, created_at, updated_at) VALUES "
            "('a2', 'Researcher', 'desc', 'You are a researcher.', 'claude', 1, '2024-01-01', '2024-01-01')"
        )
        in_memory_db.commit()

        wf_id = create_workflow("fail-wf")
        add_task(wf_id, "task1", "Researcher", {})

        cc = MagicMock()
        cc._model = "claude-sonnet-4-20250514"
        cc.chat_multi_turn.side_effect = RuntimeError("API down")

        result = run_workflow(wf_id, cc, local_client=None)
        assert result["status"] == "failed"

    def test_two_sequential_tasks(self, in_memory_db):
        from services.task_scheduler import (
            create_workflow, add_task, run_workflow
        )
        in_memory_db.execute(
            "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
            "is_builtin, created_at, updated_at) VALUES "
            "('a3', 'Writer', 'desc', 'You are a writer.', 'claude', 1, '2024-01-01', '2024-01-01')"
        )
        in_memory_db.commit()

        cc = _make_claude('{"result": "ok"}')
        wf_id = create_workflow("sequential-wf")
        t1 = add_task(wf_id, "task1", "Writer", {"data": "foo"})
        add_task(wf_id, "task2", "Writer", {"data": "bar"}, depends_on=[t1])

        result = run_workflow(wf_id, cc, local_client=None)
        assert result["status"] == "succeeded"
        statuses = {t["name"]: t["status"] for t in result["tasks"]}
        assert statuses["task1"] == "succeeded"
        assert statuses["task2"] == "succeeded"


# ── plan_workflow ─────────────────────────────────────────────────────────────

class TestPlanWorkflow:
    def test_plan_workflow_creates_tasks(self, in_memory_db):
        from services.task_scheduler import plan_workflow, get_workflow_status
        from services.prompt_library import seed_prompts

        seed_prompts()
        in_memory_db.execute(
            "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
            "is_builtin, created_at, updated_at) VALUES "
            "('a4', 'General Assistant', 'desc', 'You help.', 'auto', 1, '2024-01-01', '2024-01-01')"
        )
        in_memory_db.commit()

        plan_json = json.dumps([
            {"name": "research", "agent_role": "General Assistant",
             "depends_on": [], "description": "Do research"},
            {"name": "write",    "agent_role": "General Assistant",
             "depends_on": ["research"], "description": "Write report"},
        ])
        cc = _make_claude(plan_json)
        wf_id = plan_workflow("Write a report on AI", cc, "AI Report")
        status = get_workflow_status(wf_id)
        assert len(status["tasks"]) == 2
        names = {t["name"] for t in status["tasks"]}
        assert names == {"research", "write"}

    def test_plan_workflow_resolves_deps_to_uuids(self, in_memory_db):
        """depends_on in the DB must be UUIDs, not task names."""
        from services.task_scheduler import plan_workflow, get_workflow_status
        from services.prompt_library import seed_prompts

        seed_prompts()
        in_memory_db.execute(
            "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
            "is_builtin, created_at, updated_at) VALUES "
            "('a5', 'General Assistant', 'desc', 'You help.', 'auto', 1, '2024-01-01', '2024-01-01')"
        )
        in_memory_db.commit()

        plan_json = json.dumps([
            {"name": "step1", "agent_role": "General Assistant",
             "depends_on": [], "description": "First"},
            {"name": "step2", "agent_role": "General Assistant",
             "depends_on": ["step1"], "description": "Second"},
        ])
        cc = _make_claude(plan_json)
        wf_id = plan_workflow("Two-step goal", cc)
        status = get_workflow_status(wf_id)

        task_ids = {t["name"]: t["id"] for t in status["tasks"]}
        step2_row = in_memory_db.fetchone(
            "SELECT depends_on FROM tasks WHERE id = ?", (task_ids["step2"],)
        )
        deps = json.loads(step2_row["depends_on"])
        assert deps == [task_ids["step1"]], \
            "depends_on must contain the UUID of step1, not the string 'step1'"

    def test_plan_workflow_invalid_json_raises(self, in_memory_db):
        from services.task_scheduler import plan_workflow
        from services.prompt_library import seed_prompts
        seed_prompts()
        cc = _make_claude("not json at all")
        with pytest.raises(ValueError, match="invalid JSON"):
            plan_workflow("bad plan", cc)
