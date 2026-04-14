"""
services/task_scheduler.py — Multi-Agent Coordination.

Original functionality (UNCHANGED):
  - plan_workflow() / run_workflow() / create_workflow() / add_task()
  - Self-healing task failure handler (_handle_task_failure)
  - Dependency resolution and cycle detection

Priority 3 additions (HandoffPacket):
  - _run_task() now extracts HandoffPacket from agent responses
  - HandoffPackets logged to handoff_log table
  - Downstream tasks receive HandoffPacket context blocks
  - HANDOFF_SYSTEM_FRAGMENT injected into every workflow agent prompt

Priority 4 additions (SagaLLM Checkpoints):
  - _write_checkpoint()      — write provisional checkpoint to SQLite
  - _run_validation_gate()   — Haiku call checks artifact vs success_criteria
  - _commit_checkpoint()     — mark provisional as committed
  - _rollback_checkpoint()   — mark provisional as rolled_back
  - Saga semantics wired into _run_task() around the success path

All changes are additive — no existing function signatures changed.
"""

import concurrent.futures
import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

import db

from models import (
    HandoffPacket,
    extract_handoff_packet,
    HANDOFF_SYSTEM_FRAGMENT,
)

log = logging.getLogger("task_scheduler")

_API_SEMAPHORE = threading.Semaphore(3)


# ── Task locking (v5.0) ──────────────────────────────────────────────────────

_LOCK_TTL_SECONDS = 300  # 5 minute default lock TTL


def acquire_task_lock(task_id: str, agent_id: str, ttl: int = _LOCK_TTL_SECONDS) -> bool:
    """
    Acquire an exclusive lock on a task. Returns True if acquired.
    Expired locks are automatically released.
    """
    now = datetime.now(timezone.utc).isoformat()
    expiry = datetime.now(timezone.utc).isoformat()  # will set properly below
    from datetime import timedelta
    expiry = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()

    # Check if already locked by someone else (and not expired)
    row = db.fetchone(
        "SELECT locked_by, locked_until FROM tasks WHERE id = ?", (task_id,)
    )
    if row and row["locked_by"] and row["locked_until"]:
        if row["locked_until"] > now and row["locked_by"] != agent_id:
            log.debug("Task %s already locked by %s until %s",
                      task_id, row["locked_by"], row["locked_until"])
            return False

    db.execute(
        "UPDATE tasks SET locked_by = ?, locked_until = ?, updated_at = ? WHERE id = ?",
        (agent_id, expiry, now, task_id),
    )
    log.debug("Task %s locked by %s until %s", task_id, agent_id, expiry)
    return True


def release_task_lock(task_id: str, agent_id: str) -> None:
    """Release a task lock (only if held by this agent)."""
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "UPDATE tasks SET locked_by = NULL, locked_until = NULL, updated_at = ? "
        "WHERE id = ? AND locked_by = ?",
        (now, task_id, agent_id),
    )


# ── Agent role helpers (unchanged) ────────────────────────────────────────────

def get_available_roles() -> list[str]:
    rows = db.fetchall("SELECT DISTINCT name FROM agents")
    return [r["name"] for r in rows]


def get_role_prompt(role: str) -> str:
    agent = db.fetchone("SELECT system_prompt FROM agents WHERE name = ?", (role,))
    if agent:
        return agent["system_prompt"]
    from services.prompt_library import get_active_prompt
    return get_active_prompt("default_assistant")


# ── Workflow management (unchanged) ───────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_workflow(name: str) -> str:
    wf_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO workflows (id, name, status, created_at, updated_at) VALUES (?, ?, 'pending', ?, ?)",
        (wf_id, name, _now(), _now()),
    )
    db.commit()
    return wf_id


def add_task(
    workflow_id: str,
    name: str,
    agent_role: str,
    input_data: dict,
    depends_on: list[str] | None = None,
    max_attempts: int = 3,
) -> str:
    task_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO tasks
            (id, workflow_id, name, agent_role, status, depends_on,
             input_data, output_data, attempt_count, max_attempts, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'pending', ?, ?, '{}', 0, ?, ?, ?)
        """,
        (
            task_id, workflow_id, name, agent_role,
            json.dumps(depends_on or []),
            json.dumps(input_data),
            max_attempts, _now(), _now(),
        ),
    )
    db.commit()
    return task_id


def get_workflow_status(workflow_id: str) -> dict:
    wf = db.fetchone("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
    if not wf:
        return {}
    tasks = db.fetchall("SELECT * FROM tasks WHERE workflow_id = ?", (workflow_id,))
    runs  = db.fetchall(
        "SELECT ar.*, t.name as task_name FROM agent_runs ar "
        "JOIN tasks t ON ar.task_id = t.id WHERE t.workflow_id = ?",
        (workflow_id,),
    )
    total_input  = sum(r["input_tokens"]  or 0 for r in runs)
    total_output = sum(r["output_tokens"] or 0 for r in runs)
    cost_usd = (total_input * 3.0 + total_output * 15.0) / 1_000_000
    return {
        **dict(wf),
        "tasks": [dict(t) for t in tasks],
        "total_input_tokens":  total_input,
        "total_output_tokens": total_output,
        "estimated_cost_usd":  round(cost_usd, 4),
    }


def list_workflows(limit: int = 20) -> list[dict]:
    rows = db.fetchall(
        "SELECT id, name, status, created_at, updated_at FROM workflows ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


# ── Task execution engine ─────────────────────────────────────────────────────

def _mark_task(task_id: str, status: str, output_data: dict | None = None,
               error_message: str = "") -> None:
    db.execute(
        "UPDATE tasks SET status = ?, output_data = ?, error_message = ?, updated_at = ? WHERE id = ?",
        (status, json.dumps(output_data or {}), error_message or None, _now(), task_id),
    )
    db.execute("UPDATE tasks SET attempt_count = attempt_count + 1 WHERE id = ?", (task_id,))
    db.commit()


def _mark_workflow(workflow_id: str, status: str) -> None:
    db.execute("UPDATE workflows SET status = ?, updated_at = ? WHERE id = ?",
               (status, _now(), workflow_id))
    db.commit()


def _get_ready_tasks(workflow_id: str) -> list[dict]:
    tasks = db.fetchall(
        "SELECT * FROM tasks WHERE workflow_id = ? AND status = 'pending'",
        (workflow_id,),
    )
    succeeded_ids = {
        r["id"] for r in db.fetchall(
            "SELECT id FROM tasks WHERE workflow_id = ? AND status IN ('succeeded', 'skipped')",
            (workflow_id,),
        )
    }
    ready = []
    for t in tasks:
        deps = json.loads(t["depends_on"] or "[]")
        if all(d in succeeded_ids for d in deps):
            ready.append(dict(t))
    return ready


# ── Priority 3: HandoffPacket logging ─────────────────────────────────────────

def _log_handoff_packet(packet: HandoffPacket) -> None:
    """Persist a HandoffPacket to the handoff_log table."""
    try:
        db.execute(
            """
            INSERT INTO handoff_log
                (packet_id, workflow_id, step_index, agent_id, agent_name,
                 subtask_completed, artifact_summary, assumptions_json,
                 uncertainties_json, confidence, validation_passed,
                 validation_notes_json, duration_ms, input_tokens, output_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                packet.workflow_id,
                packet.step_index,
                packet.agent_id,
                packet.agent_name,
                packet.subtask_completed[:500],
                packet.artifact[:1000],
                json.dumps(packet.assumptions),
                json.dumps(packet.uncertainties),
                packet.confidence,
                1 if packet.validation_passed else 0,
                json.dumps(packet.validation_notes),
                packet.duration_ms,
                packet.input_tokens,
                packet.output_tokens,
                _now(),
            ),
        )
        db.commit()
    except Exception as exc:
        log.warning("handoff_log write failed: %s", exc)


def get_workflow_handoffs(workflow_id: str) -> list[dict]:
    """Return all HandoffPackets for a workflow as dicts."""
    rows = db.fetchall(
        "SELECT * FROM handoff_log WHERE workflow_id = ? ORDER BY step_index ASC, created_at ASC",
        (workflow_id,),
    )
    result = []
    for r in rows:
        d = dict(r)
        d["assumptions"]       = json.loads(d.pop("assumptions_json",  "[]"))
        d["uncertainties"]     = json.loads(d.pop("uncertainties_json","[]"))
        d["validation_notes"]  = json.loads(d.pop("validation_notes_json", "[]"))
        result.append(d)
    return result


# ── Priority 4: Saga checkpoint helpers ──────────────────────────────────────

def _write_checkpoint(
    task_id:         str,
    workflow_id:     str,
    step_index:      int,
    agent_id:        str,
    agent_name:      str,
    artifact:        str,
    success_criteria: str,
    retry_count:     int = 0,
) -> str:
    """Write a provisional checkpoint. Returns checkpoint_id."""
    checkpoint_id = str(uuid.uuid4())
    try:
        db.execute(
            """
            INSERT INTO workflow_checkpoints
                (checkpoint_id, workflow_id, step_index, task_id, agent_id, agent_name,
                 state, success_criteria, artifact_summary, retry_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'provisional', ?, ?, ?, ?)
            """,
            (
                checkpoint_id, workflow_id, step_index, task_id, agent_id, agent_name,
                success_criteria, artifact[:1000], retry_count, _now(),
            ),
        )
        db.commit()
    except Exception as exc:
        log.warning("_write_checkpoint failed: %s", exc)
    return checkpoint_id


def _run_validation_gate(
    claude_client,
    agent_name:       str,
    artifact:         str,
    success_criteria: str,
    local_client=None,
) -> dict:
    """
    Lightweight validation check: does artifact satisfy success_criteria?
    Returns {"passed": bool, "confidence": float, "reasoning": str, "gaps": [str]}.

    Uses local model first (free) with Claude as fallback, following the
    worker-judge pattern from adversarial_debate.py.
    Falls back to auto-pass if criteria is empty or both calls fail.
    """
    if not success_criteria.strip():
        return {"passed": True, "confidence": 1.0, "reasoning": "No criteria defined — auto-passed.", "gaps": []}

    system = "You are a precise quality validator. Respond ONLY with valid JSON."
    prompt = (
        f"Check whether this agent's output satisfies the stated criteria.\n\n"
        f"SUCCESS CRITERIA:\n{success_criteria}\n\n"
        f"AGENT: {agent_name}\n"
        f"OUTPUT (truncated):\n{artifact[:2000]}\n\n"
        "Respond ONLY with JSON:\n"
        '{{"passed": true, "confidence": 0.85, "reasoning": "...", "gaps": []}}\n'
        "OR\n"
        '{{"passed": false, "confidence": 0.45, "reasoning": "...", "gaps": ["specific gap 1"]}}\n\n'
        "Rules: passed=true only if ALL criteria are met. gaps must be specific — never vague."
    )

    # Try local model first (free)
    from services.task_artifacts import local_first_call
    raw = local_first_call(local_client, claude_client, system, prompt, max_tokens=512)

    if raw:
        try:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0].strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(raw[start:end + 1])
                return {
                    "passed":     bool(data.get("passed", False)),
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning":  str(data.get("reasoning", "")),
                    "gaps":       [str(g) for g in data.get("gaps", []) if g],
                }
        except Exception as exc:
            log.debug("Validation gate JSON parse failed: %s", exc)
    except Exception as exc:
        log.warning("Validation gate error: %s — auto-passing", exc)
        return {"passed": True, "confidence": 0.5, "reasoning": f"Gate failed ({exc}) — auto-passed.", "gaps": ["Validation gate unavailable"]}


def _commit_checkpoint(checkpoint_id: str, validation: dict) -> None:
    try:
        db.execute(
            """
            UPDATE workflow_checkpoints
            SET state='committed', validation_passed=1,
                confidence_score=?, validation_reasoning=?,
                known_gaps_json=?, validated_at=?, committed_at=?
            WHERE checkpoint_id=?
            """,
            (
                validation["confidence"], validation["reasoning"],
                json.dumps(validation["gaps"]), _now(), _now(), checkpoint_id,
            ),
        )
        db.commit()
    except Exception as exc:
        log.warning("_commit_checkpoint failed: %s", exc)


def _rollback_checkpoint(checkpoint_id: str, validation: dict, reason: str) -> None:
    try:
        db.execute(
            """
            UPDATE workflow_checkpoints
            SET state='rolled_back', validation_passed=0,
                confidence_score=?, validation_reasoning=?,
                known_gaps_json=?, failure_reason=?,
                validated_at=?, rolled_back_at=?
            WHERE checkpoint_id=?
            """,
            (
                validation["confidence"], validation["reasoning"],
                json.dumps(validation["gaps"]), reason[:400],
                _now(), _now(), checkpoint_id,
            ),
        )
        db.commit()
    except Exception as exc:
        log.warning("_rollback_checkpoint failed: %s", exc)


def get_workflow_checkpoints(workflow_id: str) -> list[dict]:
    """Return all saga checkpoints for a workflow."""
    rows = db.fetchall(
        "SELECT * FROM workflow_checkpoints WHERE workflow_id = ? ORDER BY step_index ASC, created_at ASC",
        (workflow_id,),
    )
    result = []
    for r in rows:
        d = dict(r)
        d["known_gaps"] = json.loads(d.pop("known_gaps_json", "[]"))
        result.append(d)
    return result


# ── _run_task (extended with HandoffPacket + Saga) ────────────────────────────

def _run_task(
    task: dict,
    claude_client,
    local_client,
    workflow_id: str,
    done_event: threading.Event,
    upstream_packets: list[HandoffPacket] | None = None,
    on_event: Callable | None = None,
    success_criteria: str = "",
) -> None:
    """
    Execute a single task.

    Priority 3: Extracts HandoffPacket from response and logs it.
    Priority 4: Wraps successful execution in saga checkpoint semantics.
    Always calls done_event.set() when finished.
    """
    task_id = task["id"]
    role    = task["agent_role"]
    name    = task["name"]

    db.execute("UPDATE tasks SET status = 'running', updated_at = ? WHERE id = ?",
               (_now(), task_id))
    db.commit()
    log.info(f"[{workflow_id[:8]}] Task '{name}' ({role}) starting…")

    system  = get_role_prompt(role)
    input_d = json.loads(task.get("input_data") or "{}")

    # ── Build upstream context (original dependency-based context) ────────────
    deps = json.loads(task.get("depends_on") or "[]")
    upstream_context = {}
    for dep_id in deps:
        dep_row = db.fetchone("SELECT name, output_data FROM tasks WHERE id = ?", (dep_id,))
        if dep_row:
            upstream_context[dep_row["name"]] = json.loads(dep_row["output_data"] or "{}")

    # ── Priority 3: prepend HandoffPacket context blocks ─────────────────────
    handoff_context_str = ""
    if upstream_packets:
        blocks = [p.to_context_block() for p in upstream_packets]
        # Add confidence warnings for low-confidence upstream outputs
        warnings = []
        for p in upstream_packets:
            if p.confidence < 0.5:
                warnings.append(
                    f"WARNING: Upstream step '{p.subtask_completed}' by {p.agent_name} "
                    f"reported LOW CONFIDENCE ({p.confidence:.0%}). "
                    f"Uncertainties: {', '.join(p.uncertainties) if p.uncertainties else 'unspecified'}. "
                    f"Verify these assumptions before proceeding."
                )
        warning_block = ""
        if warnings:
            warning_block = "## Upstream Confidence Warnings\n" + "\n".join(warnings) + "\n\n"
        handoff_context_str = warning_block + "## Context from upstream agents:\n\n" + "\n\n".join(blocks) + "\n\n"

    user_msg_parts = [json.dumps(input_d, indent=2)]
    if upstream_context:
        user_msg_parts.insert(0, "## Context from upstream tasks:\n" +
                              json.dumps(upstream_context, indent=2) + "\n\n## Your task input:")
    user_msg = "\n".join(user_msg_parts)

    # Prepend handoff context if available
    if handoff_context_str:
        user_msg = handoff_context_str + user_msg

    # Priority 3: append handoff format instruction
    user_msg = user_msg + "\n\n" + HANDOFF_SYSTEM_FRAGMENT

    agent     = db.fetchone("SELECT model_preference FROM agents WHERE name = ?", (role,))
    model_pref = agent["model_preference"] if agent else "auto"
    use_local  = (
        model_pref == "local"
        and local_client is not None
        and local_client.is_available()
    )

    run_id  = str(uuid.uuid4())
    started = _now()
    t0      = time.monotonic()

    try:
        with _API_SEMAPHORE:
            prompt_hash = hashlib.sha256(system.encode()).hexdigest()[:16]

            if use_local:
                response      = local_client.chat(system, user_msg)
                model_used    = "local"
                input_tokens  = 0
                output_tokens = 0
            else:
                result        = claude_client.chat_multi_turn(
                    system,
                    [{"role": "user", "content": user_msg}],
                    max_tokens=4096,
                )
                response      = result["text"]
                model_used    = claude_client._model
                input_tokens  = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)

        duration_ms = (time.monotonic() - t0) * 1000

        # ── Original output_data extraction (unchanged) ───────────────────────
        raw = response.strip()
        # Strip content before the <handoff> tag for JSON parsing
        handoff_start = raw.find("<handoff>")
        raw_for_json  = raw[:handoff_start].strip() if handoff_start != -1 else raw

        if raw_for_json.startswith("```"):
            raw_for_json = raw_for_json.split("```")[1]
            if raw_for_json.startswith("json"):
                raw_for_json = raw_for_json[4:]
        try:
            output_data = json.loads(raw_for_json)
        except json.JSONDecodeError:
            output_data = {"raw_output": response}

        # ── Priority 3: extract HandoffPacket ────────────────────────────────
        agent_row = db.fetchone("SELECT id FROM agents WHERE name = ?", (role,))
        agent_id  = agent_row["id"] if agent_row else role

        # ── Safety gate: scan agent output for dangerous content ─────────────
        try:
            from services.safety_gate import scan_workflow_task, RiskLevel
            safety_verdict = scan_workflow_task(
                task_name=name, agent_role=role,
                input_data=json.dumps(input_d),
                output_data=response,
            )
            if safety_verdict.level == RiskLevel.BLOCK:
                log.warning("Safety gate BLOCKED task '%s': %s", name, safety_verdict.reason)
                _mark_task(task_id, "failed",
                           {"error": f"Blocked by safety gate: {safety_verdict.reason}"},
                           error_message=f"Safety gate: {safety_verdict.reason}")
                if on_event:
                    try:
                        on_event("safety_blocked", {
                            "task_name": name, "reason": safety_verdict.reason,
                            "pattern": safety_verdict.pattern,
                        })
                    except Exception:
                        pass
                done_event.set()
                return
            if safety_verdict.level == RiskLevel.WARN:
                log.warning("Safety gate WARNING for task '%s': %s", name, safety_verdict.reason)
                if on_event:
                    try:
                        on_event("safety_warning", {
                            "task_name": name, "reason": safety_verdict.reason,
                            "pattern": safety_verdict.pattern,
                        })
                    except Exception:
                        pass
        except ImportError:
            pass  # safety_gate not available — skip

        packet = extract_handoff_packet(
            raw_response  = response,
            agent_id      = agent_id,
            agent_name    = name,
            workflow_id   = workflow_id,
            step_index    = task.get("attempt_count", 0),
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
            duration_ms   = duration_ms,
        )
        _log_handoff_packet(packet)

        # Write progress artifact to disk
        try:
            from services.task_artifacts import write_workflow_progress
            from pathlib import Path as _WP
            write_workflow_progress(
                _WP.cwd(), workflow_id, packet.step_index,
                {
                    "agent_name": packet.agent_name,
                    "subtask": packet.subtask_completed,
                    "artifact_preview": (packet.artifact or "")[:500],
                    "confidence": packet.confidence,
                    "assumptions": packet.assumptions,
                    "uncertainties": packet.uncertainties,
                },
            )
        except Exception:
            pass  # progress tracking is best-effort

        if on_event:
            try:
                on_event("handoff", packet.to_dict())
            except Exception:
                pass

        # ── Log to agent_runs ─────────────────────────────────────────────────
        finished = _now()
        db.execute(
            "INSERT INTO agent_runs (id, task_id, model, system_prompt_hash, "
            "input_tokens, output_tokens, started_at, finished_at, result_summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, task_id, model_used, prompt_hash,
             input_tokens, output_tokens, started, finished,
             str(output_data)[:200]),
        )

        # ── Priority 4: Saga checkpoint ───────────────────────────────────────
        retry_count   = task.get("attempt_count", 0)
        checkpoint_id = _write_checkpoint(
            task_id=task_id, workflow_id=workflow_id,
            step_index=task.get("attempt_count", 0),
            agent_id=agent_id, agent_name=name,
            artifact=packet.artifact or str(output_data)[:500],
            success_criteria=success_criteria,
            retry_count=retry_count,
        )

        if success_criteria.strip() and not use_local:
            validation = _run_validation_gate(
                claude_client, name, packet.artifact or str(output_data), success_criteria,
                local_client=local_client,
            )
            if validation["passed"]:
                _commit_checkpoint(checkpoint_id, validation)
                if on_event:
                    try:
                        on_event("saga_committed", {
                            "task_name": name, "checkpoint_id": checkpoint_id,
                            "confidence": validation["confidence"],
                            "reasoning": validation["reasoning"],
                        })
                    except Exception:
                        pass
            else:
                _rollback_checkpoint(checkpoint_id, validation,
                                     f"Validation failed: {validation['reasoning'][:200]}")
                if on_event:
                    try:
                        on_event("saga_rolled_back", {
                            "task_name": name, "checkpoint_id": checkpoint_id,
                            "confidence": validation["confidence"],
                            "gaps": validation["gaps"],
                        })
                    except Exception:
                        pass
                # Rollback: let the existing retry mechanism re-run this task
                # if it still has remaining attempts — just mark as pending
                current = db.fetchone("SELECT attempt_count, max_attempts FROM tasks WHERE id = ?", (task_id,))
                if current and current["attempt_count"] < current["max_attempts"] - 1:
                    db.execute(
                        "UPDATE tasks SET status='pending', error_message=?, updated_at=? WHERE id=?",
                        (f"Saga rollback: {validation['reasoning'][:200]}", _now(), task_id),
                    )
                    db.execute("UPDATE tasks SET attempt_count = attempt_count + 1 WHERE id = ?", (task_id,))
                    db.commit()
                    log.info(f"[{workflow_id[:8]}] Task '{name}' rolled back — will retry.")
                    return  # done_event.set() happens in finally
        else:
            # No criteria or local model — auto-commit the checkpoint
            _commit_checkpoint(checkpoint_id, {
                "confidence": packet.confidence, "reasoning": "Auto-committed (no criteria or local model).", "gaps": []
            })

        _mark_task(task_id, "succeeded", output_data)
        log.info(f"[{workflow_id[:8]}] Task '{name}' succeeded via {model_used}.")

    except Exception as exc:
        error_msg = str(exc)
        log.error(f"[{workflow_id[:8]}] Task '{name}' failed: {error_msg}")

        action = "fail"
        if claude_client is not None:
            try:
                task_row = db.fetchone("SELECT * FROM tasks WHERE id = ?", (task_id,))
                if task_row:
                    action = _handle_task_failure(dict(task_row), error_msg, claude_client)
            except Exception as heal_exc:
                log.warning("Self-heal lookup failed: %s", heal_exc)

        if action == "retry":
            db.execute(
                "UPDATE tasks SET status = 'pending', error_message = ?, updated_at = ? WHERE id = ?",
                (f"[retrying after: {error_msg}]", _now(), task_id),
            )
            db.commit()
            log.info(f"[{workflow_id[:8]}] Task '{name}' queued for retry.")
        elif action == "skip":
            _mark_task(task_id, "skipped",
                       output_data={"skipped": True, "reason": error_msg})
            log.info(f"[{workflow_id[:8]}] Task '{name}' skipped by coordinator.")
        else:
            _mark_task(task_id, "failed", error_message=error_msg)

        from services.error_classifier import log_error
        log_error(exc, component=f"task:{role}", workflow_id=workflow_id, task_id=task_id)

    finally:
        done_event.set()


# ── Self-healing task failure handler (unchanged) ─────────────────────────────

def _handle_task_failure(task: dict, error: str, claude_client) -> str:
    from services.prompt_library import get_active_prompt
    system = get_active_prompt("coordinator_agent")
    recovery_prompt = (
        "A task in the workflow has failed. Decide how to recover.\n\n"
        f"Task name: {task.get('name', 'unknown')}\n"
        f"Agent role: {task.get('agent_role', 'unknown')}\n"
        f"Input data: {json.dumps(json.loads(task.get('input_data') or '{}'))[:400]}\n"
        f"Error: {error[:600]}\n\n"
        "Reply with ONLY one of these JSON objects — nothing else:\n"
        '{"action": "retry", "reason": "..."}\n'
        '{"action": "skip",  "reason": "..."}\n'
        '{"action": "fail",  "reason": "..."}\n\n'
        "Use retry when the error is transient (network, rate limit, parse error).\n"
        "Use skip when the task is optional and downstream tasks can proceed without it.\n"
        "Use fail when the error is unrecoverable and the workflow goal cannot be met."
    )
    try:
        raw = claude_client.chat(system, "", recovery_prompt, max_tokens=256)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw.strip())
        action   = decision.get("action", "fail").lower()
        if action in ("retry", "skip", "fail"):
            log.info("Self-heal decision for task '%s': %s — %s",
                     task.get("name"), action, decision.get("reason", ""))
            return action
        return "fail"
    except Exception as exc:
        log.warning("Self-heal decision failed (%s); defaulting to fail", exc)
        return "fail"


# ── run_workflow (updated to pass HandoffPackets) ─────────────────────────────

def run_workflow(
    workflow_id:     str,
    claude_client,
    local_client=None,
    on_status:       Callable[[dict], None] | None = None,
    on_event:        Callable | None = None,
    success_criteria_map: dict[str, str] | None = None,
) -> dict:
    """
    Execute all tasks in a workflow, respecting dependencies.
    Runs synchronously (call from a background thread).
    Returns final workflow status dict.

    Priority 3: Collects HandoffPackets and passes them to downstream tasks.
    Priority 4: Passes success_criteria per task for validation gate.
    """
    _mark_workflow(workflow_id, "running")
    done_event  = threading.Event()
    criteria    = success_criteria_map or {}

    # Collect HandoffPackets as tasks complete (keyed by task_id)
    completed_packets: dict[str, HandoffPacket] = {}
    packets_lock = threading.Lock()

    def _get_upstream_packets(task: dict) -> list[HandoffPacket]:
        """Return HandoffPackets from all predecessor tasks."""
        deps = json.loads(task.get("depends_on") or "[]")
        with packets_lock:
            return [completed_packets[dep_id] for dep_id in deps if dep_id in completed_packets]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        in_flight: set[concurrent.futures.Future] = set()

        while True:
            ready = _get_ready_tasks(workflow_id)

            if not ready and not in_flight:
                remaining = db.fetchall(
                    "SELECT status FROM tasks WHERE workflow_id = ? "
                    "AND status NOT IN ('succeeded', 'skipped', 'failed')",
                    (workflow_id,),
                )
                if not remaining:
                    break

                failed = db.fetchall(
                    "SELECT id FROM tasks WHERE workflow_id = ? AND status = 'failed'",
                    (workflow_id,),
                )
                if failed:
                    _mark_workflow(workflow_id, "failed")
                    return get_workflow_status(workflow_id)

                log.error(f"[{workflow_id[:8]}] Workflow stalled.")
                _mark_workflow(workflow_id, "failed")
                return get_workflow_status(workflow_id)

            for task in ready:
                done_event.clear()
                upstream_pkts = _get_upstream_packets(task)
                task_criteria = criteria.get(task["id"], criteria.get(task["name"], ""))

                def _make_worker(t, pkts, crit):
                    def _worker():
                        _run_task(t, claude_client, local_client, workflow_id,
                                  done_event,
                                  upstream_packets=pkts,
                                  on_event=on_event,
                                  success_criteria=crit)
                        # Collect the packet after task completes
                        handoffs = get_workflow_handoffs(workflow_id)
                        task_handoffs = [h for h in handoffs if h.get("agent_name") == t["agent_role"] or True]
                        # Re-read the latest packet for this task from DB
                        rows = db.fetchall(
                            "SELECT * FROM handoff_log WHERE workflow_id = ? ORDER BY created_at DESC LIMIT 1",
                            (workflow_id,),
                        )
                        if rows:
                            r = dict(rows[0])
                            pkt = HandoffPacket(
                                agent_id   = r["agent_id"],
                                agent_name = r["agent_name"],
                                subtask_completed = r["subtask_completed"],
                                artifact   = r["artifact_summary"] or "",
                                assumptions  = json.loads(r.get("assumptions_json", "[]")),
                                uncertainties= json.loads(r.get("uncertainties_json", "[]")),
                                confidence = r["confidence"],
                                workflow_id = workflow_id,
                            )
                            with packets_lock:
                                completed_packets[t["id"]] = pkt
                    return _worker

                fut = pool.submit(_make_worker(task, upstream_pkts, task_criteria))
                in_flight.add(fut)

            if in_flight:
                done_event.wait(timeout=300)
                done_event.clear()
                finished = {f for f in in_flight if f.done()}
                in_flight -= finished
                if on_status:
                    on_status(get_workflow_status(workflow_id))

    failed = db.fetchall(
        "SELECT id FROM tasks WHERE workflow_id = ? AND status = 'failed'", (workflow_id,)
    )

    # ── Coordinator synthesis step ───────────────────────────────────────────
    # If workflow succeeded and has multiple completed tasks, run a final
    # synthesis step that assembles all handoff artifacts into a coherent output.
    if not failed and len(completed_packets) > 1:
        try:
            all_packets = list(completed_packets.values())
            blocks = [p.to_context_block() for p in all_packets]
            synthesis_prompt = (
                "You are the coordinator synthesizing a multi-agent workflow.\n\n"
                "## All Agent Outputs\n\n" + "\n\n".join(blocks) + "\n\n"
                "## Instructions\n"
                "Combine all outputs into one coherent, well-structured final answer. "
                "Resolve any conflicts between agents. Preserve all key findings. "
                "Note any unresolved uncertainties. Do NOT repeat redundant information."
            )
            from services.task_artifacts import local_first_call
            synthesis = local_first_call(
                local_client, claude_client,
                synthesis_prompt,
                "Synthesize the workflow results into a final answer.",
                max_tokens=2048,
            )
            if synthesis:
                db.execute(
                    "UPDATE workflows SET output_data = ?, updated_at = ? WHERE id = ?",
                    (synthesis[:5000], _now(), workflow_id),
                )
                if on_event:
                    on_event("coordinator_synthesis", {
                        "workflow_id": workflow_id,
                        "synthesis_preview": synthesis[:300],
                    })
                log.info("Coordinator synthesis completed for workflow %s", workflow_id[:8])
        except Exception as exc:
            log.debug("Coordinator synthesis skipped: %s", exc)

    final_status = "failed" if failed else "succeeded"
    _mark_workflow(workflow_id, final_status)
    return get_workflow_status(workflow_id)


# ── Coordinator: decompose user request (unchanged + success_criteria) ────────

def _validate_task_defs(task_defs: list, available_roles: list[str]) -> None:
    if not isinstance(task_defs, list) or not task_defs:
        raise ValueError("Coordinator must return a non-empty JSON array of task objects.")

    available_roles_set = set(available_roles)
    seen_names: set[str] = set()

    for i, t in enumerate(task_defs):
        if not isinstance(t, dict):
            raise ValueError(f"Task at index {i} is not a JSON object: {t!r}")
        name = t.get("name", "").strip()
        if not name:
            raise ValueError(f"Task at index {i} is missing a non-empty 'name' field.")
        if name in seen_names:
            raise ValueError(f"Duplicate task name '{name}'.")
        seen_names.add(name)
        role = t.get("agent_role", "")
        if role and role not in available_roles_set:
            log.warning(f"Task '{name}' references unknown role '{role}'; defaulting.")
            t["agent_role"] = "General Assistant"

    for t in task_defs:
        for dep in t.get("depends_on", []):
            if dep not in seen_names:
                raise ValueError(f"Task '{t['name']}' depends_on '{dep}' which is not in this plan.")

    in_degree = {t["name"]: 0 for t in task_defs}
    adj: dict[str, list[str]] = {t["name"]: [] for t in task_defs}
    for t in task_defs:
        for dep in t.get("depends_on", []):
            adj[dep].append(t["name"])
            in_degree[t["name"]] += 1

    queue   = [n for n, d in in_degree.items() if d == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for nbr in adj[node]:
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)

    if visited != len(task_defs):
        raise ValueError("Coordinator produced a task graph with a dependency cycle.")


def plan_workflow(
    user_goal: str,
    claude_client,
    workflow_name: str = "Auto workflow",
) -> str:
    """
    Ask the Coordinator agent to decompose a goal into tasks.
    Priority 3/4: Coordinator prompt now requests success_criteria per task.
    """
    from services.prompt_library import get_active_prompt

    available_roles = get_available_roles()
    system = get_active_prompt("coordinator_agent")
    prompt = (
        f"Available agent roles: {', '.join(available_roles)}\n\n"
        f"User goal: {user_goal}\n\n"
        "Return a JSON array of task objects. Each task object must have:\n"
        '  {"name": str, "agent_role": str, "description": str, "depends_on": [], "success_criteria": str}\n\n'
        "success_criteria: what the output MUST include for this task to be considered complete. "
        "Be specific — e.g. 'Must include at least 3 cited sources, a summary under 200 words, and a confidence rating.'"
    )

    raw = claude_client.chat(system, "", prompt, max_tokens=4096)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        task_defs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Coordinator returned invalid JSON: {exc}\n\nRaw output:\n{raw}") from exc

    _validate_task_defs(task_defs, available_roles)

    wf_id      = create_workflow(workflow_name)
    name_to_id = {t["name"]: str(uuid.uuid4()) for t in task_defs}

    for t in task_defs:
        task_id = name_to_id[t["name"]]
        dep_ids = [name_to_id[dep] for dep in t.get("depends_on", [])]
        db.execute(
            """
            INSERT INTO tasks
                (id, workflow_id, name, agent_role, status, depends_on,
                 input_data, output_data, attempt_count, max_attempts, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'pending', ?, ?, '{}', 0, 3, ?, ?)
            """,
            (
                task_id, wf_id,
                t["name"],
                t.get("agent_role", "General Assistant"),
                json.dumps(dep_ids),
                json.dumps({
                    "description":       t.get("description", ""),
                    "success_criteria":  t.get("success_criteria", ""),
                }),
                _now(), _now(),
            ),
        )

    db.commit()
    return wf_id


def get_success_criteria_map(workflow_id: str) -> dict[str, str]:
    """
    Extract success_criteria from task input_data for all tasks in a workflow.
    Returns {task_id: criteria_string}.
    """
    rows = db.fetchall("SELECT id, input_data FROM tasks WHERE workflow_id = ?", (workflow_id,))
    result = {}
    for r in rows:
        try:
            d = json.loads(r["input_data"] or "{}")
            crit = d.get("success_criteria", "")
            if crit:
                result[r["id"]] = crit
        except Exception:
            pass
    return result


# ── P6: Conditional routing ──────────────────────────────────────────────────

ROUTE_PROCEED_THRESHOLD  = 0.7
ROUTE_REVIEW_THRESHOLD   = 0.3


def route_after_validation(
    validation: dict,
    task: dict,
    workflow_id: str,
    on_event=None,
) -> str:
    """
    Decide what to do after a validation gate based on confidence.
    Returns: "proceed" | "review" | "halt"

    Inspired by LangGraph conditional edges — inspect state and decide
    the next step dynamically instead of following linear dependencies.
    """
    confidence = validation.get("confidence", 0.5)
    task_name = task.get("name", "unknown")

    if confidence >= ROUTE_PROCEED_THRESHOLD:
        return "proceed"

    if confidence >= ROUTE_REVIEW_THRESHOLD:
        log.info("[%s] Task '%s' confidence %.0f%% — inserting review step",
                 workflow_id[:8], task_name, confidence * 100)
        if on_event:
            try:
                on_event("route_review", {
                    "workflow_id": workflow_id,
                    "task_name": task_name,
                    "confidence": confidence,
                    "gaps": validation.get("gaps", []),
                })
            except Exception:
                pass
        return "review"

    log.warning("[%s] Task '%s' confidence %.0f%% — halting for human decision",
                workflow_id[:8], task_name, confidence * 100)
    if on_event:
        try:
            on_event("route_halt", {
                "workflow_id": workflow_id,
                "task_name": task_name,
                "confidence": confidence,
                "gaps": validation.get("gaps", []),
                "action_needed": "Human review required",
            })
        except Exception:
            pass
    return "halt"


# ── P6: Interrupt / Resume ───────────────────────────────────────────────────

_workflow_interrupts: dict[str, threading.Event] = {}
_interrupt_decisions: dict[str, str] = {}
_interrupt_lock = threading.Lock()


def interrupt_workflow(workflow_id: str, reason: str) -> None:
    """Pause a running workflow for human decision."""
    evt = threading.Event()
    with _interrupt_lock:
        _workflow_interrupts[workflow_id] = evt
    log.info("[%s] Workflow interrupted: %s", workflow_id[:8], reason)
    _mark_workflow(workflow_id, "interrupted")


def resume_workflow_decision(workflow_id: str, decision: str = "proceed") -> bool:
    """
    Resume an interrupted workflow. decision: "proceed" | "halt" | "retry"
    Returns True if workflow was interrupted and is now resuming.
    """
    with _interrupt_lock:
        evt = _workflow_interrupts.pop(workflow_id, None)
        _interrupt_decisions[workflow_id] = decision
    if evt:
        evt.set()
        if decision == "proceed":
            _mark_workflow(workflow_id, "running")
        return True
    return False


def wait_for_resume(workflow_id: str, timeout: float = 600.0) -> str:
    """Block until resume_workflow_decision() is called. Returns decision or 'timeout'."""
    with _interrupt_lock:
        evt = _workflow_interrupts.get(workflow_id)
    if not evt:
        return "proceed"
    if evt.wait(timeout=timeout):
        with _interrupt_lock:
            return _interrupt_decisions.pop(workflow_id, "proceed")
    with _interrupt_lock:
        _workflow_interrupts.pop(workflow_id, None)
        _interrupt_decisions.pop(workflow_id, None)
    return "timeout"


# ── P6: Replay from checkpoint ───────────────────────────────────────────────

def replay_workflow_from_checkpoint(
    workflow_id: str,
    checkpoint_id: str,
    claude_client,
    local_client=None,
    on_status=None,
    on_event=None,
) -> dict:
    """Resume a workflow from a specific checkpoint (time-travel)."""
    cp = db.fetchone(
        "SELECT * FROM workflow_checkpoints WHERE checkpoint_id = ? AND workflow_id = ?",
        (checkpoint_id, workflow_id),
    )
    if not cp:
        return {"error": f"Checkpoint {checkpoint_id} not found"}

    task_id = cp["task_id"]
    all_tasks = db.fetchall(
        "SELECT id, name FROM tasks WHERE workflow_id = ? ORDER BY created_at",
        (workflow_id,),
    )
    task_ids = [t["id"] for t in all_tasks]
    try:
        start_idx = task_ids.index(task_id)
    except ValueError:
        return {"error": f"Task {task_id} not found in workflow"}

    for t in all_tasks[start_idx:]:
        db.execute(
            "UPDATE tasks SET status='pending', output_data='{}', "
            "error_message=NULL, updated_at=? WHERE id=?",
            (_now(), t["id"]),
        )
    db.commit()

    log.info("[%s] Replaying from checkpoint %s", workflow_id[:8], checkpoint_id[:8])
    return run_workflow(
        workflow_id, claude_client, local_client,
        on_status=on_status, on_event=on_event,
    )
