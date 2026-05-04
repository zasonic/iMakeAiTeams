"""
services/pipeline.py — Team Pipeline Executor.

When an agent team is active, decomposes a user message into sub-tasks,
dispatches each to the appropriate specialist via HubRouter, chains
HandoffPackets between steps, and synthesises a final response.

Single-agent chat is unaffected — the pipeline only activates when the
orchestrator detects an active team (i.e. the selected agent is the
coordinator of an agent_teams row).

Uses existing infrastructure:
  - HubRouter.invoke() for all model calls (single boundary preserved)
  - HandoffPacket + HandoffValidation from models.py
  - handoff_log table from db.py
  - SSE events via on_event callback
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import db as _db
from models import (
    HandoffPacket,
    RoutingDecision,
    TaskDescriptor,
    validate_handoff_packet,
)
from services.hub_router import HubRouter
from services.redact import redact

log = logging.getLogger("iMakeAiTeams.pipeline")

# Maximum sub-tasks the coordinator can decompose into. Prevents runaway
# decomposition on adversarial or ambiguous inputs.
MAX_SUBTASKS = 6

# Maximum retries per specialist when HandoffPacket validation fails.
MAX_RETRIES_PER_STEP = 1

# Maximum HandoffPacket context injected into downstream agents (chars).
# Prevents context rot when many specialists contribute.
MAX_UPSTREAM_CONTEXT_CHARS = 12_000


@dataclass
class SubTask:
    """A single specialist assignment from the coordinator's decomposition."""
    agent_id: str
    agent_name: str
    description: str
    depends_on: list = field(default_factory=list)


@dataclass
class PipelineResult:
    """Outcome of a full pipeline run."""
    synthesis: str
    steps: list = field(default_factory=list)
    handoffs: list = field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    # Backend model name used for the synthesis step, e.g. "claude-sonnet-..."
    # or the configured local model name. The orchestrator uses this to
    # estimate cost; per-step cost attribution is a future iteration.
    synthesis_model: str = "pipeline"
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))


DECOMPOSITION_PROMPT = """You are a team coordinator. Break the user's request into sub-tasks for your specialists.

Available specialists:
{agent_list}

Return ONLY a JSON array. Each element:
{{
  "agent_id": "<id of the specialist>",
  "agent_name": "<name for display>",
  "description": "<what this specialist should do — be specific>"
}}

Rules:
- Order matters: earlier steps execute first, later steps can reference earlier results.
- Use 1 step if the task is simple enough for one specialist.
- Maximum {max_steps} steps.
- Every step must map to one of the listed specialists.
- If the task doesn't need specialisation, return a single step with yourself as the agent.
- Do NOT include a "synthesis" step — that happens automatically after all specialists finish.
"""

SYNTHESIS_PROMPT = """You are a team coordinator. Your specialists have completed their sub-tasks.
Synthesise their outputs into a single, coherent response for the user.

The user's original request: {user_message}

Specialist outputs:
{handoff_blocks}

Instructions:
- Combine the specialists' work into one clear response.
- Resolve any contradictions by noting them.
- If a specialist flagged low confidence or uncertainties, mention them briefly.
- Write as if YOU did the work — don't say "the researcher found..." unless attribution adds value.
- Keep the response focused on what the user asked for.
"""


class PipelineExecutor:
    """Executes a multi-agent pipeline for a team."""

    def __init__(self, hub_router: HubRouter, settings):
        self._hub = hub_router
        self._settings = settings

    def run(
        self,
        team_id: str,
        user_message: str,
        conversation_id: str,
        history: list,
        on_event: Optional[Callable] = None,
        on_token: Optional[Callable] = None,
    ) -> PipelineResult:
        """Execute the full pipeline: decompose -> specialists -> synthesise."""

        def emit(event_type: str, data: dict):
            if on_event:
                try:
                    on_event(event_type, data)
                except Exception:
                    pass

        pipeline_id = str(uuid.uuid4())
        emit("pipeline_started", {"pipeline_id": pipeline_id, "team_id": team_id})

        team = _db.fetchone("SELECT * FROM agent_teams WHERE id = ?", (team_id,))
        if not team:
            raise ValueError(f"Team not found: {team_id}")

        coordinator_id = team["coordinator_id"]
        coordinator_row = _db.fetchone(
            "SELECT * FROM agents WHERE id = ?", (coordinator_id,)
        )
        if not coordinator_row:
            raise ValueError(f"Coordinator not found: {coordinator_id}")
        coordinator = dict(coordinator_row)

        member_rows = _db.fetchall(
            "SELECT a.* FROM agents a "
            "JOIN agent_team_members atm ON atm.agent_id = a.id "
            "WHERE atm.team_id = ? AND a.id != ?",
            (team_id, coordinator_id),
        )
        members = [dict(m) for m in member_rows]

        if not members:
            log.info(
                "Team %s has no specialists; falling back to coordinator-only",
                team_id,
            )
            return self._single_agent_fallback(
                coordinator, user_message, history, pipeline_id, emit, on_token,
            )

        # ── Step 1: Coordinator decomposes ──────────────────────────────────
        emit("pipeline_decomposing", {"agent": coordinator["name"]})

        agent_list = "\n".join(
            f"- {m['name']} (id: {m['id']}, role: {m.get('role') or 'worker'}): "
            f"{(m.get('system_prompt') or '')[:150]}"
            for m in members
        )

        decomp_system = DECOMPOSITION_PROMPT.format(
            agent_list=agent_list,
            max_steps=MAX_SUBTASKS,
        )
        decomp_messages = [{"role": "user", "content": user_message}]

        coordinator_task = TaskDescriptor(
            text=user_message, preferred_agent_id=coordinator_id,
        )
        decomp_decision = self._hub.route_for_agent(coordinator_id, coordinator_task)
        decomp_result = self._hub.invoke(
            decomp_decision, decomp_system, decomp_messages, max_tokens=2048,
        )

        subtasks = self._parse_subtasks(decomp_result.text, members, coordinator)
        if not subtasks:
            log.warning(
                "Coordinator produced no subtasks; falling back to coordinator-only",
            )
            return self._single_agent_fallback(
                coordinator, user_message, history, pipeline_id, emit, on_token,
            )

        emit("pipeline_plan", {
            "pipeline_id": pipeline_id,
            "steps": [
                {"agent": s.agent_name, "task": s.description} for s in subtasks
            ],
        })

        # ── Step 2: Execute each sub-task ───────────────────────────────────
        handoffs: list[HandoffPacket] = []
        step_summaries: list[dict] = []

        for i, subtask in enumerate(subtasks):
            emit("pipeline_step_started", {
                "step": i + 1,
                "total": len(subtasks),
                "agent": subtask.agent_name,
                "task": subtask.description,
            })

            specialist_row = _db.fetchone(
                "SELECT * FROM agents WHERE id = ?", (subtask.agent_id,),
            )
            if not specialist_row:
                log.error("Specialist %s not found, skipping", subtask.agent_id)
                continue
            specialist = dict(specialist_row)

            specialist_system = (
                specialist.get("system_prompt") or "You are a helpful specialist."
            )

            upstream_context = self._build_upstream_context(handoffs)
            if upstream_context:
                specialist_system += "\n\n" + upstream_context

            spec_messages = [{
                "role": "user",
                "content": (
                    f"You are working as part of a team. Your specific task:\n\n"
                    f"{subtask.description}\n\n"
                    f"The user's original request was: {user_message}\n\n"
                    f"Complete your task thoroughly. Be specific and concrete in "
                    f"your output."
                ),
            }]

            spec_task = TaskDescriptor(
                text=subtask.description, preferred_agent_id=subtask.agent_id,
            )
            spec_decision = self._hub.route_for_agent(subtask.agent_id, spec_task)

            packet = self._invoke_specialist(
                spec_decision, specialist_system, spec_messages,
                subtask, pipeline_id, i,
            )

            if not packet.validation_passed and MAX_RETRIES_PER_STEP > 0:
                emit("pipeline_step_retry", {
                    "step": i + 1,
                    "agent": subtask.agent_name,
                    "reason": "; ".join(packet.validation_notes),
                })
                retry_messages = [{
                    "role": "user",
                    "content": (
                        f"Your previous response had issues: "
                        f"{'; '.join(packet.validation_notes)}\n\n"
                        f"Please redo this task:\n{subtask.description}\n\n"
                        f"Be more specific and concrete. State your uncertainties "
                        f"explicitly."
                    ),
                }]
                packet = self._invoke_specialist(
                    spec_decision, specialist_system, retry_messages,
                    subtask, pipeline_id, i,
                    is_retry=True,
                )

            self._log_handoff(packet)
            handoffs.append(packet)

            summary = {
                "step": i + 1,
                "agent": subtask.agent_name,
                "task": subtask.description,
                "confidence": packet.confidence_label,
                "validation_passed": packet.validation_passed,
                "tokens": packet.input_tokens + packet.output_tokens,
                "duration_ms": round(packet.duration_ms),
            }
            step_summaries.append(summary)
            emit("pipeline_step_complete", summary)

        # ── Step 3: Coordinator synthesises ─────────────────────────────────
        emit("pipeline_synthesising", {"agent": coordinator["name"]})

        handoff_blocks = "\n\n".join(h.to_context_block() for h in handoffs)

        synth_system = (
            coordinator.get("system_prompt") or "You are a team coordinator."
        )
        synth_messages = [{
            "role": "user",
            "content": SYNTHESIS_PROMPT.format(
                user_message=user_message,
                handoff_blocks=handoff_blocks,
            ),
        }]

        synth_decision = self._hub.route_for_agent(coordinator_id, coordinator_task)
        synth_result = self._hub.invoke(
            synth_decision, synth_system, synth_messages,
            max_tokens=4096, on_token=on_token,
        )

        emit("pipeline_complete", {
            "pipeline_id": pipeline_id,
            "steps_completed": len(step_summaries),
            "total_steps": len(subtasks),
        })

        total_in = (
            sum(h.input_tokens for h in handoffs)
            + (decomp_result.input_tokens or 0)
            + (synth_result.input_tokens or 0)
        )
        total_out = (
            sum(h.output_tokens for h in handoffs)
            + (decomp_result.output_tokens or 0)
            + (synth_result.output_tokens or 0)
        )

        return PipelineResult(
            synthesis=synth_result.text,
            steps=step_summaries,
            handoffs=handoffs,
            total_tokens_in=total_in,
            total_tokens_out=total_out,
            synthesis_model=synth_result.model_name or "pipeline",
            pipeline_id=pipeline_id,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _invoke_specialist(
        self,
        decision: RoutingDecision,
        system: str,
        messages: list,
        subtask: SubTask,
        pipeline_id: str,
        step_index: int,
        is_retry: bool = False,
    ) -> HandoffPacket:
        """Invoke a specialist and wrap the WorkerResult into a HandoffPacket."""
        start_ms = time.monotonic()
        result = self._hub.invoke(decision, system, messages, max_tokens=4096)
        elapsed_ms = (time.monotonic() - start_ms) * 1000

        # The specialists are unaware of the HandoffPacket schema (we don't
        # inject HANDOFF_SYSTEM_FRAGMENT to keep their prompts simple), so
        # we synthesise an uncertainties list. Without this, validation would
        # always fail at confidence < 0.95 and trigger a spurious retry.
        if result.had_error:
            confidence = 0.3
            uncertainties = ["Specialist invocation returned an error."]
        else:
            confidence = 0.6 if is_retry else 0.8
            uncertainties = [
                "Specialist did not self-assess; confidence is a pipeline default.",
            ]

        packet = HandoffPacket(
            agent_id=subtask.agent_id,
            agent_name=subtask.agent_name,
            subtask_completed=subtask.description,
            artifact=redact(result.text or ""),
            uncertainties=uncertainties,
            confidence=confidence,
            workflow_id=pipeline_id,
            step_index=step_index,
            raw_output=(result.text or "")[:2000],
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=elapsed_ms,
        )
        return validate_handoff_packet(packet)

    def _parse_subtasks(
        self, raw: str, members: list, coordinator: dict,
    ) -> list:
        """Parse the coordinator's JSON decomposition into SubTask objects."""
        text = (raw or "").strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            log.warning("Coordinator output is not valid JSON: %s", text[:200])
            return []

        if not isinstance(items, list):
            return []

        member_ids = {m["id"] for m in members}
        member_ids.add(coordinator["id"])

        subtasks: list[SubTask] = []
        for item in items[:MAX_SUBTASKS]:
            if not isinstance(item, dict):
                continue
            aid = item.get("agent_id", "")
            if aid not in member_ids:
                log.warning(
                    "Coordinator referenced unknown agent %s, skipping", aid,
                )
                continue
            description = str(item.get("description") or "").strip()
            if not description:
                continue
            subtasks.append(SubTask(
                agent_id=aid,
                agent_name=str(item.get("agent_name") or aid),
                description=description,
            ))

        return subtasks

    def _build_upstream_context(self, handoffs: list) -> str:
        """Build upstream context from completed HandoffPackets.

        Caps total injected text to prevent context rot on downstream agents.
        Most recent handoffs get priority — they're more likely to be directly
        relevant to the current step.
        """
        if not handoffs:
            return ""

        blocks: list[str] = []
        total_chars = 0
        for h in reversed(handoffs):
            block = h.to_context_block()
            if total_chars + len(block) > MAX_UPSTREAM_CONTEXT_CHARS:
                break
            blocks.insert(0, block)
            total_chars += len(block)

        if not blocks:
            return ""

        return (
            "## Results from earlier pipeline steps\n"
            "(These are outputs from your teammates. Build on them, don't repeat them.)\n\n"
            + "\n\n".join(blocks)
        )

    def _log_handoff(self, packet: HandoffPacket) -> None:
        """Persist a HandoffPacket to the handoff_log table."""
        from datetime import datetime, timezone
        try:
            with _db.transaction() as conn:
                conn.execute(
                    "INSERT INTO handoff_log "
                    "(packet_id, workflow_id, step_index, agent_id, agent_name, "
                    " subtask_completed, artifact_summary, assumptions_json, "
                    " uncertainties_json, confidence, validation_passed, "
                    " validation_notes_json, duration_ms, input_tokens, "
                    " output_tokens, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        packet.workflow_id,
                        packet.step_index,
                        packet.agent_id,
                        packet.agent_name,
                        packet.subtask_completed,
                        packet.artifact[:500],
                        json.dumps(packet.assumptions),
                        json.dumps(packet.uncertainties),
                        packet.confidence,
                        1 if packet.validation_passed else 0,
                        json.dumps(packet.validation_notes),
                        packet.duration_ms,
                        packet.input_tokens,
                        packet.output_tokens,
                        packet.timestamp or datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception as exc:
            log.debug("handoff_log write failed (non-fatal): %s", exc)

    def _single_agent_fallback(
        self, coordinator, user_message, history, pipeline_id, emit, on_token,
    ) -> PipelineResult:
        """Run the coordinator alone when the team has no specialists."""
        task = TaskDescriptor(
            text=user_message, preferred_agent_id=coordinator["id"],
        )
        decision = self._hub.route_for_agent(coordinator["id"], task)
        system = (
            coordinator.get("system_prompt")
            or "You are a helpful AI assistant."
        )
        messages = list(history) + [{"role": "user", "content": user_message}]
        result = self._hub.invoke(
            decision, system, messages, max_tokens=4096, on_token=on_token,
        )
        emit("pipeline_complete", {
            "pipeline_id": pipeline_id,
            "steps_completed": 0,
            "total_steps": 0,
        })
        return PipelineResult(
            synthesis=result.text,
            steps=[],
            handoffs=[],
            total_tokens_in=result.input_tokens,
            total_tokens_out=result.output_tokens,
            synthesis_model=result.model_name or "pipeline",
            pipeline_id=pipeline_id,
        )
