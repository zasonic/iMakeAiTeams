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

Layer 2 wiring (Priority 4 + Priority 6):
  - workflow_checkpoints — every specialist step is bracketed by a
    provisional → committed/rolled_back transition with retry-on-failure
    and a startup pass that marks orphaned provisional rows abandoned.
  - debate_log — opt-in adversarial challenger fires after each committed
    step and the synthesizer sees the critique alongside the artifact.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import db as _db
from models import (
    ChallengePacket,
    HandoffPacket,
    RoutingDecision,
    TaskDescriptor,
    semantic_validate_handoff,
    validate_handoff_packet,
)
from services.hub_router import HubRouter
from services.redact import redact

log = logging.getLogger("iMakeAiTeams.pipeline")

# Maximum sub-tasks the coordinator can decompose into. Prevents runaway
# decomposition on adversarial or ambiguous inputs.
MAX_SUBTASKS = 6

# Maximum retries per specialist when HandoffPacket validation fails. The
# saga commits the row on the first pass and only retries on validation
# rollback, so a value > 1 here ratchets reliability without changing the
# happy path's latency.
MAX_RETRIES_PER_STEP = 3

# Maximum HandoffPacket context injected into downstream agents (chars).
# Prevents context rot when many specialists contribute.
MAX_UPSTREAM_CONTEXT_CHARS = 12_000

# Workflow-checkpoint state vocabulary. Kept at module scope so tests and
# downstream consumers can import the strings instead of re-typing them.
CHECKPOINT_PROVISIONAL = "provisional"
CHECKPOINT_COMMITTED   = "committed"
CHECKPOINT_ROLLED_BACK = "rolled_back"
CHECKPOINT_ABANDONED   = "abandoned"


def _setting_truthy(settings, key: str, default: bool) -> bool:
    """Coerce a settings value (which may be '1'/'0'/'true'/etc.) to bool."""
    try:
        raw = settings.get(key, default)
    except Exception:
        return default
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if not s:
        return default
    return s in ("1", "true", "yes", "on")


def _str_list(raw) -> list:
    """Coerce a JSON value into a list of trimmed non-empty strings."""
    if not isinstance(raw, list):
        return []
    out: list = []
    for item in raw:
        s = str(item or "").strip()
        if s:
            out.append(s[:500])
    return out


def mark_abandoned_provisional_checkpoints() -> int:
    """Mark every still-provisional row as 'abandoned' on sidecar startup.

    A provisional row from a previous process means the sidecar died after
    opening a checkpoint but before validating it — the only honest answer
    is to declare the workflow lost. Returns the row count for logging.
    Best-effort: a DB error is swallowed and reported by the caller.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _db.transaction() as conn:
            cur = conn.execute(
                "UPDATE workflow_checkpoints "
                "SET state = ?, rolled_back_at = ?, "
                "    failure_reason = COALESCE(failure_reason, 'sidecar restart abandoned in-flight checkpoint') "
                "WHERE state = ?",
                (CHECKPOINT_ABANDONED, now, CHECKPOINT_PROVISIONAL),
            )
            return cur.rowcount or 0
    except Exception as exc:
        log.warning("mark_abandoned_provisional_checkpoints failed: %s", exc)
        return 0


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
{challenge_blocks}
Instructions:
- Combine the specialists' work into one clear response.
- Resolve any contradictions by noting them.
- If a specialist flagged low confidence or uncertainties, mention them briefly.
- If a challenger raised disputes, fact conflicts, or missing analysis above,
  weigh them and reflect the strongest critiques in your answer.
- Write as if YOU did the work — don't say "the researcher found..." unless attribution adds value.
- Keep the response focused on what the user asked for.
"""

# The challenger prompt is intentionally small and JSON-only. It runs after
# every committed step, so its latency tax shows up on every team turn — the
# shorter we keep it, the less it costs.
CHALLENGER_PROMPT = """You are an adversarial reviewer. Critique the work below.

Original task: {task}
Specialist's deliverable: {artifact}

Return ONLY a JSON object with these keys (each list may be empty):
{{
  "assumption_diffs":   ["...assumptions you'd dispute..."],
  "fact_conflicts":     ["...claims that look wrong or contradict known facts..."],
  "missing_analysis":   ["...important gaps the deliverable left out..."],
  "changed_position":   true | false,
  "revised_conclusion": "if changed_position is true, your preferred answer; otherwise empty string",
  "overall_assessment": "one sentence"
}}

Be specific. If you find nothing wrong, return all-empty lists, changed_position=false,
and overall_assessment='No material issues found.' Never invent objections to look thorough.
"""


class PipelineExecutor:
    """Executes a multi-agent pipeline for a team.

    ``claude_client`` and ``local_client`` are optional and default to None.
    When supplied they enable semantic_validate_handoff (which calls the
    local model to score whether a deliverable satisfies its task) and the
    debate-log challenger. When omitted, the executor falls back to the
    structural-only validator and skips debate entirely. Existing tests that
    construct ``PipelineExecutor(hub, settings)`` keep working unchanged.
    """

    def __init__(self, hub_router: HubRouter, settings,
                 claude_client=None, local_client=None):
        self._hub = hub_router
        self._settings = settings
        self._claude = claude_client
        self._local = local_client

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

        # ── Step 2: Execute each sub-task under the saga ────────────────────
        # Each sub-task opens a workflow_checkpoints row in 'provisional'.
        # On structural + semantic validation pass we commit it; on failure
        # we roll it back and retry up to MAX_RETRIES_PER_STEP, injecting
        # the prior failure_reason into the next prompt. After the final
        # commit (or exhaustion) we optionally run the adversarial
        # challenger and persist its ChallengePacket to debate_log.
        handoffs: list[HandoffPacket] = []
        challenges: list[ChallengePacket] = []
        step_summaries: list[dict] = []
        debate_id = str(uuid.uuid4())  # one debate per turn; many challenges
        debate_active = self._debate_should_run(user_message)

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

            spec_task = TaskDescriptor(
                text=subtask.description, preferred_agent_id=subtask.agent_id,
            )
            spec_decision = self._hub.route_for_agent(subtask.agent_id, spec_task)

            packet = self._run_step_with_saga(
                spec_decision=spec_decision,
                specialist_system=specialist_system,
                subtask=subtask,
                user_message=user_message,
                pipeline_id=pipeline_id,
                step_index=i,
                emit=emit,
            )

            self._log_handoff(packet)
            handoffs.append(packet)

            challenge = None
            if debate_active and packet.validation_passed:
                challenge = self._run_challenger(
                    subtask=subtask,
                    packet=packet,
                    pipeline_id=pipeline_id,
                    debate_id=debate_id,
                    emit=emit,
                )
                if challenge is not None:
                    challenges.append(challenge)
                    self._log_challenge(challenge)

            summary = {
                "step": i + 1,
                "agent": subtask.agent_name,
                "task": subtask.description,
                "confidence": packet.confidence_label,
                "validation_passed": packet.validation_passed,
                "tokens": packet.input_tokens + packet.output_tokens,
                "duration_ms": round(packet.duration_ms),
                "challenger_signal": (
                    challenge.has_signal() if challenge is not None else False
                ),
            }
            step_summaries.append(summary)
            emit("pipeline_step_complete", summary)

        # ── Step 3: Coordinator synthesises ─────────────────────────────────
        emit("pipeline_synthesising", {"agent": coordinator["name"]})

        handoff_blocks = "\n\n".join(h.to_context_block() for h in handoffs)
        challenge_text = "\n\n".join(
            c.to_context_block() for c in challenges if c.has_signal()
        )
        challenge_blocks = (
            "\nChallenger reviews:\n" + challenge_text + "\n"
            if challenge_text else ""
        )

        synth_system = (
            coordinator.get("system_prompt") or "You are a team coordinator."
        )
        synth_messages = [{
            "role": "user",
            "content": SYNTHESIS_PROMPT.format(
                user_message=user_message,
                handoff_blocks=handoff_blocks,
                challenge_blocks=challenge_blocks,
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

    def _run_step_with_saga(
        self,
        spec_decision: RoutingDecision,
        specialist_system: str,
        subtask: SubTask,
        user_message: str,
        pipeline_id: str,
        step_index: int,
        emit: Callable[[str, dict], None],
    ) -> HandoffPacket:
        """Run one specialist step under the workflow-checkpoints saga.

        Opens a 'provisional' checkpoint, invokes the specialist, validates
        the resulting HandoffPacket (structural + semantic when a local
        client is wired), and either commits the checkpoint or rolls it
        back. On rollback, retries up to ``max_retries`` times with the
        previous failure_reason injected into the prompt. The final packet
        — committed or exhausted — is returned to the caller.

        Always returns a HandoffPacket; never raises. Checkpoint and SSE
        writes are best-effort and never block the pipeline.
        """
        max_retries = MAX_RETRIES_PER_STEP
        checkpoint_id = self._open_checkpoint(
            pipeline_id=pipeline_id,
            step_index=step_index,
            subtask=subtask,
            max_retries=max_retries,
        )
        emit("checkpoint_state", {
            "checkpoint_id": checkpoint_id,
            "step": step_index + 1,
            "agent": subtask.agent_name,
            "state": CHECKPOINT_PROVISIONAL,
        })

        last_packet: Optional[HandoffPacket] = None
        last_failure_reason = ""
        attempt = 0
        # Total attempts = 1 initial + max_retries retries.
        for attempt in range(max_retries + 1):
            messages = self._build_specialist_messages(
                subtask=subtask,
                user_message=user_message,
                prior_failure_reason=last_failure_reason,
            )
            packet = self._invoke_specialist(
                decision=spec_decision,
                system=specialist_system,
                messages=messages,
                subtask=subtask,
                pipeline_id=pipeline_id,
                step_index=step_index,
                is_retry=attempt > 0,
            )
            last_packet = packet

            if packet.validation_passed:
                self._commit_checkpoint(checkpoint_id, packet)
                emit("checkpoint_state", {
                    "checkpoint_id": checkpoint_id,
                    "step": step_index + 1,
                    "agent": subtask.agent_name,
                    "state": CHECKPOINT_COMMITTED,
                    "confidence": packet.confidence,
                })
                return packet

            last_failure_reason = "; ".join(packet.validation_notes) or "validation failed"
            # ``retry_count`` is "retries used" — 0 on the initial attempt's
            # failure, max_retries when the last retry also fails. Counting
            # the initial attempt as a retry would push the column past
            # max_retries on exhaustion, which reads wrong in queries.
            self._rollback_checkpoint(
                checkpoint_id, packet, last_failure_reason, retry_count=attempt,
            )
            emit("checkpoint_state", {
                "checkpoint_id": checkpoint_id,
                "step": step_index + 1,
                "agent": subtask.agent_name,
                "state": CHECKPOINT_ROLLED_BACK,
                "reason": last_failure_reason,
                "retry": attempt,
                "max_retries": max_retries,
            })
            if attempt < max_retries:
                emit("pipeline_step_retry", {
                    "step": step_index + 1,
                    "agent": subtask.agent_name,
                    "reason": last_failure_reason,
                    "attempt": attempt + 2,
                })

        # Retries exhausted. Return the last packet so the caller can still
        # log it and the synthesizer can see the failure flagged.
        return last_packet  # type: ignore[return-value]

    def _build_specialist_messages(
        self, subtask: SubTask, user_message: str, prior_failure_reason: str = "",
    ) -> list:
        """Build the user-message list for a specialist invocation.

        On retry, ``prior_failure_reason`` is injected so the model knows
        what the validator complained about and can avoid repeating it.
        """
        prefix = ""
        if prior_failure_reason:
            prefix = (
                f"Your previous attempt at this task failed validation:\n"
                f"  {prior_failure_reason}\n\n"
                "Address the failure explicitly. Be more specific and concrete. "
                "State your uncertainties.\n\n"
            )
        return [{
            "role": "user",
            "content": (
                f"{prefix}"
                f"You are working as part of a team. Your specific task:\n\n"
                f"{subtask.description}\n\n"
                f"The user's original request was: {user_message}\n\n"
                f"Complete your task thoroughly. Be specific and concrete in "
                f"your output."
            ),
        }]

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
        """Invoke a specialist and wrap the WorkerResult into a HandoffPacket.

        Validates with semantic_validate_handoff when a local client is
        wired (catches off-topic / empty deliverables that the structural
        check misses); otherwise falls back to validate_handoff_packet.
        """
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
        if self._local is not None:
            return semantic_validate_handoff(packet, self._local, self._claude)
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

    # ── workflow_checkpoints (saga) ─────────────────────────────────────────
    #
    # Three states make up the happy and unhappy paths:
    #   provisional → committed   (validation passed)
    #   provisional → rolled_back (validation failed; retry window open)
    #   provisional → abandoned   (process died mid-flight; resolved at startup)
    # All writes are best-effort: a DB error here MUST NOT take down the turn.

    def _open_checkpoint(
        self, pipeline_id: str, step_index: int, subtask: SubTask,
        max_retries: int,
    ) -> str:
        """Insert a 'provisional' workflow_checkpoints row. Returns its id.

        Returns an empty string if the write failed — callers treat that as
        a no-op checkpoint (the saga still works, just without persistence).
        """
        checkpoint_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        try:
            with _db.transaction() as conn:
                conn.execute(
                    "INSERT INTO workflow_checkpoints "
                    "(checkpoint_id, workflow_id, step_index, task_id, "
                    " agent_id, agent_name, state, success_criteria, "
                    " retry_count, max_retries, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        checkpoint_id,
                        pipeline_id,
                        step_index,
                        subtask.agent_id,  # task_id ≈ owning agent for now
                        subtask.agent_id,
                        subtask.agent_name,
                        CHECKPOINT_PROVISIONAL,
                        subtask.description,
                        0,
                        max_retries,
                        now,
                    ),
                )
        except Exception as exc:
            log.debug("workflow_checkpoints open failed (non-fatal): %s", exc)
            return ""
        return checkpoint_id

    def _commit_checkpoint(self, checkpoint_id: str, packet: HandoffPacket) -> None:
        """Mark a provisional checkpoint 'committed'."""
        if not checkpoint_id:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            with _db.transaction() as conn:
                conn.execute(
                    "UPDATE workflow_checkpoints "
                    "SET state = ?, artifact_summary = ?, confidence_score = ?, "
                    "    validation_passed = 1, validation_reasoning = ?, "
                    "    validated_at = ?, committed_at = ? "
                    "WHERE checkpoint_id = ?",
                    (
                        CHECKPOINT_COMMITTED,
                        packet.artifact[:500],
                        packet.confidence,
                        "; ".join(packet.validation_notes)[:500],
                        now,
                        now,
                        checkpoint_id,
                    ),
                )
        except Exception as exc:
            log.debug("workflow_checkpoints commit failed (non-fatal): %s", exc)

    def _rollback_checkpoint(
        self, checkpoint_id: str, packet: HandoffPacket,
        failure_reason: str, retry_count: int,
    ) -> None:
        """Mark a provisional checkpoint 'rolled_back'."""
        if not checkpoint_id:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            with _db.transaction() as conn:
                conn.execute(
                    "UPDATE workflow_checkpoints "
                    "SET state = ?, artifact_summary = ?, confidence_score = ?, "
                    "    validation_passed = 0, validation_reasoning = ?, "
                    "    failure_reason = ?, retry_count = ?, "
                    "    validated_at = ?, rolled_back_at = ? "
                    "WHERE checkpoint_id = ?",
                    (
                        CHECKPOINT_ROLLED_BACK,
                        packet.artifact[:500],
                        packet.confidence,
                        "; ".join(packet.validation_notes)[:500],
                        failure_reason[:500],
                        retry_count,
                        now,
                        now,
                        checkpoint_id,
                    ),
                )
        except Exception as exc:
            log.debug("workflow_checkpoints rollback failed (non-fatal): %s", exc)

    # ── debate_log (adversarial debate) ─────────────────────────────────────

    def _debate_should_run(self, user_message: str) -> bool:
        """Decide whether the challenger fires this turn.

        Two gates:
          - debate_enabled — global on/off (default off in fresh installs)
          - debate_only_high_stakes — when true, only fire on messages
            classified as high-stakes by services.governance. When false,
            fire on every team turn (more cost, more reliability).

        The challenger needs at least the local client to run; without it
        we'd have no model to send the critique through and we silently no-op.
        """
        if self._local is None and self._claude is None:
            return False
        if not _setting_truthy(self._settings, "debate_enabled", default=False):
            return False
        if _setting_truthy(self._settings, "debate_only_high_stakes", default=True):
            try:
                from services.governance import is_high_stakes_message
            except Exception:
                return False
            return is_high_stakes_message(user_message)
        return True

    def _run_challenger(
        self,
        subtask: SubTask,
        packet: HandoffPacket,
        pipeline_id: str,
        debate_id: str,
        emit: Callable[[str, dict], None],
    ) -> Optional[ChallengePacket]:
        """Invoke the challenger and return a ChallengePacket.

        Routes through HubRouter.invoke just like a specialist so that all
        model traffic still flows through the single boundary. Failures
        (parse errors, model unavailability) are non-fatal — the pipeline
        continues without the critique. Returns ``None`` only when the
        challenger could not run at all.
        """
        # The challenger reuses the specialist's agent_id so HubRouter can
        # score+authorize it. A separate "challenger" agent could be added
        # later; for now reusing the same worker keeps the wiring simple.
        decision = self._hub.route_for_agent(
            subtask.agent_id,
            TaskDescriptor(
                text=subtask.description, preferred_agent_id=subtask.agent_id,
            ),
        )
        emit("challenger_started", {
            "step": packet.step_index + 1,
            "agent": subtask.agent_name,
        })
        start_ms = time.monotonic()
        challenger_user = CHALLENGER_PROMPT.format(
            task=subtask.description[:500],
            artifact=packet.artifact[:1500],
        )
        result = self._hub.invoke(
            decision,
            "You are an adversarial reviewer. Output JSON only.",
            [{"role": "user", "content": challenger_user}],
            max_tokens=600,
        )
        elapsed_ms = (time.monotonic() - start_ms) * 1000
        challenge_id = str(uuid.uuid4())

        if result.had_error or not result.text:
            packet_out = ChallengePacket(
                challenge_id=challenge_id,
                debate_id=debate_id,
                workflow_id=pipeline_id,
                agent_id=subtask.agent_id,
                agent_name=f"Challenger of {subtask.agent_name}",
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                duration_ms=elapsed_ms,
                parse_failed=True,
            )
            emit("challenger_complete", {
                "step": packet.step_index + 1,
                "signal": False,
                "parse_failed": True,
            })
            return packet_out

        verdict = self._parse_challenger_json(result.text)
        if verdict is None:
            return ChallengePacket(
                challenge_id=challenge_id,
                debate_id=debate_id,
                workflow_id=pipeline_id,
                agent_id=subtask.agent_id,
                agent_name=f"Challenger of {subtask.agent_name}",
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                duration_ms=elapsed_ms,
                parse_failed=True,
            )

        challenge = ChallengePacket(
            challenge_id=challenge_id,
            debate_id=debate_id,
            workflow_id=pipeline_id,
            agent_id=subtask.agent_id,
            agent_name=f"Challenger of {subtask.agent_name}",
            assumption_diffs=_str_list(verdict.get("assumption_diffs")),
            fact_conflicts=_str_list(verdict.get("fact_conflicts")),
            missing_analysis=_str_list(verdict.get("missing_analysis")),
            changed_position=bool(verdict.get("changed_position")),
            revised_conclusion=(
                str(verdict.get("revised_conclusion") or "").strip() or None
            ),
            overall_assessment=str(verdict.get("overall_assessment") or "")[:500],
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=elapsed_ms,
        )
        emit("challenger_complete", {
            "step": packet.step_index + 1,
            "signal": challenge.has_signal(),
            "parse_failed": False,
        })
        return challenge

    @staticmethod
    def _parse_challenger_json(raw: str) -> Optional[dict]:
        text = (raw or "").strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        qstart = text.find("{")
        qend = text.rfind("}")
        if qstart == -1 or qend == -1 or qend <= qstart:
            return None
        try:
            parsed = json.loads(text[qstart:qend + 1])
        except (ValueError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _log_challenge(self, challenge: ChallengePacket) -> None:
        """Persist a ChallengePacket to the debate_log table."""
        try:
            with _db.transaction() as conn:
                conn.execute(
                    "INSERT INTO debate_log "
                    "(challenge_id, debate_id, workflow_id, agent_id, agent_name, "
                    " assumption_diffs_json, fact_conflicts_json, "
                    " missing_analysis_json, changed_position, revised_conclusion, "
                    " overall_assessment, input_tokens, output_tokens, "
                    " duration_ms, parse_failed, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        challenge.challenge_id,
                        challenge.debate_id,
                        challenge.workflow_id,
                        challenge.agent_id,
                        challenge.agent_name,
                        json.dumps(challenge.assumption_diffs),
                        json.dumps(challenge.fact_conflicts),
                        json.dumps(challenge.missing_analysis),
                        1 if challenge.changed_position else 0,
                        challenge.revised_conclusion,
                        challenge.overall_assessment,
                        challenge.input_tokens,
                        challenge.output_tokens,
                        challenge.duration_ms,
                        1 if challenge.parse_failed else 0,
                        challenge.timestamp,
                    ),
                )
        except Exception as exc:
            log.debug("debate_log write failed (non-fatal): %s", exc)

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
