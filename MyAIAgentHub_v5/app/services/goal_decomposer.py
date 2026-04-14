"""
services/goal_decomposer.py — Dynamic Goal Decomposition Engine (#2)

When the router classifies a message as "complex" AND the message contains
multiple distinct objectives, this engine:
  1. Sends the message to Claude with a decomposition prompt
  2. Receives a task graph as JSON
  3. Executes each step sequentially, feeding prior outputs as context
  4. Streams thinking timeline events for each step
  5. Combines all step outputs into a final coherent response

Falls through to normal single-pass chat for simple messages.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger("MyAIEnv.decomposer")

# ── Prompts ────────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """You are a task planning assistant. Analyze the user's message and determine if it contains multiple distinct objectives that would benefit from step-by-step execution.

If YES (multi-step): Return a JSON task graph:
{
  "multi_step": true,
  "goal_summary": "One sentence describing the overall goal",
  "steps": [
    {"step": 1, "task": "clear description of this step", "depends_on": [], "output_key": "step1_result"},
    {"step": 2, "task": "clear description of this step", "depends_on": [1], "output_key": "step2_result"}
  ]
}

If NO (single step): Return:
{"multi_step": false}

Rules:
- Only decompose if there are 2+ genuinely distinct subtasks
- Maximum 6 steps — combine related tasks
- Each step must be independently executable
- depends_on lists which step numbers must complete first (usually sequential)
- Return ONLY valid JSON, no markdown, no explanation"""

_EXECUTE_STEP_SYSTEM = """You are executing step {step_num} of {total_steps} in a multi-step task.

Overall goal: {goal_summary}

Your specific task for this step: {task_description}

{prior_context}

Instructions:
- Focus ONLY on this specific step's task
- Be thorough and detailed in your output
- Your output will be passed to subsequent steps as context
- Do NOT attempt to complete future steps — stay focused on this step only"""

_SYNTHESIZE_SYSTEM = """You are synthesizing the results of a {step_count}-step analysis into a single coherent response.

Overall goal: {goal_summary}

{step_results}

Instructions:
- Combine all step results into a cohesive, well-structured final answer
- Do not repeat redundant information
- Preserve all important findings, data, and recommendations from each step
- Format the response clearly with appropriate structure (headers, lists, etc.)
- Write as if giving one complete, unified answer — not a list of step outputs"""

# ── Patterns that suggest multi-step intent ────────────────────────────────────

_MULTI_STEP_PATTERNS = [
    r'\band\b.{3,50}\band\b',          # "research X and compare Y and write Z"
    r'\bthen\b.{3,50}\bthen\b',        # "do X then do Y then do Z"
    r'\b(first|second|third|finally)\b',
    r'\b\d+\.\s+\w',                   # numbered list in request
    r'\bstep[s]?\b',
    r'\b(research|analyze|compare|summarize|write|create|find|list|calculate).{5,80}(and|then|also|plus|additionally).{5,80}(research|analyze|compare|summarize|write|create|find|list|calculate)\b',
    r'\bmultiple\b.*\b(topics?|aspects?|parts?|sections?|components?)\b',
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _MULTI_STEP_PATTERNS]


@dataclass
class StepResult:
    step: int
    task: str
    output: str
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0.0
    output_key: str = ""


@dataclass
class DecompositionResult:
    was_decomposed: bool
    goal_summary: str = ""
    steps_completed: list = field(default_factory=list)  # list[StepResult]
    final_response: str = ""
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    error: Optional[str] = None


def _looks_multi_step(message: str) -> bool:
    """Heuristic pre-filter: does this message look like it has multiple objectives?"""
    if len(message.split()) < 8:
        return False
    for pat in _COMPILED_PATTERNS:
        if pat.search(message):
            return True
    return False


def _parse_decomposition(raw: str) -> Optional[dict]:
    """Parse the decomposition JSON from Claude's response."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Find JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        log.debug("Decomposition JSON parse failed: %s", raw[:200])
        return None


class GoalDecomposer:
    """
    Dynamic Goal Decomposition Engine.

    Usage:
        decomposer = GoalDecomposer(claude_client=..., local_client=...)
        result = decomposer.decompose_and_execute(
            message="research CRMs, compare pricing, write summary",
            system_prompt="You are a helpful assistant",
            messages=[...],  # conversation history
            complexity="complex",
            on_event=_emit_event,
            on_token=_on_token,
        )
        if result.was_decomposed:
            return result.final_response
        else:
            # fall through to normal single-pass chat
    """

    def __init__(self, claude_client, local_client):
        self.claude = claude_client
        self.local = local_client

    def decompose_and_execute(
        self,
        message: str,
        system_prompt: str,
        messages: list,
        complexity: str,
        on_event: Optional[Callable] = None,
        on_token: Optional[Callable] = None,
    ) -> DecompositionResult:
        """
        Main entry point. Returns DecompositionResult.
        If was_decomposed=False, caller should use normal chat path.
        """
        def _emit(event_type: str, data: dict) -> None:
            if on_event:
                try:
                    on_event(event_type, data)
                except Exception:
                    pass

        # ── Guard: only decompose complex messages with multi-step signals ────
        if complexity not in ("complex", "medium"):
            return DecompositionResult(was_decomposed=False)

        if not _looks_multi_step(message):
            return DecompositionResult(was_decomposed=False)

        # ── Step 1: Ask Claude to decompose the goal ──────────────────────────
        _emit("goal_decomposing", {
            "label": "Analyzing goal complexity…",
            "message_preview": message[:80],
        })

        try:
            decomp_result = self.claude.chat_multi_turn(
                system=_DECOMPOSE_SYSTEM,
                messages=[{"role": "user", "content": message}],
                max_tokens=800,
            )
            decomp_text = decomp_result.get("text", "")
            decomp_data = _parse_decomposition(decomp_text)
        except Exception as exc:
            log.warning("Goal decomposition API call failed: %s", exc)
            return DecompositionResult(was_decomposed=False, error=str(exc))

        if not decomp_data:
            return DecompositionResult(was_decomposed=False)

        if not decomp_data.get("multi_step"):
            return DecompositionResult(was_decomposed=False)

        steps = decomp_data.get("steps", [])
        if not steps or len(steps) < 2:
            return DecompositionResult(was_decomposed=False)

        goal_summary = decomp_data.get("goal_summary", message[:80])
        total_steps = len(steps)

        _emit("goal_decomposed", {
            "goal_summary": goal_summary,
            "step_count": total_steps,
            "steps": [{"step": s["step"], "task": s["task"]} for s in steps],
        })

        log.info("Goal decomposed into %d steps: %s", total_steps, goal_summary)

        # ── Step 2: Execute each step sequentially ────────────────────────────
        step_results: list[StepResult] = []
        total_tokens_in = 0
        total_tokens_out = 0

        for step_def in steps:
            step_num = step_def.get("step", len(step_results) + 1)
            task_desc = step_def.get("task", "")
            output_key = step_def.get("output_key", f"step{step_num}_result")

            _emit("step_started", {
                "step": step_num,
                "total": total_steps,
                "task": task_desc,
            })

            # Build prior context from completed steps
            prior_parts = []
            if step_results:
                prior_parts.append("## Results from previous steps:\n")
                for prev in step_results:
                    prior_parts.append(
                        f"### Step {prev.step}: {prev.task}\n{prev.output}\n"
                    )
            prior_context = "".join(prior_parts)

            # Build the step execution system prompt
            step_system = _EXECUTE_STEP_SYSTEM.format(
                step_num=step_num,
                total_steps=total_steps,
                goal_summary=goal_summary,
                task_description=task_desc,
                prior_context=prior_context,
            )

            # Include original conversation history as context
            step_messages = list(messages)  # copy history
            step_messages.append({"role": "user", "content": message})

            t0 = time.time()
            try:
                step_resp = self.claude.chat_multi_turn(
                    system=step_system,
                    messages=step_messages,
                    max_tokens=2048,
                )
                step_output = step_resp.get("text", "")
                tokens_in = step_resp.get("input_tokens", 0)
                tokens_out = step_resp.get("output_tokens", 0)
            except Exception as exc:
                log.error("Step %d execution failed: %s", step_num, exc)
                _emit("step_error", {
                    "step": step_num,
                    "task": task_desc,
                    "error": str(exc),
                })
                # Return partial result rather than failing entirely
                return DecompositionResult(
                    was_decomposed=True,
                    goal_summary=goal_summary,
                    steps_completed=step_results,
                    final_response="",
                    total_tokens_in=total_tokens_in,
                    total_tokens_out=total_tokens_out,
                    error=f"Step {step_num} failed: {exc}",
                )

            duration_ms = (time.time() - t0) * 1000
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out

            sr = StepResult(
                step=step_num,
                task=task_desc,
                output=step_output,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=duration_ms,
                output_key=output_key,
            )
            step_results.append(sr)

            _emit("step_completed", {
                "step": step_num,
                "total": total_steps,
                "task": task_desc,
                "output_preview": step_output[:150],
                "tokens_out": tokens_out,
                "duration_ms": round(duration_ms),
            })

            log.info("Step %d/%d completed: %s (%.0fms, %d tokens)",
                     step_num, total_steps, task_desc[:50], duration_ms, tokens_out)

        # ── Step 3: Synthesize all step outputs into a final response ─────────
        _emit("synthesizing_steps", {
            "step_count": total_steps,
            "label": "Combining results…",
        })

        step_results_text = "\n\n".join(
            f"## Step {sr.step}: {sr.task}\n{sr.output}"
            for sr in step_results
        )

        synth_system = _SYNTHESIZE_SYSTEM.format(
            step_count=total_steps,
            goal_summary=goal_summary,
            step_results=step_results_text,
        )

        try:
            # Stream the synthesis back if we have an on_token callback
            if on_token:
                final_text, usage = self.claude.stream_multi_turn(
                    system=synth_system,
                    messages=[{"role": "user", "content": f"Please synthesize the above step results for this goal: {goal_summary}"}],
                    on_token=on_token,
                    max_tokens=3000,
                )
                if usage:
                    total_tokens_in += getattr(usage, "input_tokens", 0) or 0
                    total_tokens_out += getattr(usage, "output_tokens", 0) or 0
            else:
                synth_resp = self.claude.chat_multi_turn(
                    system=synth_system,
                    messages=[{"role": "user", "content": f"Please synthesize the above step results for this goal: {goal_summary}"}],
                    max_tokens=3000,
                )
                final_text = synth_resp.get("text", "")
                total_tokens_in += synth_resp.get("input_tokens", 0)
                total_tokens_out += synth_resp.get("output_tokens", 0)
        except Exception as exc:
            log.error("Synthesis failed: %s", exc)
            # Fall back to concatenating step results
            final_text = f"# {goal_summary}\n\n" + "\n\n---\n\n".join(
                f"**Step {sr.step}: {sr.task}**\n\n{sr.output}"
                for sr in step_results
            )

        _emit("decomposition_complete", {
            "steps_completed": total_steps,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
        })

        return DecompositionResult(
            was_decomposed=True,
            goal_summary=goal_summary,
            steps_completed=step_results,
            final_response=final_text,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
        )
