"""
services/qwen_thinking.py — Phase 3: Qwen3 hybrid thinking glue.

Qwen3 toggles its built-in reasoning by accepting the directives
``/think`` and ``/no_think`` in the prompt:

  - ``/no_think`` — fast path. Used by the hub for routing decisions
    (target < 1s). Output is the answer only, no <think> block.
  - ``/think`` — deep path. Used by workers for reasoning. Output is
    a <think>…</think> block followed by the final answer.

We surface both behaviors through a single LocalClient (one GGUF), so the
hub and workers share the same model. The "thinking budget" is a
per-agent ``max_tokens`` ceiling — coarse but portable across LM Studio
versions, and aligned with how Qwen3 emits reasoning inline.

Module layout:
  - ``THINK_DIRECTIVE`` / ``NO_THINK_DIRECTIVE`` — the magic strings.
  - ``with_think_directive()`` — pure helper that prepends the directive.
  - ``worker_think()``         — invoked by the hub for worker calls.
  - ``no_think_route()``       — LLM fallback wired into HubRouter.
  - ``strip_think_block()``    — convenience for callers that want the
    final answer only, with reasoning extracted separately.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from models import RoutingDecision, TaskDescriptor

log = logging.getLogger("MyAIEnv.qwen_thinking")

THINK_DIRECTIVE: str = "/think"
NO_THINK_DIRECTIVE: str = "/no_think"

# Routing fallback should respond fast — keep its budget tiny.
ROUTING_MAX_TOKENS: int = 256

# Used by ``strip_think_block``. Qwen3 wraps reasoning in <think>…</think>.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


# ── Prompt helpers ──────────────────────────────────────────────────────────

def with_think_directive(system: str, *, think: bool) -> str:
    """Prepend the appropriate thinking directive to a system prompt.

    The directive is on its own first line so it survives downstream prompt
    transformations that operate on later sections (memory, ToM, tools).
    """
    directive = THINK_DIRECTIVE if think else NO_THINK_DIRECTIVE
    base = (system or "").lstrip("\n")
    return f"{directive}\n{base}".rstrip() + "\n"


def strip_think_block(text: str) -> tuple[str, str]:
    """Split a Qwen3 response into (answer, reasoning).

    If no ``<think>`` block exists, ``reasoning`` is the empty string and
    ``answer`` is the original text untouched.
    """
    if not text:
        return "", ""
    match = _THINK_BLOCK_RE.search(text)
    if not match:
        return text, ""
    reasoning = match.group(0)
    answer = (text[:match.start()] + text[match.end():]).lstrip()
    # Strip the surrounding tags in the returned reasoning so callers don't
    # re-emit them when logging.
    reasoning_inner = reasoning.replace("<think>", "").replace("</think>", "").strip()
    return answer, reasoning_inner


# ── Worker invocation (used by HubRouter._invoke_local in Phase 3) ─────────

def worker_think(
    local_client,
    system: str,
    messages: list,
    *,
    budget_tokens: int,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """Run a worker call with Qwen3 ``/think`` mode and a token budget.

    ``budget_tokens=0`` opts out of thinking entirely (uses ``/no_think``).
    The budget is enforced as ``max_tokens`` — Qwen3 emits its reasoning
    inside the response, so the cap covers think + answer.
    """
    think = budget_tokens > 0
    full_system = with_think_directive(system, think=think)
    if on_token:
        return local_client.stream_multi_turn(
            full_system, messages, on_token,
            max_tokens=max(budget_tokens, 256),
        )
    return local_client.chat_multi_turn(
        full_system, messages, max_tokens=max(budget_tokens, 256),
    )


# ── LLM fallback for HubRouter (no_think mode) ──────────────────────────────

_ROUTING_SYSTEM = (
    "You are the routing brain for a hub of specialized AI agents. "
    "Given a task and a list of candidate agents (each with skills), pick the "
    "single best agent for the task. Reply with ONLY a JSON object on one line: "
    '{"agent_id": "<id>", "reason": "one short clause"}. '
    "If no agent is a clean fit, pick the one with the closest domain match."
)


def _summarize_agents(rows: list[dict]) -> str:
    """Compact JSON list of candidate agents for the routing prompt."""
    summary = [
        {
            "id":     r.get("id", ""),
            "name":   r.get("name", ""),
            "role":   r.get("role", "custom"),
            "skills": _parse_skill_names(r.get("skills")),
        }
        for r in rows
    ]
    return json.dumps(summary, separators=(",", ":"))


def _parse_skill_names(raw) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [str(item.get("name", "")) for item in data
            if isinstance(item, dict) and item.get("name")]


def make_no_think_router(local_client, agents_provider: Callable[[], list[dict]]):
    """Build a ``RoutingDecision`` callable for HubRouter.llm_fallback.

    ``agents_provider`` returns the current list of agent rows (each at least
    ``id``, ``name``, ``role``, ``skills``, ``model_preference``). Closing
    over a callable keeps the registry the single source of truth without
    forcing this module to depend on it.
    """

    def _route(task: TaskDescriptor) -> RoutingDecision:
        return no_think_route(local_client, agents_provider(), task)

    return _route


def no_think_route(
    local_client,
    candidate_agents: list[dict],
    task: TaskDescriptor,
) -> RoutingDecision:
    """Ask Qwen3 (in ``/no_think`` mode) which agent to dispatch to.

    Returns a RoutingDecision pointing at the chosen agent. On a parse failure
    or unknown agent_id, falls back to the first candidate so chat keeps moving
    (with ``used_fallback=True`` and a clear ``reasoning`` string).
    """
    if not candidate_agents:
        raise ValueError("no_think_route requires at least one candidate agent")

    full_system = with_think_directive(_ROUTING_SYSTEM, think=False)
    user_msg = (
        f"TASK:\n{task.text.strip()}\n\n"
        f"REQUIRED SKILLS: {list(task.required_skills) or 'unspecified'}\n"
        f"REQUIRED SCOPES: {list(task.required_scopes) or 'unspecified'}\n\n"
        f"CANDIDATE AGENTS:\n{_summarize_agents(candidate_agents)}\n"
    )
    raw = local_client.chat(full_system, user_msg, max_tokens=ROUTING_MAX_TOKENS)
    answer, _ = strip_think_block(raw or "")

    chosen_id = ""
    reason = ""
    try:
        # Tolerate bare JSON or fenced code.
        clean = answer.strip().strip("`")
        if clean.lower().startswith("json"):
            clean = clean[4:].lstrip()
        # Find the first {...} block.
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end > start:
            data = json.loads(clean[start:end + 1])
            chosen_id = str(data.get("agent_id", "")).strip()
            reason = str(data.get("reason", "")).strip()
    except (json.JSONDecodeError, TypeError):
        chosen_id = ""

    by_id = {r["id"]: r for r in candidate_agents if r.get("id")}
    picked = by_id.get(chosen_id)
    if picked is None:
        picked = candidate_agents[0]
        reason = (
            f"Qwen routing returned an unknown agent_id; defaulting to first "
            f"candidate ({picked.get('name', picked.get('id', ''))})."
        )
        log.warning(
            "qwen_thinking.no_think_route: unknown agent_id %r — falling back",
            chosen_id,
        )

    backend = _backend_from_pref(picked.get("model_preference"))
    return RoutingDecision(
        agent_id=picked["id"],
        backend=backend,
        score=0.5,  # fallback routes are by definition low-confidence
        reasoning=reason or "qwen no_think route",
        used_fallback=True,
        skill_matched="",
    )


def _backend_from_pref(pref: str | None) -> str:
    p = (pref or "auto").lower()
    if p == "claude":
        return "claude"
    if p == "local":
        return "local"
    return "claude"
