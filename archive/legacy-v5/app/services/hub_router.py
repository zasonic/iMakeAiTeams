"""
services/hub_router.py — Phase 1: Centralized hub routing.

Single boundary that selects a worker (agent) for a TaskDescriptor and is the
only call site permitted to invoke ``claude_client`` / ``local_client`` for
worker work.

Routing strategy:
  1. ``route_for_agent(agent_id, task)`` — caller specifies the agent; router
     validates skill/scope authorization, then returns a RoutingDecision.
  2. ``route(task)`` — caller does not specify an agent; router scores all
     skill-declaring agents by deterministic skill match (target p99 < 50ms).
     If no agent's score exceeds ``MIN_SKILL_MATCH_SCORE``, the LLM fallback
     hook fires (filled in by Phase 3 — see ``_llm_fallback``).
  3. ``invoke(decision, system, messages, ...)`` — dispatches the chosen
     decision to the right model client and returns a uniform WorkerResult.

Authz model:
  A task's ``required_scopes`` must be a subset of the chosen worker's declared
  scopes for the matched skill. If not, routing raises ``AuthorizationError``
  rather than silently downgrading.

Design notes:
  - No I/O during ``route()`` other than a single agents-table read; scoring
    is pure Python.
  - ``invoke()`` is the only place under ``app/services/`` (besides the
    orchestrator's bootstrap) allowed to call worker model methods. The Phase 1
    test suite enforces this via static AST inspection.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

import db as _db
from models import (
    ExecutionTarget,
    RoutingDecision,
    Skill,
    TaskDescriptor,
    WorkerResult,
)

log = logging.getLogger("MyAIEnv.hub_router")

# Below this score, deterministic routing fails over to the LLM fallback.
MIN_SKILL_MATCH_SCORE: float = 0.5


class AuthorizationError(RuntimeError):
    """Raised when a task's required_scopes are not a subset of the worker's."""


class HubRouter:
    """The hub's single boundary for worker selection and invocation."""

    def __init__(
        self,
        claude_client,
        local_client,
        settings,
        llm_fallback: Optional[Callable[[TaskDescriptor], RoutingDecision]] = None,
    ):
        self._claude = claude_client
        self._local = local_client
        self._settings = settings
        # Phase 3 wires Qwen /no_think here; Phase 1 leaves it None and routing
        # raises if it would be needed without a fallback configured.
        self._llm_fallback = llm_fallback

    # ── Skill scoring (deterministic, no LLM) ────────────────────────────────

    @staticmethod
    def _parse_skills(raw: str | None) -> list[Skill]:
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        out: list[Skill] = []
        for item in data:
            if isinstance(item, dict) and item.get("name"):
                out.append(Skill.from_dict(item))
        return out

    @staticmethod
    def _score_match(declared: list[Skill], task: TaskDescriptor) -> tuple[float, str]:
        """
        Return (score, matched_skill_name).

        Score combines:
          - Required-skill coverage: fraction of task.required_skills present in
            declared skill names. Zero means no overlap → unrouteable.
          - Scope fit: required_scopes must be a subset of the declared skill's
            scopes for at least one matched skill.
          - Specificity bonus: agents with fewer total skills get a small boost
            so a generalist doesn't outrank a specialist on equal coverage.
        """
        if not task.required_skills:
            # No skills required: any agent matches with neutral score.
            return (0.5, "")
        declared_names = {s.name for s in declared}
        matched = [s for s in declared if s.name in task.required_skills]
        if not matched:
            return (0.0, "")

        coverage = len(matched) / len(task.required_skills)

        # Pick the matched skill that satisfies scopes; if none, scope=0.
        scope_ok_skill = next(
            (s for s in matched if set(task.required_scopes).issubset(set(s.scopes))),
            None,
        )
        if scope_ok_skill is None:
            return (0.0, "")  # no skill can satisfy scopes → unauthorized

        # Specificity: 1.0 when agent declares only the matched skill,
        # decays as the agent declares unrelated skills.
        specificity = 1.0 / max(1, len(declared_names))
        score = (0.7 * coverage) + (0.3 * specificity)
        # Clamp to [0, 1].
        score = max(0.0, min(1.0, score))
        return (score, scope_ok_skill.name)

    # ── Public routing API ──────────────────────────────────────────────────

    def route_for_agent(self, agent_id: str, task: TaskDescriptor) -> RoutingDecision:
        """Caller-specified agent path. Validates authz; never runs the LLM."""
        row = _db.fetchone(
            "SELECT id, model_preference, skills, thinking_budget FROM agents WHERE id = ?",
            (agent_id,),
        )
        if not row:
            raise AuthorizationError(f"Unknown agent: {agent_id}")

        declared = self._parse_skills(row["skills"])
        score, matched = self._score_match(declared, task)

        # Authz: if the task declared required_skills/scopes, the agent must
        # cover them. If the task is open (no required_skills), we accept any
        # agent — this preserves the existing chat flow where the user picks
        # an agent freely.
        if task.required_skills and score == 0.0:
            raise AuthorizationError(
                f"Agent {agent_id} cannot satisfy required skills "
                f"{list(task.required_skills)} with scopes "
                f"{list(task.required_scopes)}"
            )

        backend = self._resolve_backend(row["model_preference"], task.backend_hint)
        # Phase 3: Cap the per-agent thinking budget by the global ceiling.
        budget = self._capped_budget(row)
        return RoutingDecision(
            agent_id=row["id"],
            backend=backend,
            score=score if task.required_skills else 1.0,
            reasoning=f"caller-selected agent {agent_id}",
            used_fallback=False,
            skill_matched=matched,
            thinking_budget=budget,
        )

    def _capped_budget(self, row) -> int:
        """Resolve a per-agent thinking budget capped by the global setting."""
        try:
            agent_budget = int(row["thinking_budget"] or 0)
        except (KeyError, IndexError, ValueError, TypeError):
            agent_budget = 0
        if agent_budget <= 0:
            return 0
        cap = int(self._settings.get("qwen_thinking_global_budget_cap", 8192) or 0)
        if cap <= 0:
            return agent_budget
        return min(agent_budget, cap)

    def route(self, task: TaskDescriptor) -> RoutingDecision:
        """No-agent-specified path. Picks by skill match across all agents."""
        if task.preferred_agent_id:
            return self.route_for_agent(task.preferred_agent_id, task)

        rows = _db.fetchall(
            "SELECT id, model_preference, skills, thinking_budget FROM agents "
            "WHERE skills IS NOT NULL AND skills != '[]'"
        )

        best_row = None
        best_backend: str = "claude"
        best_score: float = 0.0
        best_skill: str = ""
        for r in rows:
            declared = self._parse_skills(r["skills"])
            score, matched = self._score_match(declared, task)
            if score > best_score:
                best_score = score
                best_row = r
                best_backend = self._resolve_backend(r["model_preference"], task.backend_hint)
                best_skill = matched

        if best_row is not None and best_score >= MIN_SKILL_MATCH_SCORE:
            return RoutingDecision(
                agent_id=best_row["id"],
                backend=best_backend,
                score=best_score,
                reasoning=f"skill-match on '{best_skill}' (score {best_score:.2f})",
                used_fallback=False,
                skill_matched=best_skill,
                thinking_budget=self._capped_budget(best_row),
            )

        # No deterministic winner — use LLM fallback if Phase 3 wired one.
        if self._llm_fallback is None:
            raise RoutingError(
                f"No agent declared a skill matching {list(task.required_skills)}; "
                "LLM fallback not configured."
            )
        decision = self._llm_fallback(task)
        return RoutingDecision(
            agent_id=decision.agent_id,
            backend=decision.backend,
            score=decision.score,
            reasoning=decision.reasoning,
            used_fallback=True,
            skill_matched=decision.skill_matched,
        )

    # ── Worker invocation (only call site for model clients) ────────────────

    def invoke(
        self,
        decision: RoutingDecision,
        system: str,
        messages: list,
        max_tokens: int = 4096,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> WorkerResult:
        """Dispatch a routed task to its model client. Single source of truth."""
        if decision.backend == "claude":
            return self._invoke_claude(system, messages, max_tokens, on_token)
        return self._invoke_local(
            system, messages, max_tokens, on_token,
            thinking_budget=int(decision.thinking_budget or 0),
        )

    def target_for(self, decision: RoutingDecision, max_tokens: int) -> ExecutionTarget:
        """Public helper so the orchestrator can build an ExecutionTarget."""
        if decision.backend == "claude":
            return ExecutionTarget(
                backend="claude",
                model_name=getattr(self._claude, "_model", "claude"),
                max_tokens=max_tokens,
            )
        return ExecutionTarget(
            backend="local",
            model_name=self._settings.get("default_local_model", "local"),
            max_tokens=min(max_tokens, 2048),
        )

    # ── Backend resolution (mirrors orchestrator's pre-Phase-1 logic) ────────

    @staticmethod
    def _resolve_backend(model_preference: str | None, hint: Optional[str]) -> str:
        # Agent preference is a hard constraint; backend_hint is advisory and
        # only consulted when the agent is configured for ``auto``. This
        # matches the pre-Phase-1 orchestrator logic where model_preference
        # short-circuited the TaskRouter.
        pref = (model_preference or "auto").lower()
        if pref == "claude":
            return "claude"
        if pref == "local":
            return "local"
        if hint in ("claude", "local"):
            return hint
        return "claude"

    # ── Private dispatch (the only model-client call sites in services/) ────

    def _invoke_claude(self, system, messages, max_tokens, on_token) -> WorkerResult:
        try:
            if on_token:
                text, usage = self._claude.stream_multi_turn(
                    system, messages, on_token, max_tokens=max_tokens,
                )
                tokens_in = getattr(usage, "input_tokens", 0) or 0 if usage else 0
                tokens_out = getattr(usage, "output_tokens", 0) or 0 if usage else 0
                return WorkerResult(
                    text=text,
                    backend="claude",
                    model_name=getattr(self._claude, "_model", "claude"),
                    input_tokens=tokens_in,
                    output_tokens=tokens_out,
                )
            result = self._claude.chat_multi_turn(system, messages, max_tokens=max_tokens)
            return WorkerResult(
                text=result["text"],
                backend="claude",
                model_name=getattr(self._claude, "_model", "claude"),
                input_tokens=int(result.get("input_tokens", 0)),
                output_tokens=int(result.get("output_tokens", 0)),
            )
        except Exception as exc:
            log.error("Claude invocation failed: %s", exc)
            return WorkerResult(
                text=f"[Error: {exc}]",
                backend="claude",
                model_name=getattr(self._claude, "_model", "claude"),
                had_error=True,
            )

    def _invoke_local(self, system, messages, max_tokens, on_token,
                      thinking_budget: int = 0) -> WorkerResult:
        local_max = min(max_tokens, 2048)
        try:
            if thinking_budget > 0:
                # Phase 3: route through qwen_thinking so the /think directive
                # and per-agent budget cap are applied. Imported lazily so the
                # router module stays import-cheap and qwen_thinking can use
                # types defined in models.py without a cycle.
                from services import qwen_thinking
                budget = min(thinking_budget, local_max)
                text = qwen_thinking.worker_think(
                    self._local, system, messages,
                    budget_tokens=budget, on_token=on_token,
                )
            elif on_token:
                text = self._local.stream_multi_turn(
                    system, messages, on_token, max_tokens=local_max,
                )
            else:
                text = self._local.chat_multi_turn(
                    system, messages, max_tokens=local_max,
                )
            return WorkerResult(
                text=text or "",
                backend="local",
                model_name=self._settings.get("default_local_model", "local"),
            )
        except Exception as exc:
            log.error("Local invocation failed: %s", exc)
            return WorkerResult(
                text=f"[Error: {exc}]",
                backend="local",
                model_name=self._settings.get("default_local_model", "local"),
                had_error=True,
            )


class RoutingError(RuntimeError):
    """Raised when no agent can be routed and no fallback is available."""
