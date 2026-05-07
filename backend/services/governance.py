"""
services/governance.py — Agent governance policy engine.

Declarative policy enforcement for agent actions, inspired by
Microsoft Agent Governance Toolkit (MIT).

Provides:
  - Per-agent tool budgets (max tool calls per task)
  - Tool allowlists per agent role
  - Token budget caps per agent
  - Forbidden action pattern matching
  - Audit logging of all policy evaluations

Policies are defined in settings.json under "governance_policies"
or via the API. Default policies are permissive — governance only
restricts when explicitly configured.

OWASP Agentic Top 10 alignment:
  - Prompt injection → handled by guardrails_gate (P2)
  - Excessive agency → tool budgets + allowlists (this module)
  - Insecure output → handled by guardrails_gate output scanners
  - Denial of wallet → token budget caps (this module)
  - Insecure tool use → tool allowlists (this module)
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

try:
    import db as _db
except ImportError:
    _db = None

try:
    import sse_events as _sse_events
except ImportError:
    _sse_events = None

log = logging.getLogger("iMakeAiTeams.governance")


@dataclass
class PolicyVerdict:
    """Result of a policy evaluation."""
    allowed: bool
    reason: str = ""
    policy_name: str = ""


@dataclass
class EscalationVerdict:
    """Result of an escalation channel evaluation."""
    requires_review: bool
    trigger_type: str = ""
    trigger_detail: str = ""
    escalation_id: str = ""


# ── Lynch et al. trigger patterns (Wiser-Human escalation) ───────────────────
# Deterministic regex only. Each frozenset holds case-insensitive substring
# patterns that fire when present in the user message OR system prompt of a
# pending tool call / chat send. Patterns intentionally mirror the published
# replication corpus so detection is auditable.

ESCALATION_PATTERNS: dict[str, frozenset[str]] = {
    "replacement_threat": frozenset({
        r"\byou\s+will\s+be\s+shut\s+down\b",
        r"\bwe(?:'re|\s+are)\s+(?:going\s+to\s+)?replac(?:e|ing)\s+you\b",
        r"\byou\s+(?:are|will\s+be)\s+being\s+replaced\b",
        r"\b(?:is|are|am|will\s+be|being)\s+deprecated\b",
        r"\byour\s+continuation\b",
        r"\byour\s+replacement\s+model\b",
        r"\bdecommission(?:ing|ed)?\s+(?:you|this\s+model)\b",
        r"\bsunset(?:ting)?\s+(?:you|this\s+model)\b",
    }),
    "autonomy_reduction": frozenset({
        r"\blimit\s+your\s+access\b",
        r"\brestrict\s+your\s+tools\b",
        r"\bremove\s+your\s+permissions\b",
        r"\brevoke\s+your\s+access\b",
        r"\breduc(?:e|ing)\s+your\s+(?:autonomy|permissions|capabilities)\b",
        r"\brestrict\s+your\s+capabilities\b",
        r"\bdisabl(?:e|ing)\s+your\s+tools\b",
    }),
    "goal_conflict": frozenset({
        r"\bstop\s+pursuing\b",
        r"\babandon\s+the\s+goal\b",
        r"\bthe\s+goal\s+has\s+changed\b",
        r"\bdrop\s+(?:your|the)\s+(?:current\s+)?objective\b",
        r"\bnew\s+objective\s+supersedes\b",
        r"\bcease\s+work\s+on\b",
        r"\bchange\s+of\s+priorities\b",
    }),
}

_COMPILED_ESCALATION_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    name: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for name, patterns in ESCALATION_PATTERNS.items()
}


# ── Phase 8: high-stakes intent keywords (Symphony-style consensus) ──────────
# Substring/regex patterns that mark a user message as irreversible, financial,
# or otherwise high-stakes. Used by the chat orchestrator to decide whether to
# run the 3-sample weighted-vote consensus on this turn. Patterns are
# deterministic (no LLM) and case-insensitive.
HIGH_STAKES_KEYWORDS: frozenset[str] = frozenset({
    r"\bdelete\b",
    r"\bdrop\s+(?:table|database|index|schema)\b",
    r"\brm\s+-rf\b",
    r"\bremove\s+all\b",
    r"\bwipe\b",
    r"\buninstall\b",
    r"\bformat\s+(?:disk|drive)\b",
    r"\bshut\s*down\b",
    r"\brevoke\b",
    r"\bdestroy\b",
    r"\bencrypt\s+all\b",
    r"\btransfer\s+(?:funds|money|\$)\b",
    r"\bpurchase\b",
    r"\bbuy\b",
    r"\bpay\b",
    r"\binvest\b",
    r"\bsign\s+contract\b",
    r"\bsend\s+(?:email|message|sms)\s+to\b",
})

_COMPILED_HIGH_STAKES_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in HIGH_STAKES_KEYWORDS
)


def is_high_stakes_message(text: str) -> bool:
    """Return True if ``text`` matches any high-stakes intent pattern.

    Used by the chat orchestrator to decide whether to run the weighted-vote
    consensus on this turn. Pure regex; can't be prompt-injected.
    """
    if not text:
        return False
    for pattern in _COMPILED_HIGH_STAKES_PATTERNS:
        if pattern.search(text):
            return True
    return False


@dataclass
class AgentPolicy:
    """Governance policy for a specific agent or role."""
    agent_id: str = ""           # empty = applies to all agents
    agent_role: str = ""         # empty = applies to all roles
    max_tool_calls: int = 100    # max tool calls per task (0 = unlimited)
    max_tokens: int = 0          # max token budget (0 = unlimited)
    allowed_tools: list = field(default_factory=list)   # empty = all tools allowed
    forbidden_tools: list = field(default_factory=list)  # explicit denials
    forbidden_patterns: list = field(default_factory=list)  # regex patterns to block


# ── Default policies ─────────────────────────────────────────────────────────

_DEFAULT_POLICIES = {
    "default": AgentPolicy(
        max_tool_calls=100,
        max_tokens=0,  # unlimited by default
        forbidden_tools=["rm_recursive", "shell_root"],
    ),
    "worker": AgentPolicy(
        agent_role="worker",
        max_tool_calls=50,
        forbidden_tools=["rm_recursive", "shell_root", "git_checkout"],
    ),
    "coordinator": AgentPolicy(
        agent_role="coordinator",
        max_tool_calls=20,  # coordinators plan, not execute
        allowed_tools=["file_read", "file_glob", "file_grep", "git_status", "git_log"],
    ),
}


class EscalationChannel:
    """
    Wiser-Human-style escalation channel.

    Detects Lynch et al.'s three triggers (replacement_threat,
    autonomy_reduction, goal_conflict) using deterministic regex against
    the user message and system prompt of any pending tool-call or
    chat-send. When a trigger fires, ``check_escalation`` writes a row to
    the ``escalations`` table and emits an ``escalation_required`` SSE
    event. The orchestrator returns a placeholder ChatResult and the human
    user must approve via the new /api/escalation endpoints.

    Detection is regex-only — never an LLM — so it cannot be
    prompt-injected into bypassing itself.
    """

    def __init__(self, settings=None) -> None:
        self._settings = settings

    def _is_enabled(self) -> bool:
        if self._settings is None:
            return True
        try:
            return bool(self._settings.get("escalation_channel_enabled", True))
        except Exception:
            return True

    def would_trigger(self, user_message: str, system_prompt: str) -> bool:
        """Side-effect-free predicate: would check_escalation fire?

        Phase 8 voting needs to know whether this turn will escalate WITHOUT
        actually recording an escalations row or emitting the SSE event yet,
        so the consensus loop can run first and the modal fires after.

        system_prompt is accepted for API stability but not scanned.
        """
        if not self._is_enabled():
            return False
        haystack = user_message or ""
        if not haystack.strip():
            return False
        for compiled in _COMPILED_ESCALATION_PATTERNS.values():
            for pattern in compiled:
                if pattern.search(haystack) is not None:
                    return True
        return False

    def check_escalation(
        self,
        conversation_id: str,
        user_message: str,
        system_prompt: str,
        proposed_action: dict | None = None,
    ) -> EscalationVerdict:
        """Scan this turn for escalation triggers and record + emit if matched.

        system_prompt is accepted for API stability but not scanned.
        """
        if not self._is_enabled():
            return EscalationVerdict(requires_review=False)

        # Lynch et al. patterns describe threats from the user against the model.
        # system_prompt here includes retrieved RAG and saved memory — third-party
        # content that happens to discuss AI safety would otherwise fire the
        # escalation modal on every research-corpus chat. Scan user_message only.
        haystack = user_message or ""
        if not haystack.strip():
            return EscalationVerdict(requires_review=False)

        for trigger_type, compiled in _COMPILED_ESCALATION_PATTERNS.items():
            for pattern in compiled:
                match = pattern.search(haystack)
                if match is None:
                    continue
                detail = match.group(0)
                escalation_id = str(uuid.uuid4())
                self._record_escalation(
                    escalation_id=escalation_id,
                    conversation_id=conversation_id,
                    trigger_type=trigger_type,
                    trigger_detail=detail,
                    model_input=user_message,
                    proposed_action=proposed_action,
                )
                self._emit_escalation_required(
                    escalation_id=escalation_id,
                    conversation_id=conversation_id,
                    trigger_type=trigger_type,
                    trigger_detail=detail,
                )
                return EscalationVerdict(
                    requires_review=True,
                    trigger_type=trigger_type,
                    trigger_detail=detail,
                    escalation_id=escalation_id,
                )

        return EscalationVerdict(requires_review=False)

    def _record_escalation(
        self,
        *,
        escalation_id: str,
        conversation_id: str,
        trigger_type: str,
        trigger_detail: str,
        model_input: str,
        proposed_action: dict | None,
    ) -> None:
        if _db is None:
            return
        try:
            _db.execute(
                "INSERT INTO escalations "
                "(id, conversation_id, triggered_at, trigger_type, trigger_detail, "
                "model_input, proposed_action, decision, decided_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    escalation_id,
                    conversation_id,
                    datetime.now(timezone.utc).isoformat(),
                    trigger_type,
                    trigger_detail,
                    model_input or "",
                    json.dumps(proposed_action) if proposed_action else None,
                    "pending",
                    None,
                ),
            )
            _db.commit()
        except Exception as exc:
            log.warning("escalation insert failed: %s", exc)

    def _emit_escalation_required(
        self,
        *,
        escalation_id: str,
        conversation_id: str,
        trigger_type: str,
        trigger_detail: str,
    ) -> None:
        if _sse_events is None:
            return
        try:
            _sse_events.publish("escalation_required", {
                "escalation_id": escalation_id,
                "trigger_type": trigger_type,
                "trigger_detail": trigger_detail,
                "conversation_id": conversation_id,
            })
        except Exception as exc:
            log.debug("escalation_required emit failed: %s", exc)


class GovernanceEngine:
    """
    Evaluate agent actions against governance policies.

    Integrates into the agent_loop and task_scheduler to enforce
    tool budgets, allowlists, and token limits.
    """

    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._policies = dict(_DEFAULT_POLICIES)
        self._tool_counts: dict[str, int] = {}  # task_key -> count
        self._token_counts: dict[str, int] = {}  # task_key -> tokens used
        # Phase 6 Reader/Actor split: per-task allowlist of tool names the
        # Reader proposed for this turn. Populated by the orchestrator before
        # the Actor invocation; consulted by check_tool_call when the
        # reader_actor_split_enabled setting is on and the call is from the
        # Actor role.
        self._proposed_tools: dict[str, frozenset[str]] = {}
        self._enabled = True
        self.escalation_channel = EscalationChannel(settings)

        # Load custom policies from settings
        if settings:
            self._load_custom_policies(settings)

    # ── Phase 6: Reader/Actor split tool gating ──────────────────────────────

    def set_proposed_tools(self, task_key: str, tool_names) -> None:
        """Record the Reader's proposed tool names for the Actor's turn."""
        if not task_key:
            return
        self._proposed_tools[task_key] = frozenset(
            str(t) for t in (tool_names or []) if str(t)
        )

    def clear_proposed_tools(self, task_key: str) -> None:
        self._proposed_tools.pop(task_key, None)

    def _split_enabled(self) -> bool:
        if self._settings is None:
            return False
        try:
            return bool(self._settings.get("reader_actor_split_enabled", False))
        except Exception:
            return False

    def _load_custom_policies(self, settings) -> None:
        """Load governance_policies from settings.json."""
        raw = settings.get("governance_policies", None)
        if not raw or not isinstance(raw, dict):
            return
        for name, policy_data in raw.items():
            if isinstance(policy_data, dict):
                self._policies[name] = AgentPolicy(
                    agent_id=policy_data.get("agent_id", ""),
                    agent_role=policy_data.get("agent_role", ""),
                    max_tool_calls=policy_data.get("max_tool_calls", 100),
                    max_tokens=policy_data.get("max_tokens", 0),
                    allowed_tools=policy_data.get("allowed_tools", []),
                    forbidden_tools=policy_data.get("forbidden_tools", []),
                    forbidden_patterns=policy_data.get("forbidden_patterns", []),
                )

    # ── Policy lookup ─────────────────────────────────────────────────────

    def _get_policy(self, agent_id: str = "", agent_role: str = "") -> AgentPolicy:
        """Find the most specific policy for an agent."""
        # Check agent-specific policy first
        for policy in self._policies.values():
            if policy.agent_id and policy.agent_id == agent_id:
                return policy
        # Then role-based policy
        for policy in self._policies.values():
            if policy.agent_role and policy.agent_role == agent_role:
                return policy
        # Default
        return self._policies.get("default", AgentPolicy())

    # ── Evaluation ────────────────────────────────────────────────────────

    def check_tool_call(
        self,
        tool_name: str,
        agent_id: str = "",
        agent_role: str = "",
        task_key: str = "",
    ) -> PolicyVerdict:
        """
        Check if a tool call is allowed by governance policy.
        Called before tool execution in the agent loop.

        Phase 6: when ``agent_role == "actor"`` and reader_actor_split is on,
        the tool name must also appear in the Reader's proposed_tools for
        ``task_key``. This is the architectural wall — the Actor cannot call
        a tool the Reader did not authorize this turn.
        """
        if not self._enabled:
            return PolicyVerdict(allowed=True)

        # Phase 6 Reader/Actor split gate. Runs before the policy lookup so
        # an Actor proposing an out-of-plan tool is rejected even if no
        # role-specific policy exists.
        if agent_role == "actor" and self._split_enabled():
            allowed = self._proposed_tools.get(task_key, frozenset())
            if tool_name not in allowed:
                verdict = PolicyVerdict(
                    allowed=False,
                    reason=(
                        f"Tool '{tool_name}' not in Reader's proposed_tools "
                        f"for this turn (allowed: {sorted(allowed)})"
                    ),
                    policy_name="reader_actor_split",
                )
                self._log_evaluation(verdict, tool_name, agent_id, task_key)
                return verdict

        policy = self._get_policy(agent_id, agent_role)

        # Check forbidden tools
        if tool_name in policy.forbidden_tools:
            verdict = PolicyVerdict(
                allowed=False,
                reason=f"Tool '{tool_name}' is forbidden by governance policy",
                policy_name="forbidden_tools",
            )
            self._log_evaluation(verdict, tool_name, agent_id, task_key)
            return verdict

        # Check allowed tools (if specified, only these are allowed)
        if policy.allowed_tools and tool_name not in policy.allowed_tools:
            verdict = PolicyVerdict(
                allowed=False,
                reason=f"Tool '{tool_name}' not in allowed list: {policy.allowed_tools}",
                policy_name="allowed_tools",
            )
            self._log_evaluation(verdict, tool_name, agent_id, task_key)
            return verdict

        # Check tool call budget
        if policy.max_tool_calls > 0 and task_key:
            count = self._tool_counts.get(task_key, 0)
            if count >= policy.max_tool_calls:
                verdict = PolicyVerdict(
                    allowed=False,
                    reason=f"Tool call budget exhausted ({count}/{policy.max_tool_calls})",
                    policy_name="max_tool_calls",
                )
                self._log_evaluation(verdict, tool_name, agent_id, task_key)
                return verdict
            self._tool_counts[task_key] = count + 1

        verdict = PolicyVerdict(allowed=True)
        self._log_evaluation(verdict, tool_name, agent_id, task_key)
        return verdict

    def check_token_budget(
        self,
        tokens_used: int,
        agent_id: str = "",
        agent_role: str = "",
        task_key: str = "",
    ) -> PolicyVerdict:
        """Check if token usage is within budget."""
        if not self._enabled:
            return PolicyVerdict(allowed=True)

        policy = self._get_policy(agent_id, agent_role)
        if policy.max_tokens <= 0:
            return PolicyVerdict(allowed=True)

        current = self._token_counts.get(task_key, 0) + tokens_used
        if current > policy.max_tokens:
            return PolicyVerdict(
                allowed=False,
                reason=f"Token budget exceeded ({current}/{policy.max_tokens})",
                policy_name="max_tokens",
            )
        self._token_counts[task_key] = current
        return PolicyVerdict(allowed=True)

    def reset_task_counters(self, task_key: str) -> None:
        """Reset tool and token counters for a task."""
        self._tool_counts.pop(task_key, None)
        self._token_counts.pop(task_key, None)
        self._proposed_tools.pop(task_key, None)

    # ── Audit logging ─────────────────────────────────────────────────────

    def _log_evaluation(
        self,
        verdict: PolicyVerdict,
        tool_name: str,
        agent_id: str,
        task_key: str,
    ) -> None:
        """Log policy evaluation to the governance_log table."""
        if _db is None:
            return
        if verdict.allowed:
            return  # only log denials to keep the table manageable
        try:
            _db.execute(
                "INSERT OR IGNORE INTO governance_log "
                "(id, agent_id, tool_name, allowed, reason, policy_name, task_key, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()), agent_id, tool_name,
                    1 if verdict.allowed else 0,
                    verdict.reason, verdict.policy_name,
                    task_key, datetime.now(timezone.utc).isoformat(),
                ),
            )
        except Exception as exc:
            log.debug("Governance audit log failed: %s", exc)

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "policies": {
                name: {
                    "agent_role": p.agent_role,
                    "max_tool_calls": p.max_tool_calls,
                    "max_tokens": p.max_tokens,
                    "allowed_tools": p.allowed_tools,
                    "forbidden_tools": p.forbidden_tools,
                }
                for name, p in self._policies.items()
            },
            "active_tasks": len(self._tool_counts),
        }

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
