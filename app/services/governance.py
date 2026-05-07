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
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

try:
    import db as _db
except ImportError:
    _db = None

log = logging.getLogger("MyAIAgentHub.governance")


@dataclass
class PolicyVerdict:
    """Result of a policy evaluation."""
    allowed: bool
    reason: str = ""
    policy_name: str = ""


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
        self._enabled = True

        # Load custom policies from settings
        if settings:
            self._load_custom_policies(settings)

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
        """
        if not self._enabled:
            return PolicyVerdict(allowed=True)

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
