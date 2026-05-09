"""
services/camel/exceptions.py — CaMeL-specific error types.

Three failure modes the interpreter and adapter raise:

  CamelToolDenied      — governance refused the tool call. The pipeline
                         records it as a blocked_call without aborting
                         the plan; later tool calls may still succeed.
  CapabilityViolation  — the plan tried to use an UNTRUSTED value in a
                         position that would let attacker data drive
                         control flow (function name, attribute name)
                         OR exceeded max_steps. The interpreter aborts.
  PlanParseError       — the privileged-LLM output failed ``ast.parse``,
                         or the plan contained a node type that is not
                         on the allow-list.
"""

from __future__ import annotations


class CamelError(Exception):
    """Base class so callers can catch all CaMeL-specific errors at once."""


class CamelToolDenied(CamelError):
    """Raised when GovernanceEngine refuses a tool call inside the plan."""

    def __init__(self, reason: str, policy_name: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.policy_name = policy_name


class CapabilityViolation(CamelError):
    """Raised when the plan would consume an UNTRUSTED value as a function
    name, attribute name, or otherwise drive control flow from data, OR
    when the interpreter exceeds its max_steps budget."""

    def __init__(self, message: str, ast_node_repr: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.ast_node_repr = ast_node_repr


class PlanParseError(CamelError):
    """Raised when the privileged-LLM plan source is not parseable as
    restricted Python (syntax error, or a disallowed AST node such as
    Import / FunctionDef / For / While)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
