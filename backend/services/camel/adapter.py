"""
services/camel/adapter.py — Tool-executor adapter for the CaMeL interpreter.

The interpreter does not know about governance, the execution bridge, or
provenance tagging. It calls a single ``tool_executor(name, args, kwargs)``
closure and expects a ``CapabilityTaggedResult`` back. This module is the
seam between that abstract interface and the host:

  1. Run governance.check_tool_call (per-task budget + allowlist + the
     Reader/Actor split's proposed_tools when applicable).
  2. Dispatch to ``execution_bridge.run_tool(name, args, kwargs)``. The
     bridge is treated as an opaque object; tests can pass a stub with a
     callable ``run_tool`` attribute to drive the closure deterministically.
  3. Wrap the bridge's return value in a ``CapabilityTaggedResult`` whose
     tags come from ``capabilities_for_tool(name)``.

When governance denies the call the closure raises CamelToolDenied so the
interpreter records it as a blocked_call and the plan can keep running.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .capabilities import CapabilityTaggedResult, capabilities_for_tool
from .exceptions import CamelToolDenied

log = logging.getLogger("iMakeAiTeams.camel.adapter")


def make_tool_executor_for_turn(
    *,
    agent_id: str,
    conversation_id: str,
    governance: Any,
    execution_bridge: Any,
) -> Callable[..., CapabilityTaggedResult]:
    """Build the closure passed to CamelInterpreter as ``tool_executor``.

    Parameters mirror the per-turn data already plumbed through the
    chat orchestrator: ``governance`` is the existing GovernanceEngine
    instance, ``execution_bridge`` is whatever object exposes
    ``run_tool(name, args, kwargs)``. When either is None the closure
    falls back to a permissive / no-op path so unit tests for the
    interpreter can construct a self-contained executor.

    The returned closure has the signature

        executor(tool_name: str, args: list, kwargs: dict)
            -> CapabilityTaggedResult

    and raises ``CamelToolDenied`` for governance refusals.
    """

    def _executor(tool_name: str, args: list, kwargs: dict) -> CapabilityTaggedResult:
        if governance is not None:
            try:
                verdict = governance.check_tool_call(
                    tool_name=tool_name,
                    agent_id=agent_id,
                    task_key=conversation_id,
                )
            except Exception as exc:
                log.debug("governance.check_tool_call raised, treating as deny: %s", exc)
                raise CamelToolDenied(
                    reason=f"governance check raised: {exc}",
                    policy_name="exception",
                )
            if not getattr(verdict, "allowed", True):
                raise CamelToolDenied(
                    reason=getattr(verdict, "reason", "denied by policy"),
                    policy_name=getattr(verdict, "policy_name", ""),
                )

        if execution_bridge is None or not hasattr(execution_bridge, "run_tool"):
            value = None
        else:
            try:
                value = execution_bridge.run_tool(tool_name, list(args), dict(kwargs))
            except CamelToolDenied:
                raise
            except Exception as exc:
                log.warning("execution_bridge.run_tool(%s) failed: %s", tool_name, exc)
                value = f"[tool {tool_name} error: {exc}]"

        caps = capabilities_for_tool(tool_name)
        return CapabilityTaggedResult(value=value, capabilities=caps)

    return _executor
