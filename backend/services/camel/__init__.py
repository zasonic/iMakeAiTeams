"""
services/camel/ — Phase 12: CaMeL (Defeating Prompt Injections by Design).

Clean-room reimplementation of the algorithm described in
arXiv 2503.18813 (DeepMind/ETH, March 2025). The pattern:

  - Privileged LLM (P-LLM) reads the user message and emits a PLAN as
    restricted Python source.
  - The plan can call tools by name with arguments. Tool results are
    values that flow through the plan.
  - Quarantined LLM (Q-LLM) processes untrusted retrieved data. Its
    outputs inherit the UNTRUSTED capability tag.
  - An interpreter walks the plan AST. Values from tools / Q-LLM carry
    capability tags; the interpreter REJECTS any control-flow operation
    that would consume a tagged value (so the model cannot be tricked
    into running attacker-provided code via the data path).

Public surface:
    - capabilities.Capability / CapabilityTaggedResult / capabilities_for_tool
    - exceptions.CamelToolDenied / CapabilityViolation / PlanParseError
    - adapter.make_tool_executor_for_turn
    - interpreter.CamelInterpreter / InterpreterResult
    - pipeline.camel_plan_and_execute
    - prompts.P_LLM_SYSTEM_PROMPT / Q_LLM_SYSTEM_PROMPT
"""

from .capabilities import (
    Capability,
    CapabilityTaggedResult,
    UNTRUSTED_TOOLS,
    capabilities_for_tool,
)
from .exceptions import (
    CamelToolDenied,
    CapabilityViolation,
    PlanParseError,
)
from .interpreter import CamelInterpreter, InterpreterResult
from .pipeline import camel_plan_and_execute
from .prompts import P_LLM_SYSTEM_PROMPT, Q_LLM_SYSTEM_PROMPT
from .adapter import make_tool_executor_for_turn

__all__ = [
    "Capability",
    "CapabilityTaggedResult",
    "UNTRUSTED_TOOLS",
    "capabilities_for_tool",
    "CamelToolDenied",
    "CapabilityViolation",
    "PlanParseError",
    "CamelInterpreter",
    "InterpreterResult",
    "camel_plan_and_execute",
    "P_LLM_SYSTEM_PROMPT",
    "Q_LLM_SYSTEM_PROMPT",
    "make_tool_executor_for_turn",
]
