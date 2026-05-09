"""
services/camel/capabilities.py — Capability tags for CaMeL values.

A capability tag travels with every value that flows through the
interpreter. The two tags in the implementation:

  TRUSTED   — produced by constants in the privileged plan, by tools
              that the host trusts, or by composing only TRUSTED values.
  UNTRUSTED — produced by tools that read external / attacker-controlled
              surfaces (RAG, web fetch, MCP, file read) AND by every
              quarantined-LLM call.

The interpreter uses these tags to reject control-flow operations that
would consume an UNTRUSTED value. An attacker who plants a directive
inside a retrieved document cannot redirect the plan, because the
directive becomes a value that is structurally barred from being used as
a function name or as an attribute lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Capability(str, Enum):
    """The capability lattice. Order is irrelevant — set union is used."""
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class CapabilityTaggedResult:
    """A value paired with the capability tags that produced it.

    Equality compares the underlying value AND tags, so two tagged
    results with the same value but different provenance are not equal.
    The class is frozen so closures can hash it where the value type is
    hashable (e.g. literal ints, strings, frozensets).
    """
    value: Any
    capabilities: frozenset[Capability]

    def __post_init__(self) -> None:
        # Defensive: callers sometimes pass plain sets / lists; normalise
        # to frozenset so the dataclass stays hashable when value is too.
        if not isinstance(self.capabilities, frozenset):
            object.__setattr__(self, "capabilities", frozenset(self.capabilities))

    @property
    def is_trusted(self) -> bool:
        return Capability.UNTRUSTED not in self.capabilities

    @property
    def is_untrusted(self) -> bool:
        return Capability.UNTRUSTED in self.capabilities


# Tools whose output is treated as untrusted by default. Anything that
# crosses a network or filesystem boundary qualifies. The ``mcp_*``
# convention catches every dynamically registered MCP server tool.
UNTRUSTED_TOOLS: frozenset[str] = frozenset({
    "web_fetch",
    "rag_search",
    "file_read",
})


def capabilities_for_tool(tool_name: str) -> frozenset[Capability]:
    """Return the capability set that a tool's output should carry.

    Defaults to ``{TRUSTED}`` for unknown tools. Names that match the
    untrusted-tools set or that start with the ``mcp_`` prefix return
    ``{UNTRUSTED}`` so any value coming back from external sources is
    quarantined automatically.
    """
    if not tool_name:
        return frozenset({Capability.TRUSTED})
    if tool_name in UNTRUSTED_TOOLS or tool_name.startswith("mcp_"):
        return frozenset({Capability.UNTRUSTED})
    return frozenset({Capability.TRUSTED})


def merge_capabilities(*tagged: "CapabilityTaggedResult") -> frozenset[Capability]:
    """Union the capability sets of all operands. Used for BinOp / Compare /
    BoolOp / UnaryOp / Subscript so tags propagate through derived values."""
    out: frozenset[Capability] = frozenset()
    for t in tagged:
        if isinstance(t, CapabilityTaggedResult):
            out = out | t.capabilities
    if not out:
        out = frozenset({Capability.TRUSTED})
    return out
