"""
models.py — Typed data contracts for iMakeAiTeams.

All core data structures are frozen dataclasses. Internal code passes these
typed objects. Only api.py converts to dicts at the JS boundary.

Original stage 5 additions (UNCHANGED):
  - RouteDecision, ChatResult, TokenUsage, StreamEvent (Improvement 1)
  - PermissionDenial, ToolPermissionContext (Improvement 4)
  - HistoryEvent, SessionHistory (Improvement 5)
  - ExecutionTarget (Improvement 6)

Priority 3 additions (NEW — additive only):
  - HandoffPacket         — structured inter-agent handoff
  - HandoffValidation     — validation result for a HandoffPacket
  - extract_handoff_packet() — parse <handoff> block from agent response
  - validate_handoff_packet() — validate and annotate a HandoffPacket
  - HANDOFF_SYSTEM_FRAGMENT  — injected into every workflow agent prompt
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


# ── Improvement 1: Core data contracts (UNCHANGED) ────────────────────────────

@dataclass(frozen=True)
class RouteDecision:
    model: str          # "claude" | "local"
    complexity: str     # "simple" | "medium" | "complex"
    reasoning: str = ""
    confidence: float = 1.0   # 0.0–1.0, UAR-inspired epistemic signal
    needs_context: bool = False  # True when model signals it needs more info

    @classmethod
    def from_json(cls, raw: str) -> "RouteDecision":
        import json
        try:
            clean = raw.strip().strip("`")
            if clean.startswith("json"):
                clean = clean[4:]
            d = json.loads(clean)
            conf = d.get("confidence", 0.8)
            try:
                conf = max(0.0, min(1.0, float(conf)))
            except (TypeError, ValueError):
                conf = 0.8
            return cls(
                model=d.get("model", "claude"),
                complexity=d.get("complexity", "complex"),
                reasoning=d.get("reasoning", ""),
                confidence=conf,
                needs_context=bool(d.get("needs_context", False)),
            )
        except Exception:
            return cls(model="claude", complexity="complex",
                       reasoning="parse failed", confidence=0.0)


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str
    route_reason: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    message_id: str
    budget_warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, inp: int, out: int, cost: float) -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + inp,
            output_tokens=self.output_tokens + out,
            cost_usd=self.cost_usd + cost,
        )

    def combine(self, other: "TokenUsage") -> "TokenUsage":
        """Combine two TokenUsage instances — useful for aggregating workflow costs."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class StreamEvent:
    """Typed streaming event sent to the frontend."""
    event_type: str          # "message_start" | "route_decided" | "memory_recalled" | "token" | "message_done" | "error"
    conversation_id: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.event_type, "conversation_id": self.conversation_id, **self.data}


# ── Improvement 4: Tool permission context (UNCHANGED) ────────────────────────

@dataclass(frozen=True)
class PermissionDenial:
    tool_name: str
    reason: str


@dataclass(frozen=True)
class ToolPermissionContext:
    deny_names: frozenset[str] = field(default_factory=frozenset)
    deny_prefixes: tuple[str, ...] = ()

    @classmethod
    def from_iterables(cls, deny_names: list[str] | None = None,
                       deny_prefixes: list[str] | None = None) -> "ToolPermissionContext":
        return cls(
            deny_names=frozenset(n.lower() for n in (deny_names or [])),
            deny_prefixes=tuple(p.lower() for p in (deny_prefixes or [])),
        )

    def blocks(self, name: str) -> bool:
        lowered = name.lower()
        return lowered in self.deny_names or any(lowered.startswith(p) for p in self.deny_prefixes)


# ── Improvement 5: Session history / transcript (UNCHANGED) ──────────────────

@dataclass
class HistoryEvent:
    event_type: str    # "routing", "memory_recall", "fact_extracted", "summarized", "error"
    detail: str
    timestamp: str


@dataclass
class SessionHistory:
    events: list[HistoryEvent] = field(default_factory=list)

    def add(self, event_type: str, detail: str) -> None:
        self.events.append(HistoryEvent(
            event_type=event_type,
            detail=detail,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

    def recent(self, n: int = 20) -> list[HistoryEvent]:
        return self.events[-n:]


# ── Improvement 6: Execution target (UNCHANGED) ──────────────────────────────

@dataclass(frozen=True)
class ExecutionTarget:
    backend: str        # "claude" | "local"
    model_name: str
    max_tokens: int


# ── Priority 3: HandoffPacket (NEW) ──────────────────────────────────────────

# System prompt fragment injected into every workflow agent's prompt.
# Tells the agent to append a structured <handoff> block after its main output.
HANDOFF_SYSTEM_FRAGMENT = """
---
## Required Output Format for Workflow Handoffs

After completing your assigned subtask, append this block at the very end of your response:

<handoff>
{
  "subtask_completed": "One sentence: what you were asked to do",
  "artifact": "Your key finding or deliverable. Summarize if very long.",
  "assumptions": ["Every assumption you made that was not explicitly stated"],
  "uncertainties": ["Everything you are not certain about"],
  "confidence": 0.85,
  "date_scope": null,
  "domain_scope": null
}
</handoff>

Rules:
- assumptions: list EVERY interpretive choice you made. Empty list = you made none.
- uncertainties: if confidence < 0.95, this list CANNOT be empty. Silence = overconfidence.
- confidence: your honest 0.0–1.0 assessment. Be accurate.
- The handoff block is appended AFTER your main work output, not instead of it.
---
"""

HANDOFF_OPEN_TAG  = "<handoff>"
HANDOFF_CLOSE_TAG = "</handoff>"


@dataclass
class HandoffPacket:
    """
    Typed inter-agent handoff packet.

    NOT frozen — fields are annotated after validation.
    """
    agent_id:          str
    agent_name:        str
    subtask_completed: str
    artifact:          str
    assumptions:       list = field(default_factory=list)
    uncertainties:     list = field(default_factory=list)
    confidence:        float = 1.0
    date_scope:        Optional[str] = None
    domain_scope:      Optional[str] = None
    workflow_id:       Optional[str] = None
    step_index:        int = 0
    timestamp:         str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_output:        str = ""
    input_tokens:      int = 0
    output_tokens:     int = 0
    duration_ms:       float = 0.0
    validation_passed: bool = True
    validation_notes:  list = field(default_factory=list)

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.85:
            return "HIGH"
        if self.confidence >= 0.60:
            return "MEDIUM"
        return "LOW"

    def to_context_block(self) -> str:
        """Format for injection into downstream agent prompts."""
        lines = [
            f"## Upstream result from {self.agent_name}",
            f"**Subtask completed:** {self.subtask_completed}",
            "",
            f"**Artifact:**",
            self.artifact,
            "",
        ]
        if self.assumptions:
            lines.append("**Assumptions (treat as unverified):**")
            for a in self.assumptions:
                lines.append(f"- {a}")
            lines.append("")
        if self.uncertainties:
            lines.append("**Uncertainties flagged:**")
            for u in self.uncertainties:
                lines.append(f"- {u}")
            lines.append("")
        lines.append(f"**Confidence:** {self.confidence:.0%}")
        if self.date_scope:
            lines.append(f"**Date scope:** {self.date_scope}")
        if self.domain_scope:
            lines.append(f"**Domain scope:** {self.domain_scope}")
        if not self.validation_passed:
            lines.append("")
            lines.append("⚠️ **This handoff failed validation — review carefully before proceeding.**")
        lines.append("---")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "subtask_completed": self.subtask_completed,
            "artifact": self.artifact[:500],
            "assumptions": self.assumptions,
            "uncertainties": self.uncertainties,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
            "date_scope": self.date_scope,
            "domain_scope": self.domain_scope,
            "workflow_id": self.workflow_id,
            "step_index": self.step_index,
            "validation_passed": self.validation_passed,
            "validation_notes": self.validation_notes,
            "duration_ms": self.duration_ms,
        }


@dataclass
class HandoffValidation:
    passed:   bool
    errors:   list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @classmethod
    def validate(cls, packet: "HandoffPacket") -> "HandoffValidation":
        errors:   list = []
        warnings: list = []

        if not packet.subtask_completed.strip():
            errors.append("subtask_completed is empty.")
        if not packet.artifact.strip():
            errors.append("artifact is empty.")
        if not 0.0 <= packet.confidence <= 1.0:
            errors.append(f"confidence={packet.confidence} out of range.")
        if packet.confidence < 0.95 and not packet.uncertainties:
            errors.append(
                f"Confidence={packet.confidence:.0%} but uncertainties list is empty. "
                "An agent that is not fully confident MUST list its uncertainties."
            )
        if packet.confidence >= 0.95 and not packet.uncertainties:
            warnings.append("Agent reported near-full confidence with no uncertainties — verify this is warranted.")

        return cls(passed=len(errors) == 0, errors=errors, warnings=warnings)


def validate_handoff_packet(packet: HandoffPacket) -> HandoffPacket:
    """Validate a HandoffPacket in-place. Returns the packet."""
    result = HandoffValidation.validate(packet)
    packet.validation_passed = result.passed
    packet.validation_notes  = result.errors + result.warnings
    return packet


# ── Phase 1: Hub routing contracts (NEW) ─────────────────────────────────────
#
# These types support the deterministic HubRouter that selects a worker by
# declared skill match. They are distinct from RouteDecision above, which
# selects a *model backend* (Claude vs local) for a single chat exchange.
# RouteDecision answers "which model"; the types below answer "which worker".


@dataclass(frozen=True)
class Skill:
    """A capability declared by an agent. Matched against a TaskDescriptor."""
    name:   str
    scopes: tuple[str, ...] = ()  # e.g. ("read",), ("read", "write")

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        raw_scopes = d.get("scopes", []) or []
        return cls(
            name=str(d.get("name", "")).strip(),
            scopes=tuple(str(s).strip() for s in raw_scopes if str(s).strip()),
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "scopes": list(self.scopes)}


@dataclass(frozen=True)
class TaskDescriptor:
    """A unit of work submitted to the hub for routing."""
    text:             str
    required_skills:  tuple[str, ...] = ()    # any-of match
    required_scopes:  tuple[str, ...] = ()    # subset of chosen skill's scopes
    preferred_agent_id: Optional[str] = None  # caller's hint; still authz'd
    backend_hint:     Optional[str] = None    # "claude" | "local" | None


@dataclass(frozen=True)
class RoutingDecision:
    """Result of HubRouter.route() — names the chosen worker and why."""
    agent_id:    str           # selected worker
    backend:     str           # "claude" | "local"
    score:       float         # 0.0-1.0 specificity of match
    reasoning:   str           # human-readable selection reason
    used_fallback: bool = False  # True if LLM /no_think fallback fired
    skill_matched: str = ""    # which declared skill won
    # Phase 3: per-decision Qwen3 thinking budget. 0 means "no thinking" —
    # local dispatch goes through the plain path with no /think directive,
    # preserving compatibility with non-Qwen local models.
    thinking_budget: int = 0


@dataclass(frozen=True)
class WorkerResult:
    """Output of HubRouter.invoke() — wraps the model response uniformly."""
    text:          str
    backend:       str
    model_name:    str
    input_tokens:  int = 0
    output_tokens: int = 0
    had_error:     bool = False


def extract_handoff_packet(
    raw_response: str,
    agent_id:     str,
    agent_name:   str,
    workflow_id:  Optional[str] = None,
    step_index:   int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_ms:  float = 0.0,
) -> HandoffPacket:
    """
    Extract a HandoffPacket from an agent's raw text response.
    If no <handoff> block found, returns a degraded packet with the full
    response as the artifact so the workflow can continue.
    """
    start = raw_response.find(HANDOFF_OPEN_TAG)
    end   = raw_response.rfind(HANDOFF_CLOSE_TAG)

    if start == -1 or end == -1 or end <= start:
        return HandoffPacket(
            agent_id=agent_id, agent_name=agent_name,
            subtask_completed="(agent did not report subtask — see artifact)",
            artifact=raw_response.strip(),
            uncertainties=["Agent did not produce a structured handoff — output reliability unknown."],
            confidence=0.5,
            workflow_id=workflow_id, step_index=step_index,
            raw_output=raw_response,
            input_tokens=input_tokens, output_tokens=output_tokens,
            duration_ms=duration_ms,
            validation_passed=False,
            validation_notes=["No <handoff> block found — confidence set to 0.5 as conservative default."],
        )

    json_str = raw_response[start + len(HANDOFF_OPEN_TAG): end].strip()
    main_output = raw_response[:start].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return HandoffPacket(
            agent_id=agent_id, agent_name=agent_name,
            subtask_completed="(handoff JSON parse failed)",
            artifact=main_output or raw_response,
            uncertainties=[f"HandoffPacket JSON was malformed: {exc}"],
            confidence=0.4,
            workflow_id=workflow_id, step_index=step_index,
            raw_output=raw_response,
            input_tokens=input_tokens, output_tokens=output_tokens,
            duration_ms=duration_ms,
            validation_passed=False,
            validation_notes=[f"JSON parse error: {exc}"],
        )

    packet = HandoffPacket(
        agent_id=agent_id, agent_name=agent_name,
        subtask_completed=str(data.get("subtask_completed", "")).strip(),
        artifact=str(data.get("artifact", main_output)).strip() or main_output,
        assumptions=[str(a) for a in data.get("assumptions", []) if a],
        uncertainties=[str(u) for u in data.get("uncertainties", []) if u],
        confidence=float(data.get("confidence", 0.5)),
        date_scope=data.get("date_scope") or None,
        domain_scope=data.get("domain_scope") or None,
        workflow_id=workflow_id, step_index=step_index,
        raw_output=raw_response,
        input_tokens=input_tokens, output_tokens=output_tokens,
        duration_ms=duration_ms,
    )
    return validate_handoff_packet(packet)
