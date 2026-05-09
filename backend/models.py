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
                       reasoning="parse failed", confidence=0.5)


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


def semantic_validate_handoff(
    packet: HandoffPacket,
    local_client,
    claude_client=None,
) -> HandoffPacket:
    """
    Structural + semantic validation of a HandoffPacket.

    Runs `validate_handoff_packet()` first, then if structural validation
    passed, asks the local model to score whether the deliverables actually
    satisfy the requested subtask. A score below 4 flags the packet
    (validation_passed=False) and appends a note. The semantic check is
    silently skipped (no flag) when the local model is unavailable or its
    response can't be parsed — never blocks the handoff.
    """
    import logging as _logging
    _log = _logging.getLogger("iMakeAiTeams.models")

    validate_handoff_packet(packet)
    if not packet.validation_passed:
        return packet

    try:
        from services.task_artifacts import local_first_call
    except Exception as exc:
        _log.debug("semantic_validate_handoff: local_first_call import failed: %s", exc)
        return packet

    system = (
        "You are a quality reviewer. Given a task description and deliverables, "
        "score whether the deliverables satisfy the task. Return ONLY JSON: "
        '{"score": 0-10, "reason": "one sentence"}'
    )
    user_message = (
        f"TASK: {packet.subtask_completed[:300]}\n"
        f"DELIVERABLES: {packet.artifact[:500]}"
    )

    try:
        raw = local_first_call(local_client, claude_client, system, user_message, max_tokens=120)
    except Exception as exc:
        _log.debug("semantic_validate_handoff: local_first_call raised: %s", exc)
        return packet
    if not raw:
        return packet

    qstart = raw.find("{")
    qend = raw.rfind("}")
    if qstart == -1 or qend == -1 or qend <= qstart:
        _log.debug("semantic_validate_handoff: no JSON object in response")
        return packet
    try:
        verdict = json.loads(raw[qstart:qend + 1])
    except (ValueError, TypeError) as exc:
        _log.debug("semantic_validate_handoff: JSON parse failed: %s", exc)
        return packet

    try:
        score = float(verdict.get("score", 10))
    except (TypeError, ValueError):
        _log.debug("semantic_validate_handoff: non-numeric score")
        return packet
    reason = str(verdict.get("reason", "")).strip()[:200]

    if score < 4:
        packet.validation_notes.append(
            f"Semantic quality check: score {score:.1f}/10 — {reason}"
        )
        packet.validation_passed = False
    return packet


# Symphony-inspired proof-of-work threshold. Below this score, the local model
# judges that the handoff's deliverables don't satisfy the requested subtask.
# The packet is flagged (validation_passed=False) but the workflow is NOT
# blocked — downstream agents see the warning in to_context_block() and can
# decide whether to proceed.
HANDOFF_POW_THRESHOLD = 4.0


def proof_of_work_validate_handoff(
    packet: HandoffPacket,
    task_description: str,
    local_client,
) -> HandoffPacket:
    """
    Symphony-inspired semantic validation: judge whether the handoff's
    deliverables actually satisfy the stated subtask, using the local model
    (free). Updates packet.validation_passed and packet.validation_notes
    in place. Returns the same packet.

    No-op (returns packet unchanged) when:
      - local_client is None
      - local_client.is_available() returns False
      - the local model fails to produce parseable JSON

    This is a quality gate, not a security gate — it should never block a
    workflow on its own. Callers compose this with validate_handoff_packet()
    to get both structural AND semantic validation.
    """
    if not local_client:
        return packet
    try:
        if not local_client.is_available():
            return packet
    except Exception:
        return packet

    system = (
        "You are a quality auditor. Given a task description and an agent's "
        "self-reported deliverables, judge whether the deliverables actually "
        "satisfy the task. Be strict: empty, evasive, or off-topic outputs "
        "should score low. Respond with ONLY a JSON object: "
        '{"score": 0-10, "reason": "..."}'
    )
    user = (
        f"TASK: {task_description[:300]}\n\n"
        f"AGENT'S CLAIMED SUBTASK: {packet.subtask_completed[:200]}\n"
        f"AGENT'S DELIVERABLE: {packet.artifact[:600]}\n"
        f"AGENT'S SELF-REPORTED CONFIDENCE: {packet.confidence:.0%}"
    )

    try:
        raw = local_client.chat(system, user, max_tokens=120)
    except Exception:
        return packet
    if not raw:
        return packet

    qstart = raw.find("{")
    qend   = raw.rfind("}")
    if qstart == -1 or qend == -1 or qend <= qstart:
        return packet
    try:
        verdict = json.loads(raw[qstart:qend + 1])
    except (ValueError, TypeError):
        return packet

    try:
        score = float(verdict.get("score", 10))
    except (TypeError, ValueError):
        return packet
    reason = str(verdict.get("reason", "")).strip()[:200]

    if score < HANDOFF_POW_THRESHOLD:
        packet.validation_passed = False
        packet.validation_notes.append(
            f"proof-of-work failed: scored {score:.1f}/10 — {reason}"
        )
    else:
        packet.validation_notes.append(
            f"proof-of-work passed: scored {score:.1f}/10"
        )
    return packet


# ── Priority 6: Adversarial debate (Du 2024) ─────────────────────────────────
#
# A ChallengePacket is the output of a "challenger" agent that critiques a
# committed HandoffPacket. The synthesizer receives both the original artifact
# and the challenge so it can resolve disagreements before answering the user.
# Fields mirror the debate_log columns one-for-one so persistence is a single
# straight-line INSERT.


@dataclass
class ChallengePacket:
    """Adversarial critique of a committed HandoffPacket."""
    challenge_id:        str
    debate_id:           str          # one debate per turn; ties many challenges together
    workflow_id:         Optional[str]
    agent_id:            str          # the challenger's id
    agent_name:          str          # the challenger's display name
    assumption_diffs:    list = field(default_factory=list)   # assumptions the challenger disputes
    fact_conflicts:      list = field(default_factory=list)   # claimed facts that conflict
    missing_analysis:    list = field(default_factory=list)   # gaps the original missed
    changed_position:    bool = False                          # would the challenger draw a different conclusion?
    revised_conclusion:  Optional[str] = None                  # the challenger's preferred answer (optional)
    overall_assessment:  str = ""                              # one-sentence summary
    input_tokens:        int = 0
    output_tokens:       int = 0
    duration_ms:         float = 0.0
    parse_failed:        bool = False                          # JSON parse failure → packet is best-effort, no signal
    timestamp:           str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def has_signal(self) -> bool:
        """Did the challenger find anything worth surfacing to the synthesizer?"""
        if self.parse_failed:
            return False
        return bool(
            self.assumption_diffs or self.fact_conflicts
            or self.missing_analysis or self.changed_position
            or (self.revised_conclusion or "").strip()
        )

    def to_context_block(self) -> str:
        """Format for injection into the synthesizer prompt."""
        if not self.has_signal():
            return ""
        lines = [f"## Challenger review by {self.agent_name}"]
        if self.overall_assessment:
            lines.append(f"**Assessment:** {self.overall_assessment}")
        if self.assumption_diffs:
            lines.append("**Disputed assumptions:**")
            for a in self.assumption_diffs:
                lines.append(f"- {a}")
        if self.fact_conflicts:
            lines.append("**Fact conflicts:**")
            for f in self.fact_conflicts:
                lines.append(f"- {f}")
        if self.missing_analysis:
            lines.append("**Missing analysis:**")
            for m in self.missing_analysis:
                lines.append(f"- {m}")
        if self.changed_position and self.revised_conclusion:
            lines.append(f"**Challenger's preferred conclusion:** {self.revised_conclusion}")
        lines.append("---")
        return "\n".join(lines)


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


# Phase 6: Hackett et al. (ACL 2025) Reader/Actor split. The Reader produces
# this structured plan; the Actor executes against it without ever seeing the
# raw user message or raw retrieved data.
@dataclass(frozen=True)
class ReaderOutput:
    intent:          str                      # what the user is asking
    constraints:     tuple[str, ...] = ()     # explicit constraints from the message
    relevant_facts:  tuple[str, ...] = ()     # facts the Reader believes are relevant
    proposed_tools:  tuple[str, ...] = ()     # tool NAMES the Reader proposes
    red_flags:       tuple[str, ...] = ()     # suspicious patterns in retrieved data

    def to_json(self) -> str:
        return json.dumps({
            "intent": self.intent,
            "constraints": list(self.constraints),
            "relevant_facts": list(self.relevant_facts),
            "proposed_tools": list(self.proposed_tools),
            "red_flags": list(self.red_flags),
        }, ensure_ascii=False)

    @classmethod
    def from_raw(cls, raw: str) -> "ReaderOutput":
        """Parse the Reader's JSON output. Tolerant of stray fences/prose."""
        if not raw:
            return cls(intent="")
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return cls(intent="")
        try:
            d = json.loads(cleaned[start:end + 1])
        except (TypeError, ValueError):
            return cls(intent="")
        if not isinstance(d, dict):
            return cls(intent="")

        def _str_list(v) -> tuple[str, ...]:
            if not isinstance(v, list):
                return ()
            return tuple(str(x) for x in v if isinstance(x, (str, int, float)))

        return cls(
            intent=str(d.get("intent", "")),
            constraints=_str_list(d.get("constraints")),
            relevant_facts=_str_list(d.get("relevant_facts")),
            proposed_tools=_str_list(d.get("proposed_tools")),
            red_flags=_str_list(d.get("red_flags")),
        )


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
