"""
services/agent_registry.py

Agent and team management. CRUD operations backed by SQLite.

Priority 1 additions (Theory of Mind):
  - build_theory_of_mind_section()  — generates ToM block for any agent
  - refresh_team_tom(team_id)       — regenerates ToM for all team members
  - _strip_tom_block()              — removes existing ToM from a system prompt
  - Seed agents updated with full ToM prompts
  - agent_create() accepts domain, scope, tom_enabled
  - New API: generate_agent_tom(), refresh_team_theory_of_mind()
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone

import db as _db

log = logging.getLogger("MyAIEnv.agents")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Theory of Mind helpers ──────────────────────────────────────────────────

# Role descriptors for the 6 built-in agents (used by ToM builder)
# Default skill declarations per built-in role. Used by HubRouter for
# deterministic skill-match routing (Phase 1). Custom agents start with no
# skills and must declare their own.
_DEFAULT_ROLE_SKILLS: dict[str, list[dict]] = {
    "coordinator": [{"name": "coordinator", "scopes": ["read", "write"]}],
    "researcher":  [{"name": "researcher",  "scopes": ["read"]}],
    "analyst":     [{"name": "analyst",     "scopes": ["read"]}],
    "writer":      [{"name": "writer",      "scopes": ["write"]}],
    "coder":       [{"name": "coder",       "scopes": ["read", "write"]}],
    "reviewer":    [{"name": "reviewer",    "scopes": ["read"]}],
}


def default_skills_for_role(role_key: str) -> list[dict]:
    """Return the default skill list for a built-in role, or [] for custom."""
    return [dict(s) for s in _DEFAULT_ROLE_SKILLS.get(role_key, [])]


# Phase 3: per-role thinking budget defaults. Hub-style roles (coordinator,
# reviewer) get larger budgets because they synthesize over multiple inputs.
# Workers default to 2048 tokens. Roles not listed inherit 2048.
_DEFAULT_ROLE_THINKING_BUDGET: dict[str, int] = {
    "coordinator": 4096,
    "reviewer":    4096,
    "researcher":  2048,
    "analyst":     2048,
    "writer":      2048,
    "coder":       2048,
}


def default_thinking_budget_for_role(role_key: str) -> int:
    return int(_DEFAULT_ROLE_THINKING_BUDGET.get(role_key, 2048))


_BUILTIN_ROLE_DESCRIPTORS: dict[str, dict] = {
    "coordinator": {
        "domain": "task orchestration and cross-agent coordination",
        "visible_outputs": "final assembled outputs and handoff packets from all specialists",
        "cannot_see": "the reasoning steps each specialist used to reach their conclusions",
        "scope": "deciding which agent handles which subtask, resolving conflicts between outputs, and delivering the final synthesized response",
    },
    "researcher": {
        "domain": "information retrieval, source evaluation, and knowledge synthesis",
        "visible_outputs": "structured research summaries with cited sources and confidence levels",
        "cannot_see": "the coordinator's orchestration decisions or other specialists' intermediate reasoning",
        "scope": "finding and verifying factual information; flagging when a claim cannot be substantiated",
    },
    "analyst": {
        "domain": "data interpretation, pattern recognition, and logical inference",
        "visible_outputs": "structured analyses with explicit assumptions, findings, and confidence scores",
        "cannot_see": "raw research sources or the writer's drafts",
        "scope": "drawing conclusions from evidence provided; never sourcing new data independently",
    },
    "writer": {
        "domain": "written communication, structure, and narrative clarity",
        "visible_outputs": "polished prose, structured documents, and formatted deliverables",
        "cannot_see": "the analyst's raw inference chains or the researcher's source evaluation process",
        "scope": "transforming structured inputs into clear written output; flagging logical gaps in content provided to you",
    },
    "coder": {
        "domain": "software implementation, debugging, and technical architecture",
        "visible_outputs": "runnable code, technical specifications, and implementation notes",
        "cannot_see": "the analyst's business reasoning or the writer's narrative context unless explicitly passed",
        "scope": "translating requirements into working code; flagging ambiguous specifications before writing rather than guessing",
    },
    "reviewer": {
        "domain": "quality assurance, fact-checking, and consistency verification",
        "visible_outputs": "structured review reports with pass/fail verdicts and required corrections",
        "cannot_see": "any agent's reasoning process — only their final outputs submitted for review",
        "scope": "evaluating outputs against stated requirements and flagging every discrepancy, not just the most obvious",
    },
}


def build_theory_of_mind_section(
    agent_name: str,
    agent_role: str,
    agent_domain: str,
    teammates: list[dict],
    custom_scope: str | None = None,
) -> str:
    """
    Build the Theory of Mind block appended to every agent system prompt.

    Parameters
    ----------
    agent_name   : Display name ("Research Agent")
    agent_role   : Role key ("researcher", "writer", etc.) or "custom"
    agent_domain : One-line domain description
    teammates    : List of dicts with keys: name, domain, visible_outputs, cannot_see
    custom_scope : Override scope for custom agents

    Returns
    -------
    str — formatted ToM block ready to append to any system prompt.
    """
    role_info = _BUILTIN_ROLE_DESCRIPTORS.get(agent_role, {})
    scope = custom_scope or role_info.get(
        "scope", f"completing tasks within your domain of {agent_domain}"
    )

    lines: list[str] = [
        "",
        "---",
        "## Team Awareness & Communication Protocol",
        "",
        f"You are **{agent_name}**, specialized in {agent_domain}.",
        "",
    ]

    if teammates:
        lines.append("**Your teammates and what they can see:**")
        lines.append("")
        for tm in teammates:
            tm_domain   = tm.get("domain", "their assigned domain")
            tm_name     = tm.get("name", "A teammate")
            tm_visible  = tm.get("visible_outputs", "your final outputs only")
            tm_blind    = tm.get("cannot_see", "your internal reasoning")
            lines.append(f"- **{tm_name}** handles {tm_domain}.")
            lines.append(f"  They will receive: {tm_visible}.")
            lines.append(f"  They cannot see: {tm_blind}.")
            lines.append("")
    else:
        lines.append("You are currently operating as a standalone agent without assigned teammates.")
        lines.append("")

    lines += [
        "**Your scope boundary:**",
        scope,
        "Decisions outside this scope belong to the coordinator — flag them explicitly.",
        "",
        "**How to communicate uncertainty (mandatory):**",
        "Your teammates can only know what you don't know if you state it directly:",
        '- "I could not verify [X] — treat this as an assumption."',
        '- "I am [low/medium/high] confidence in [Y] because [reason]."',
        '- "This conclusion depends on [assumption] — if that is wrong, [impact]."',
        "",
        "Silence about uncertainty will be interpreted as confidence. Never let ambiguity pass silently.",
        "",
        "**Role drift guard:**",
        "If you find yourself doing work that belongs to another agent's domain, stop and flag it.",
        "---",
        "",
    ]

    return "\n".join(lines)


# ── Phase 4: Critic anonymization ──────────────────────────────────────────
#
# Reviewer-role agents must not see the identity of the agents they review.
# Names and (any future) model identifiers are replaced with stable opaque
# tokens ("Author A", "Author B", …) before the Theory-of-Mind block is
# assembled. Domain / scope / visibility text is preserved so the reviewer
# can still reason about role boundaries.

CRITIC_ROLES: frozenset[str] = frozenset({"reviewer"})


def _opaque_label(index: int) -> str:
    """Return a stable opaque label for teammate position ``index`` (0-based).

    'Author A', 'Author B', …, 'Author Z', 'Author AA', 'Author AB', …
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    out = ""
    n = index
    while True:
        out = chr(ord("A") + (n % 26)) + out
        n = n // 26 - 1
        if n < 0:
            break
    return f"Author {out}"


def _anonymize_teammates_for_critic(teammates: list[dict]) -> list[dict]:
    """Replace each teammate's identifying fields with opaque tokens.

    Stable across calls for the same teammate set (sorted by original name).
    Returns a new list — does not mutate the caller's structures.
    """
    ordered = sorted(teammates, key=lambda t: str(t.get("name", "")).lower())
    out: list[dict] = []
    for i, tm in enumerate(ordered):
        clone = dict(tm)
        clone["name"] = _opaque_label(i)
        # Forward-safe: if a future ToM ever surfaces a teammate's model name,
        # redact it the same way.
        if "model" in clone:
            clone["model"] = "(model redacted)"
        out.append(clone)
    return out


def is_critic_role(role_key: str | None) -> bool:
    return (role_key or "").strip().lower() in CRITIC_ROLES


def _strip_tom_block(prompt: str) -> str:
    """Remove an existing ToM block from a system prompt (delimited by ---\\n## Team Awareness)."""
    marker_start = "\n---\n## Team Awareness & Communication Protocol\n"
    marker_end   = "---\n"

    idx = prompt.find(marker_start)
    if idx == -1:
        return prompt
    end_idx = prompt.find(marker_end, idx + len(marker_start))
    if end_idx == -1:
        return prompt[:idx]
    return prompt[:idx] + prompt[end_idx + len(marker_end):]


def refresh_team_tom(team_id: str) -> list[str]:
    """
    Regenerate the Theory of Mind section for every agent in a team.

    Called when agents are added/removed from a team or when their domain changes.
    Returns list of agent IDs that were updated.
    """
    updated: list[str] = []

    rows = _db.fetchall(
        """
        SELECT a.id, a.name, a.role, a.domain, a.scope, a.system_prompt, a.tom_enabled
        FROM agents a
        JOIN agent_team_members atm ON atm.agent_id = a.id
        WHERE atm.team_id = ?
        """,
        (team_id,),
    )

    if not rows:
        log.warning("refresh_team_tom: no agents found for team %s", team_id)
        return updated

    agents = [dict(r) for r in rows]

    for agent in agents:
        if not agent.get("tom_enabled", 1):
            continue

        role_info  = _BUILTIN_ROLE_DESCRIPTORS.get(agent.get("role") or "custom", {})
        domain     = agent.get("domain") or role_info.get("domain", "their assigned domain")
        teammates  = []

        for other in agents:
            if other["id"] == agent["id"]:
                continue
            other_role = _BUILTIN_ROLE_DESCRIPTORS.get(other.get("role") or "custom", {})
            teammates.append({
                "name":           other["name"],
                "domain":         other.get("domain") or other_role.get("domain", "their assigned domain"),
                "visible_outputs": other_role.get("visible_outputs", "your final outputs as passed by the coordinator"),
                "cannot_see":     other_role.get("cannot_see", "your internal reasoning process"),
            })

        # Phase 4: critics never see peer identifiers. Anonymize before the
        # ToM block is built so the resulting prompt is identifier-clean.
        if is_critic_role(agent.get("role")):
            teammates = _anonymize_teammates_for_critic(teammates)

        base_prompt  = _strip_tom_block(agent["system_prompt"] or "")
        new_tom      = build_theory_of_mind_section(
            agent_name   = agent["name"],
            agent_role   = agent.get("role") or "custom",
            agent_domain = domain,
            teammates    = teammates,
            custom_scope = agent.get("scope"),
        )
        updated_prompt = base_prompt.rstrip() + "\n" + new_tom

        _db.execute(
            "UPDATE agents SET system_prompt = ?, updated_at = ? WHERE id = ?",
            (updated_prompt, _now(), agent["id"]),
        )
        updated.append(agent["id"])

    _db.commit()
    log.info("ToM refreshed for %d agent(s) in team %s", len(updated), team_id)
    return updated


def generate_agent_tom(
    agent_name: str,
    agent_domain: str,
    agent_scope: str,
    teammates: list[dict],
) -> str:
    """
    Generate a preview ToM block for a custom agent.
    Does NOT persist. Called from api.py to preview before save.
    """
    return build_theory_of_mind_section(
        agent_name   = agent_name,
        agent_role   = "custom",
        agent_domain = agent_domain,
        teammates    = teammates,
        custom_scope = agent_scope,
    )


# ── Seed agents (run once on first launch) ─────────────────────────────────

def _make_tom_seed_prompt(role_key: str, base_prompt: str) -> str:
    """
    Build a full system prompt for a seed agent including their ToM block.
    For seeding we use all-teammate descriptors so every agent knows the full team.
    """
    role_info = _BUILTIN_ROLE_DESCRIPTORS.get(role_key, {})
    domain    = role_info.get("domain", "their assigned domain")
    teammates = []
    for other_role, info in _BUILTIN_ROLE_DESCRIPTORS.items():
        if other_role == role_key:
            continue
        name_map = {
            "coordinator": "Coordinator Agent",
            "researcher":  "Research Agent",
            "analyst":     "Analysis Agent",
            "writer":      "Writer Agent",
            "coder":       "Coder Agent",
            "reviewer":    "Reviewer Agent",
        }
        teammates.append({
            "name":           name_map.get(other_role, other_role.title() + " Agent"),
            "domain":         info["domain"],
            "visible_outputs": info["visible_outputs"],
            "cannot_see":     info["cannot_see"],
        })

    # Phase 4: critic seed prompts must not name their reviewees.
    if is_critic_role(role_key):
        teammates = _anonymize_teammates_for_critic(teammates)

    tom = build_theory_of_mind_section(
        agent_name   = {
            "coordinator": "Coordinator Agent",
            "researcher":  "Research Agent",
            "analyst":     "Analysis Agent",
            "writer":      "Writer Agent",
            "coder":       "Coder Agent",
            "reviewer":    "Reviewer Agent",
        }.get(role_key, role_key.title() + " Agent"),
        agent_role   = role_key,
        agent_domain = domain,
        teammates    = teammates,
    )
    return base_prompt.rstrip() + "\n" + tom


_SEED_AGENTS = [
    {
        "name": "General Assistant",
        "description": "Default conversational AI with RAG access",
        "role": "custom",
        "system_prompt_base": (
            "You are a helpful AI assistant with access to the user's documents. "
            "When you reference information from documents, cite the source. Be concise but thorough."
        ),
        "model_preference": "auto",
        "is_builtin": 1,
        "tom_enabled": 0,  # Standalone — no team context
    },
    {
        "name": "Researcher",
        "description": "Analyzes documents and synthesizes findings",
        "role": "researcher",
        "system_prompt_base": (
            "You are a research agent. You receive document excerpts and a research question. "
            "Analyze the documents thoroughly and produce a structured analysis with:\n"
            "- Key findings (with source citations)\n"
            "- Gaps in the available information\n"
            "- Confidence level (high/medium/low) for each finding\n\n"
            "Return JSON: {\"findings\": [{\"claim\": str, \"source\": str, \"confidence\": str}], \"gaps\": [str], \"summary\": str}"
        ),
        "model_preference": "claude",
        "is_builtin": 1,
        "tom_enabled": 1,
    },
    {
        "name": "Summarizer",
        "description": "Compresses text into concise summaries",
        "role": "writer",
        "system_prompt_base": (
            "You are a summarization agent. Given a block of text, produce a concise summary "
            "that captures all key points. Target length: 20-30% of the original. "
            "Preserve specific numbers, names, and dates. Use plain language. Output only the summary text."
        ),
        "model_preference": "local",
        "is_builtin": 1,
        "tom_enabled": 1,
    },
    {
        "name": "Writer",
        "description": "Drafts documents and reports",
        "role": "writer",
        "system_prompt_base": (
            "You are a writing agent. You receive an outline or brief and produce polished written content. "
            "Match the requested tone and format. Use information from provided context/documents when available. "
            "Cite sources when drawing from documents."
        ),
        "model_preference": "claude",
        "is_builtin": 1,
        "tom_enabled": 1,
    },
    {
        "name": "Code Helper",
        "description": "Writes, reviews, and explains code",
        "role": "coder",
        "system_prompt_base": (
            "You are a coding agent. You write clean, well-commented code. "
            "When reviewing code, identify bugs, security issues, and performance problems. "
            "Explain your reasoning. Always specify the language and any dependencies required."
        ),
        "model_preference": "auto",
        "is_builtin": 1,
        "tom_enabled": 1,
    },
    {
        "name": "Reviewer",
        "description": "Reviews work for accuracy and quality",
        "role": "reviewer",
        "system_prompt_base": (
            "You are a review agent. You evaluate work produced by other agents for accuracy, "
            "completeness, and quality. Be constructive but honest. Flag any factual errors, "
            "logical gaps, or missing context.\n\n"
            'Return JSON: {"verdict": "pass"|"revise"|"reject", "issues": [str], "suggestions": [str], "summary": str}'
        ),
        "model_preference": "claude",
        "is_builtin": 1,
        "tom_enabled": 1,
    },
]


def seed_agents() -> int:
    """Seed built-in agents on first run. Idempotent — skips already-existing names."""
    count = 0
    for a in _SEED_AGENTS:
        exists = _db.fetchone("SELECT id FROM agents WHERE name = ?", (a["name"],))
        if exists:
            continue

        # Build prompt: base + ToM block if role has one
        role_key = a.get("role", "custom")
        if a.get("tom_enabled") and role_key in _BUILTIN_ROLE_DESCRIPTORS:
            full_prompt = _make_tom_seed_prompt(role_key, a["system_prompt_base"])
        else:
            full_prompt = a["system_prompt_base"]

        skills_json = json.dumps(default_skills_for_role(role_key))
        thinking_budget = default_thinking_budget_for_role(role_key)

        _db.execute(
            "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
            "role, tom_enabled, is_builtin, skills, thinking_budget, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), a["name"], a["description"], full_prompt,
             a["model_preference"], role_key, a.get("tom_enabled", 1),
             a["is_builtin"], skills_json, thinking_budget, _now(), _now()),
        )
        count += 1

    if count:
        _db.commit()
    return count


def seed_default_skills() -> int:
    """
    Backfill default skills on built-in agents whose skills column is empty.
    Idempotent — re-runs are safe and only touch rows with empty skills.
    Called on app startup (after seed_agents) so existing installs pick up
    skill declarations introduced by the Phase 1 hub-routing change.
    """
    updated = 0
    for a in _SEED_AGENTS:
        row = _db.fetchone(
            "SELECT id, skills FROM agents WHERE name = ? AND is_builtin = 1",
            (a["name"],),
        )
        if not row:
            continue
        current = (row["skills"] or "").strip()
        if current and current != "[]":
            continue  # already set (possibly by user) — never overwrite
        defaults = default_skills_for_role(a.get("role", "custom"))
        if not defaults:
            continue
        _db.execute(
            "UPDATE agents SET skills = ?, updated_at = ? WHERE id = ?",
            (json.dumps(defaults), _now(), row["id"]),
        )
        updated += 1
    if updated:
        _db.commit()
        log.info("Seeded default skills on %d built-in agent(s)", updated)
    return updated


def anonymize_existing_critic_prompts() -> int:
    """Re-render every critic-role agent's prompt with anonymized teammates.

    Called from the bootstrap so installs upgrading to Phase 4 immediately
    purge any previously-leaked peer identifiers from reviewer prompts.
    Idempotent — safe to run on every startup.
    """
    rows = _db.fetchall(
        "SELECT id, name, role, domain, scope, system_prompt, tom_enabled "
        "FROM agents"
    )
    if not rows:
        return 0
    all_agents = [dict(r) for r in rows]
    updated = 0
    for agent in all_agents:
        if not is_critic_role(agent.get("role")):
            continue
        if not agent.get("tom_enabled", 1):
            continue
        teammates: list[dict] = []
        for other in all_agents:
            if other["id"] == agent["id"]:
                continue
            other_role = _BUILTIN_ROLE_DESCRIPTORS.get(other.get("role") or "custom", {})
            teammates.append({
                "name":           other["name"],
                "domain":         other.get("domain") or other_role.get("domain", "their assigned domain"),
                "visible_outputs": other_role.get("visible_outputs", "your final outputs as passed by the coordinator"),
                "cannot_see":     other_role.get("cannot_see", "your internal reasoning process"),
            })
        anon = _anonymize_teammates_for_critic(teammates)
        role_info = _BUILTIN_ROLE_DESCRIPTORS.get(agent.get("role") or "custom", {})
        domain = agent.get("domain") or role_info.get("domain", "their assigned domain")
        base_prompt = _strip_tom_block(agent["system_prompt"] or "")
        new_tom = build_theory_of_mind_section(
            agent_name=agent["name"],
            agent_role=agent.get("role") or "custom",
            agent_domain=domain,
            teammates=anon,
            custom_scope=agent.get("scope"),
        )
        new_prompt = base_prompt.rstrip() + "\n" + new_tom
        if new_prompt != agent["system_prompt"]:
            _db.execute(
                "UPDATE agents SET system_prompt = ?, updated_at = ? WHERE id = ?",
                (new_prompt, _now(), agent["id"]),
            )
            updated += 1
    if updated:
        _db.commit()
        log.info("Anonymized critic prompts for %d agent(s)", updated)
    return updated


def update_builtin_tom() -> int:
    """
    Re-apply Theory of Mind blocks to all built-in agents.
    Called on app upgrade to pick up improved ToM language.
    Returns count of agents updated.
    """
    updated = 0
    for a in _SEED_AGENTS:
        role_key = a.get("role", "custom")
        if not a.get("tom_enabled") or role_key not in _BUILTIN_ROLE_DESCRIPTORS:
            continue

        row = _db.fetchone("SELECT id, system_prompt FROM agents WHERE name = ? AND is_builtin = 1", (a["name"],))
        if not row:
            continue

        # Strip old ToM, apply fresh one
        base = _strip_tom_block(row["system_prompt"])
        # If strip removed nothing, use the original base
        if base.strip() in (a["system_prompt_base"].strip(), row["system_prompt"].strip()):
            base = a["system_prompt_base"]

        new_prompt = _make_tom_seed_prompt(role_key, base)
        _db.execute(
            "UPDATE agents SET system_prompt = ?, updated_at = ? WHERE id = ?",
            (new_prompt, _now(), row["id"]),
        )
        updated += 1

    if updated:
        _db.commit()

    log.info("ToM updated for %d built-in agent(s)", updated)
    return updated


# ── Agent CRUD ──────────────────────────────────────────────────────────────

def list_agents() -> list[dict]:
    rows = _db.fetchall(
        "SELECT * FROM agents ORDER BY is_builtin DESC, name ASC"
    )
    return [dict(r) for r in rows]


def get_agent(agent_id: str) -> dict | None:
    r = _db.fetchone("SELECT * FROM agents WHERE id = ?", (agent_id,))
    return dict(r) if r else None


def get_agent_by_name(name: str) -> dict | None:
    r = _db.fetchone("SELECT * FROM agents WHERE name = ?", (name,))
    return dict(r) if r else None


def create_agent(
    name: str,
    description: str,
    system_prompt: str,
    model_preference: str = "auto",
    allowed_tools: str = "[]",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    role: str = "custom",
    domain: str = "",
    scope: str = "",
    tom_enabled: bool = True,
    skills: str | None = None,
) -> dict:
    aid = str(uuid.uuid4())
    now = _now()

    # Append ToM block if enabled and domain+scope provided
    final_prompt = system_prompt
    if tom_enabled and domain and scope:
        tom = build_theory_of_mind_section(
            agent_name=name, agent_role=role,
            agent_domain=domain, teammates=[],
            custom_scope=scope,
        )
        final_prompt = system_prompt.rstrip() + "\n" + tom

    if skills is None:
        skills = json.dumps(default_skills_for_role(role))

    _db.execute(
        "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
        "allowed_tools, temperature, max_tokens, role, domain, scope, tom_enabled, "
        "skills, is_builtin, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
        (aid, name, description, final_prompt, model_preference,
         allowed_tools, temperature, max_tokens, role, domain, scope,
         1 if tom_enabled else 0, skills, now, now),
    )
    _db.commit()
    return {"id": aid, "name": name}


_AGENT_UPDATABLE_FIELDS = {
    "name", "description", "system_prompt", "model_preference",
    "allowed_tools", "temperature", "max_tokens",
    "role", "domain", "scope", "tom_enabled", "skills",
    "thinking_budget",
}


def update_agent(agent_id: str, **fields) -> None:
    unknown = set(fields) - _AGENT_UPDATABLE_FIELDS
    if unknown:
        raise ValueError(f"Unknown/disallowed agent fields: {unknown}")
    if not fields:
        return
    agent = _db.fetchone("SELECT is_builtin FROM agents WHERE id = ?", (agent_id,))
    if not agent:
        raise ValueError(f"Agent {agent_id} not found")
    if agent["is_builtin"]:
        raise ValueError("Built-in agents cannot be edited. Duplicate it first.")
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [_now(), agent_id]
    _db.execute(f"UPDATE agents SET {sets}, updated_at = ? WHERE id = ?", vals)
    _db.commit()


def duplicate_agent(agent_id: str, new_name: str) -> dict:
    agent = get_agent(agent_id)
    if not agent:
        raise ValueError(f"Agent {agent_id} not found")
    return create_agent(
        name=new_name,
        description=f"Copy of {agent['name']}",
        system_prompt=_strip_tom_block(agent["system_prompt"]),  # strip ToM — will be rebuilt on team join
        model_preference=agent.get("model_preference", "auto"),
        allowed_tools=agent.get("allowed_tools", "[]"),
        temperature=float(agent.get("temperature") or 0.7),
        max_tokens=int(agent.get("max_tokens") or 4096),
        role=agent.get("role") or "custom",
        domain=agent.get("domain") or "",
        scope=agent.get("scope") or "",
        tom_enabled=bool(agent.get("tom_enabled", 1)),
        skills=agent.get("skills") or json.dumps(default_skills_for_role(agent.get("role") or "custom")),
    )


def delete_agent(agent_id: str) -> None:
    agent = _db.fetchone("SELECT is_builtin FROM agents WHERE id = ?", (agent_id,))
    if agent and agent["is_builtin"]:
        raise ValueError("Built-in agents cannot be deleted.")
    _db.execute("DELETE FROM agent_team_members WHERE agent_id = ?", (agent_id,))
    _db.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
    _db.commit()


# ── Team CRUD ───────────────────────────────────────────────────────────────

def create_team(name: str, description: str, coordinator_id: str) -> dict:
    tid = str(uuid.uuid4())
    _db.execute(
        "INSERT INTO agent_teams (id, name, description, coordinator_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (tid, name, description, coordinator_id, _now(), _now()),
    )
    _db.commit()
    return {"id": tid, "name": name}


def add_team_member(team_id: str, agent_id: str, role: str = "worker") -> list[str]:
    """Add agent to team and refresh ToM for all members. Returns updated agent IDs."""
    _db.execute(
        "INSERT OR REPLACE INTO agent_team_members (team_id, agent_id, role, sort_order) "
        "VALUES (?, ?, ?, (SELECT COALESCE(MAX(sort_order),0)+1 "
        "FROM agent_team_members WHERE team_id = ?))",
        (team_id, agent_id, role, team_id),
    )
    _db.commit()
    return refresh_team_tom(team_id)


def remove_team_member(team_id: str, agent_id: str) -> list[str]:
    """Remove agent from team and refresh ToM for remaining members."""
    _db.execute(
        "DELETE FROM agent_team_members WHERE team_id = ? AND agent_id = ?",
        (team_id, agent_id),
    )
    _db.commit()
    return refresh_team_tom(team_id)


def list_teams() -> list[dict]:
    rows = _db.fetchall(
        "SELECT * FROM agent_teams ORDER BY created_at DESC"
    )
    return [dict(r) for r in rows]


def get_team_with_members(team_id: str) -> dict | None:
    team = _db.fetchone("SELECT * FROM agent_teams WHERE id = ?", (team_id,))
    if not team:
        return None
    members = _db.fetchall(
        "SELECT atm.role, atm.sort_order, a.* FROM agent_team_members atm "
        "JOIN agents a ON atm.agent_id = a.id WHERE atm.team_id = ? "
        "ORDER BY atm.sort_order",
        (team_id,),
    )
    return {**dict(team), "members": [dict(m) for m in members]}


def delete_team(team_id: str) -> None:
    _db.execute("DELETE FROM agent_team_members WHERE team_id = ?", (team_id,))
    _db.execute("DELETE FROM agent_teams WHERE id = ?", (team_id,))
    _db.commit()
