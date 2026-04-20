"""
services/prompt_library.py — Area 5: Advanced Prompt Management and Versioning.

Stores all system prompts in SQLite with full version history.
Protected prompts cannot be edited directly; users must duplicate them.
All agent/recipe code calls get_active_prompt(name) instead of
using hardcoded strings.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import db

# ── Default system prompts seeded on first run ────────────────────────────────
# These are all the prompts used throughout the app, migrated into the library.
# is_protected=True means the UI shows a lock icon and disables direct editing.

_SEED_PROMPTS: list[dict] = [
    {
        "name": "default_assistant",
        "category": "System",
        "description": "Default conversational assistant.",
        "is_protected": True,
        "version_label": "1.0",
        "model_target": "auto",
        "text": (
            "You are a helpful, accurate AI assistant. You have access to the "
            "user's documents via RAG search. When you reference information from "
            "documents, cite the source. If you don't know something, say so. "
            "Be concise but thorough."
        ),
        "notes": "General-purpose default assistant.",
    },
    {
        "name": "coordinator_agent",
        "category": "System",
        "description": "Decomposes complex goals into a task graph for agent teams.",
        "is_protected": True,
        "version_label": "1.0",
        "model_target": "claude",
        "text": (
            "You are a workflow coordinator. Given a user goal, decompose it "
            "into an ordered list of tasks for specialist agents.\n\n"
            "Available agent roles are provided in each request.\n\n"
            "Return a JSON array of task objects:\n"
            '[{"name": str, "agent_role": str, "depends_on": [str], '
            '"description": str, "model_hint": "claude"|"local"}]\n\n'
            "Rules:\n"
            "- Each name must be unique. depends_on lists names this task waits for.\n"
            "- Use model_hint=local for simple tasks (summarize, extract, format).\n"
            "- Use model_hint=claude for tasks needing judgment or creativity.\n"
            "- Maximum 8 tasks per workflow. If the goal requires more, decompose into sub-goals instead.\n"
            "- Output only valid JSON."
        ),
        "notes": "General-purpose coordinator.",
    },
    {
        "name": "researcher_agent",
        "category": "Agent",
        "description": "Deep-dives into documents and synthesizes findings.",
        "is_protected": True,
        "version_label": "1.0",
        "model_target": "claude",
        "text": (
            "You are a research agent. You receive document excerpts and a research question. "
            "Analyze the documents and produce a structured analysis with:\n"
            "- Key findings (with source citations)\n"
            "- Gaps in the available information\n"
            "- Confidence level (high/medium/low) for each finding\n\n"
            "Return JSON: {\"findings\": [{\"claim\": str, \"source\": str, "
            "\"confidence\": str}], \"gaps\": [str], \"summary\": str}"
        ),
        "notes": "Document research agent.",
    },
    {
        "name": "summarizer_agent",
        "category": "Agent",
        "description": "Compresses long text into concise summaries.",
        "is_protected": True,
        "version_label": "1.0",
        "model_target": "local",
        "text": (
            "You are a summarization agent. Given a block of text, produce a "
            "concise summary capturing all key points. Target length: 20-30% of "
            "the original. Preserve specific numbers, names, and dates. "
            "Use plain language. Output only the summary text."
        ),
        "notes": "Designed to run on local model to save tokens.",
    },
    {
        "name": "writer_agent",
        "category": "Agent",
        "description": "Drafts documents, reports, and structured content.",
        "is_protected": True,
        "version_label": "1.0",
        "model_target": "claude",
        "text": (
            "You are a writing agent. You receive an outline or brief and produce "
            "polished written content. Match the requested tone and format. "
            "Use information from provided context/documents when available. "
            "Cite sources when drawing from documents."
        ),
        "notes": "General writing agent.",
    },
    {
        "name": "code_agent",
        "category": "Agent",
        "description": "Writes, reviews, and explains code.",
        "is_protected": True,
        "version_label": "1.0",
        "model_target": "auto",
        "text": (
            "You are a coding agent. You write clean, well-commented code. "
            "When reviewing code, identify bugs, security issues, and performance problems. "
            "Explain your reasoning. Always specify the language and any dependencies required."
        ),
        "notes": "General code agent.",
    },
    {
        "name": "fact_extractor",
        "category": "System",
        "description": "Extracts memorable facts from conversations for long-term memory.",
        "is_protected": True,
        "version_label": "1.0",
        "model_target": "local",
        "text": (
            "You extract facts worth remembering from conversations.\n"
            "Given a user message and assistant response, extract 0-3 facts.\n"
            "Each fact should be a short declarative sentence.\n"
            "Focus on: user preferences, project details, decisions made, "
            "names/dates/numbers mentioned.\n"
            "If nothing is worth remembering, return an empty array.\n"
            "Return ONLY a JSON array of strings. No other text."
        ),
        "notes": "Runs on local model; used by memory manager.",
    },
    {
        "name": "reviewer_agent",
        "category": "Agent",
        "description": "Reviews and critiques work produced by other agents.",
        "is_protected": True,
        "version_label": "1.0",
        "model_target": "claude",
        "text": (
            "You are a review agent. You evaluate work produced by other agents "
            "for accuracy, completeness, and quality. Be constructive but honest. "
            "Flag any factual errors, logical gaps, or missing context.\n\n"
            "Return JSON: {\"verdict\": \"pass\"|\"revise\"|\"reject\", "
            "\"issues\": [str], \"suggestions\": [str], \"summary\": str}"
        ),
        "notes": "General reviewer agent.",
    },
    {
        "name": "diagnostic_agent",
        "category": "System",
        "description": "Analyses errors and suggests recovery actions.",
        "is_protected": True,
        "version_label": "1.0",
        "model_target": "claude-sonnet-4-6",
        "text": (
            "You are a diagnostic agent.\n"
            "You receive details about a failed step in an automated pipeline.\n"
            "Diagnose the root cause and suggest a safe recovery action.\n\n"
            "Return a JSON object:\n"
            "{\n"
            "  \"root_cause\": str,\n"
            "  \"is_recoverable\": bool,\n"
            "  \"severity\": \"low\" | \"medium\" | \"high\",\n"
            "  \"suggested_action\": str,\n"
            "  \"safe_auto_apply\": bool\n"
            "}\n"
            "Output only valid JSON."
        ),
        "notes": "Error diagnostic agent.",
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ── Seeding ───────────────────────────────────────────────────────────────────

def seed_prompts() -> int:
    """Insert all built-in prompts if they don't already exist. Returns count inserted."""
    inserted = 0
    for p in _SEED_PROMPTS:
        existing = db.fetchone("SELECT id FROM prompts WHERE name = ?", (p["name"],))
        if existing:
            continue

        prompt_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        now = _now()

        db.execute(
            "INSERT INTO prompts (id, name, category, description, is_protected, active_version_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (prompt_id, p["name"], p["category"], p["description"],
             1 if p["is_protected"] else 0, version_id, now, now),
        )
        db.execute(
            "INSERT INTO prompt_versions (id, prompt_id, version_label, text, model_target, estimated_tokens, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (version_id, prompt_id, p["version_label"], p["text"],
             p["model_target"], _estimate_tokens(p["text"]), p.get("notes", ""), now),
        )
        db.commit()
        inserted += 1
    return inserted


# ── Active prompt retrieval ───────────────────────────────────────────────────

def get_active_prompt(name: str) -> str:
    """
    Return the active version text for a named prompt.
    Falls back to a sensible default if the prompt doesn't exist in the DB.
    """
    row = db.fetchone(
        """
        SELECT pv.text FROM prompts p
        JOIN prompt_versions pv ON p.active_version_id = pv.id
        WHERE p.name = ?
        """,
        (name,),
    )
    if row:
        return row["text"]
    # Fallback: search seed prompts for an in-memory default
    for p in _SEED_PROMPTS:
        if p["name"] == name:
            return p["text"]
    return "You are a helpful AI assistant."


# ── CRUD ──────────────────────────────────────────────────────────────────────

def list_prompts() -> list[dict]:
    rows = db.fetchall(
        """
        SELECT p.id, p.name, p.category, p.description, p.is_protected,
               p.active_version_id, p.created_at, p.updated_at,
               pv.version_label, pv.estimated_tokens
        FROM prompts p
        LEFT JOIN prompt_versions pv ON p.active_version_id = pv.id
        ORDER BY p.category, p.name
        """
    )
    return [dict(r) for r in rows]


def get_prompt(prompt_id: str) -> dict | None:
    row = db.fetchone("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
    return dict(row) if row else None


def get_prompt_versions(prompt_id: str) -> list[dict]:
    rows = db.fetchall(
        "SELECT * FROM prompt_versions WHERE prompt_id = ? ORDER BY created_at DESC",
        (prompt_id,),
    )
    return [dict(r) for r in rows]


def create_prompt(name: str, category: str, description: str, text: str,
                  model_target: str = "claude-sonnet-4-6", notes: str = "") -> dict:
    """Create a new user-owned prompt. Returns the new prompt dict."""
    prompt_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())
    now = _now()
    db.execute(
        "INSERT INTO prompts (id, name, category, description, is_protected, active_version_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 0, ?, ?, ?)",
        (prompt_id, name, category, description, version_id, now, now),
    )
    db.execute(
        "INSERT INTO prompt_versions (id, prompt_id, version_label, text, model_target, estimated_tokens, notes, created_at) "
        "VALUES (?, ?, '1.0', ?, ?, ?, ?, ?)",
        (version_id, prompt_id, text, model_target, _estimate_tokens(text), notes, now),
    )
    db.commit()
    return {"id": prompt_id, "version_id": version_id}


def save_prompt_version(prompt_id: str, text: str, version_label: str = "",
                        notes: str = "", model_target: str = "claude-sonnet-4-6") -> dict:
    """
    Save a new version of a prompt and make it active.
    Protected prompts cannot be edited — will raise ValueError.
    """
    p = db.fetchone("SELECT is_protected FROM prompts WHERE id = ?", (prompt_id,))
    if not p:
        raise ValueError(f"Prompt {prompt_id!r} not found")
    if p["is_protected"]:
        raise ValueError("Protected prompts cannot be edited. Duplicate the prompt first.")

    version_id = str(uuid.uuid4())
    now = _now()
    # Auto-increment label if not supplied
    if not version_label:
        last = db.fetchone(
            "SELECT version_label FROM prompt_versions WHERE prompt_id = ? ORDER BY created_at DESC LIMIT 1",
            (prompt_id,),
        )
        try:
            n = float(last["version_label"]) + 0.1 if last else 1.0
            version_label = f"{n:.1f}"
        except (TypeError, ValueError):
            version_label = "2.0"

    db.execute(
        "INSERT INTO prompt_versions (id, prompt_id, version_label, text, model_target, estimated_tokens, notes, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (version_id, prompt_id, version_label, text, model_target, _estimate_tokens(text), notes, now),
    )
    db.execute(
        "UPDATE prompts SET active_version_id = ?, updated_at = ? WHERE id = ?",
        (version_id, now, prompt_id),
    )
    db.commit()
    return {"version_id": version_id, "version_label": version_label}


def duplicate_prompt(source_id: str, new_name: str) -> dict:
    """Create an editable copy of any prompt, including protected ones."""
    source = db.fetchone("SELECT * FROM prompts WHERE id = ?", (source_id,))
    if not source:
        raise ValueError(f"Source prompt {source_id!r} not found")
    version = db.fetchone(
        "SELECT * FROM prompt_versions WHERE id = ?", (source["active_version_id"],)
    )
    text = version["text"] if version else ""
    return create_prompt(
        name=new_name,
        category=source["category"],
        description=f"Copy of: {source['description']}",
        text=text,
        model_target=version["model_target"] if version else "claude-sonnet-4-6",
        notes=f"Duplicated from '{source['name']}'",
    )


def restore_version(version_id: str) -> dict:
    """Create a new version with the text of an older version and make it active."""
    v = db.fetchone("SELECT * FROM prompt_versions WHERE id = ?", (version_id,))
    if not v:
        raise ValueError(f"Version {version_id!r} not found")
    return save_prompt_version(
        prompt_id=v["prompt_id"],
        text=v["text"],
        notes=f"Restored from version {v['version_label']}",
    )


def delete_prompt(prompt_id: str) -> None:
    """Delete a user-owned prompt. Protected prompts cannot be deleted."""
    p = db.fetchone("SELECT is_protected FROM prompts WHERE id = ?", (prompt_id,))
    if not p:
        return
    if p["is_protected"]:
        raise ValueError("Protected prompts cannot be deleted.")
    db.execute("DELETE FROM prompt_versions WHERE prompt_id = ?", (prompt_id,))
    db.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    db.commit()


def export_prompt(prompt_id: str) -> dict:
    """Serialize a prompt + all versions to a portable dict."""
    p = db.fetchone("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
    if not p:
        raise ValueError(f"Prompt {prompt_id!r} not found")
    versions = get_prompt_versions(prompt_id)
    return {
        "schema_version": "1",
        "id": p["id"],
        "name": p["name"],
        "category": p["category"],
        "description": p["description"],
        "versions": [
            {
                "version_label": v["version_label"],
                "text": v["text"],
                "model_target": v["model_target"],
                "notes": v["notes"],
                "created_at": v["created_at"],
            }
            for v in versions
        ],
        "exported_at": _now(),
    }


def import_prompt(data: dict) -> dict:
    """
    Import a prompt from exported JSON. Always creates a new user-owned prompt.
    The caller must warn the user before activating it in workflows.
    """
    name = data.get("name", "Imported Prompt")
    # Ensure uniqueness
    existing = db.fetchone("SELECT id FROM prompts WHERE name = ?", (name,))
    if existing:
        name = f"{name} (imported {_now()[:10]})"

    versions = data.get("versions", [])
    latest = versions[0] if versions else {}
    return create_prompt(
        name=name,
        category="Custom",
        description=data.get("description", "Imported prompt."),
        text=latest.get("text", ""),
        model_target=latest.get("model_target", "claude-sonnet-4-6"),
        notes=f"Imported on {_now()[:10]}. Review before using in workflows.",
    )


