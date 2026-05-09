"""routes/prompt_templates.py — user-saved prompt templates (PR 18).

Distinct from the legacy ``/api/prompts`` routes (which wrap the
versioned orchestrator system-prompt library). Templates here are
user-authored snippets and system prompts surfaced through the
slash-command picker in chat and the Settings "Set as default system
prompt" action. CRUD over the ``prompt_templates`` table created by the
``phase14.prompt_templates`` migration.

Mounted at ``/api/prompt-templates`` so the legacy ``/api/prompts``
router keeps working unchanged.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import db as _db

router = APIRouter()

KIND_VALUES = ("snippet", "system_prompt")
TITLE_MAX = 100
BODY_MAX = 10_000


# ── Pydantic IO models ────────────────────────────────────────────────────────


class PromptTemplateOut(BaseModel):
    id: str
    title: str
    body: str
    kind: str
    tags: str
    created_at: str
    updated_at: str
    use_count: int


class PromptTemplateCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=TITLE_MAX)
    body: str = Field(..., min_length=1, max_length=BODY_MAX)
    kind: Literal["snippet", "system_prompt"]
    tags: str | None = ""


class PromptTemplateUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=TITLE_MAX)
    body: str | None = Field(default=None, min_length=1, max_length=BODY_MAX)
    kind: Literal["snippet", "system_prompt"] | None = None
    tags: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _row_to_out(row: sqlite3.Row) -> PromptTemplateOut:
    return PromptTemplateOut(
        id=row["id"],
        title=row["title"],
        body=row["body"],
        kind=row["kind"],
        tags=row["tags"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        use_count=int(row["use_count"] or 0),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_or_404(template_id: str) -> sqlite3.Row:
    row = _db.fetchone(
        "SELECT id, title, body, kind, tags, created_at, updated_at, use_count "
        "FROM prompt_templates WHERE id = ?",
        (template_id,),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="prompt template not found")
    return row


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("", response_model=list[PromptTemplateOut])
async def list_templates() -> list[PromptTemplateOut]:
    rows = _db.fetchall(
        "SELECT id, title, body, kind, tags, created_at, updated_at, use_count "
        "FROM prompt_templates "
        "ORDER BY use_count DESC, updated_at DESC"
    )
    return [_row_to_out(r) for r in rows]


@router.get("/{template_id}", response_model=PromptTemplateOut)
async def get_template(template_id: str) -> PromptTemplateOut:
    return _row_to_out(_fetch_or_404(template_id))


@router.post("", response_model=PromptTemplateOut)
async def create_template(body: PromptTemplateCreate) -> PromptTemplateOut:
    template_id = uuid.uuid4().hex
    now = _now_iso()
    tags = (body.tags or "").strip()
    with _db.transaction() as conn:
        conn.execute(
            "INSERT INTO prompt_templates "
            "(id, title, body, kind, tags, created_at, updated_at, use_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (template_id, body.title, body.body, body.kind, tags, now, now),
        )
    return _row_to_out(_fetch_or_404(template_id))


@router.put("/{template_id}", response_model=PromptTemplateOut)
async def update_template(
    template_id: str, body: PromptTemplateUpdate
) -> PromptTemplateOut:
    existing = _fetch_or_404(template_id)
    new_title = body.title if body.title is not None else existing["title"]
    new_body = body.body if body.body is not None else existing["body"]
    new_kind = body.kind if body.kind is not None else existing["kind"]
    new_tags = (body.tags if body.tags is not None else (existing["tags"] or "")).strip()
    now = _now_iso()
    with _db.transaction() as conn:
        conn.execute(
            "UPDATE prompt_templates "
            "SET title = ?, body = ?, kind = ?, tags = ?, updated_at = ? "
            "WHERE id = ?",
            (new_title, new_body, new_kind, new_tags, now, template_id),
        )
    return _row_to_out(_fetch_or_404(template_id))


@router.delete("/{template_id}")
async def delete_template(template_id: str) -> dict:
    _fetch_or_404(template_id)
    with _db.transaction() as conn:
        conn.execute("DELETE FROM prompt_templates WHERE id = ?", (template_id,))
    return {"ok": True}


@router.post("/{template_id}/use", response_model=PromptTemplateOut)
async def use_template(template_id: str) -> PromptTemplateOut:
    _fetch_or_404(template_id)
    now = _now_iso()
    with _db.transaction() as conn:
        conn.execute(
            "UPDATE prompt_templates "
            "SET use_count = use_count + 1, updated_at = ? "
            "WHERE id = ?",
            (now, template_id),
        )
    return _row_to_out(_fetch_or_404(template_id))
