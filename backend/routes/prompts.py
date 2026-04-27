"""Prompt-library routes — wrap the prompt_* methods on the API facade."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ._helpers import get_api

router = APIRouter()


class PromptSaveIn(BaseModel):
    prompt_id: str
    text: str
    notes: str = ""


class PromptCreateIn(BaseModel):
    name: str
    category: str
    description: str
    text: str
    model_target: str = "auto"


class PromptDuplicateIn(BaseModel):
    source_id: str
    new_name: str


class PromptRestoreIn(BaseModel):
    version_id: str


class PromptImportIn(BaseModel):
    data: dict


@router.get("")
async def list_prompts(request: Request) -> list:
    return get_api(request).prompt_list()


@router.get("/{prompt_id}/versions")
async def versions(prompt_id: str, request: Request) -> list:
    return get_api(request).prompt_versions(prompt_id)


@router.post("/save")
async def save(body: PromptSaveIn, request: Request) -> dict:
    return get_api(request).prompt_save(body.prompt_id, body.text, body.notes)


@router.post("/create")
async def create(body: PromptCreateIn, request: Request) -> dict:
    return get_api(request).prompt_create(
        body.name, body.category, body.description, body.text, body.model_target,
    )


@router.post("/duplicate")
async def duplicate(body: PromptDuplicateIn, request: Request) -> dict:
    return get_api(request).prompt_duplicate(body.source_id, body.new_name)


@router.post("/restore")
async def restore(body: PromptRestoreIn, request: Request) -> dict:
    return get_api(request).prompt_restore_version(body.version_id)


@router.post("/delete/{prompt_id}")
async def delete(prompt_id: str, request: Request) -> dict:
    return get_api(request).prompt_delete(prompt_id)


@router.get("/{prompt_id}/export")
async def export(prompt_id: str, request: Request) -> dict:
    return get_api(request).prompt_export(prompt_id)


@router.post("/import")
async def import_prompt(body: PromptImportIn, request: Request) -> dict:
    return get_api(request).prompt_import(body.data)
