"""Memory routes — wrap core/api/memory.MemoryAPI."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ._helpers import get_api

router = APIRouter()


class SaveMemoryIn(BaseModel):
    content: str
    category: str = "fact"


class SemanticSearchIn(BaseModel):
    query: str
    top_k: int = 5


class DocumentSearchIn(BaseModel):
    query: str
    top_k: int = 10
    doc_type: str = ""


@router.post("/save")
async def save_memory(body: SaveMemoryIn, request: Request) -> dict:
    return get_api(request).save_memory(body.content, body.category)


@router.post("/search_memories")
async def search_memories(body: SemanticSearchIn, request: Request) -> list:
    return get_api(request).search_memories_semantic(body.query, body.top_k)


@router.post("/search_documents")
async def search_documents(body: DocumentSearchIn, request: Request) -> list:
    return get_api(request).search_documents_semantic(
        body.query, body.top_k, body.doc_type,
    )


@router.get("/semantic_available")
async def semantic_available(request: Request) -> dict:
    return {"available": bool(get_api(request).semantic_search_available())}


@router.get("/stale")
async def stale(request: Request, days: int = 30) -> list:
    return get_api(request).get_stale_memories(days)


@router.post("/delete/{entry_id}")
async def delete_entry(entry_id: str, request: Request) -> dict:
    return get_api(request).delete_memory_entry(entry_id)
