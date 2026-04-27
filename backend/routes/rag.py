"""RAG routes — wrap core/api/rag.RagAPI."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ._helpers import get_api

router = APIRouter()


class FolderIn(BaseModel):
    folder_path: str


class FileIn(BaseModel):
    file_path: str


class TextIn(BaseModel):
    text: str
    source: str = "manual"


class SearchIn(BaseModel):
    query: str
    top_k: int = 5


class HybridSearchIn(BaseModel):
    query: str
    top_k: int = 5
    method: str = "hybrid"
    doc_type: str = ""


@router.post("/index_folder")
async def index_folder(body: FolderIn, request: Request) -> dict:
    return get_api(request).build_rag_index(body.folder_path)


@router.post("/add_file")
async def add_file(body: FileIn, request: Request) -> dict:
    return get_api(request).rag_add_file(body.file_path)


@router.post("/add_text")
async def add_text(body: TextIn, request: Request) -> dict:
    return get_api(request).rag_add_text(body.text, body.source)


@router.post("/clear")
async def clear(request: Request) -> dict:
    return get_api(request).rag_clear()


@router.get("/status")
async def status(request: Request) -> dict:
    return get_api(request).rag_status()


@router.post("/search")
async def search(body: SearchIn, request: Request) -> list:
    return get_api(request).rag_search(body.query, body.top_k)


@router.post("/search_hybrid")
async def search_hybrid(body: HybridSearchIn, request: Request) -> list:
    return get_api(request).rag_search_hybrid(
        body.query, body.top_k, body.method, body.doc_type,
    )


@router.get("/bm25_corpus_size")
async def bm25_corpus_size(request: Request) -> dict:
    return get_api(request).bm25_corpus_size()
