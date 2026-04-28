"""Chat routes — wrap core/api/chat.ChatAPI.

Streaming behavior: chat_send fires a thread that emits chat_token /
chat_event / chat_done events through events_sse. The renderer's EventSource
on /api/events drains those — there's no per-request SSE here, just a JSON
ack that the work was kicked off.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ._helpers import get_api

router = APIRouter()


class ChatSendIn(BaseModel):
    conversation_id: str
    user_message: str
    agent_id: str = ""


class ChatNewIn(BaseModel):
    agent_id: str = ""
    title: str = "New conversation"


class ChatRenameIn(BaseModel):
    conversation_id: str
    title: str


class ChatBranchIn(BaseModel):
    conversation_id: str
    from_message_id: str


class ChatExportIn(BaseModel):
    conversation_id: str
    fmt: str = "markdown"


class ChatThinkingIn(BaseModel):
    user_message: str
    budget_tokens: int = 10000


class ChatStopIn(BaseModel):
    conversation_id: str = ""


@router.post("/send")
async def send(body: ChatSendIn, request: Request) -> dict:
    # `chat_send` is decorated with @rate_limit_chat; when refused it returns
    # `{"error": ...}` instead of spawning the worker thread. Forward that to
    # the renderer so the UI can show the message instead of hanging on an
    # SSE stream that will never start.
    result = get_api(request).chat_send(body.conversation_id, body.user_message, body.agent_id)
    if isinstance(result, dict) and result.get("error"):
        return {"ok": False, "conversation_id": body.conversation_id, **result}
    return {"ok": True, "conversation_id": body.conversation_id}


@router.post("/stop")
async def stop(request: Request, body: ChatStopIn | None = None) -> dict:
    conversation_id = body.conversation_id if body else ""
    get_api(request).chat_stop(conversation_id)
    return {"ok": True}


@router.post("/new_conversation")
async def new_conversation(body: ChatNewIn, request: Request) -> dict:
    return get_api(request).chat_new_conversation(body.agent_id, body.title)


@router.get("/conversations")
async def list_conversations(request: Request, limit: int = 30) -> list:
    return get_api(request).chat_list_conversations(limit=limit)


@router.get("/messages/{conversation_id}")
async def get_messages(
    conversation_id: str, request: Request, limit: int = 100,
) -> list:
    return get_api(request).chat_get_messages(conversation_id, limit=limit)


@router.post("/rename_conversation")
async def rename(body: ChatRenameIn, request: Request) -> dict:
    return get_api(request).chat_rename_conversation(body.conversation_id, body.title)


@router.post("/delete_conversation/{conversation_id}")
async def delete(conversation_id: str, request: Request) -> dict:
    return get_api(request).chat_delete_conversation(conversation_id)


@router.post("/branch_conversation")
async def branch(body: ChatBranchIn, request: Request) -> dict:
    return get_api(request).chat_branch_conversation(
        body.conversation_id, body.from_message_id,
    )


@router.post("/export_conversation")
async def export(body: ChatExportIn, request: Request) -> dict:
    return get_api(request).chat_export_conversation(body.conversation_id, body.fmt)


@router.get("/token_stats")
async def token_stats(request: Request) -> dict:
    return get_api(request).chat_token_stats()


@router.get("/router_stats")
async def router_stats(request: Request) -> dict:
    return get_api(request).get_router_stats()


@router.post("/ask_with_thinking")
async def thinking(body: ChatThinkingIn, request: Request) -> dict:
    get_api(request).ask_with_thinking(body.user_message, body.budget_tokens)
    return {"ok": True}
