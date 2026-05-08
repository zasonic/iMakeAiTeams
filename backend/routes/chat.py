"""Chat routes — wrap core/api/chat.ChatAPI.

Streaming behavior: chat_send fires a thread that emits chat_token /
chat_event / chat_done events through sse_events. The renderer's EventSource
on /api/events drains those — there's no per-request SSE here, just a JSON
ack that the work was kicked off.
"""

from __future__ import annotations

import html as _html
import json as _json
from datetime import datetime, timezone
from typing import Optional

import db as _db
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ._helpers import get_api

router = APIRouter()


class ChatSendIn(BaseModel):
    conversation_id: str
    user_message: str = Field(..., max_length=200_000)
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
    user_message: str = Field(..., max_length=200_000)
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


# ── Conversation export (PR 7) ───────────────────────────────────────────────
#
# Three formats served straight off the DB rows so the rendering stays simple
# and the route is testable without spinning up the chat orchestrator. The PDF
# path only goes as far as HTML — the renderer hands that HTML to Electron's
# main process, which uses webContents.printToPDF to produce the bytes (no new
# dependency, no headless Chromium spawn from Python).

_PDF_HTML_STYLE = (
    "body{font-family:Georgia,'Times New Roman',serif;line-height:1.5;"
    "color:#111;margin:2.5em;font-size:12pt;}"
    "h1{font-size:20pt;margin-bottom:0.2em;border-bottom:1px solid #888;"
    "padding-bottom:0.2em;}"
    "h2{font-size:13pt;margin-top:1.4em;margin-bottom:0.4em;}"
    ".meta{color:#666;font-size:10pt;font-style:italic;margin-bottom:1.5em;}"
    ".turn{margin-bottom:1.4em;page-break-inside:avoid;}"
    ".role{font-weight:bold;font-size:11pt;text-transform:uppercase;"
    "letter-spacing:0.05em;color:#444;}"
    ".content{margin-top:0.4em;white-space:pre-wrap;}"
    "code,pre{font-family:'SFMono-Regular',Menlo,Consolas,monospace;"
    "font-size:10.5pt;background:#f4f4f4;}"
    "pre{padding:0.6em 0.8em;border-radius:4px;overflow-x:auto;"
    "white-space:pre-wrap;}"
    "code{padding:0.1em 0.3em;border-radius:3px;}"
    "hr{border:none;border-top:1px solid #ddd;margin:1.4em 0;}"
)


def _load_export_data(conversation_id: str) -> tuple[dict, list[dict]]:
    """Fetch (conv, messages) or raise 404. Same query shape the orchestrator
    uses, just hoisted into the route layer so each format renders directly
    from these rows."""
    conv = _db.fetchone(
        "SELECT id, title, created_at FROM conversations WHERE id = ?",
        (conversation_id,),
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    rows = _db.fetchall(
        "SELECT role, content, model_used, cost_usd, created_at "
        "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    )
    return dict(conv), [dict(r) for r in rows]


def _role_label(role: str) -> str:
    if role == "user":
        return "You"
    if role == "assistant":
        return "Assistant"
    return role.capitalize() or "Message"


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@router.get("/conversations/{conversation_id}/export.md")
async def export_markdown(conversation_id: str, request: Request) -> Response:
    conv, messages = _load_export_data(conversation_id)
    title = conv["title"] or "Conversation"
    parts = [f"# {title}", "", f"*Exported {_now_utc_str()}*", ""]
    for msg in messages:
        parts.append(f"## {_role_label(msg['role'])}")
        parts.append("")
        parts.append(msg["content"] or "")
        parts.append("")
        parts.append("---")
        parts.append("")
    return Response(
        content="\n".join(parts),
        media_type="text/markdown; charset=utf-8",
    )


@router.get("/conversations/{conversation_id}/export.json")
async def export_json(conversation_id: str, request: Request) -> Response:
    _conv, messages = _load_export_data(conversation_id)
    # Raw record shape, no transformation — the test contract is "JSON parses
    # as a list" and the dict keys match what /api/chat/messages/<id> returns.
    payload = _json.dumps(messages, ensure_ascii=False, indent=2)
    return Response(content=payload, media_type="application/json")


@router.get("/conversations/{conversation_id}/export.pdf-html")
async def export_pdf_html(conversation_id: str, request: Request) -> Response:
    conv, messages = _load_export_data(conversation_id)
    title = conv["title"] or "Conversation"
    body_parts = [
        f"<h1>{_html.escape(title)}</h1>",
        f"<p class=\"meta\">Exported {_html.escape(_now_utc_str())}</p>",
    ]
    for msg in messages:
        role = _html.escape(_role_label(msg["role"]))
        content = _html.escape(msg["content"] or "")
        body_parts.append(
            "<div class=\"turn\">"
            f"<div class=\"role\">{role}</div>"
            f"<div class=\"content\">{content}</div>"
            "</div>"
            "<hr/>"
        )
    doc = (
        "<!DOCTYPE html>"
        "<html><head><meta charset=\"utf-8\">"
        f"<title>{_html.escape(title)}</title>"
        f"<style>{_PDF_HTML_STYLE}</style>"
        "</head><body>"
        + "".join(body_parts)
        + "</body></html>"
    )
    return Response(content=doc, media_type="text/html; charset=utf-8")
