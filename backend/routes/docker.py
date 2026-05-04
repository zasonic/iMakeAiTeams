"""Docker / Power Mode routes.

Mounted at /api/docker. All endpoints require the bearer token. The handlers
reach the lifecycle manager and execution bridge via
``request.app.state.container.docker`` and ``.execution_bridge`` — both are
created in backend/server.py at startup.

Endpoints:
  GET  /docker/status     — Docker / WSL2 / OpenClaw detection snapshot
  POST /docker/start      — render compose, ``docker compose up -d``, poll health
  POST /docker/stop       — ``docker compose down``
  POST /docker/restart    — stop + start
  GET  /docker/health     — single OpenClaw API health probe
  POST /docker/classify   — route a message to chat or execution (LLM classifier)
  POST /docker/execute    — submit an execution task (streams via /api/events)
  POST /docker/cancel     — cancel an in-flight Power Mode task
  POST /docker/approve    — allow / deny a pending sensitive operation
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


def _container(request: Request):
    return request.app.state.container


def _docker(request: Request):
    mgr = getattr(_container(request), "docker", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Power Mode is unavailable in this build.")
    return mgr


def _bridge(request: Request):
    bridge = getattr(_container(request), "execution_bridge", None)
    if bridge is None:
        raise HTTPException(status_code=503, detail="Power Mode is unavailable in this build.")
    return bridge


def _classifier(request: Request):
    cls = getattr(_container(request), "execution_classifier", None)
    if cls is None:
        raise HTTPException(status_code=503, detail="Power Mode classifier unavailable.")
    return cls


# ── Models ───────────────────────────────────────────────────────────────────

class ExecuteIn(BaseModel):
    conversation_id: str
    user_message: str


class ClassifyIn(BaseModel):
    user_message: str
    conversation_id: str = ""


class CancelIn(BaseModel):
    task_id: str


class ApproveIn(BaseModel):
    task_id: str
    approval_id: str
    allow: bool


# ── Status / lifecycle ──────────────────────────────────────────────────────

@router.get("/status")
async def status(request: Request) -> dict:
    mgr = _docker(request)
    snap = await asyncio.to_thread(mgr.status)
    return snap.to_dict()


@router.post("/start")
async def start(request: Request) -> dict:
    # `start_openclaw` is synchronous and may block for tens of seconds while
    # `docker compose up -d` runs and the gateway becomes healthy. Run it on
    # a worker thread so the asyncio event loop (and other in-flight requests
    # like /docker/status polled by the renderer) keep moving.
    mgr = _docker(request)
    r = await asyncio.to_thread(mgr.start_openclaw)
    out = {"ok": r.ok, "error": r.error,
           "gateway_url": r.gateway_url, "detail": r.detail}
    if not out["ok"]:
        raise HTTPException(status_code=409, detail=out.get("error") or "start failed")
    return out


@router.post("/stop")
async def stop(request: Request) -> dict:
    mgr = _docker(request)
    r = await asyncio.to_thread(mgr.stop_openclaw)
    return {"ok": r.ok, "error": r.error, "detail": r.detail}


@router.post("/restart")
async def restart(request: Request) -> dict:
    mgr = _docker(request)
    r = await asyncio.to_thread(mgr.restart_openclaw)
    if not r.ok:
        raise HTTPException(status_code=409, detail=r.error or "restart failed")
    return {"ok": r.ok, "error": r.error, "gateway_url": r.gateway_url,
            "detail": r.detail}


@router.get("/health")
async def health(request: Request) -> dict:
    mgr = _docker(request)
    return {
        "ok": await asyncio.to_thread(mgr.health_check),
        "gateway_url": mgr.gateway_url(),
    }


# ── Classification + execution ──────────────────────────────────────────────

@router.post("/classify")
async def classify(body: ClassifyIn, request: Request) -> dict:
    classifier = _classifier(request)
    # The classifier may call out to an LLM for ambiguous messages — keep the
    # event loop responsive by running it on a worker thread.
    return await asyncio.to_thread(classifier.classify, body.user_message)


@router.post("/execute")
async def execute(body: ExecuteIn, request: Request) -> dict:
    bridge = _bridge(request)
    api = _container(request).api
    history: list[dict] = []
    try:
        if body.conversation_id:
            rows = await asyncio.to_thread(
                api.chat_get_messages, body.conversation_id, 20,
            )
            for r in rows or []:
                role = r.get("role") if isinstance(r, dict) else None
                content = r.get("content") if isinstance(r, dict) else None
                if role in ("user", "assistant") and content:
                    history.append({"role": role, "content": content})
    except Exception:
        # Missing history is non-fatal — OpenClaw can run without it.
        history = []
    return await asyncio.to_thread(
        bridge.submit, body.conversation_id, body.user_message, history,
    )


@router.post("/cancel")
async def cancel(body: CancelIn, request: Request) -> dict:
    # cancel makes a best-effort POST to the OpenClaw gateway, so run it on a
    # worker thread to keep the event loop responsive.
    return await asyncio.to_thread(_bridge(request).cancel, body.task_id)


@router.post("/approve")
async def approve(body: ApproveIn, request: Request) -> dict:
    return _bridge(request).approve(body.task_id, body.approval_id, body.allow)
