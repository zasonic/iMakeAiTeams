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

import threading

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
    return _docker(request).status().to_dict()


@router.post("/start")
async def start(request: Request) -> dict:
    # Run the (potentially long) start in a worker thread so we don't tie up
    # the asyncio loop. The handler still waits for the result so the caller
    # gets a clear ok/error verdict; live progress is published via SSE.
    mgr = _docker(request)
    result_holder: dict = {}

    def _work():
        try:
            r = mgr.start_openclaw()
            result_holder["result"] = {"ok": r.ok, "error": r.error,
                                       "gateway_url": r.gateway_url,
                                       "detail": r.detail}
        except Exception as exc:
            result_holder["result"] = {"ok": False, "error": str(exc)}

    t = threading.Thread(target=_work, daemon=True, name="docker-start")
    t.start()
    t.join()
    out = result_holder.get("result", {"ok": False, "error": "no result"})
    if not out.get("ok"):
        raise HTTPException(status_code=409, detail=out.get("error", "start failed"))
    return out


@router.post("/stop")
async def stop(request: Request) -> dict:
    r = _docker(request).stop_openclaw()
    return {"ok": r.ok, "error": r.error, "detail": r.detail}


@router.post("/restart")
async def restart(request: Request) -> dict:
    r = _docker(request).restart_openclaw()
    if not r.ok:
        raise HTTPException(status_code=409, detail=r.error or "restart failed")
    return {"ok": r.ok, "error": r.error, "gateway_url": r.gateway_url,
            "detail": r.detail}


@router.get("/health")
async def health(request: Request) -> dict:
    mgr = _docker(request)
    return {
        "ok": mgr.health_check(),
        "gateway_url": mgr.gateway_url(),
    }


# ── Classification + execution ──────────────────────────────────────────────

@router.post("/classify")
async def classify(body: ClassifyIn, request: Request) -> dict:
    classifier = _classifier(request)
    return classifier.classify(body.user_message)


@router.post("/execute")
async def execute(body: ExecuteIn, request: Request) -> dict:
    bridge = _bridge(request)
    api = _container(request).api
    history: list[dict] = []
    try:
        if body.conversation_id:
            rows = api.chat_get_messages(body.conversation_id, limit=20)
            history = [
                {"role": r.get("role"), "content": r.get("content")}
                for r in rows if r.get("role") in ("user", "assistant")
            ]
    except Exception:
        # Missing history is non-fatal — OpenClaw can run without it.
        history = []
    return bridge.submit(body.conversation_id, body.user_message, history)


@router.post("/cancel")
async def cancel(body: CancelIn, request: Request) -> dict:
    return _bridge(request).cancel(body.task_id)


@router.post("/approve")
async def approve(body: ApproveIn, request: Request) -> dict:
    return _bridge(request).approve(body.task_id, body.approval_id, body.allow)
