"""Lifecycle (agent shutdown gates) routes — wrap core/api/lifecycle.LifecycleAPI."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ._helpers import get_api

router = APIRouter()


class TokenIn(BaseModel):
    token: str


class ShutdownDemoIn(BaseModel):
    target_id: str = "agent-b"
    requester_id: str = "agent-a"
    reason: str = "demo"


@router.post("/confirm")
async def confirm(body: TokenIn, request: Request) -> dict:
    return get_api(request).confirm_shutdown(body.token)


@router.post("/deny")
async def deny(body: TokenIn, request: Request) -> dict:
    return get_api(request).deny_shutdown(body.token)


@router.get("/audit")
async def audit(request: Request, limit: int = 100) -> list:
    return get_api(request).list_lifecycle_audit(limit)


@router.post("/demo_shutdown")
async def demo(body: ShutdownDemoIn, request: Request) -> dict:
    return get_api(request).request_agent_shutdown_demo(
        body.target_id, body.requester_id, body.reason,
    )
