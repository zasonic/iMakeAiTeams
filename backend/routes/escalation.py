"""Escalation routes — wrap core/api/escalation.EscalationAPI (Phase 5)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ._helpers import get_api

router = APIRouter()


@router.get("/pending")
async def pending(request: Request) -> list:
    return get_api(request).list_pending_escalations()


@router.post("/{escalation_id}/approve")
async def approve(escalation_id: str, request: Request) -> dict:
    return get_api(request).approve_escalation(escalation_id)


@router.post("/{escalation_id}/deny")
async def deny(escalation_id: str, request: Request) -> dict:
    return get_api(request).deny_escalation(escalation_id)
