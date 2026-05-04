"""System routes — diagnostics, health checks, hardware probe, security, changelog,
error logs.

These are the methods that didn't fit into a domain sub-API and live directly
on the API facade.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ._helpers import get_api

router = APIRouter()


class FirewallIn(BaseModel):
    enabled: bool


class TestConnIn(BaseModel):
    backend: str  # "ollama" | "lmstudio"


class FetchModelsIn(BaseModel):
    backend: str


class HealthCheckIn(BaseModel):
    skip_api: bool = False


class OpenUrlIn(BaseModel):
    url: str


@router.get("/service_status")
async def service_status(request: Request) -> dict:
    return get_api(request).service_status()


@router.post("/probe_hardware")
async def probe_hardware(request: Request) -> dict:
    get_api(request).probe_hardware()
    return {"ok": True}


@router.post("/test_connection")
async def test_connection(body: TestConnIn, request: Request) -> dict:
    get_api(request).test_connection(body.backend)
    return {"ok": True}


@router.post("/fetch_chat_models")
async def fetch_chat_models(body: FetchModelsIn, request: Request) -> dict:
    get_api(request).fetch_chat_models(body.backend)
    return {"ok": True}


@router.post("/run_health_check")
async def run_health_check(body: HealthCheckIn, request: Request) -> dict:
    get_api(request).run_health_check(skip_api=body.skip_api)
    return {"ok": True}


@router.get("/error_logs")
async def error_logs(request: Request, limit: int = 50) -> list:
    return get_api(request).get_error_logs(limit)


@router.post("/error_logs/{record_id}/resolve")
async def resolve_error(record_id: str, request: Request) -> dict:
    return get_api(request).mark_error_resolved(record_id)


@router.post("/export_diagnostics")
async def export_diagnostics(request: Request) -> dict:
    get_api(request).export_diagnostics()
    return {"ok": True}


@router.get("/changelog")
async def changelog(request: Request) -> dict:
    return get_api(request).get_changelog()


@router.post("/changelog/seen")
async def changelog_seen(request: Request) -> dict:
    return get_api(request).mark_changelog_seen()


@router.get("/security/status")
async def security_status(request: Request) -> dict:
    return get_api(request).security_get_status()


@router.post("/security/firewall")
async def security_firewall(body: FirewallIn, request: Request) -> dict:
    return get_api(request).security_toggle_firewall(body.enabled)


@router.get("/security/scan_log")
async def security_scan_log(
    request: Request, limit: int = 50, verdict_filter: str = "",
) -> list:
    return get_api(request).security_get_scan_log(limit, verdict_filter)


@router.post("/open_url")
async def open_url(body: OpenUrlIn, request: Request) -> dict:
    get_api(request).open_url(body.url)
    return {"ok": True}
