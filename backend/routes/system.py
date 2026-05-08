"""System routes — diagnostics, health checks, hardware probe, security, changelog,
error logs.

These are the methods that didn't fit into a domain sub-API and live directly
on the API facade.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Request
from pydantic import BaseModel

import sse_events
from services.bundled_server import BundledServerError, DEFAULT_MODEL_ID

from ._helpers import get_api

log = logging.getLogger("iMakeAiTeams.routes.system")

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


class ActiveLocalModelIn(BaseModel):
    model_id: str


class BundledDownloadIn(BaseModel):
    # Optional — defaults to the catalog's recommended Quick Start model.
    model_id: str = ""


class BundledStartIn(BaseModel):
    model_id: str = ""


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


@router.post("/canary/reset/{model_id:path}")
async def canary_reset(model_id: str, request: Request) -> dict:
    return get_api(request).canary_reset(model_id)


@router.get("/local_models")
async def local_models(request: Request) -> dict:
    api = get_api(request)
    client = api.local_client
    models = client.list_local_models() if client is not None else []
    current = api._settings.get("default_local_model", "") or ""
    return {"models": models, "current": current}


@router.post("/local_model/active")
async def set_active_local_model(body: ActiveLocalModelIn, request: Request) -> dict:
    api = get_api(request)
    api._settings.set("default_local_model", body.model_id)
    return {"current": api._settings.get("default_local_model", "") or "", "ok": True}


# ── Phase 9: Bundled llama.cpp server endpoints ──────────────────────────────


_bundled_download_lock = threading.Lock()
_bundled_download_running = False


def _run_bundled_download(api, model_id: str) -> None:
    """Background-thread worker for POST /bundled/download.

    Streams progress to the renderer via SSE events:
      - ``bundled_download_progress`` {bytes_done, bytes_total, model_id}
      - ``bundled_download_complete`` {model_id, file_path}
      - ``bundled_download_error``    {model_id, error}

    On success, also persists ``local_backend_mode = "bundled"`` and
    ``bundled_model_id`` so the next startup auto-rebinds the server.
    """
    global _bundled_download_running
    bs = api.bundled_server
    if bs is None:
        sse_events.publish("bundled_download_error", {
            "model_id": model_id,
            "error": "bundled server is not initialised",
        })
        with _bundled_download_lock:
            _bundled_download_running = False
        return

    def _on_progress(done: int, total: int) -> None:
        sse_events.publish("bundled_download_progress", {
            "model_id":   model_id,
            "bytes_done": done,
            "bytes_total": total,
        })

    try:
        result = bs.download_model(model_id, on_progress=_on_progress)
        api._settings.set("local_backend_mode", "bundled")
        api._settings.set("bundled_model_id", model_id)
        sse_events.publish("bundled_download_complete", {
            "model_id":  model_id,
            "file_path": result.get("file_path", ""),
            "size_bytes": result.get("expected_size_bytes", 0),
        })
    except BundledServerError as exc:
        sse_events.publish("bundled_download_error", {
            "model_id": model_id,
            "error":    str(exc),
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("bundled download crashed: %s", exc, exc_info=True)
        sse_events.publish("bundled_download_error", {
            "model_id": model_id,
            "error":    f"unexpected error: {exc}",
        })
    finally:
        with _bundled_download_lock:
            _bundled_download_running = False


@router.post("/bundled/download")
async def bundled_download(body: BundledDownloadIn, request: Request) -> dict:
    """Kick off the bundled-model download in a background thread.

    Returns immediately with ``{ok, model_id}`` — actual progress flows over
    SSE. Concurrent calls are rejected so a refresh-happy user can't queue
    five downloads of the same 2.5 GB file.
    """
    global _bundled_download_running
    api = get_api(request)
    model_id = body.model_id or DEFAULT_MODEL_ID

    with _bundled_download_lock:
        if _bundled_download_running:
            return {"ok": False, "error": "download already in progress",
                    "model_id": model_id}
        _bundled_download_running = True

    threading.Thread(
        target=_run_bundled_download,
        args=(api, model_id),
        daemon=True,
        name=f"bundled-download-{model_id}",
    ).start()
    return {"ok": True, "model_id": model_id}


@router.post("/bundled/start")
async def bundled_start(body: BundledStartIn, request: Request) -> dict:
    """Start the bundled llama-server. Returns the bound port on success."""
    api = get_api(request)
    bs = api.bundled_server
    if bs is None:
        return {"ok": False, "error": "bundled server is not initialised"}

    model_id = body.model_id or api._settings.get("bundled_model_id", "") or DEFAULT_MODEL_ID
    try:
        import db as _db_module
        row = _db_module.get_db().execute(
            "SELECT file_path FROM bundled_models WHERE model_id = ?",
            (model_id,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": f"model {model_id} not downloaded"}
        file_path = row["file_path"] if isinstance(row, dict) else row[0]
        port = bs.start(file_path, model_id=model_id)
    except BundledServerError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        log.warning("bundled start crashed: %s", exc, exc_info=True)
        return {"ok": False, "error": f"unexpected error: {exc}"}

    api._settings.set("local_backend_mode", "bundled")
    api._settings.set("bundled_model_id", model_id)
    return {"ok": True, "port": port, "model_id": model_id}


@router.post("/bundled/stop")
async def bundled_stop(request: Request) -> dict:
    api = get_api(request)
    bs = api.bundled_server
    if bs is None:
        return {"ok": False, "error": "bundled server is not initialised"}
    bs.stop()
    return {"ok": True}


@router.get("/bundled/status")
async def bundled_status(request: Request) -> dict:
    api = get_api(request)
    bs = api.bundled_server
    if bs is None:
        return {"running": False, "port": None, "model_id": None,
                "available": False}
    return {
        "running":   bs.is_running(),
        "port":      bs.port(),
        "model_id":  bs.model_id() or (api._settings.get("bundled_model_id", "") or None),
        "available": True,
    }
