"""Voice routes (PR 17) — speech-to-text + text-to-speech via bundled binaries.

Four endpoints share the /api/voice/* prefix:

  POST /api/voice/transcribe     multipart audio upload → {text}
  POST /api/voice/synthesize     JSON {text}            → audio/wav bytes
  GET  /api/voice/assets/status                         → {stt_ready, tts_ready}
  POST /api/voice/assets/download                       → SSE-driven download

The transcribe / synthesize endpoints fail-closed when the requested model is
not yet downloaded, with a message that points the user at Settings → Voice
or the VoiceSetupModal. The /assets/download endpoint runs in a background
thread and emits SSE events so the wizard can render a progress bar — same
pattern as POST /api/system/bundled/download.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

import sse_events
from core import paths
from services.voice import (
    DEFAULT_STT_MODEL_ID,
    DEFAULT_TTS_VOICE_ID,
    VoiceService,
    VoiceServiceError,
)

from ._helpers import get_api

log = logging.getLogger("iMakeAiTeams.voice.routes")

router = APIRouter()


# Hard cap on inbound audio. ~30 seconds of mono PCM at 16 kHz is about
# 1 MB; we allow 25 MB so the user can drop a longer clip without a
# perplexing 413 if Whisper happens to handle it.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Hard cap on inbound text. Piper happily synthesizes paragraphs, but a
# multi-thousand-character message takes long enough that we want a clear
# error rather than a 60s timeout. The renderer should send one paragraph
# at a time.
MAX_SYNTHESIZE_CHARS = 5000


class SynthesizeIn(BaseModel):
    text: str
    voice_id: str = ""


class AssetsDownloadIn(BaseModel):
    stt_model_id: str = ""
    tts_voice_id: str = ""


# ── Voice service helper ─────────────────────────────────────────────────────


def _get_voice(request: Request) -> VoiceService:
    """Lazily instantiate (and memoize) a VoiceService on the app container.

    The container only hard-binds the BundledServer right now; voice is
    additive and dormant unless a route is actually called, so we attach it
    on first use rather than burdening sidecar boot with another service
    construction.
    """
    container = request.app.state.container
    voice = getattr(container, "voice", None)
    if voice is None:
        voice = VoiceService(container.settings)
        container.voice = voice
    return voice


# ── Background-thread download worker ────────────────────────────────────────


_download_lock = threading.Lock()
_download_running = False


def _emit_progress(asset_kind: str, asset_id: str) -> "callable":
    def _on(done: int, total: int) -> None:
        sse_events.publish("voice_assets_progress", {
            "kind":       asset_kind,
            "asset_id":   asset_id,
            "bytes_done": done,
            "bytes_total": total,
        })
    return _on


def _run_download(voice: VoiceService, stt_id: str, tts_id: str) -> None:
    global _download_running
    try:
        if stt_id:
            try:
                voice.ensure_stt_model(stt_id, on_progress=_emit_progress("stt", stt_id))
                sse_events.publish("voice_assets_complete", {
                    "kind": "stt", "asset_id": stt_id,
                })
            except VoiceServiceError as exc:
                sse_events.publish("voice_assets_error", {
                    "kind": "stt", "asset_id": stt_id, "error": str(exc),
                })
                return
        if tts_id:
            try:
                voice.ensure_tts_voice(tts_id, on_progress=_emit_progress("tts", tts_id))
                sse_events.publish("voice_assets_complete", {
                    "kind": "tts", "asset_id": tts_id,
                })
            except VoiceServiceError as exc:
                sse_events.publish("voice_assets_error", {
                    "kind": "tts", "asset_id": tts_id, "error": str(exc),
                })
                return
        sse_events.publish("voice_assets_done", {
            "stt_ready": voice.stt_ready(stt_id) if stt_id else True,
            "tts_ready": voice.tts_ready(tts_id) if tts_id else True,
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("voice download crashed: %s", exc, exc_info=True)
        sse_events.publish("voice_assets_error", {
            "kind": "unknown", "asset_id": "",
            "error": f"unexpected error: {exc}",
        })
    finally:
        with _download_lock:
            _download_running = False


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    """Receive a wav upload, run whisper-cli, return the transcript text."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Audio too large ({len(raw) // (1024 * 1024)} MB). "
                f"Maximum {MAX_AUDIO_BYTES // (1024 * 1024)} MB."
            ),
        )

    api = get_api(request)
    model_id = api._settings.get("stt_model_id") or DEFAULT_STT_MODEL_ID
    voice = _get_voice(request)

    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=str(paths.voice_dir()), delete=False,
    ) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        try:
            text = voice.transcribe(tmp_path, model_id)
        except VoiceServiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return {"text": text}


@router.post("/synthesize")
async def synthesize(body: SynthesizeIn, request: Request) -> Response:
    """Synthesize ``text`` into a wav blob via piper. Returns audio/wav."""
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    if len(text) > MAX_SYNTHESIZE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Text too long ({len(text)} chars). "
                f"Maximum {MAX_SYNTHESIZE_CHARS}."
            ),
        )

    api = get_api(request)
    voice_id = (body.voice_id or "").strip() \
        or api._settings.get("tts_voice_id") \
        or DEFAULT_TTS_VOICE_ID
    voice = _get_voice(request)

    try:
        wav = voice.synthesize(text, voice_id)
    except VoiceServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=wav, media_type="audio/wav")


@router.get("/assets/status")
async def assets_status(request: Request) -> dict:
    """Return whether the user's configured STT model + TTS voice are
    downloaded and ready to use."""
    api = get_api(request)
    voice = _get_voice(request)
    stt_id = api._settings.get("stt_model_id") or DEFAULT_STT_MODEL_ID
    tts_id = api._settings.get("tts_voice_id") or DEFAULT_TTS_VOICE_ID
    return {
        "stt_ready":   voice.stt_ready(stt_id),
        "tts_ready":   voice.tts_ready(tts_id),
        "stt_model_id": stt_id,
        "tts_voice_id": tts_id,
    }


@router.post("/assets/download")
async def assets_download(body: AssetsDownloadIn, request: Request) -> dict:
    """Kick off a background download for the configured (or supplied) STT +
    TTS assets. Progress flows over SSE; the response is just an ack so the
    renderer knows whether the download started or was already running."""
    global _download_running
    api = get_api(request)
    voice = _get_voice(request)

    stt_id = (body.stt_model_id or "").strip() \
        or api._settings.get("stt_model_id") \
        or DEFAULT_STT_MODEL_ID
    tts_id = (body.tts_voice_id or "").strip() \
        or api._settings.get("tts_voice_id") \
        or DEFAULT_TTS_VOICE_ID

    # Validate up front so an unknown id surfaces as an HTTP 400 instead of a
    # background SSE error the user might miss.
    try:
        voice.resolve_stt(stt_id)
        voice.resolve_tts(tts_id)
    except VoiceServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _download_lock:
        if _download_running:
            return {"ok": False, "error": "download already in progress",
                    "stt_model_id": stt_id, "tts_voice_id": tts_id}
        _download_running = True

    threading.Thread(
        target=_run_download,
        args=(voice, stt_id, tts_id),
        daemon=True,
        name=f"voice-download-{stt_id}-{tts_id}",
    ).start()
    return {"ok": True, "stt_model_id": stt_id, "tts_voice_id": tts_id}
