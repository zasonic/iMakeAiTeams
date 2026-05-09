"""tests/test_voice_routes.py — HTTP-level tests for the PR 17 voice routes.

Mounts only the voice router on a minimal FastAPI app and feeds it a stub
container so the routes can resolve a Settings object + lazily attach a
VoiceService. The VoiceService methods that touch subprocesses are
monkeypatched per-test so we exercise the route shape without spawning
whisper-cli or piper.
"""

from __future__ import annotations

import struct
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.settings import Settings
from routes import voice as voice_routes
from server import BearerAuthMiddleware
from services import voice as voice_module
from services.voice import VoiceService, VoiceServiceError


TOKEN = "test-token-voice"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _make_wav_bytes() -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<4h", 0, 1, -1, 0))
    return buf.getvalue()


@pytest.fixture
def voice_root(tmp_path, monkeypatch):
    root = tmp_path / "voice"
    stt = root / "stt"
    tts = root / "tts"
    root.mkdir()
    stt.mkdir()
    tts.mkdir()
    from core import paths as _paths
    monkeypatch.setattr(_paths, "voice_dir", lambda: root)
    monkeypatch.setattr(_paths, "stt_models_dir", lambda: stt)
    monkeypatch.setattr(_paths, "tts_voices_dir", lambda: tts)
    monkeypatch.setattr(
        _paths, "voice_assets_catalog_path",
        lambda: tmp_path / "missing_catalog.json",
    )
    monkeypatch.setattr(
        voice_module.paths, "voice_assets_catalog_path",
        lambda: tmp_path / "missing_catalog.json",
    )
    monkeypatch.setattr(voice_module.paths, "voice_dir", lambda: root)
    monkeypatch.setattr(voice_module.paths, "stt_models_dir", lambda: stt)
    monkeypatch.setattr(voice_module.paths, "tts_voices_dir", lambda: tts)
    return root


@pytest.fixture
def app(in_memory_db, voice_root, tmp_path):
    a = FastAPI()
    a.add_middleware(BearerAuthMiddleware, expected_token=TOKEN)
    a.include_router(voice_routes.router, prefix="/api/voice")

    settings = Settings(tmp_path / "settings.json")
    fake_api = MagicMock()
    fake_api._settings = settings
    fake_container = MagicMock()
    fake_container.api = fake_api
    fake_container.settings = settings
    fake_container.voice = None  # forces lazy construction in _get_voice
    a.state.container = fake_container
    return a


# ── Auth ─────────────────────────────────────────────────────────────────────


class TestAuth:
    def test_unauthenticated_transcribe_returns_401(self, app):
        client = TestClient(app)
        resp = client.post(
            "/api/voice/transcribe",
            files={"file": ("a.wav", BytesIO(b"\x00"), "audio/wav")},
        )
        assert resp.status_code == 401

    def test_unauthenticated_synthesize_returns_401(self, app):
        client = TestClient(app)
        resp = client.post("/api/voice/synthesize", json={"text": "hi"})
        assert resp.status_code == 401

    def test_unauthenticated_status_returns_401(self, app):
        client = TestClient(app)
        resp = client.get("/api/voice/assets/status")
        assert resp.status_code == 401

    def test_unauthenticated_download_returns_401(self, app):
        client = TestClient(app)
        resp = client.post("/api/voice/assets/download", json={})
        assert resp.status_code == 401


# ── /assets/status ───────────────────────────────────────────────────────────


class TestAssetsStatus:
    def test_reports_not_ready_when_files_missing(self, app):
        client = TestClient(app)
        resp = client.get("/api/voice/assets/status", headers=_auth())
        assert resp.status_code == 200
        data = resp.json()
        assert data["stt_ready"] is False
        assert data["tts_ready"] is False
        assert data["stt_model_id"]
        assert data["tts_voice_id"]


# ── /transcribe ──────────────────────────────────────────────────────────────


class TestTranscribe:
    def test_returns_text_for_valid_audio(self, app, monkeypatch):
        monkeypatch.setattr(
            VoiceService, "transcribe",
            lambda self, audio_path, model_id: "hello world",
        )
        client = TestClient(app)
        resp = client.post(
            "/api/voice/transcribe",
            files={"file": ("clip.wav", BytesIO(_make_wav_bytes()), "audio/wav")},
            headers=_auth(),
        )
        assert resp.status_code == 200
        assert resp.json() == {"text": "hello world"}

    def test_empty_upload_returns_400(self, app):
        client = TestClient(app)
        resp = client.post(
            "/api/voice/transcribe",
            files={"file": ("clip.wav", BytesIO(b""), "audio/wav")},
            headers=_auth(),
        )
        assert resp.status_code == 400

    def test_voice_service_error_returns_503(self, app, monkeypatch):
        def _raise(self, audio_path, model_id):
            raise VoiceServiceError("model not downloaded")
        monkeypatch.setattr(VoiceService, "transcribe", _raise)
        client = TestClient(app)
        resp = client.post(
            "/api/voice/transcribe",
            files={"file": ("clip.wav", BytesIO(_make_wav_bytes()), "audio/wav")},
            headers=_auth(),
        )
        assert resp.status_code == 503
        assert "not downloaded" in resp.json()["detail"]


# ── /synthesize ──────────────────────────────────────────────────────────────


class TestSynthesize:
    def test_returns_wav_bytes(self, app, monkeypatch):
        wav = _make_wav_bytes()
        monkeypatch.setattr(
            VoiceService, "synthesize",
            lambda self, text, voice_id: wav,
        )
        client = TestClient(app)
        resp = client.post(
            "/api/voice/synthesize",
            json={"text": "hello"},
            headers=_auth(),
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content[:4] == b"RIFF"
        assert resp.content[8:12] == b"WAVE"

    def test_empty_text_returns_400(self, app):
        client = TestClient(app)
        resp = client.post(
            "/api/voice/synthesize",
            json={"text": "   "},
            headers=_auth(),
        )
        assert resp.status_code == 400

    def test_oversized_text_returns_400(self, app):
        client = TestClient(app)
        resp = client.post(
            "/api/voice/synthesize",
            json={"text": "x" * (voice_routes.MAX_SYNTHESIZE_CHARS + 1)},
            headers=_auth(),
        )
        assert resp.status_code == 400


# ── /assets/download ─────────────────────────────────────────────────────────


class TestAssetsDownload:
    def test_invalid_stt_id_returns_400(self, app):
        client = TestClient(app)
        resp = client.post(
            "/api/voice/assets/download",
            json={"stt_model_id": "does-not-exist"},
            headers=_auth(),
        )
        assert resp.status_code == 400

    def test_starts_background_download_and_acks(self, app, monkeypatch):
        # Reset the global lock so tests in any order don't interfere.
        voice_routes._download_running = False

        spawned: list[tuple] = []
        original_thread = voice_routes.threading.Thread

        class _NoopThread:
            def __init__(self, *args, **kwargs):
                spawned.append((args, kwargs))

            def start(self):
                pass

        monkeypatch.setattr(voice_routes.threading, "Thread", _NoopThread)
        client = TestClient(app)

        resp = client.post(
            "/api/voice/assets/download", json={}, headers=_auth(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["stt_model_id"]
        assert body["tts_voice_id"]
        assert spawned, "background thread was never spawned"

    def test_concurrent_download_rejects(self, app, monkeypatch):
        voice_routes._download_running = True
        client = TestClient(app)
        resp = client.post(
            "/api/voice/assets/download", json={}, headers=_auth(),
        )
        # Reset for downstream tests
        voice_routes._download_running = False
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert "in progress" in body["error"]
