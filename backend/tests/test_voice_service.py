"""tests/test_voice_service.py — exercise VoiceService download lifecycle
and a tiny smoke check around transcribe / synthesize.

We don't fetch from Hugging Face. ``ensure_stt_model`` / ``ensure_tts_voice``
accept a ``_requests`` kwarg so tests can pass a fake namespace whose
``get`` returns canned bytes. The catalog's expected sha256 is monkeypatched
to match (or mismatch) so we can drive both the success and the corruption
paths without ever opening a socket.

``transcribe`` / ``synthesize`` spawn external binaries; the unit tests here
drive the pre-spawn validation paths (binary missing, model missing) and
the route-shape contract. End-to-end transcription is gated behind a
``@pytest.mark.skipif`` that reads a local install path — out of scope for
CI but useful for manual verification.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.settings import Settings
from services import voice as voice_module
from services.voice import (
    DEFAULT_STT_MODEL_ID,
    DEFAULT_TTS_VOICE_ID,
    VoiceService,
    VoiceServiceError,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


class _FakeStreamResponse:
    """Mimics requests.Response for ``with`` + iter_content."""

    def __init__(self, content: bytes, *, status_code: int = 200,
                 content_length: int | None = None) -> None:
        self._content = content
        self.status_code = status_code
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def iter_content(self, chunk_size: int = 1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i: i + chunk_size]


class _FakeRequests:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        # Map a substring of the URL → payload so a single fake can serve
        # both halves of the Piper download.
        self._payloads = payloads
        self.calls: list[str] = []

    def get(self, url, *, stream=False, timeout=None):
        self.calls.append(url)
        for needle, payload in self._payloads.items():
            if needle in url:
                return _FakeStreamResponse(payload,
                                            content_length=len(payload))
        return _FakeStreamResponse(b"", status_code=404)


def _make_wav_bytes(samples: list[int]) -> bytes:
    """Build a tiny mono-16-bit-16k wav so the synthesize header check passes."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return buf.getvalue()


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    """Redirect voice paths into tmp_path so tests don't pollute userData."""
    voice_root = tmp_path / "voice"
    stt = voice_root / "stt"
    tts = voice_root / "tts"
    voice_root.mkdir()
    stt.mkdir()
    tts.mkdir()
    monkeypatch.setattr(voice_module.paths, "voice_dir", lambda: voice_root)
    monkeypatch.setattr(voice_module.paths, "stt_models_dir", lambda: stt)
    monkeypatch.setattr(voice_module.paths, "tts_voices_dir", lambda: tts)
    monkeypatch.setattr(
        voice_module.paths, "voice_assets_catalog_path",
        lambda: tmp_path / "missing_catalog.json",
    )
    return tmp_path


@pytest.fixture
def voice_service(patched_dirs, tmp_path):
    s = Settings(tmp_path / "settings.json")
    return VoiceService(s)


# ── Catalog ──────────────────────────────────────────────────────────────────


class TestCatalog:
    def test_unknown_stt_raises(self, voice_service):
        with pytest.raises(VoiceServiceError):
            voice_service.resolve_stt("nonexistent")

    def test_unknown_tts_raises(self, voice_service):
        with pytest.raises(VoiceServiceError):
            voice_service.resolve_tts("nonexistent")

    def test_default_stt_resolves(self, voice_service):
        entry = voice_service.resolve_stt(DEFAULT_STT_MODEL_ID)
        assert entry["filename"]
        assert entry["repo"]

    def test_default_tts_resolves(self, voice_service):
        entry = voice_service.resolve_tts(DEFAULT_TTS_VOICE_ID)
        assert entry["model_filename"]
        assert entry["config_filename"]

    def test_stt_ready_false_when_missing(self, voice_service):
        assert voice_service.stt_ready(DEFAULT_STT_MODEL_ID) is False

    def test_tts_ready_false_when_missing(self, voice_service):
        assert voice_service.tts_ready(DEFAULT_TTS_VOICE_ID) is False


# ── Download ─────────────────────────────────────────────────────────────────


class TestEnsureSttModel:
    def test_writes_partial_then_renames(
        self, voice_service, patched_dirs, monkeypatch, in_memory_db,
    ):
        payload = b"GGML" + b"\x01" * 4096
        sha = hashlib.sha256(payload).hexdigest()
        monkeypatch.setattr(
            voice_service, "resolve_stt",
            lambda mid: {
                "repo":                "ggerganov/whisper.cpp",
                "filename":            "ggml-base.en.bin",
                "expected_sha256":     sha,
                "expected_size_bytes": len(payload),
            },
        )
        fake = _FakeRequests({"ggml-base.en.bin": payload})
        progress: list[tuple[int, int]] = []

        result_path = voice_service.ensure_stt_model(
            DEFAULT_STT_MODEL_ID,
            on_progress=lambda d, t: progress.append((d, t)),
            _requests=fake,
        )

        target = patched_dirs / "voice" / "stt" / "ggml-base.en.bin"
        partial = target.with_suffix(target.suffix + ".partial")
        assert result_path == target
        assert target.exists()
        assert not partial.exists()
        assert progress, "progress callback never fired"

        # DB row was inserted with the right sha256
        rows = in_memory_db.get_db().execute(
            "SELECT asset_id, asset_type, sha256 FROM voice_assets"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == DEFAULT_STT_MODEL_ID
        assert rows[0][1] == "stt"
        assert rows[0][2] == sha

    def test_sha_mismatch_drops_partial(
        self, voice_service, patched_dirs, monkeypatch, in_memory_db,
    ):
        payload = b"GGML" + b"\x02" * 32
        wrong_sha = "0" * 64
        monkeypatch.setattr(
            voice_service, "resolve_stt",
            lambda mid: {
                "repo":                "ggerganov/whisper.cpp",
                "filename":            "ggml-base.en.bin",
                "expected_sha256":     wrong_sha,
                "expected_size_bytes": len(payload),
            },
        )
        fake = _FakeRequests({"ggml-base.en.bin": payload})

        with pytest.raises(VoiceServiceError, match="sha256 mismatch"):
            voice_service.ensure_stt_model(
                DEFAULT_STT_MODEL_ID, _requests=fake,
            )

        target = patched_dirs / "voice" / "stt" / "ggml-base.en.bin"
        partial = target.with_suffix(target.suffix + ".partial")
        assert not partial.exists()
        assert not target.exists()

        rows = in_memory_db.get_db().execute(
            "SELECT asset_id FROM voice_assets"
        ).fetchall()
        assert rows == []

    def test_existing_valid_file_short_circuits(
        self, voice_service, patched_dirs, monkeypatch, in_memory_db,
    ):
        payload = b"already-here"
        sha = hashlib.sha256(payload).hexdigest()
        target = patched_dirs / "voice" / "stt" / "ggml-base.en.bin"
        target.write_bytes(payload)
        monkeypatch.setattr(
            voice_service, "resolve_stt",
            lambda mid: {
                "repo":                "ggerganov/whisper.cpp",
                "filename":            "ggml-base.en.bin",
                "expected_sha256":     sha,
                "expected_size_bytes": len(payload),
            },
        )
        fake = _FakeRequests({"ggml-base.en.bin": b"should-not-be-fetched"})

        result = voice_service.ensure_stt_model(
            DEFAULT_STT_MODEL_ID, _requests=fake,
        )
        assert result == target
        assert fake.calls == [], "no HTTP request should have fired"


class TestEnsureTtsVoice:
    def test_downloads_both_files(
        self, voice_service, patched_dirs, monkeypatch, in_memory_db,
    ):
        model_payload = b"ONNX" + b"\x03" * 4096
        cfg_payload = b'{"audio": {"sample_rate": 22050}}'
        model_sha = hashlib.sha256(model_payload).hexdigest()
        cfg_sha = hashlib.sha256(cfg_payload).hexdigest()
        monkeypatch.setattr(
            voice_service, "resolve_tts",
            lambda vid: {
                "repo":                "rhasspy/piper-voices",
                "hf_subpath":          "en/en_US/amy/medium",
                "model_filename":      "en_US-amy-medium.onnx",
                "config_filename":     "en_US-amy-medium.onnx.json",
                "model_sha256":        model_sha,
                "model_size_bytes":    len(model_payload),
                "config_sha256":       cfg_sha,
                "config_size_bytes":   len(cfg_payload),
            },
        )
        fake = _FakeRequests({
            "en_US-amy-medium.onnx.json": cfg_payload,
            "en_US-amy-medium.onnx":      model_payload,  # matches both due to substring
        })
        # The substring match would make .onnx fetch return cfg_payload; map
        # by full filename instead so order doesn't matter:
        fake._payloads = {
            "en_US-amy-medium.onnx.json": cfg_payload,
            "%2Fen_US-amy-medium.onnx?": model_payload,
        }
        # The above doesn't match either URL — replace with a smarter fake
        # that distinguishes by file extension.
        fake._payloads = {".onnx.json": cfg_payload, ".onnx": model_payload}

        # Order of substring lookup matters since ".onnx" matches both. Make
        # the fake check the longer key first.
        original_get = fake.get

        def _smart_get(url, *, stream=False, timeout=None):
            fake.calls.append(url)
            if ".onnx.json" in url:
                return _FakeStreamResponse(cfg_payload, content_length=len(cfg_payload))
            if ".onnx" in url:
                return _FakeStreamResponse(model_payload, content_length=len(model_payload))
            return _FakeStreamResponse(b"", status_code=404)
        fake.get = _smart_get

        model_path, config_path = voice_service.ensure_tts_voice(
            DEFAULT_TTS_VOICE_ID, _requests=fake,
        )

        assert model_path.exists()
        assert config_path.exists()
        assert model_path.read_bytes() == model_payload
        assert config_path.read_bytes() == cfg_payload

        rows = in_memory_db.get_db().execute(
            "SELECT asset_id, asset_type FROM voice_assets ORDER BY asset_id"
        ).fetchall()
        kinds = {r[1] for r in rows}
        assert kinds == {"tts"}
        assert len(rows) == 2


# ── Subprocess: pre-spawn validation ─────────────────────────────────────────


class TestTranscribeValidation:
    def test_missing_binary_raises(self, voice_service, monkeypatch, tmp_path):
        monkeypatch.setattr(
            voice_module.paths, "whisper_binary",
            lambda: tmp_path / "no-such-whisper.exe",
        )
        with pytest.raises(VoiceServiceError, match="not found"):
            voice_service.transcribe(tmp_path / "any.wav", DEFAULT_STT_MODEL_ID)

    def test_missing_model_raises(self, voice_service, monkeypatch, tmp_path):
        binary = tmp_path / "whisper.exe"
        binary.write_bytes(b"\x00")
        monkeypatch.setattr(
            voice_module.paths, "whisper_binary", lambda: binary,
        )
        # No model on disk at the resolved path → fail closed.
        with pytest.raises(VoiceServiceError, match="not downloaded"):
            voice_service.transcribe(tmp_path / "any.wav", DEFAULT_STT_MODEL_ID)


class TestSynthesizeValidation:
    def test_missing_binary_raises(self, voice_service, monkeypatch, tmp_path):
        monkeypatch.setattr(
            voice_module.paths, "piper_binary",
            lambda: tmp_path / "no-such-piper.exe",
        )
        with pytest.raises(VoiceServiceError, match="not found"):
            voice_service.synthesize("hello", DEFAULT_TTS_VOICE_ID)

    def test_missing_voice_raises(self, voice_service, monkeypatch, tmp_path):
        binary = tmp_path / "piper.exe"
        binary.write_bytes(b"\x00")
        monkeypatch.setattr(
            voice_module.paths, "piper_binary", lambda: binary,
        )
        with pytest.raises(VoiceServiceError, match="not downloaded"):
            voice_service.synthesize("hello", DEFAULT_TTS_VOICE_ID)

    def test_empty_text_raises(self, voice_service, monkeypatch, tmp_path):
        binary = tmp_path / "piper.exe"
        binary.write_bytes(b"\x00")
        monkeypatch.setattr(
            voice_module.paths, "piper_binary", lambda: binary,
        )
        # Stage the voice so we get past the model-missing check.
        entry = voice_service.resolve_tts(DEFAULT_TTS_VOICE_ID)
        (voice_module.paths.tts_voices_dir() / entry["model_filename"]).write_bytes(b"\x00")
        (voice_module.paths.tts_voices_dir() / entry["config_filename"]).write_bytes(b"{}")
        with pytest.raises(VoiceServiceError, match="empty"):
            voice_service.synthesize("   ", DEFAULT_TTS_VOICE_ID)


class TestSynthesizeWavCheck:
    """Stub Popen so the wav-header check is the only thing under test."""

    def test_invalid_wav_header_raises(self, voice_service, monkeypatch, tmp_path):
        binary = tmp_path / "piper.exe"
        binary.write_bytes(b"\x00")
        monkeypatch.setattr(
            voice_module.paths, "piper_binary", lambda: binary,
        )
        entry = voice_service.resolve_tts(DEFAULT_TTS_VOICE_ID)
        (voice_module.paths.tts_voices_dir() / entry["model_filename"]).write_bytes(b"\x00")
        (voice_module.paths.tts_voices_dir() / entry["config_filename"]).write_bytes(b"{}")

        class _FakeProc:
            returncode = 0
            stdout = b"NOT-A-WAV"
            stderr = b""

        monkeypatch.setattr(voice_module.subprocess, "run",
                            lambda *a, **k: _FakeProc())
        with pytest.raises(VoiceServiceError, match="wav header"):
            voice_service.synthesize("hello", DEFAULT_TTS_VOICE_ID)

    def test_valid_wav_header_returns_bytes(self, voice_service, monkeypatch, tmp_path):
        binary = tmp_path / "piper.exe"
        binary.write_bytes(b"\x00")
        monkeypatch.setattr(
            voice_module.paths, "piper_binary", lambda: binary,
        )
        entry = voice_service.resolve_tts(DEFAULT_TTS_VOICE_ID)
        (voice_module.paths.tts_voices_dir() / entry["model_filename"]).write_bytes(b"\x00")
        (voice_module.paths.tts_voices_dir() / entry["config_filename"]).write_bytes(b"{}")

        wav = _make_wav_bytes([0, 1, 2, 3, 0, -1, -2, -3])

        class _FakeProc:
            returncode = 0
            stdout = wav
            stderr = b""

        monkeypatch.setattr(voice_module.subprocess, "run",
                            lambda *a, **k: _FakeProc())
        result = voice_service.synthesize("hello", DEFAULT_TTS_VOICE_ID)
        assert result == wav
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"


# ── End-to-end (skipped unless local install present) ────────────────────────


def _local_whisper_binary() -> Path | None:
    """Return a Whisper.cpp binary if one is sitting in the source tree."""
    candidate = (
        Path(__file__).resolve().parents[2]
        / "branding" / "sidecar-bundle" / "whisper"
        / ("whisper-cli.exe" if sys.platform == "win32" else "whisper-cli")
    )
    return candidate if candidate.exists() else None


def _local_piper_binary() -> Path | None:
    candidate = (
        Path(__file__).resolve().parents[2]
        / "branding" / "sidecar-bundle" / "piper"
        / ("piper.exe" if sys.platform == "win32" else "piper")
    )
    return candidate if candidate.exists() else None


@pytest.mark.skipif(
    _local_whisper_binary() is None,
    reason="whisper-cli binary not present in branding/sidecar-bundle/whisper/",
)
class TestTranscribeEndToEnd:
    def test_smoke(self):
        # Placeholder: a real round-trip needs a Whisper model on disk too.
        # Skipped on CI; left as a hook for manual smoke checks.
        pytest.skip("requires a downloaded Whisper model — manual run only")


@pytest.mark.skipif(
    _local_piper_binary() is None,
    reason="piper binary not present in branding/sidecar-bundle/piper/",
)
class TestSynthesizeEndToEnd:
    def test_smoke(self):
        pytest.skip("requires a downloaded Piper voice — manual run only")
