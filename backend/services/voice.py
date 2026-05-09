"""
services/voice.py — Whisper.cpp + Piper bundled-binary lifecycle (PR 17).

VoiceService owns four concerns:

  1. **STT model download** (`ensure_stt_model`). Streams a Whisper.cpp .bin
     from Hugging Face into ``userData/voice/stt/``, validates sha256 against
     the catalog, writes atomically (`.partial` → rename), and inserts a
     ``voice_assets`` row.

  2. **TTS voice download** (`ensure_tts_voice`). Streams the .onnx + .json
     pair for a Piper voice into ``userData/voice/tts/``. Same atomic-write
     and sha256-check pattern as the STT downloader; one row per file is
     written into ``voice_assets``.

  3. **Subprocess transcription** (`transcribe`). Spawns ``whisper-cli`` with
     ``--output-txt --no-timestamps`` against the supplied wav, captures the
     produced .txt content, and returns the transcribed string. Bounded by a
     hard timeout — a hung child is killed and the call raises
     ``VoiceServiceError``.

  4. **Subprocess synthesis** (`synthesize`). Spawns ``piper`` with the
     downloaded voice, pipes the text on stdin, captures the wav bytes from
     stdout. Same timeout + error contract as ``transcribe``.

Design notes:

* Mirrors the BundledServer (services/bundled_server.py) contract: download
  on first feature use, sha256-validate, record in DB, then keep the file
  around. Voice binaries themselves ship bundled in the installer (the spec
  trades 25 MB of installer growth for zero-setup-on-clean-install).
* All subprocess calls go through ``argv``-only Popen — no shell
  interpolation, no untrusted strings reaching ``cmd.exe``.
* ``VoiceServiceError`` is the typed failure surface; routes catch it and
  surface a human-readable message via the SSE event stream and HTTP body.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from core import paths
from core.settings import Settings

log = logging.getLogger("iMakeAiTeams.voice")


# ── Catalog of bundled-binary-companion voice assets ─────────────────────────
#
# Each entry maps a stable id to its remote source and pinned sha256. The
# build pipeline overwrites this catalog at branding/sidecar-bundle/
# voice_assets.json; runtime falls through to these defaults when the catalog
# file is missing (source checkouts, dev runs).

_DEFAULT_STT_MODELS: dict[str, dict] = {
    "whisper-base.en": {
        "repo":                "ggerganov/whisper.cpp",
        "filename":            "ggml-base.en.bin",
        "expected_sha256":     "",
        "expected_size_bytes": 0,
    },
}

_DEFAULT_TTS_VOICES: dict[str, dict] = {
    "en_US-amy-medium": {
        "repo":                "rhasspy/piper-voices",
        "hf_subpath":          "en/en_US/amy/medium",
        "model_filename":      "en_US-amy-medium.onnx",
        "config_filename":     "en_US-amy-medium.onnx.json",
        "model_sha256":        "",
        "model_size_bytes":    0,
        "config_sha256":       "",
        "config_size_bytes":   0,
    },
}

DEFAULT_STT_MODEL_ID = "whisper-base.en"
DEFAULT_TTS_VOICE_ID = "en_US-amy-medium"

_HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"

_PARTIAL_SUFFIX = ".partial"
_DOWNLOAD_TIMEOUT_SEC = 60
_DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB

# Subprocess timeouts. Whisper transcription scales with audio length; we
# allow roughly 5x real time as a generous ceiling for base.en on CPU. Piper
# synthesis is bounded by output length, which we cap at 5000 characters at
# the route level; 60s is more than enough for that.
_TRANSCRIBE_TIMEOUT_SEC = 600
_SYNTHESIZE_TIMEOUT_SEC = 60


class VoiceServiceError(Exception):
    """Raised when a VoiceService method has to fail closed."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_catalog() -> dict[str, dict]:
    """Read the build-pipeline voice catalog if present; else use the
    in-source defaults. The returned shape is ``{"stt": {...}, "tts": {...}}``.
    """
    catalog_path = paths.voice_assets_catalog_path()
    base: dict[str, dict] = {
        "stt": {k: dict(v) for k, v in _DEFAULT_STT_MODELS.items()},
        "tts": {k: dict(v) for k, v in _DEFAULT_TTS_VOICES.items()},
    }
    if not catalog_path.exists():
        return base
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("voice: failed to read catalog %s: %s", catalog_path, exc)
        return base
    if not isinstance(data, dict):
        return base
    for kind in ("stt", "tts"):
        section = data.get(kind)
        if not isinstance(section, dict):
            continue
        for asset_id, entry in section.items():
            if not isinstance(entry, dict):
                continue
            merged = dict(base[kind].get(asset_id, {}))
            merged.update(entry)
            base[kind][asset_id] = merged
    return base


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_DOWNLOAD_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("voice: could not unlink %s: %s", path, exc)


# ── Service ──────────────────────────────────────────────────────────────────


class VoiceService:
    """Voice asset lifecycle + subprocess dispatch for Whisper.cpp + Piper."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()

    # ── Catalog ──────────────────────────────────────────────────────────────

    def list_known_assets(self) -> dict[str, dict]:
        return _load_catalog()

    def resolve_stt(self, model_id: str) -> dict:
        catalog = _load_catalog()
        stt = catalog.get("stt", {})
        if model_id not in stt:
            raise VoiceServiceError(f"unknown STT model: {model_id}")
        return dict(stt[model_id])

    def resolve_tts(self, voice_id: str) -> dict:
        catalog = _load_catalog()
        tts = catalog.get("tts", {})
        if voice_id not in tts:
            raise VoiceServiceError(f"unknown TTS voice: {voice_id}")
        return dict(tts[voice_id])

    # ── Asset readiness probes (used by /assets/status) ──────────────────────

    def stt_ready(self, model_id: str) -> bool:
        try:
            entry = self.resolve_stt(model_id)
        except VoiceServiceError:
            return False
        target = paths.stt_models_dir() / entry["filename"]
        return target.exists()

    def tts_ready(self, voice_id: str) -> bool:
        try:
            entry = self.resolve_tts(voice_id)
        except VoiceServiceError:
            return False
        model = paths.tts_voices_dir() / entry["model_filename"]
        cfg = paths.tts_voices_dir() / entry["config_filename"]
        return model.exists() and cfg.exists()

    # ── Download ─────────────────────────────────────────────────────────────

    def ensure_stt_model(
        self,
        model_id: str,
        on_progress: Callable[[int, int], None] | None = None,
        *,
        _requests: object = requests,
    ) -> Path:
        """Download a Whisper.cpp .bin into userData/voice/stt/ if not present.

        Atomic: writes to ``<file>.partial`` then renames on sha256 match.
        Returns the on-disk path once the file is ready.
        """
        entry = self.resolve_stt(model_id)
        target_dir = paths.stt_models_dir()
        target = target_dir / entry["filename"]

        if target.exists() and entry.get("expected_size_bytes"):
            # Re-validate against the catalog so a half-written or replaced
            # file doesn't pass silently. Unknown size = trust on-disk.
            if target.stat().st_size == entry["expected_size_bytes"]:
                if not entry.get("expected_sha256") or \
                        _sha256_file(target) == entry["expected_sha256"]:
                    self._record_asset(model_id, "stt", target,
                                       sha256=entry.get("expected_sha256", ""),
                                       size_bytes=int(entry.get("expected_size_bytes") or 0))
                    return target
        elif target.exists():
            return target

        url = _HF_RESOLVE.format(
            repo=urllib.parse.quote(entry["repo"], safe="/"),
            path=urllib.parse.quote(entry["filename"]),
        )
        downloaded = self._stream_to_disk(
            url, target,
            expected_sha256=entry.get("expected_sha256", ""),
            expected_size_bytes=int(entry.get("expected_size_bytes") or 0),
            on_progress=on_progress,
            _requests=_requests,
        )
        self._record_asset(
            model_id, "stt", target,
            sha256=downloaded["sha256"],
            size_bytes=downloaded["size_bytes"],
        )
        return target

    def ensure_tts_voice(
        self,
        voice_id: str,
        on_progress: Callable[[int, int], None] | None = None,
        *,
        _requests: object = requests,
    ) -> tuple[Path, Path]:
        """Download a Piper voice (.onnx + .json) into userData/voice/tts/.

        Returns ``(model_path, config_path)``. Atomic per file.
        """
        entry = self.resolve_tts(voice_id)
        target_dir = paths.tts_voices_dir()
        model_target = target_dir / entry["model_filename"]
        config_target = target_dir / entry["config_filename"]

        # Two-file progress: report cumulative bytes across both files so the
        # UI bar advances monotonically. The denominator is the catalog sum.
        total_expected = (
            int(entry.get("model_size_bytes") or 0)
            + int(entry.get("config_size_bytes") or 0)
        )
        cumulative = {"done": 0}

        def _wrap_progress(current_done: int, current_total: int) -> None:
            if on_progress is None:
                return
            try:
                on_progress(cumulative["done"] + current_done,
                            total_expected or current_total)
            except Exception:  # noqa: BLE001
                log.debug("ensure_tts_voice: on_progress raised", exc_info=True)

        # Model file
        if not (model_target.exists()
                and (not entry.get("model_size_bytes")
                     or model_target.stat().st_size == entry["model_size_bytes"])):
            model_url = _HF_RESOLVE.format(
                repo=urllib.parse.quote(entry["repo"], safe="/"),
                path=urllib.parse.quote(
                    f"{entry['hf_subpath']}/{entry['model_filename']}"),
            )
            downloaded = self._stream_to_disk(
                model_url, model_target,
                expected_sha256=entry.get("model_sha256", ""),
                expected_size_bytes=int(entry.get("model_size_bytes") or 0),
                on_progress=_wrap_progress,
                _requests=_requests,
            )
            self._record_asset(
                f"{voice_id}.onnx", "tts", model_target,
                sha256=downloaded["sha256"],
                size_bytes=downloaded["size_bytes"],
            )
        cumulative["done"] += int(entry.get("model_size_bytes") or 0)

        # Config file
        if not (config_target.exists()
                and (not entry.get("config_size_bytes")
                     or config_target.stat().st_size == entry["config_size_bytes"])):
            config_url = _HF_RESOLVE.format(
                repo=urllib.parse.quote(entry["repo"], safe="/"),
                path=urllib.parse.quote(
                    f"{entry['hf_subpath']}/{entry['config_filename']}"),
            )
            downloaded = self._stream_to_disk(
                config_url, config_target,
                expected_sha256=entry.get("config_sha256", ""),
                expected_size_bytes=int(entry.get("config_size_bytes") or 0),
                on_progress=_wrap_progress,
                _requests=_requests,
            )
            self._record_asset(
                f"{voice_id}.json", "tts", config_target,
                sha256=downloaded["sha256"],
                size_bytes=downloaded["size_bytes"],
            )

        return model_target, config_target

    def _stream_to_disk(
        self,
        url: str,
        target: Path,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
        on_progress: Callable[[int, int], None] | None,
        _requests: object,
    ) -> dict:
        """Stream ``url`` to ``target`` atomically. Returns the actual sha256
        + size_bytes once verified. Raises VoiceServiceError on any failure.
        """
        partial = target.with_suffix(target.suffix + _PARTIAL_SUFFIX)
        if partial.exists():
            _safe_unlink(partial)

        hasher = hashlib.sha256()
        bytes_done = 0
        bytes_total = expected_size_bytes or 0

        try:
            with _requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC) as resp:
                resp.raise_for_status()
                hdr_total = resp.headers.get("Content-Length")
                if hdr_total and hdr_total.isdigit():
                    bytes_total = int(hdr_total)
                with open(partial, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                        if not chunk:
                            continue
                        out.write(chunk)
                        hasher.update(chunk)
                        bytes_done += len(chunk)
                        if on_progress is not None:
                            try:
                                on_progress(bytes_done, bytes_total)
                            except Exception:  # noqa: BLE001
                                log.debug("voice: on_progress raised",
                                          exc_info=True)
        except (requests.RequestException, OSError) as exc:
            _safe_unlink(partial)
            raise VoiceServiceError(f"download failed: {exc}") from exc

        actual_sha = hasher.hexdigest()
        if expected_sha256 and actual_sha != expected_sha256:
            _safe_unlink(partial)
            raise VoiceServiceError(
                f"sha256 mismatch for {target.name}: "
                f"expected {expected_sha256}, got {actual_sha}"
            )

        try:
            os.replace(partial, target)
        except OSError as exc:
            _safe_unlink(partial)
            raise VoiceServiceError(
                f"could not move into place: {exc}"
            ) from exc

        return {"sha256": actual_sha, "size_bytes": bytes_done}

    def _record_asset(
        self, asset_id: str, asset_type: str, target: Path,
        *, sha256: str, size_bytes: int,
    ) -> None:
        """Insert/update a ``voice_assets`` row. Idempotent on re-runs."""
        try:
            import db as _db_module  # noqa: PLC0415
            conn = _db_module.get_db()
            conn.execute(
                """INSERT INTO voice_assets
                   (asset_id, asset_type, file_path, sha256, size_bytes, downloaded_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(asset_id) DO UPDATE SET
                     asset_type    = excluded.asset_type,
                     file_path     = excluded.file_path,
                     sha256        = excluded.sha256,
                     size_bytes    = excluded.size_bytes,
                     downloaded_at = excluded.downloaded_at""",
                (asset_id, asset_type, str(target), sha256, int(size_bytes),
                 _utc_now_iso()),
            )
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("voice: could not record %s in DB: %s", asset_id, exc)

    # ── Subprocess: transcribe ───────────────────────────────────────────────

    def transcribe(self, audio_path: Path, model_id: str) -> str:
        """Spawn whisper-cli against ``audio_path`` and return the transcript.

        ``audio_path`` should be a 16-bit PCM wav at 16 kHz — Whisper.cpp's
        native input. The route layer is responsible for any conversion.
        """
        binary = paths.whisper_binary()
        if not binary.exists():
            raise VoiceServiceError(
                f"whisper-cli binary not found at {binary}. "
                "The installer ships it under resources/backend/whisper/; "
                "if you're running from source, populate "
                "branding/sidecar-bundle/whisper/."
            )
        try:
            entry = self.resolve_stt(model_id)
        except VoiceServiceError:
            raise
        model_path = paths.stt_models_dir() / entry["filename"]
        if not model_path.exists():
            raise VoiceServiceError(
                f"STT model {model_id} not downloaded. "
                "Open Settings → Voice and click Download."
            )
        if not Path(audio_path).exists():
            raise VoiceServiceError(f"audio file missing: {audio_path}")

        # whisper-cli writes <wav>.txt next to the input when called with
        # --output-txt. Use a temp dir so concurrent calls don't collide.
        with tempfile.TemporaryDirectory(prefix="whisper-", dir=str(paths.voice_dir())) as work:
            work_dir = Path(work)
            staged_wav = work_dir / "input.wav"
            try:
                staged_wav.write_bytes(Path(audio_path).read_bytes())
            except OSError as exc:
                raise VoiceServiceError(
                    f"could not stage audio for whisper: {exc}",
                ) from exc

            cmd = [
                str(binary),
                "-m", str(model_path),
                "-f", str(staged_wav),
                "--output-txt",
                "--no-timestamps",
                "--language", "en",
            ]
            text_path = staged_wav.with_suffix(".wav.txt")

            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            try:
                proc = subprocess.run(  # noqa: S603 — argv only, no shell
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=_TRANSCRIBE_TIMEOUT_SEC,
                    creationflags=creation_flags,
                )
            except subprocess.TimeoutExpired as exc:
                raise VoiceServiceError(
                    "transcription timed out — try a shorter clip"
                ) from exc
            except OSError as exc:
                raise VoiceServiceError(
                    f"could not spawn whisper-cli: {exc}",
                ) from exc

            if proc.returncode != 0:
                stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
                raise VoiceServiceError(
                    f"whisper-cli failed (exit {proc.returncode}): "
                    f"{stderr.strip()[:500]}"
                )

            try:
                transcript = text_path.read_text(encoding="utf-8")
            except OSError:
                # Some whisper-cli builds emit text on stdout instead of a
                # sidecar file; fall back to the captured stdout in that case.
                transcript = (proc.stdout or b"").decode("utf-8", errors="replace")

            return transcript.strip()

    # ── Subprocess: synthesize ───────────────────────────────────────────────

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Spawn piper, pipe ``text`` on stdin, return the wav bytes from stdout.

        Caller is responsible for capping ``text`` length — we don't truncate
        because a cut-off sentence sounds worse than a clean refusal.
        """
        binary = paths.piper_binary()
        if not binary.exists():
            raise VoiceServiceError(
                f"piper binary not found at {binary}. "
                "The installer ships it under resources/backend/piper/; "
                "if you're running from source, populate "
                "branding/sidecar-bundle/piper/."
            )
        entry = self.resolve_tts(voice_id)
        model_path = paths.tts_voices_dir() / entry["model_filename"]
        config_path = paths.tts_voices_dir() / entry["config_filename"]
        if not model_path.exists() or not config_path.exists():
            raise VoiceServiceError(
                f"TTS voice {voice_id} not downloaded. "
                "Open Settings → Voice and click Download."
            )
        if not text or not text.strip():
            raise VoiceServiceError("text to synthesize is empty")

        # Piper writes a fully-framed wav (with header) when --output_file is
        # ``-``; the renderer plays it back through the Audio API.
        cmd = [
            str(binary),
            "--model", str(model_path),
            "--config", str(config_path),
            "--output_file", "-",
            "--quiet",
        ]

        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            proc = subprocess.run(  # noqa: S603 — argv only, no shell
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_SYNTHESIZE_TIMEOUT_SEC,
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired as exc:
            raise VoiceServiceError(
                "synthesis timed out — try a shorter passage"
            ) from exc
        except OSError as exc:
            raise VoiceServiceError(
                f"could not spawn piper: {exc}",
            ) from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
            raise VoiceServiceError(
                f"piper failed (exit {proc.returncode}): "
                f"{stderr.strip()[:500]}"
            )

        wav = proc.stdout or b""
        # A wav file always starts with "RIFF....WAVE". A bare PCM stream is
        # a sign of an old piper build — we surface that as an error so the
        # frontend doesn't try to play raw PCM through the Audio API.
        if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            raise VoiceServiceError(
                "piper did not return a wav header — bundled binary may be stale"
            )
        return wav
