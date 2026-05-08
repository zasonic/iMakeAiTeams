"""
services/bundled_server.py — Lifecycle for the bundled llama.cpp server.

The BundledServer owns three concerns:

  1. **Model download** (`download_model`). Streams a GGUF from a known-good
     source (Hugging Face), validates sha256 against the catalog, writes
     atomically (`.partial` → rename), and inserts a `bundled_models` row.

  2. **Subprocess lifecycle** (`start` / `stop` / `is_running`). Spawns
     ``llama-server`` on a free localhost port, parses the port from the
     child's startup log, writes it to a port file for crash-recovery, and
     reaps the child on shutdown.

  3. **Catalog resolution** (`resolve_model`). Reads the build-pipeline
     catalog from ``paths.bundled_models_catalog_path()`` if present;
     otherwise falls back to live Hugging Face metadata lookup so source
     checkouts still work.

Design notes:

* Bundled mode is additive. Ollama/LM Studio code paths are untouched —
  ``LocalClient`` decides which backend to call based on
  ``settings.local_backend_mode``.
* The server binds to 127.0.0.1 only. We do not need a Bearer token because
  ``LocalClient`` is the only consumer; the reach surface is loopback only.
* All public methods log on failure and never raise — the wizard surfaces
  errors via the SSE event stream, not exception propagation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from core import paths
from core.settings import Settings

log = logging.getLogger("iMakeAiTeams.bundled")

# ── Catalog of known-good models ─────────────────────────────────────────────
#
# Each model_id maps to the Hugging Face source. ``expected_sha256`` and
# ``expected_size_bytes`` may be empty/None at source-checkout time; in that
# case the runtime fetches them live from HF metadata. The build pipeline
# (dev/build-installer.bat) overwrites the catalog at
# ``branding/bundled_models.json`` with both fields populated, so packaged
# installers ship a fully pinned catalog.
_DEFAULT_MODELS: dict[str, dict] = {
    "Qwen3-4B-Instruct-Q4_K_M": {
        "repo":     "Qwen/Qwen3-4B-Instruct-GGUF",
        "filename": "Qwen3-4B-Instruct-Q4_K_M.gguf",
        # Filled in by the build pipeline via HF metadata lookup; an empty
        # sha256 forces the live lookup path at download time.
        "expected_sha256":     "",
        "expected_size_bytes": 0,
        "context_length":      32768,
    },
}

DEFAULT_MODEL_ID = "Qwen3-4B-Instruct-Q4_K_M"

_HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{filename}"
_HF_TREE_API = "https://huggingface.co/api/models/{repo}/tree/main"

_PARTIAL_SUFFIX = ".partial"
_DOWNLOAD_TIMEOUT_SEC = 60
_DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB
_PORT_PATTERN = re.compile(r"listening (?:on|at) [^:]+:(\d{2,5})", re.IGNORECASE)


class BundledServerError(Exception):
    """Raised by BundledServer when a public method has to fail closed."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_catalog() -> dict[str, dict]:
    """Read the build-pipeline catalog if present; else use the in-source defaults."""
    catalog_path = paths.bundled_models_catalog_path()
    if catalog_path.exists():
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged: dict[str, dict] = {}
                for mid, defaults in _DEFAULT_MODELS.items():
                    merged[mid] = {**defaults, **(data.get(mid) or {})}
                # Catalog entries not in defaults are still picked up so the
                # build script can add models without a code change. Skip
                # keys starting with "_" so the build pipeline can stash
                # metadata (e.g. release tag) without it looking like a model.
                for mid, entry in data.items():
                    if mid.startswith("_"):
                        continue
                    if mid not in merged and isinstance(entry, dict):
                        merged[mid] = entry
                return merged
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("bundled: failed to read catalog %s: %s", catalog_path, exc)
    return {k: dict(v) for k, v in _DEFAULT_MODELS.items()}


def _hf_metadata_lookup(repo: str, filename: str, *,
                        timeout: float = 10.0) -> tuple[str, int]:
    """Return (sha256, size_bytes) for a single LFS file in an HF repo.

    Raises ``BundledServerError`` if the API call fails or the file isn't
    found. Used as a fallback when the catalog has no pinned sha256.
    """
    try:
        url = _HF_TREE_API.format(repo=urllib.parse.quote(repo, safe="/"))
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        items = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise BundledServerError(
            f"could not fetch HF metadata for {repo}: {exc}"
        ) from exc

    for entry in items:
        if not isinstance(entry, dict):
            continue
        if entry.get("path") != filename:
            continue
        lfs = entry.get("lfs") or {}
        sha = lfs.get("sha256") or lfs.get("oid") or ""
        size = lfs.get("size") or entry.get("size") or 0
        if not sha or not size:
            raise BundledServerError(
                f"HF metadata for {repo}/{filename} is missing sha256/size"
            )
        return str(sha), int(size)

    raise BundledServerError(
        f"file {filename} not found in HF repo {repo}"
    )


def _bind_free_port() -> int:
    """Return a free localhost TCP port and immediately release it.

    There's a small TOCTOU window before llama-server binds, but the
    alternative (handing it a pre-bound socket) doesn't work cross-process
    on Windows. The wizard's smoke chat will surface a clear error if the
    port was claimed by something else in that gap.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


# ── Class ────────────────────────────────────────────────────────────────────


class BundledServer:
    """Owns the bundled llama-server child process and download lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._model_id: str | None = None

    # ── Catalog ──────────────────────────────────────────────────────────────

    def list_known_models(self) -> dict[str, dict]:
        """Return the merged catalog (defaults + build-pipeline JSON)."""
        return _load_catalog()

    def resolve_model(self, model_id: str) -> dict:
        """Return the full catalog entry for ``model_id``, with live HF lookup
        filling in missing sha256/size_bytes on the fly. Raises
        ``BundledServerError`` if ``model_id`` is unknown."""
        catalog = _load_catalog()
        if model_id not in catalog:
            raise BundledServerError(f"unknown bundled model: {model_id}")
        entry = dict(catalog[model_id])
        if not entry.get("expected_sha256") or not entry.get("expected_size_bytes"):
            sha, size = _hf_metadata_lookup(entry["repo"], entry["filename"])
            entry["expected_sha256"] = sha
            entry["expected_size_bytes"] = size
        return entry

    # ── Download ────────────────────────────────────────────────────────────

    def download_model(
        self,
        model_id: str,
        on_progress: Callable[[int, int], None] | None = None,
        *,
        _requests: object = requests,
    ) -> dict:
        """Download a known-good GGUF and record it in the bundled_models table.

        Atomic: writes to ``<file>.partial`` and renames on success; deletes
        the partial on sha256 mismatch. ``on_progress(done, total)`` fires
        every chunk so the wizard can render a progress bar. Returns the
        catalog entry merged with ``{"file_path": str, "downloaded_at": iso}``.

        Raises ``BundledServerError`` on any failure. The DB row is only
        written after sha256 verification succeeds.
        """
        entry = self.resolve_model(model_id)
        target_dir = paths.bundled_models_dir()
        target = target_dir / entry["filename"]
        partial = target.with_suffix(target.suffix + _PARTIAL_SUFFIX)

        if target.exists() and target.stat().st_size == entry["expected_size_bytes"]:
            # Already present; verify sha256 once before declaring victory.
            actual = _sha256_file(target)
            if actual == entry["expected_sha256"]:
                self._record_downloaded(model_id, target, entry)
                return {**entry, "file_path": str(target),
                        "downloaded_at": _utc_now_iso(), "skipped": True}

        url = _HF_RESOLVE.format(
            repo=urllib.parse.quote(entry["repo"], safe="/"),
            filename=urllib.parse.quote(entry["filename"]),
        )

        # Fresh download — wipe any stale partial so we never resume against
        # a different file that happened to share the name.
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass

        hasher = hashlib.sha256()
        bytes_done = 0
        bytes_total = int(entry["expected_size_bytes"]) or 0

        try:
            with _requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC) as resp:
                resp.raise_for_status()
                # Some HF redirects don't include Content-Length; fall back
                # to the catalog-supplied size for the progress denominator.
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
                                # Progress callbacks must never abort a download.
                                log.debug("download_model: on_progress raised", exc_info=True)
        except (requests.RequestException, OSError) as exc:
            _safe_unlink(partial)
            raise BundledServerError(f"download failed: {exc}") from exc

        actual_sha = hasher.hexdigest()
        if actual_sha != entry["expected_sha256"]:
            _safe_unlink(partial)
            raise BundledServerError(
                f"sha256 mismatch for {entry['filename']}: "
                f"expected {entry['expected_sha256']}, got {actual_sha}"
            )

        try:
            os.replace(partial, target)
        except OSError as exc:
            _safe_unlink(partial)
            raise BundledServerError(f"could not move into place: {exc}") from exc

        self._record_downloaded(model_id, target, entry)
        self._settings.set("bundled_model_id", model_id)
        return {
            **entry,
            "file_path":     str(target),
            "downloaded_at": _utc_now_iso(),
            "skipped":       False,
        }

    def _record_downloaded(self, model_id: str, target: Path, entry: dict) -> None:
        """Insert/update the ``bundled_models`` row. Idempotent on re-runs."""
        try:
            import db as _db_module  # noqa: PLC0415  (lazy to keep import light)
            conn = _db_module.get_db()
            conn.execute(
                """INSERT INTO bundled_models
                   (model_id, file_path, size_bytes, sha256, downloaded_at, last_loaded_at)
                   VALUES (?, ?, ?, ?, ?, NULL)
                   ON CONFLICT(model_id) DO UPDATE SET
                     file_path     = excluded.file_path,
                     size_bytes    = excluded.size_bytes,
                     sha256        = excluded.sha256,
                     downloaded_at = excluded.downloaded_at""",
                (
                    model_id,
                    str(target),
                    int(entry["expected_size_bytes"]),
                    str(entry["expected_sha256"]),
                    _utc_now_iso(),
                ),
            )
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("bundled: could not record %s in DB: %s", model_id, exc)

    # ── Subprocess lifecycle ─────────────────────────────────────────────────

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def port(self) -> int | None:
        with self._lock:
            return self._port

    def model_id(self) -> str | None:
        with self._lock:
            return self._model_id

    def start(self, model_path: str, *, model_id: str | None = None) -> int:
        """Spawn llama-server bound to a random localhost port.

        Returns the bound port. Raises ``BundledServerError`` if the binary
        is missing or the child fails to bind. Idempotent: calling start()
        twice in a row with the same model returns the existing port.
        """
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                if self._port is not None and (model_id or self._model_id) == self._model_id:
                    return self._port
                # Different model requested — stop the old child first.
                self._stop_locked()

            binary = paths.bundled_server_binary()
            if not binary.exists():
                raise BundledServerError(
                    f"llama-server binary not found at {binary}. "
                    "The installer ships it under resources/backend/llama-server/; "
                    "if you're running from source, populate "
                    "branding/sidecar-bundle/llama-server/."
                )
            if not Path(model_path).exists():
                raise BundledServerError(f"model file missing: {model_path}")

            port = _bind_free_port()
            cmd = [
                str(binary),
                "-m", str(model_path),
                "--host", "127.0.0.1",
                "--port", str(port),
                "--ctx-size", "8192",
                "--n-gpu-layers", "0",
            ]

            try:
                # stdin closed so the child doesn't block on a parent terminal.
                # creationflags on Windows hides the console window even when
                # the parent is launched from File Explorer.
                creation_flags = 0
                if sys.platform == "win32":
                    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                proc = subprocess.Popen(  # noqa: S603 — argv only, no shell
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                    text=True,
                )
            except OSError as exc:
                raise BundledServerError(f"could not spawn llama-server: {exc}") from exc

            self._proc = proc
            self._port = port
            self._model_id = model_id

            # Persist the port for crash-recovery probes from the wizard.
            try:
                paths.bundled_server_port_file().write_text(str(port), encoding="utf-8")
            except OSError as exc:
                log.warning("bundled: could not write port file: %s", exc)

            # Drain stdout in a daemon thread so the pipe never blocks.
            threading.Thread(
                target=_drain_pipe,
                args=(proc, log),
                daemon=True,
                name="bundled-server-stdout",
            ).start()

            # Update DB last_loaded_at if we recognise the model.
            if model_id:
                self._touch_last_loaded(model_id)

            return port

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                if sys.platform == "win32":
                    # taskkill /T to also reap any helper processes the child
                    # may have spawned (e.g. CUDA worker threads).
                    subprocess.run(  # noqa: S603,S607
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001
            log.warning("bundled: stop() raised: %s", exc, exc_info=True)
        finally:
            self._proc = None
            self._port = None
            self._model_id = None
            try:
                paths.bundled_server_port_file().unlink(missing_ok=True)
            except OSError:
                pass

    def _touch_last_loaded(self, model_id: str) -> None:
        try:
            import db as _db_module  # noqa: PLC0415
            conn = _db_module.get_db()
            conn.execute(
                "UPDATE bundled_models SET last_loaded_at = ? WHERE model_id = ?",
                (_utc_now_iso(), model_id),
            )
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            log.debug("bundled: could not update last_loaded_at: %s", exc)


# ── Helpers ──────────────────────────────────────────────────────────────────


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
        log.debug("bundled: could not unlink %s: %s", path, exc)


def _drain_pipe(proc: subprocess.Popen, logger: logging.Logger) -> None:
    """Forward llama-server stdout/stderr into our log so the user can
    diagnose startup failures via app.log without staring at a terminal."""
    if proc.stdout is None:
        return
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.info("[llama-server] %s", line)
    except Exception as exc:  # noqa: BLE001
        logger.debug("bundled: stdout drain ended: %s", exc)
