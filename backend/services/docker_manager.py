"""
services/docker_manager.py — Power Mode (v3) Docker + OpenClaw lifecycle.

Manages Docker / WSL2 detection and the OpenClaw container that executes
delegated tasks for Power Mode. Everything in this module is *additive*: if
Docker isn't installed or Power Mode is disabled, the rest of the app runs
exactly as it did in v2.

Surface:
  DockerManager.status()                 → current detection snapshot
  DockerManager.start_openclaw(emit)     → render compose + `docker compose up`
  DockerManager.stop_openclaw(emit)      → `docker compose down`
  DockerManager.health_check()           → poll OpenClaw API once
  DockerManager.gateway_url()            → "http://127.0.0.1:<port>"
  DockerManager.workspace_dir()          → effective workspace path

All shell-outs go through `subprocess.run` with a hard timeout. We never call
`shell=True`, never interpolate user input into a command line, and never bind
anything off 127.0.0.1.

Errors are surfaced via SSE through the ``emit`` callable supplied by the
caller (the API facade); callers also receive a return value with the same
information so the caller can pick a recovery path.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("MyAIEnv.docker_manager")


# ── Defaults & constants ─────────────────────────────────────────────────────

DEFAULT_OPENCLAW_IMAGE = "coollabsio/openclaw:latest"
DEFAULT_GATEWAY_PORT = 18789
CONTAINER_NAME = "imakeaiteams-openclaw"
COMPOSE_FILENAME = "docker-compose.yml"
HEALTH_TIMEOUT_SEC = 60.0
HEALTH_POLL_INTERVAL_SEC = 1.5
SHELL_TIMEOUT_SEC = 30.0
MAX_RESTART_ATTEMPTS = 3
RESTART_BACKOFF_BASE_SEC = 2.0


# ── Status payloads ──────────────────────────────────────────────────────────

@dataclass
class DockerStatus:
    """Snapshot of host capabilities + OpenClaw container state.

    Serialized straight to JSON for the renderer's status panel; field names
    match the React side (see SettingsPanel.tsx Power Mode section).
    """

    wsl_installed: bool = False
    docker_installed: bool = False
    docker_running: bool = False
    openclaw_running: bool = False
    openclaw_healthy: bool = False
    gpu_available: bool = False
    platform: str = ""
    detail: str = ""
    last_error: str = ""
    gateway_url: str = ""
    workspace_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "wsl_installed": self.wsl_installed,
            "docker_installed": self.docker_installed,
            "docker_running": self.docker_running,
            "openclaw_running": self.openclaw_running,
            "openclaw_healthy": self.openclaw_healthy,
            "gpu_available": self.gpu_available,
            "platform": self.platform,
            "detail": self.detail,
            "last_error": self.last_error,
            "gateway_url": self.gateway_url,
            "workspace_dir": self.workspace_dir,
        }


@dataclass
class StartResult:
    ok: bool
    error: str = ""
    gateway_url: str = ""
    detail: str = ""


# ── Subprocess helpers ───────────────────────────────────────────────────────

def _run(cmd: list[str], *, timeout: float = SHELL_TIMEOUT_SEC) -> subprocess.CompletedProcess:
    """Run a command without a shell, with a strict timeout, capturing output.

    On Windows, ``CREATE_NO_WINDOW`` keeps a console flash from popping up when
    the sidecar is launched from a packaged installer.
    """
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=creationflags,
        check=False,
    )


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


# ── DockerManager ────────────────────────────────────────────────────────────

EmitFn = Callable[[str, dict], None]


class DockerManager:
    """Single owner of Docker detection + OpenClaw container lifecycle.

    Thread-safety: all state-mutating calls take an internal lock. Status
    reads return a copy so callers can serialize without holding the lock.
    """

    def __init__(
        self,
        settings,
        user_data_dir: Path,
        templates_dir: Path,
        emit: Optional[EmitFn] = None,
    ) -> None:
        self._settings = settings
        self._user_data_dir = Path(user_data_dir)
        self._templates_dir = Path(templates_dir)
        self._emit = emit or (lambda _e, _p: None)

        self._lock = threading.Lock()
        self._last_status = DockerStatus(platform=platform.system().lower())
        self._compose_path = self._user_data_dir / "openclaw" / COMPOSE_FILENAME
        self._data_dir = self._user_data_dir / "openclaw-data"
        self._restart_attempts = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def status(self, *, refresh: bool = True) -> DockerStatus:
        """Return the current status, refreshing from the host by default."""
        if refresh:
            self._refresh_status()
        with self._lock:
            return DockerStatus(**self._last_status.to_dict())

    def gateway_url(self) -> str:
        port = int(self._settings.get("power_mode_gateway_port", DEFAULT_GATEWAY_PORT) or DEFAULT_GATEWAY_PORT)
        return f"http://127.0.0.1:{port}"

    def workspace_dir(self) -> Path:
        configured = (self._settings.get("power_mode_workspace") or "").strip()
        if configured:
            return Path(configured)
        return Path.home() / "Documents" / "iMakeAiTeams-Workspace"

    def start_openclaw(self) -> StartResult:
        """Render compose, run `docker compose up -d`, then poll until healthy."""
        with self._lock:
            return self._start_locked()

    def stop_openclaw(self) -> StartResult:
        with self._lock:
            return self._stop_locked()

    def restart_openclaw(self) -> StartResult:
        """Manual restart — clears the auto-restart attempt counter."""
        self._restart_attempts = 0
        self.stop_openclaw()
        return self.start_openclaw()

    def health_check(self) -> bool:
        """Single health probe against the OpenClaw API. Never raises."""
        try:
            import urllib.request
            url = f"{self.gateway_url()}/health"
            with urllib.request.urlopen(url, timeout=2) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def shutdown(self) -> None:
        """Best-effort container teardown on app quit. Never raises."""
        try:
            self._stop_locked()
        except Exception as exc:
            log.debug("docker_manager.shutdown: %s", exc)

    # ── Detection ───────────────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        plat = platform.system().lower()
        wsl_installed = False
        docker_installed = False
        docker_running = False
        openclaw_running = False
        openclaw_healthy = False
        gpu_available = False
        last_error = ""

        # WSL2 — Windows only. Treat any non-empty `wsl --status` as installed.
        if plat == "windows":
            try:
                r = _run(["wsl", "--status"])
                wsl_installed = r.returncode == 0 and bool(r.stdout.strip())
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                last_error = f"wsl probe: {exc}"

        # Docker CLI present on PATH? (Windows checks both host docker and WSL.)
        if _which("docker"):
            docker_installed = True
            try:
                r = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
                docker_running = r.returncode == 0 and bool(r.stdout.strip())
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                last_error = f"docker info: {exc}"

        if plat == "windows" and not docker_running and wsl_installed:
            try:
                r = _run(["wsl", "-d", "Ubuntu", "--", "docker", "info",
                          "--format", "{{.ServerVersion}}"])
                if r.returncode == 0 and r.stdout.strip():
                    docker_installed = True
                    docker_running = True
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                if not last_error:
                    last_error = f"wsl docker: {exc}"

        # GPU availability — best effort. Any nvidia-smi success counts.
        if _which("nvidia-smi"):
            try:
                r = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
                gpu_available = r.returncode == 0 and bool(r.stdout.strip())
            except (subprocess.TimeoutExpired, FileNotFoundError):
                gpu_available = False

        if docker_running:
            openclaw_running = self._container_running()
            if openclaw_running:
                openclaw_healthy = self.health_check()

        with self._lock:
            self._last_status = DockerStatus(
                wsl_installed=wsl_installed,
                docker_installed=docker_installed,
                docker_running=docker_running,
                openclaw_running=openclaw_running,
                openclaw_healthy=openclaw_healthy,
                gpu_available=gpu_available,
                platform=plat,
                detail=self._build_detail(
                    wsl_installed, docker_installed, docker_running,
                    openclaw_running, openclaw_healthy,
                ),
                last_error=last_error,
                gateway_url=self.gateway_url(),
                workspace_dir=str(self.workspace_dir()),
            )

    @staticmethod
    def _build_detail(wsl: bool, di: bool, dr: bool, oc: bool, ocho: bool) -> str:
        if not di:
            return "Docker is not installed. Install Docker Desktop to enable Power Mode."
        if not dr:
            return "Docker is installed but not running. Start Docker Desktop and try again."
        if not oc:
            return "Docker is running. OpenClaw is not started yet."
        if not ocho:
            return "OpenClaw container is up but the gateway hasn't responded yet."
        return "OpenClaw is ready."

    def _container_running(self) -> bool:
        try:
            r = _run([
                "docker", "ps", "--filter", f"name=^{CONTAINER_NAME}$",
                "--format", "{{.Names}}",
            ])
            return r.returncode == 0 and CONTAINER_NAME in (r.stdout or "")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    # ── Compose render & lifecycle (lock held) ──────────────────────────────

    def _start_locked(self) -> StartResult:
        # Refresh first so we don't fight a half-initialized cache.
        self._refresh_status()
        if not self._last_status.docker_installed:
            msg = ("Power Mode requires Docker Desktop. "
                   "Install it from https://www.docker.com/products/docker-desktop/ "
                   "and click Re-check.")
            self._publish_status(error=msg)
            return StartResult(ok=False, error=msg)
        if not self._last_status.docker_running:
            msg = "Docker Desktop is installed but not running. Start it and click Re-check."
            self._publish_status(error=msg)
            return StartResult(ok=False, error=msg)

        try:
            self._render_compose()
        except Exception as exc:
            msg = f"Could not write docker-compose.yml: {exc}"
            log.warning(msg, exc_info=True)
            self._publish_status(error=msg)
            return StartResult(ok=False, error=msg)

        self._emit("power_mode_event", {
            "phase": "starting",
            "message": "Starting OpenClaw container…",
        })

        cmd = ["docker", "compose", "-f", str(self._compose_path), "up", "-d"]
        try:
            r = _run(cmd, timeout=120.0)
        except subprocess.TimeoutExpired as exc:
            msg = f"docker compose up timed out: {exc}"
            log.warning(msg)
            self._publish_status(error=msg)
            return StartResult(ok=False, error=msg)
        except FileNotFoundError as exc:
            msg = f"docker compose not found on PATH: {exc}"
            self._publish_status(error=msg)
            return StartResult(ok=False, error=msg)

        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "docker compose up failed").strip()
            self._publish_status(error=msg)
            return StartResult(ok=False, error=msg)

        # Poll the OpenClaw gateway until /health returns 200 or we give up.
        deadline = time.time() + HEALTH_TIMEOUT_SEC
        while time.time() < deadline:
            if self.health_check():
                self._restart_attempts = 0
                self._refresh_status()
                self._publish_status()
                self._emit("power_mode_event", {
                    "phase": "ready",
                    "message": "OpenClaw is ready.",
                    "gateway_url": self.gateway_url(),
                })
                return StartResult(ok=True, gateway_url=self.gateway_url(),
                                   detail="OpenClaw is ready.")
            time.sleep(HEALTH_POLL_INTERVAL_SEC)

        msg = ("OpenClaw container started but never became healthy within "
               f"{int(HEALTH_TIMEOUT_SEC)}s. Check `docker logs "
               f"{CONTAINER_NAME}` for details.")
        self._refresh_status()
        self._publish_status(error=msg)
        return StartResult(ok=False, error=msg)

    def _stop_locked(self) -> StartResult:
        if not self._compose_path.exists():
            # Nothing to stop. Silently succeed.
            self._refresh_status()
            self._publish_status()
            return StartResult(ok=True, detail="No compose file present.")

        cmd = ["docker", "compose", "-f", str(self._compose_path), "down"]
        try:
            r = _run(cmd, timeout=60.0)
        except subprocess.TimeoutExpired as exc:
            msg = f"docker compose down timed out: {exc}"
            self._publish_status(error=msg)
            return StartResult(ok=False, error=msg)
        except FileNotFoundError as exc:
            msg = f"docker compose not found on PATH: {exc}"
            self._publish_status(error=msg)
            return StartResult(ok=False, error=msg)

        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "docker compose down failed").strip()
            self._publish_status(error=msg)
            return StartResult(ok=False, error=msg)

        self._refresh_status()
        self._publish_status()
        self._emit("power_mode_event", {
            "phase": "stopped",
            "message": "OpenClaw stopped.",
        })
        return StartResult(ok=True, detail="OpenClaw stopped.")

    def _publish_status(self, *, error: str = "") -> None:
        snap = DockerStatus(**self._last_status.to_dict())
        if error:
            snap.last_error = error
        self._emit("power_mode_status", snap.to_dict())

    # ── Compose rendering ───────────────────────────────────────────────────

    def _render_compose(self) -> Path:
        """Render the Jinja2 compose template into the user-data dir."""
        template_path = self._templates_dir / f"{COMPOSE_FILENAME}.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"missing template: {template_path}")

        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
        except ImportError as exc:
            raise RuntimeError(
                "Power Mode needs the 'jinja2' Python package; "
                "reinstall the sidecar dependencies."
            ) from exc

        env = Environment(
            loader=FileSystemLoader(str(self._templates_dir)),
            autoescape=select_autoescape(default=False),
            keep_trailing_newline=True,
        )
        tmpl = env.get_template(f"{COMPOSE_FILENAME}.j2")

        ws_dir = self.workspace_dir()
        ws_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._compose_path.parent.mkdir(parents=True, exist_ok=True)

        provider = (self._settings.get("power_mode_model_provider") or "anthropic").strip()
        model = (self._settings.get("power_mode_model_name") or "claude-sonnet-4-6").strip()
        api_key = self._settings.get("power_mode_api_key") or self._settings.get("claude_api_key") or ""
        gateway_port = int(self._settings.get("power_mode_gateway_port", DEFAULT_GATEWAY_PORT) or DEFAULT_GATEWAY_PORT)

        memory_limit_mb = self._memory_limit_mb()

        rendered = tmpl.render(
            image=DEFAULT_OPENCLAW_IMAGE,
            container_name=CONTAINER_NAME,
            gateway_port=gateway_port,
            provider=provider,
            model=model,
            api_key=api_key,
            workspace_dir=str(ws_dir),
            data_dir=str(self._data_dir),
            memory_limit_mb=memory_limit_mb,
            extra_env={},
        )
        self._compose_path.write_text(rendered, encoding="utf-8")
        return self._compose_path

    @staticmethod
    def _memory_limit_mb() -> int:
        """Cap OpenClaw at half of installed RAM (clamped to a sane range)."""
        try:
            import psutil
            total = psutil.virtual_memory().total
            mb = int(total / (1024 * 1024) * 0.5)
            return max(1024, min(mb, 16384))
        except Exception:
            return 4096

    # ── Auto-restart with exponential backoff (called by health watcher) ────

    def watch_and_restart(self, stop_event: threading.Event) -> None:
        """Background loop: if OpenClaw dies, retry up to MAX_RESTART_ATTEMPTS."""
        while not stop_event.is_set():
            stop_event.wait(15.0)
            if stop_event.is_set():
                return
            if not self._settings.get("power_mode_enabled"):
                continue
            if self.health_check():
                self._restart_attempts = 0
                continue
            if self._restart_attempts >= MAX_RESTART_ATTEMPTS:
                self._emit("power_mode_event", {
                    "phase": "fatal",
                    "message": ("OpenClaw failed to recover after "
                                f"{MAX_RESTART_ATTEMPTS} attempts. Disable Power "
                                "Mode and check Docker Desktop."),
                })
                return
            backoff = RESTART_BACKOFF_BASE_SEC * (2 ** self._restart_attempts)
            self._restart_attempts += 1
            self._emit("power_mode_event", {
                "phase": "restarting",
                "attempt": self._restart_attempts,
                "message": f"OpenClaw is unhealthy; restart attempt {self._restart_attempts}/{MAX_RESTART_ATTEMPTS}",
            })
            stop_event.wait(backoff)
            if stop_event.is_set():
                return
            self.start_openclaw()
