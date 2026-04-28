"""
backend/server.py — FastAPI sidecar entrypoint.

Lifecycle (matches the contract Electron's SidecarManager expects):
1. Parse `--token <uuid>` (required) and `--user-data <path>` (optional).
2. Bind a uvicorn server on 127.0.0.1 with port=0 (OS-assigned).
3. Print `PORT=<n>` to stdout and flush, so Electron can read it line-by-line.
4. Mount routers, then print `READY` and flush.
5. Serve until POST /shutdown is hit (or the parent kills us).

All routes (except GET /health with no token) require a Bearer token.
SSE endpoints accept the token via ?token= query string since EventSource
doesn't support custom headers.

Network: binds 127.0.0.1 only — never 0.0.0.0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import signal
import socket
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Make `import core`, `import services`, `import db`, etc. resolve from this dir
# whether we run as `python backend/server.py` or as a frozen exe.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import events_sse  # noqa: E402
from core import paths  # noqa: E402
from core.events import EventBus  # noqa: E402
from core.first_run import needs_first_run  # noqa: E402
from core.settings import Settings  # noqa: E402
from core.api import API  # noqa: E402
import db as _db_module  # noqa: E402

log = logging.getLogger("sidecar")


# ── Auth middleware ──────────────────────────────────────────────────────────

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject any request that doesn't carry the right Bearer token.

    /health is intentionally token-optional so Electron's pre-ready poller can
    detect liveness before it has been told the token. Every other route must
    present ``Authorization: Bearer <token>``. Electron's main process injects
    that header for the renderer via a webRequest hook, so EventSource works
    without a query-string token (which would otherwise leak into history,
    referers, and access logs).
    """

    def __init__(self, app, *, expected_token: str) -> None:
        super().__init__(app)
        self._expected = expected_token

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/health":
            return await call_next(request)

        supplied = ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()

        if not supplied or not secrets.compare_digest(supplied, self._expected):
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        return await call_next(request)


# ── App container ────────────────────────────────────────────────────────────

class _AppContainer:
    """Holds the shared services that route handlers reach into."""

    def __init__(self, user_data: Path | None) -> None:
        # Honor --user-data so Electron's userData dir wins over platformdirs
        # when bundled inside a packaged installer.
        if user_data is not None:
            user_data.mkdir(parents=True, exist_ok=True)
            os.environ["MYAI_USER_DATA"] = str(user_data)
            # paths.user_dir() reads from platformdirs; for now the existing
            # code path is preserved — userData under platformdirs already
            # matches Electron's convention on Windows. The env var is exposed
            # so future changes can pick it up.

        # Run the v5 → v6 migration before logging is configured (matches the
        # invariant in the legacy main.py).
        paths.migrate_v5_user_dir()
        user_dir = paths.user_dir()
        paths.migrate_legacy_install(_HERE, user_dir)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(paths.log_path(), encoding="utf-8"),
            ],
        )

        self.settings = Settings(paths.settings_path())
        self.bus = EventBus()
        self.api = API(self.settings, self.bus, _HERE, log)

        # ── Power Mode (v3) — additive OpenClaw delegation ────────────────
        # All three handles are constructed unconditionally; their lifecycle
        # work is gated by the user's `power_mode_enabled` setting so v2 users
        # never touch Docker. Failure to construct is non-fatal: routes/docker
        # surfaces a 503 with a plain-English message.
        self.docker = None
        self.execution_bridge = None
        self.execution_classifier = None
        self._docker_watch_stop = threading.Event()
        try:
            from services.docker_manager import DockerManager
            from services.execution_bridge import ExecutionBridge
            from services.hub_router import ExecutionClassifier

            templates_dir = _HERE / "templates"
            self.docker = DockerManager(
                settings=self.settings,
                user_data_dir=paths.user_dir(),
                templates_dir=templates_dir,
                emit=events_sse.publish,
            )
            self.execution_bridge = ExecutionBridge(
                docker_manager=self.docker,
                settings=self.settings,
                emit=events_sse.publish,
            )
            self.execution_classifier = ExecutionClassifier(
                claude_client=getattr(self.api, "_claude", None),
            )

            # Auto-start OpenClaw if the user opted in. Always run on a
            # daemon thread so a missing Docker install can't block sidecar
            # boot.
            if self.settings.get("power_mode_enabled") and self.settings.get("power_mode_autostart"):
                threading.Thread(
                    target=self.docker.start_openclaw,
                    daemon=True,
                    name="power-mode-autostart",
                ).start()

            # Background health watcher — only acts when Power Mode is on.
            threading.Thread(
                target=self.docker.watch_and_restart,
                args=(self._docker_watch_stop,),
                daemon=True,
                name="power-mode-watch",
            ).start()
        except Exception as exc:
            log.warning("Power Mode services failed to initialise: %s", exc, exc_info=True)

        # Match the legacy main.py post-load behavior: kick off heavy services
        # in a background thread so /health responds quickly.
        self.api.start_deferred_init()

    def shutdown(self) -> None:
        try:
            self._docker_watch_stop.set()
        except Exception:
            pass
        if self.execution_bridge is not None:
            try:
                self.execution_bridge.shutdown()
            except Exception as exc:
                log.warning("execution_bridge.shutdown raised: %s", exc, exc_info=True)
        if self.docker is not None:
            try:
                self.docker.shutdown()
            except Exception as exc:
                log.warning("docker.shutdown raised: %s", exc, exc_info=True)
        try:
            self.api.shutdown()
        except Exception as exc:
            log.warning("api.shutdown raised: %s", exc, exc_info=True)


# ── FastAPI factory ──────────────────────────────────────────────────────────

def build_app(token: str, user_data: Path | None) -> tuple[FastAPI, _AppContainer]:
    container = _AppContainer(user_data)

    app = FastAPI(title="iMakeAiTeams Sidecar", version="1.0.0")

    # CORS: Electron's renderer runs on file:// or http://localhost in dev.
    # Allow only localhost origins; the Bearer middleware is the real gate.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "app://-",            # electron-vite production
            "file://",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.add_middleware(BearerAuthMiddleware, expected_token=token)

    # Stash the container on the app so route handlers can reach it via
    # request.app.state.container.
    app.state.container = container
    app.state.shutdown_event = asyncio.Event()

    @app.on_event("startup")
    async def _startup() -> None:
        events_sse.attach_loop(asyncio.get_running_loop())

    # Register routers
    from routes import (
        agents as agents_routes,
        chat as chat_routes,
        docker as docker_routes,
        echo as echo_routes,
        events as events_routes,
        health as health_routes,
        lifecycle as lifecycle_routes,
        mcp as mcp_routes,
        memory as memory_routes,
        prompts as prompts_routes,
        rag as rag_routes,
        settings as settings_routes,
        system as system_routes,
    )

    app.include_router(health_routes.router)
    app.include_router(echo_routes.router, prefix="/api")
    app.include_router(events_routes.router, prefix="/api")
    app.include_router(chat_routes.router, prefix="/api/chat")
    app.include_router(agents_routes.router, prefix="/api/agents")
    app.include_router(memory_routes.router, prefix="/api/memory")
    app.include_router(rag_routes.router, prefix="/api/rag")
    app.include_router(settings_routes.router, prefix="/api/settings")
    app.include_router(mcp_routes.router, prefix="/api/mcp")
    app.include_router(lifecycle_routes.router, prefix="/api/lifecycle")
    app.include_router(prompts_routes.router, prefix="/api/prompts")
    app.include_router(system_routes.router, prefix="/api/system")
    app.include_router(docker_routes.router, prefix="/api/docker")

    @app.post("/shutdown")
    async def _shutdown(request: Request) -> dict:
        container.shutdown()
        request.app.state.shutdown_event.set()
        return {"ok": True}

    return app, container


# ── Port discovery ───────────────────────────────────────────────────────────

def _bind_free_port() -> tuple[socket.socket, int]:
    """Bind a TCP socket on 127.0.0.1:0 and return (socket, assigned_port).

    Uvicorn 0.30+ accepts an `fd=` kwarg so we can hand it the bound socket
    directly, sidestepping the race where another process grabs the port
    between us calling getsockname() and uvicorn calling bind().
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    return sock, port


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="server")
    parser.add_argument("--token", required=True, help="Bearer auth token")
    parser.add_argument("--user-data", default="", help="Override userData dir")
    args = parser.parse_args(argv)

    user_data = Path(args.user_data) if args.user_data else None

    app, container = build_app(args.token, user_data)

    sock, port = _bind_free_port()

    # Print the line Electron's stdout reader is grepping for. Flush
    # immediately — the parent uses line-by-line iteration and a buffered
    # write would deadlock the handshake.
    print(f"PORT={port}", flush=True)

    # Hand uvicorn the already-bound socket via fd= so we never release the
    # port between getsockname() and serve(). Closing + rebinding (the old
    # behavior) had a brief TOCTOU window where another local process could
    # claim the port; advertising a port we no longer hold then made
    # Electron poll a stranger's /health.
    config = uvicorn.Config(
        app,
        fd=sock.fileno(),
        log_level="info",
        access_log=False,
        loop="asyncio",
        http="h11",
        ws="none",
        lifespan="on",
        # No reload — Electron's electron-vite handles HMR for the renderer;
        # the sidecar is restarted explicitly when the user clicks "Restart
        # Backend" or by sending POST /shutdown then respawning.
        reload=False,
    )
    server = uvicorn.Server(config)

    async def _run() -> None:
        # Print READY *after* startup events have fired (CORS + auth middleware
        # registered, services warmed up to first-paint readiness).
        async def _emit_ready() -> None:
            # `serve()` blocks until shutdown; this watcher races with the
            # built-in startup event so we only print when the app is truly
            # serving.
            while not server.started:
                await asyncio.sleep(0.05)
            print("READY", flush=True)

        ready_task = asyncio.create_task(_emit_ready())
        shutdown_task = asyncio.create_task(app.state.shutdown_event.wait())

        serve_task = asyncio.create_task(server.serve())

        # If POST /shutdown fires, ask uvicorn to exit cleanly. If uvicorn
        # exits on its own (parent killed us), we just fall through.
        done, _ = await asyncio.wait(
            {serve_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if shutdown_task in done:
            server.should_exit = True
            await serve_task
        ready_task.cancel()

    # Graceful SIGTERM: uvicorn already handles SIGINT on POSIX, but on
    # Windows the parent uses taskkill /T which sends a CTRL_BREAK — install
    # a no-op handler so the default abort path is replaced with our cleanup.
    def _sig(_signum, _frame):
        try:
            container.shutdown()
        finally:
            os._exit(0)

    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _sig)
        except (OSError, ValueError):
            pass

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        container.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
