"""
End-to-end smoke tests — drive the full app through the real PyWebView
bridge and verify a chat message round-trips to a response.

Two tests:
  - test_smoke_source:    spawns ``python app/main.py`` directly.
  - test_smoke_packaged:  spawns the PyInstaller-built binary.

Both share the same driver. The app side is instrumented by
``app/core/smoke_harness.py``, which activates only when
MYAI_SMOKE_TEST=1 — normal launches are unaffected.

Network and the Anthropic SDK are never reached: the harness stubs
ChatOrchestrator.send() with a canned ChatResult.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "app"
MAIN_PY = APP_DIR / "main.py"

LOADED_TIMEOUT_S = 60.0
RESPONSE_TIMEOUT_S = 30.0
SHUTDOWN_TIMEOUT_S = 15.0

EXPECTED_REPLY = "smoke-ok"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _isolated_env(smoke_dir: Path) -> dict:
    """Build a subprocess env that isolates user-data paths into smoke_dir."""
    env = os.environ.copy()
    env["MYAI_SMOKE_TEST"] = "1"
    env["MYAI_SMOKE_DIR"] = str(smoke_dir)

    home = smoke_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home)
    # Windows user-data roots (paths.user_dir uses these)
    env["APPDATA"] = str(home / "AppData" / "Roaming")
    env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    # Linux XDG roots
    env["XDG_DATA_HOME"] = str(home / ".local" / "share")
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    return env


def _maybe_xvfb(argv: list[str]) -> list[str]:
    """On headless Linux, prepend xvfb-run so the webview has a display."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        xvfb = shutil.which("xvfb-run")
        if xvfb is None:
            pytest.skip("Linux without $DISPLAY and no xvfb-run on PATH")
        return [xvfb, "-a", "--"] + argv
    return argv


def _read_events(events_path: Path) -> list[dict]:
    if not events_path.exists():
        return []
    out = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _find_event(events: list[dict], name: str) -> Optional[dict]:
    for e in events:
        if e.get("event") == name:
            return e
    return None


def _dump_tail(path: Path, head: int = 50) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"<unreadable: {exc}>"
    lines = data.splitlines()
    if len(lines) <= head:
        return data
    return "\n".join(lines[-head:])


def _drive_smoke(argv: list[str], cwd: Path) -> dict:
    """Run the full handshake against ``argv``. Returns the chat_done payload."""
    with tempfile.TemporaryDirectory(prefix="myai-smoke-") as td:
        smoke_dir = Path(td)
        env = _isolated_env(smoke_dir)
        events_path = smoke_dir / "events.jsonl"
        loaded_flag = smoke_dir / "loaded.flag"
        command_path = smoke_dir / "command.json"

        full_argv = _maybe_xvfb(argv)

        proc = subprocess.Popen(
            full_argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        def _fail(msg: str) -> "pytest.fail":
            try:
                out, err = proc.communicate(timeout=3)
                out_s = out.decode("utf-8", errors="replace")
                err_s = err.decode("utf-8", errors="replace")
            except Exception:
                out_s = err_s = "<no output captured>"
            events_tail = _dump_tail(events_path)
            return pytest.fail(
                f"{msg}\n"
                f"---- stdout ----\n{out_s}\n"
                f"---- stderr ----\n{err_s}\n"
                f"---- events tail ----\n{events_tail}\n",
                pytrace=False,
            )

        try:
            # 1. Wait for the loaded event.
            deadline = time.monotonic() + LOADED_TIMEOUT_S
            while time.monotonic() < deadline:
                if loaded_flag.exists():
                    break
                if proc.poll() is not None:
                    _fail(f"App exited before loaded (rc={proc.returncode}).")
                time.sleep(0.25)
            else:
                _fail(f"loaded.flag never appeared within {LOADED_TIMEOUT_S}s.")

            # 2. Drive a chat message through the real pywebview.api bridge.
            command_path.write_text(
                json.dumps({
                    "method": "chat_send",
                    "args": ["smoke-cid", "hello from the smoke test", ""],
                }),
                encoding="utf-8",
            )

            # 3. Tail events until chat_done arrives.
            deadline = time.monotonic() + RESPONSE_TIMEOUT_S
            chat_done = None
            while time.monotonic() < deadline:
                chat_done = _find_event(_read_events(events_path), "chat_done")
                if chat_done is not None:
                    break
                if proc.poll() is not None:
                    _fail(f"App crashed while awaiting chat_done (rc={proc.returncode}).")
                time.sleep(0.15)
            if chat_done is None:
                _fail(f"chat_done event never appeared within {RESPONSE_TIMEOUT_S}s.")

            payload = chat_done.get("payload") or {}
            return payload
        finally:
            # Graceful shutdown, then hard-kill as backstop.
            try:
                command_path.write_text(
                    json.dumps({"method": "_shutdown", "args": []}),
                    encoding="utf-8",
                )
            except Exception:
                pass
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass


# ── Locate packaged binary ────────────────────────────────────────────────────

def _packaged_binary() -> Optional[Path]:
    env_override = os.environ.get("MYAI_PACKAGED_BINARY", "").strip()
    if env_override:
        p = Path(env_override)
        return p if p.exists() else None

    dist = REPO_ROOT / "dist"
    candidates: list[Path] = []
    if sys.platform == "win32":
        candidates += [
            dist / "MyAIAgentHub" / "MyAIAgentHub.exe",
            dist / "MyAIAgentHub-lite" / "MyAIAgentHub-lite.exe",
        ]
    elif sys.platform == "darwin":
        candidates += [
            dist / "MyAI Agent Hub.app" / "Contents" / "MacOS" / "MyAIAgentHub",
            dist / "MyAI Agent Hub Lite.app" / "Contents" / "MacOS" / "MyAIAgentHub-lite",
        ]
    else:
        candidates += [
            dist / "MyAIAgentHub" / "MyAIAgentHub",
            dist / "MyAIAgentHub-lite" / "MyAIAgentHub-lite",
        ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_smoke_source():
    """End-to-end: spawn main.py, wait for loaded, send a chat message,
    assert chat_done comes back with the expected payload."""
    if not MAIN_PY.exists():
        pytest.skip(f"{MAIN_PY} not found")
    payload = _drive_smoke([sys.executable, str(MAIN_PY)], cwd=APP_DIR)
    assert payload.get("text") == EXPECTED_REPLY, (
        f"expected text={EXPECTED_REPLY!r}, got payload={payload!r}"
    )
    assert payload.get("conversation_id") == "smoke-cid"


def test_smoke_packaged():
    """End-to-end against the PyInstaller build — catches missing modules,
    wrong model paths, broken --collect-all entries, etc."""
    binary = _packaged_binary()
    if binary is None:
        pytest.skip(
            "packaged binary not found. Set $MYAI_PACKAGED_BINARY or run "
            "`pyinstaller build/MyAIAgentHub.spec --noconfirm --clean` first."
        )
    payload = _drive_smoke([str(binary)], cwd=binary.parent)
    assert payload.get("text") == EXPECTED_REPLY, (
        f"expected text={EXPECTED_REPLY!r}, got payload={payload!r}"
    )
    assert payload.get("conversation_id") == "smoke-cid"
