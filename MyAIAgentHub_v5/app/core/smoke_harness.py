"""
core/smoke_harness.py — Opt-in test hook for end-to-end smoke tests.

Activates only when MYAI_SMOKE_TEST=1. Otherwise ``install()`` is a no-op
and the app runs completely normally. See tests/test_smoke_end_to_end.py
for the driver.

Protocol (all under MYAI_SMOKE_DIR):
  - events.jsonl   Every API._emit call, appended line-by-line as JSON.
  - loaded.flag    Written after window.events.loaded fires.
  - command.json   Driver writes one of:
                     {"method": "<api_method>", "args": [...]}   — dispatched via
                                                                   window.evaluate_js
                                                                   so the real JS↔Py
                                                                   bridge is exercised
                     {"method": "_shutdown", "args": []}         — window.destroy()
                   File is deleted after dispatch.

The harness also replaces api._chat.send with a stub that returns a
deterministic ChatResult. This keeps the entire chat_send worker path
(rate limiter, sanitizer, event emission, run_in_thread) live — only the
LLM call itself is bypassed so tests don't need network or API keys.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path


_log = logging.getLogger("MyAIAgentHub.smoke")

SMOKE_ENV = "MYAI_SMOKE_TEST"
SMOKE_DIR_ENV = "MYAI_SMOKE_DIR"
SMOKE_REPLY = "smoke-ok"


def install(api, window) -> None:
    """Wire up the smoke-test hooks. No-op unless MYAI_SMOKE_TEST=1."""
    if os.environ.get(SMOKE_ENV, "").lower() not in ("1", "true", "yes"):
        return

    smoke_dir = os.environ.get(SMOKE_DIR_ENV, "").strip()
    if not smoke_dir:
        _log.warning("smoke harness active but %s is empty — disabling", SMOKE_DIR_ENV)
        return

    dir_ = Path(smoke_dir)
    dir_.mkdir(parents=True, exist_ok=True)
    events_path = dir_ / "events.jsonl"
    loaded_flag = dir_ / "loaded.flag"
    command_path = dir_ / "command.json"

    write_lock = threading.Lock()

    # ── 1. Stub the LLM call ──────────────────────────────────────────────────
    _install_chat_stub(api)

    # ── 2. Wrap _emit so every event is also tee'd to events.jsonl ────────────
    original_emit = api._emit

    def _tee_emit(event, payload=None):
        try:
            with write_lock:
                with events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {"event": event, "payload": payload, "ts": time.time()},
                        default=str,
                    ) + "\n")
        except Exception as exc:
            _log.debug("smoke tee failed for %r: %s", event, exc)
        return original_emit(event, payload)

    api._emit = _tee_emit

    # ── 3. On window loaded, drop the flag and start the command poller ──────
    def _on_smoke_loaded():
        try:
            loaded_flag.write_text("ok", encoding="utf-8")
        except Exception as exc:
            _log.warning("could not write loaded.flag: %s", exc)
        threading.Thread(
            target=_command_poller,
            args=(window, command_path, events_path, write_lock),
            daemon=True,
            name="smoke-cmd-poller",
        ).start()

    window.events.loaded += _on_smoke_loaded
    _log.info("smoke harness installed; dir=%s", dir_)


def _install_chat_stub(api) -> None:
    """Replace api._chat.send with a canned ChatResult returner."""
    if getattr(api, "_chat", None) is None:
        _log.warning("smoke harness: api._chat is None, cannot stub send()")
        return

    from models import ChatResult

    def _stub_send(conversation_id, user_message, agent_id=None,
                   on_token=None, on_event=None, **_kwargs):
        # Exercise the token path so chat_token events still fire end-to-end.
        if on_token is not None:
            try:
                on_token(SMOKE_REPLY)
            except Exception:
                pass
        return ChatResult(
            text=SMOKE_REPLY,
            model="smoke",
            route_reason="smoke-harness-stub",
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            message_id="smoke-msg",
            budget_warning="",
        )

    api._chat.send = _stub_send


def _command_poller(window, command_path: Path, events_path: Path,
                    write_lock: threading.Lock) -> None:
    """Watch for command.json, dispatch the method through window.pywebview.api."""
    while True:
        try:
            if command_path.exists():
                raw = command_path.read_text(encoding="utf-8")
                try:
                    command_path.unlink()
                except Exception:
                    pass
                try:
                    cmd = json.loads(raw)
                except Exception as exc:
                    _log_error(events_path, write_lock, f"bad command json: {exc}")
                    continue

                method = str(cmd.get("method", ""))
                args = cmd.get("args", []) or []

                if method == "_shutdown":
                    try:
                        window.destroy()
                    except Exception as exc:
                        _log_error(events_path, write_lock, f"destroy failed: {exc}")
                    return

                if not method or not method.replace("_", "").isalnum():
                    _log_error(events_path, write_lock, f"rejected method: {method!r}")
                    continue

                args_js = ", ".join(json.dumps(a) for a in args)
                js = f"window.pywebview.api.{method}({args_js});"
                try:
                    window.evaluate_js(js)
                except Exception as exc:
                    _log_error(events_path, write_lock, f"evaluate_js failed: {exc}")
        except Exception as exc:
            _log_error(events_path, write_lock, f"poller loop error: {exc}")
        time.sleep(0.1)


def _log_error(events_path: Path, write_lock: threading.Lock, msg: str) -> None:
    try:
        with write_lock:
            with events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"event": "_smoke_error", "payload": {"error": msg},
                     "ts": time.time()},
                ) + "\n")
    except Exception:
        pass
