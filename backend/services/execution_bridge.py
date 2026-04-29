"""
services/execution_bridge.py — Power Mode (v3) bridge to OpenClaw's task API.

Translates a chat message into an OpenClaw task, opens a streaming connection
to the OpenClaw gateway, and re-emits OpenClaw events through the existing
``sse_events`` pipeline so ChatView.tsx can render them as execution cards.

Event mapping (OpenClaw → renderer):
    thinking      → power_mode_step  {kind:"thinking",  ...}
    tool_call     → power_mode_step  {kind:"tool_call", ...}
    file_write    → power_mode_step  {kind:"file_write", ...}
    shell_command → power_mode_step  {kind:"shell",     ...}
    web           → power_mode_step  {kind:"web",       ...}
    approval      → power_mode_approval {token, ...}
    result        → power_mode_message  {text}
    error         → power_mode_error    {error}
    done          → power_mode_done

The bridge is single-task per conversation: a new send cancels any in-flight
task on that conversation. Approval prompts auto-deny after 60 s.

This module is *additive*: it never invokes the existing chat orchestrator,
agent registry, or memory system, and it never runs unless Power Mode has been
explicitly enabled and OpenClaw has reported healthy.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import sse_events

log = logging.getLogger("MyAIEnv.execution_bridge")


APPROVAL_TIMEOUT_SEC = 60.0
STREAM_CONNECT_TIMEOUT_SEC = 10.0
HISTORY_TURNS = 5  # Last N messages of context to send with the task.


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionTask:
    task_id: str
    conversation_id: str
    started_at: float
    cancel_event: threading.Event = field(default_factory=threading.Event)
    pending_approvals: dict[str, threading.Event] = field(default_factory=dict)
    pending_decisions: dict[str, str] = field(default_factory=dict)
    openclaw_task_id: str = ""


# ── Bridge ───────────────────────────────────────────────────────────────────

class ExecutionBridge:
    """Owns the lifecycle of in-flight Power Mode tasks.

    Construction is cheap; the heavy work (HTTP client, streaming) happens
    inside ``submit`` on a daemon worker thread.
    """

    def __init__(
        self,
        docker_manager,
        settings,
        emit: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self._docker = docker_manager
        self._settings = settings
        self._emit = emit or sse_events.publish
        self._lock = threading.Lock()
        self._tasks: dict[str, ExecutionTask] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def submit(
        self,
        conversation_id: str,
        user_message: str,
        history: Optional[list[dict]] = None,
    ) -> dict:
        """Queue a task for OpenClaw and return immediately.

        Streaming results arrive on the existing /api/events SSE channel. The
        return value is a tiny ack so the renderer can correlate steps with a
        task_id.
        """
        if not self._settings.get("power_mode_enabled"):
            err = "Power Mode is disabled. Enable it in Settings to delegate execution."
            self._emit("power_mode_error", {
                "conversation_id": conversation_id, "error": err,
            })
            return {"ok": False, "error": err}

        if not self._docker.health_check():
            err = ("OpenClaw isn't running. Open Settings → Power Mode and "
                   "click Start to bring it up.")
            self._emit("power_mode_error", {
                "conversation_id": conversation_id, "error": err,
            })
            return {"ok": False, "error": err}

        # Cancel any existing task on this conversation so steps don't collide.
        with self._lock:
            stale = [t for t in self._tasks.values() if t.conversation_id == conversation_id]
        for t in stale:
            self._cancel_task(t, reason="superseded by new request")

        task = ExecutionTask(
            task_id=f"pm_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            started_at=time.time(),
        )
        with self._lock:
            self._tasks[task.task_id] = task

        self._emit("power_mode_started", {
            "task_id": task.task_id,
            "conversation_id": conversation_id,
            "message": user_message,
        })

        thread = threading.Thread(
            target=self._run_task,
            args=(task, user_message, history or []),
            daemon=True,
            name=f"power-mode-{task.task_id}",
        )
        thread.start()
        return {"ok": True, "task_id": task.task_id}

    def cancel(self, task_id: str) -> dict:
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return {"ok": False, "error": "unknown task_id"}
        self._cancel_task(task, reason="cancelled by user")
        return {"ok": True}

    def approve(self, task_id: str, approval_id: str, allow: bool) -> dict:
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return {"ok": False, "error": "unknown task_id"}
        evt = task.pending_approvals.get(approval_id)
        if not evt:
            return {"ok": False, "error": "unknown approval_id"}
        task.pending_decisions[approval_id] = "allow" if allow else "deny"
        evt.set()
        return {"ok": True}

    def shutdown(self) -> None:
        with self._lock:
            tasks = list(self._tasks.values())
        for t in tasks:
            self._cancel_task(t, reason="sidecar shutting down")

    # ── Internals ───────────────────────────────────────────────────────────

    def _cancel_task(self, task: ExecutionTask, *, reason: str) -> None:
        if task.cancel_event.is_set():
            return
        task.cancel_event.set()
        # Unblock any pending approval waits.
        for evt in list(task.pending_approvals.values()):
            evt.set()
        # Best-effort cancel call to OpenClaw.
        if task.openclaw_task_id:
            try:
                import httpx
                with httpx.Client(timeout=5.0) as client:
                    client.post(
                        f"{self._docker.gateway_url()}/tasks/"
                        f"{task.openclaw_task_id}/cancel",
                        headers=self._auth_headers(),
                    )
            except Exception as exc:
                log.debug("openclaw cancel failed: %s", exc)
        self._emit("power_mode_done", {
            "task_id": task.task_id,
            "conversation_id": task.conversation_id,
            "cancelled": True,
            "reason": reason,
        })
        with self._lock:
            self._tasks.pop(task.task_id, None)

    def _run_task(
        self,
        task: ExecutionTask,
        user_message: str,
        history: list[dict],
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            self._emit("power_mode_error", {
                "task_id": task.task_id,
                "conversation_id": task.conversation_id,
                "error": ("httpx is missing from the sidecar. Reinstall the "
                          f"backend dependencies. ({exc})"),
            })
            with self._lock:
                self._tasks.pop(task.task_id, None)
            return

        body = self._build_request_body(user_message, history)
        url_post = f"{self._docker.gateway_url()}/tasks"

        try:
            with httpx.Client(timeout=httpx.Timeout(STREAM_CONNECT_TIMEOUT_SEC,
                                                   read=None)) as client:
                resp = client.post(url_post, json=body, headers=self._auth_headers())
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"OpenClaw rejected the task ({resp.status_code}): "
                        f"{resp.text.strip() or 'no detail'}"
                    )
                payload = resp.json()
                task.openclaw_task_id = str(payload.get("task_id") or payload.get("id") or "")
                if not task.openclaw_task_id:
                    raise RuntimeError("OpenClaw response missing task_id")

                stream_url = (
                    f"{self._docker.gateway_url()}/tasks/"
                    f"{task.openclaw_task_id}/stream"
                )
                with client.stream("GET", stream_url,
                                   headers=self._auth_headers(),
                                   timeout=httpx.Timeout(None)) as s:
                    if s.status_code >= 400:
                        raise RuntimeError(
                            f"OpenClaw stream rejected ({s.status_code})"
                        )
                    self._consume_stream(task, s)

        except Exception as exc:
            if not task.cancel_event.is_set():
                self._emit("power_mode_error", {
                    "task_id": task.task_id,
                    "conversation_id": task.conversation_id,
                    "error": _humanize(exc),
                })
        finally:
            if not task.cancel_event.is_set():
                self._emit("power_mode_done", {
                    "task_id": task.task_id,
                    "conversation_id": task.conversation_id,
                    "cancelled": False,
                })
            with self._lock:
                self._tasks.pop(task.task_id, None)

    def _consume_stream(self, task: ExecutionTask, response) -> None:
        """Parse OpenClaw's NDJSON / SSE-ish stream and re-emit each event.

        OpenClaw advertises an NDJSON stream where each line is a complete JSON
        object. We accept either ``data: {...}`` SSE framing or bare NDJSON;
        anything we can't parse is logged and skipped (never crashes the loop).
        """
        for raw in response.iter_lines():
            if task.cancel_event.is_set():
                return
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line.startswith(":"):
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                log.debug("execution_bridge: dropped malformed line: %s", line[:200])
                continue
            self._dispatch_event(task, evt)

    def _dispatch_event(self, task: ExecutionTask, evt: dict) -> None:
        kind = str(evt.get("type") or evt.get("event") or "").lower()
        base = {
            "task_id": task.task_id,
            "conversation_id": task.conversation_id,
            "step_id": str(evt.get("step_id") or evt.get("id") or
                           f"step_{uuid.uuid4().hex[:8]}"),
        }

        if kind in ("thinking", "plan", "planning"):
            self._emit("power_mode_step", {**base, "kind": "thinking",
                                           "title": evt.get("title", "Planning"),
                                           "detail": evt.get("text", ""),
                                           "status": evt.get("status", "running")})
            return
        if kind in ("tool_call", "tool"):
            self._emit("power_mode_step", {**base, "kind": "tool_call",
                                           "title": evt.get("name", "Tool"),
                                           "args": evt.get("args"),
                                           "result": evt.get("result"),
                                           "status": evt.get("status", "running")})
            return
        if kind in ("file_write", "file_op", "file"):
            self._emit("power_mode_step", {**base, "kind": "file_write",
                                           "path": evt.get("path", ""),
                                           "preview": evt.get("preview", ""),
                                           "bytes": evt.get("bytes"),
                                           "status": evt.get("status", "done")})
            return
        if kind in ("shell", "shell_command", "command"):
            self._emit("power_mode_step", {**base, "kind": "shell",
                                           "command": evt.get("command", ""),
                                           "stdout": evt.get("stdout", ""),
                                           "stderr": evt.get("stderr", ""),
                                           "exit_code": evt.get("exit_code"),
                                           "status": evt.get("status", "done")})
            return
        if kind in ("web", "browse", "browser"):
            self._emit("power_mode_step", {**base, "kind": "web",
                                           "url": evt.get("url", ""),
                                           "title": evt.get("title", ""),
                                           "summary": evt.get("summary", ""),
                                           "status": evt.get("status", "done")})
            return
        if kind in ("approval", "approval_request"):
            self._handle_approval(task, evt)
            return
        if kind in ("result", "message", "final"):
            self._emit("power_mode_message", {**base,
                                              "text": evt.get("text", ""),
                                              "model": evt.get("model")})
            return
        if kind in ("error", "failure"):
            self._emit("power_mode_error", {**base,
                                            "error": _humanize(evt.get("error") or
                                                               evt.get("message") or
                                                               "Unknown error")})
            return
        if kind in ("done", "complete", "completed"):
            # Stream-level done; emitted by run_task wrapper.
            return

        # Unknown — pass through so the UI can render a generic step.
        self._emit("power_mode_step", {**base, "kind": "other",
                                       "title": kind or "step",
                                       "detail": json.dumps(evt)[:512],
                                       "status": evt.get("status", "done")})

    def _handle_approval(self, task: ExecutionTask, evt: dict) -> None:
        """Block until the user clicks Allow/Deny or the timeout elapses."""
        approval_id = str(evt.get("approval_id") or evt.get("id") or
                          f"appr_{uuid.uuid4().hex[:8]}")
        wait_evt = threading.Event()
        task.pending_approvals[approval_id] = wait_evt

        self._emit("power_mode_approval", {
            "task_id": task.task_id,
            "conversation_id": task.conversation_id,
            "approval_id": approval_id,
            "summary": evt.get("summary", ""),
            "details": evt.get("details", {}),
            "danger": evt.get("danger", "medium"),
            "timeout_sec": APPROVAL_TIMEOUT_SEC,
        })

        wait_evt.wait(APPROVAL_TIMEOUT_SEC)

        decision = task.pending_decisions.get(approval_id)
        task.pending_approvals.pop(approval_id, None)
        if decision is None:
            decision = "deny"
            self._emit("power_mode_event", {
                "task_id": task.task_id,
                "approval_id": approval_id,
                "phase": "approval_timeout",
                "message": "Approval timed out — auto-denied.",
            })

        try:
            import httpx
            with httpx.Client(timeout=10.0) as client:
                client.post(
                    f"{self._docker.gateway_url()}/tasks/"
                    f"{task.openclaw_task_id}/approve",
                    json={"approval_id": approval_id, "decision": decision},
                    headers=self._auth_headers(),
                )
        except Exception as exc:
            log.warning("openclaw approve POST failed: %s", exc)
            self._emit("power_mode_error", {
                "task_id": task.task_id,
                "conversation_id": task.conversation_id,
                "error": f"Could not send approval to OpenClaw: {_humanize(exc)}",
            })

    def _build_request_body(self, user_message: str, history: list[dict]) -> dict:
        trimmed: list[dict] = []
        for row in history[-HISTORY_TURNS:]:
            role = row.get("role")
            text = row.get("content") or row.get("text") or ""
            if role and text:
                trimmed.append({"role": role, "content": str(text)[:8000]})
        return {
            "task": user_message,
            "context": trimmed,
            "workspace": str(self._docker.workspace_dir()),
            "require_approval": True,
            "client": "imakeaiteams-v3",
        }

    def _auth_headers(self) -> dict:
        # All traffic to OpenClaw goes through a localhost-only Caddy gateway
        # that requires `Authorization: Bearer <secret>`. The secret is
        # generated and persisted by docker_manager at compose-render time
        # and read back from disk; if it's missing here the gateway isn't
        # running and the request will fail at the TCP layer regardless.
        headers = {"X-Client": "imakeaiteams"}
        token = self._docker.gateway_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


# ── Helpers ──────────────────────────────────────────────────────────────────

def _humanize(exc: Any) -> str:
    if isinstance(exc, Exception):
        # Translate the most common networking failures into plain English so
        # the chat surface doesn't dump a stack trace at the user.
        name = type(exc).__name__
        msg = str(exc) or name
        if "ConnectError" in name or "Connection refused" in msg:
            return ("Couldn't reach OpenClaw. Open Settings → Power Mode and "
                    "make sure the gateway is running.")
        if "ReadTimeout" in name or "TimeoutException" in name:
            return ("OpenClaw stopped responding. Try again, or restart it "
                    "from Settings → Power Mode.")
        return f"{name}: {msg}"
    return str(exc)
