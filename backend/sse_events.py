"""
sse_events.py — Process-wide event pump that bridges legacy `_emit(event, payload)`
calls (originally PyWebView's `window.__emit`) to a Server-Sent Events stream.

The sidecar runs FastAPI/uvicorn; routes that fan out work to threads (chat
streaming, hardware probing, health checks, etc.) used to call
`window.evaluate_js("window.__emit(...)")` to push results to the renderer.
That window doesn't exist anymore — instead, those events go onto a thread-safe
queue, and the GET /api/events SSE endpoint drains it to whichever EventSource
the renderer has open.

Design notes:
- Single global queue (FIFO). Only one renderer EventSource is expected at a
  time; if we ever support multiple, switch to a fanout broadcaster.
- `publish()` is callable from any thread (worker pool, asyncio loop,
  pywebview-shaped legacy callbacks) and never blocks.
- Backpressure: the queue has a soft cap; if the renderer is gone and events
  pile up, the oldest events are dropped to keep memory bounded.
- Format: each event is `{"event": str, "data": json_serializable}`. The SSE
  endpoint serializes this as `event: <name>\\ndata: <json>\\n\\n`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from threading import Lock
from typing import Any, Deque

log = logging.getLogger("sse_events")

# Cap on backlog when no consumer is attached. ~30 minutes of chat tokens at a
# reasonable token/sec rate fits well below this; anything older is junk.
_MAX_BACKLOG = 4096


class _EventQueue:
    """Thread-safe queue with an asyncio.Event for readers waiting for new items."""

    def __init__(self) -> None:
        self._items: Deque[dict] = deque()
        self._lock = Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._signal: asyncio.Event | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio loop that the SSE endpoint runs on."""
        self._loop = loop
        self._signal = asyncio.Event()

    def publish(self, event: str, payload: Any) -> None:
        """Append (event, payload) to the queue. Safe to call from any thread."""
        try:
            data = payload if payload is not None else {}
            json.dumps(data)  # validate serializable up front
        except (TypeError, ValueError) as exc:
            log.debug("publish: dropping non-serializable payload for %s: %s", event, exc)
            return
        with self._lock:
            self._items.append({"event": event, "data": data})
            while len(self._items) > _MAX_BACKLOG:
                self._items.popleft()
        loop, signal = self._loop, self._signal
        if loop is not None and signal is not None:
            try:
                loop.call_soon_threadsafe(signal.set)
            except RuntimeError:
                pass

    async def drain(self) -> list[dict]:
        """Wait for at least one event, then return everything currently queued."""
        assert self._signal is not None, "attach_loop() must be called first"
        await self._signal.wait()
        with self._lock:
            out = list(self._items)
            self._items.clear()
            self._signal.clear()
        return out

    def snapshot_size(self) -> int:
        with self._lock:
            return len(self._items)


_queue = _EventQueue()


def attach_loop(loop: asyncio.AbstractEventLoop) -> None:
    _queue.attach_loop(loop)


def publish(event: str, payload: Any = None) -> None:
    """Module-level entry point used by the legacy `_emit` shim."""
    _queue.publish(event, payload)


async def drain() -> list[dict]:
    return await _queue.drain()


def queue_size() -> int:
    return _queue.snapshot_size()
