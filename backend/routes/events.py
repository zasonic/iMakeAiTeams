"""GET /api/events — Server-Sent Events stream of backend → renderer events.

Replaces the old PyWebView `window.__emit(...)` mechanism. The renderer opens
one EventSource at boot and consumes events forever; on disconnect it
auto-reconnects. The token is supplied as a query string (`?token=...`)
because EventSource doesn't allow custom headers.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import events_sse

router = APIRouter()


def _format_sse(event: str, data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    # Each SSE message is `event: <name>\ndata: <json>\n\n`.
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


_KEEPALIVE_INTERVAL_S = 20.0
_KEEPALIVE_FRAME = b": keepalive\n\n"


@router.get("/events")
async def events_stream() -> StreamingResponse:
    async def _gen():
        # Send a hello frame so the renderer knows the stream is live before
        # anything is published.
        yield _format_sse("hello", {"queue_size": events_sse.queue_size()})
        try:
            while True:
                # Wait for a publish OR the keepalive timeout, whichever
                # comes first. The keepalive is an SSE comment line — valid
                # by spec, ignored by EventSource — but it forces a write
                # through the socket so intermediate proxies / Chromium's
                # idle-stream detector don't decide the connection is dead.
                try:
                    batch = await asyncio.wait_for(
                        events_sse.drain(),
                        timeout=_KEEPALIVE_INTERVAL_S,
                    )
                except asyncio.TimeoutError:
                    yield _KEEPALIVE_FRAME
                    continue
                for item in batch:
                    yield _format_sse(item["event"], item["data"])
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
