"""
Simple synchronous event bus and typed event dataclasses.

EventBus.publish(event) dispatches to all subscribers registered for
that exact event type (class-based dispatch).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Type


# ── Event base ────────────────────────────────────────────────────────────────

class Event:
    """Base class for all events."""


# ── Concrete event types ──────────────────────────────────────────────────────

@dataclass
class StatusEvent(Event):
    text: str


@dataclass
class LogEvent(Event):
    level: str        # "info" | "warning" | "error"
    text: str


@dataclass
class ProgressEvent(Event):
    label: str
    pct: float         # 0.0 – 100.0


@dataclass
class DownloadUpdateEvent(Event):
    item_id: str
    pct: float
    speed: str = ""
    eta: str = ""
    total_size: int = 0
    downloaded: int = 0


@dataclass
class DownloadDoneEvent(Event):
    item_id: str
    filename: str
    dest: str


@dataclass
class DownloadErrorEvent(Event):
    item_id: str
    text: str


@dataclass
class ChatTokenEvent(Event):
    item_id: str
    token: str


@dataclass
class ChatDoneEvent(Event):
    item_id: str
    full_text: str


@dataclass
class ChatErrorEvent(Event):
    item_id: str
    error: str


@dataclass
class RAGBuiltEvent(Event):
    chunk_count: int


@dataclass
class ThinkingResultEvent(Event):
    thinking: str
    answer: str


# ── Bus ───────────────────────────────────────────────────────────────────────

class EventBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._handlers: Dict[Type[Event], List[Callable]] = {}
        self._attached: List[Any] = []

    def subscribe(self, event_type: Type[Event], handler: Callable) -> None:
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._handlers.get(type(event), []))
        for h in handlers:
            try:
                h(event)
            except Exception:
                pass

    def emit(self, event_name: str, data: Any = None) -> None:
        """String-keyed convenience method for lightweight events.

        Dispatches to handlers registered under the string key (not a class).
        Used by ChannelManager for channel_response / channel_status events.
        """
        with self._lock:
            handlers = list(self._handlers.get(event_name, []))
        for h in handlers:
            try:
                h(data)
            except Exception:
                pass

    def subscribe_event(self, event_name: str, handler: Callable) -> None:
        """Subscribe to a string-keyed event emitted via emit()."""
        with self._lock:
            self._handlers.setdefault(event_name, []).append(handler)

    def attach(self, obj: Any) -> None:
        """Store a reference to an object (e.g. the main window) for later use."""
        self._attached.append(obj)
