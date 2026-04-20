"""
core/service_guard.py — Short-circuit decorator for API methods whose backing
service may have failed during API._safe_init.

When a service reports ok=False in self._status, decorated methods return a
caller-supplied ``default`` (matching the method's documented return type)
and emit a ``service_unavailable`` event so the frontend can show a toast.

This lives in its own module so tests can import it without pulling in the
full ``core.api`` dependency chain (anthropic, pywebview, etc.).
"""

from __future__ import annotations

import functools
from typing import Any, Callable


def requires(service_name: str, default: Any = None) -> Callable:
    """Decorate an API method with a service-availability short-circuit.

    Usage::

        @requires("chat_orchestrator", default=[])
        def chat_list_conversations(self, limit=30):
            return self._chat.list_conversations(limit=limit)

    If ``self._status[service_name].ok`` is False (or the entry is missing),
    the method body is not executed, ``default`` is returned, and a
    ``service_unavailable`` event carrying the failing service name, the
    stored error string, and the method name is emitted via ``self._emit``.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            status = self._status.get(service_name, {})
            if not status.get("ok"):
                self._emit("service_unavailable", {
                    "service": service_name,
                    "error": status.get("error"),
                    "method": fn.__name__,
                })
                return default
            return fn(self, *args, **kwargs)
        return wrapped
    return deco
