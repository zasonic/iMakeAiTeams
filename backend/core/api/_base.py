"""
core/api/_base.py — Shared base class for domain sub-APIs.

Each domain module (chat, agents, memory, rag, settings) exposes a class
whose instances hold a reference to the facade `API` instance. Attribute
access is forwarded to the facade via `__getattr__`, so method bodies can
use `self._claude`, `self._settings`, `self._emit`, etc. unchanged.

The facade owns the canonical state (service handles, settings, status dict,
event emitter). Sub-APIs never shadow those attributes — they only call
through. This keeps a single source of truth and matches how the original
monolithic API class was structured.
"""

from __future__ import annotations


class BaseAPI:
    """Sub-APIs inherit this and gain facade-attribute passthrough.

    The `@_requires` and rate-limit decorators depend on `self._status` and
    `self._emit`; both resolve via `__getattr__` to the facade's live values.
    """

    def __init__(self, facade):
        # Bypass __getattr__ so storing the facade reference never recurses.
        object.__setattr__(self, "_facade", facade)

    def __getattr__(self, name):
        # Only called when normal lookup misses — safe to proxy every read
        # (service handles, settings, emit callable) to the facade.
        return getattr(self._facade, name)
