"""
services/rate_limiter.py — In-memory rate limiter for the API bridge.

Prevents runaway API costs from buggy or malicious frontend behavior.
Thread-safe. No external dependencies.

Usage in api.py:
    from services.rate_limiter import rate_limit_chat, rate_limit

    class API:
        @rate_limit_chat
        def chat_send(self, conversation_id, message, agent_id=None):
            ...
"""

import time
import threading
from collections import deque
from functools import wraps


class RateLimiter:
    """Sliding-window rate limiter. Thread-safe."""

    def __init__(self, max_calls: int, window_seconds: float):
        self._max = max_calls
        self._window = window_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self) -> bool:
        """Return True if the call is allowed, False if rate-limited."""
        now = time.monotonic()
        with self._lock:
            while self._calls and self._calls[0] < now - self._window:
                self._calls.popleft()
            if len(self._calls) >= self._max:
                return False
            self._calls.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


# ── Pre-configured limiters ──────────────────────────────────────────────────

_chat_limiter = RateLimiter(max_calls=10, window_seconds=60)     # 10 chats/min
_general_limiter = RateLimiter(max_calls=120, window_seconds=60) # 120 calls/min


def rate_limit_chat(func):
    """Decorator for chat-send API methods. 10 calls/minute."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _chat_limiter.check():
            return {"error": "Rate limited — too many messages. Please wait a moment."}
        return func(self, *args, **kwargs)
    return wrapper


def rate_limit(func):
    """Decorator for general API methods. 120 calls/minute."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _general_limiter.check():
            return {"error": "Rate limited. Please slow down."}
        return func(self, *args, **kwargs)
    return wrapper
