"""
tests/test_rate_limiter.py — Tests for the API rate limiter.

Run: pytest tests/test_rate_limiter.py -v
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.rate_limiter import RateLimiter


def test_allows_under_limit():
    rl = RateLimiter(max_calls=3, window_seconds=1.0)
    assert rl.check() is True
    assert rl.check() is True
    assert rl.check() is True
    assert rl.check() is False


def test_blocks_over_limit():
    rl = RateLimiter(max_calls=1, window_seconds=1.0)
    assert rl.check() is True
    assert rl.check() is False
    assert rl.check() is False


def test_window_expiry():
    rl = RateLimiter(max_calls=1, window_seconds=0.1)
    assert rl.check() is True
    assert rl.check() is False
    time.sleep(0.15)
    assert rl.check() is True


def test_reset():
    rl = RateLimiter(max_calls=1, window_seconds=10.0)
    assert rl.check() is True
    assert rl.check() is False
    rl.reset()
    assert rl.check() is True


def test_thread_safety():
    import threading
    rl = RateLimiter(max_calls=100, window_seconds=1.0)
    results = []

    def worker():
        results.append(rl.check())

    threads = [threading.Thread(target=worker) for _ in range(150)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    allowed = sum(1 for r in results if r)
    blocked = sum(1 for r in results if not r)
    assert allowed == 100
    assert blocked == 50
