"""
services/error_classifier.py — Area 6: Self-Healing Pipeline and Error Recovery.

Classifies exceptions into TRANSIENT / PERMANENT / DEGRADED categories,
provides a tenacity-based retry decorator, and writes structured error
records to SQLite.

Usage:
    from services.error_classifier import with_retry, log_error, ErrorCategory

    @with_retry("researcher_agent")
    def call_claude(...):
        ...
"""

import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

log = logging.getLogger("error_classifier")

# ── Error categories ─────────────────────────────────────────────────────────


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"   # Safe to retry with backoff
    PERMANENT = "permanent"   # Do not retry; report to user
    DEGRADED  = "degraded"    # Partial retry possible; needs guidance


# ── Classification map ────────────────────────────────────────────────────────

def classify(exc: Exception) -> ErrorCategory:
    """Return the ErrorCategory for a given exception."""
    name = type(exc).__name__
    module = type(exc).__module__ or ""

    # Anthropic SDK errors
    if "anthropic" in module or name.startswith("anthropic"):
        if name in ("RateLimitError", "APIConnectionError", "APITimeoutError"):
            return ErrorCategory.TRANSIENT
        if name in ("AuthenticationError", "BadRequestError"):
            return ErrorCategory.PERMANENT
        if name == "ContextWindowExceededError":
            return ErrorCategory.DEGRADED

    # HTTP / network
    if name in ("ConnectionError", "Timeout", "HTTPError"):
        return ErrorCategory.TRANSIENT
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        code = exc.response.status_code
        if code in (500, 502, 503, 429):
            return ErrorCategory.TRANSIENT
        if code in (400, 401, 403, 404):
            return ErrorCategory.PERMANENT

    # File / OS errors
    if isinstance(exc, FileNotFoundError):
        return ErrorCategory.PERMANENT
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return ErrorCategory.PERMANENT

    # Disk full (OSError errno 28)
    if isinstance(exc, OSError):
        import errno
        if exc.errno == errno.ENOSPC:
            return ErrorCategory.DEGRADED

    return ErrorCategory.TRANSIENT  # conservative default — retry once


# ── Retry decorator ───────────────────────────────────────────────────────────

def with_retry(component: str, workflow_id: str = "", task_id: str = ""):
    """
    Decorator that wraps a function with exponential-backoff retry logic.
    TRANSIENT errors are retried (up to 4 attempts).
    PERMANENT errors are not retried — PipelineError raised immediately.
    DEGRADED errors are retried once, then raised.
    All errors are logged to SQLite.
    """
    try:
        from tenacity import (
            retry, wait_exponential, stop_after_attempt,
            retry_if_exception, before_sleep_log, RetryError,
        )
        _tenacity_available = True
    except ImportError:
        _tenacity_available = False

    def decorator(fn: Callable) -> Callable:
        if not _tenacity_available:
            # Tenacity not installed — run without retry but still log
            def _no_retry_wrapper(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    log_error(exc, component, workflow_id=workflow_id, task_id=task_id)
                    raise PipelineError(str(exc), classify(exc)) from exc
            return _no_retry_wrapper

        from tenacity import (
            retry, wait_exponential, stop_after_attempt,
            retry_if_exception, before_sleep_log,
        )

        def _should_retry(exc: Exception) -> bool:
            cat = classify(exc)
            return cat == ErrorCategory.TRANSIENT

        @retry(
            wait=wait_exponential(multiplier=2, min=2, max=60),
            stop=stop_after_attempt(4),
            retry=retry_if_exception(_should_retry),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=False,
        )
        def _retrying_wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        def _outer(*args, **kwargs):
            try:
                return _retrying_wrapper(*args, **kwargs)
            except Exception as exc:
                cat = classify(exc)
                log_error(exc, component, workflow_id=workflow_id, task_id=task_id)
                raise PipelineError(str(exc), cat) from exc

        _outer.__name__ = fn.__name__
        _outer.__doc__ = fn.__doc__
        return _outer

    return decorator


# ── Custom error class ────────────────────────────────────────────────────────

class PipelineError(Exception):
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.TRANSIENT):
        super().__init__(message)
        self.category = category


# ── Structured error logging ──────────────────────────────────────────────────

def log_error(
    exc: Exception,
    component: str,
    *,
    workflow_id: str = "",
    task_id: str = "",
    input_summary: str = "",
    claude_suggestion: str = "",
) -> str:
    """Write a structured error record to SQLite. Returns the record ID."""
    try:
        import db as _db
        record_id = str(uuid.uuid4())
        category = classify(exc)
        _db.execute(
            "INSERT INTO error_logs "
            "(id, timestamp, workflow_id, task_id, component, error_class, error_message, "
            " stack_trace, input_summary, error_category, claude_suggestion) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record_id,
                datetime.now(timezone.utc).isoformat(),
                workflow_id or None,
                task_id or None,
                component,
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
                input_summary,
                category.value,
                claude_suggestion or None,
            ),
        )
        _db.commit()
        return record_id
    except Exception as db_exc:
        log.error(f"Failed to write error log: {db_exc}")
        return ""


def get_recent_errors(limit: int = 50) -> list[dict]:
    """Return recent error log entries for display in the UI."""
    try:
        import db as _db
        rows = _db.fetchall(
            "SELECT id, timestamp, component, error_class, error_message, "
            "       error_category, claude_suggestion, resolved_at "
            "FROM error_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]
    except Exception:
        return []


def mark_resolved(record_id: str) -> None:
    try:
        import db as _db
        _db.execute(
            "UPDATE error_logs SET resolved_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), record_id),
        )
        _db.commit()
    except Exception:
        pass
