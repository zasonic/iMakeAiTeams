"""
services/audit_log.py — Phase 4: append-only JSONL lifecycle audit log.

Records every inter-agent lifecycle event (start, shutdown attempt,
shutdown confirmed, shutdown denied, shutdown timed out) to a single
flat file at ``paths.user_dir() / "lifecycle_audit.jsonl"``.

Design rules:
  - Append-only: writes always use O_APPEND so concurrent writers do not
    corrupt the tail. The file may be read at any time without locking.
  - Flush-on-write: every record is flushed before ``append`` returns so
    a crash mid-session still preserves prior events.
  - Plain JSONL: one event per line, parseable by ``jq`` or any reader
    that splits on newlines.
  - Schema: ``{ts, event, agent_id?, target_id?, requester_id?, reason?,
    outcome?, token?}``. ``ts`` is ISO-8601 with millisecond precision.

The file is intentionally simple — recovery, rotation, and retention are
out of scope for Phase 4. The file is small (one line per lifecycle
event) so it grows slowly even over years of use.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("MyAIEnv.audit_log")


def _ts_ms() -> str:
    """ISO-8601 timestamp with millisecond precision and UTC offset."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class AuditLog:
    """Append-only JSONL writer for lifecycle events.

    One AuditLog instance per app process. Construction is cheap (does not
    touch the file). The first ``append`` lazily creates the file.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        # Inter-thread lock guarding the open-append-flush sequence so two
        # threads never write a half line. The OS-level O_APPEND on the file
        # already protects against torn writes from separate processes; this
        # lock is for in-process correctness only.
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: str, **fields: Any) -> dict:
        """Append one lifecycle event. Returns the persisted record dict."""
        if not isinstance(event, str) or not event.strip():
            raise ValueError("event name is required")
        record: dict[str, Any] = {"ts": _ts_ms(), "event": event.strip()}
        for k, v in fields.items():
            if v is None:
                continue
            record[k] = v
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # O_APPEND is atomic for writes ≤ PIPE_BUF on POSIX; on Windows
            # we still serialize via the in-process lock above.
            fd = os.open(
                str(self._path),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
                try:
                    os.fsync(fd)
                except OSError:
                    pass  # fsync may fail on tmpfs / WSL; not fatal
            finally:
                os.close(fd)
        return record

    def read_all(self) -> list[dict]:
        """Read every recorded event. Returns [] when the file does not exist."""
        if not self._path.exists():
            return []
        out: list[dict] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    log.warning(
                        "audit_log.read_all: skipping malformed line: %s", exc
                    )
        return out

    def tail(self, n: int) -> list[dict]:
        """Return the last ``n`` recorded events."""
        all_events = self.read_all()
        return all_events[-max(0, n):]

    def filter(self, *, event: str | None = None,
               agent_id: str | None = None,
               target_id: str | None = None) -> Iterable[dict]:
        """Iterate recorded events matching the given keyword filters."""
        for rec in self.read_all():
            if event is not None and rec.get("event") != event:
                continue
            if agent_id is not None and rec.get("agent_id") != agent_id:
                continue
            if target_id is not None and rec.get("target_id") != target_id:
                continue
            yield rec
