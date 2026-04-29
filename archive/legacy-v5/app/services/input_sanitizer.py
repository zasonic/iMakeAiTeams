"""
services/input_sanitizer.py — LlamaFirewall Input Sanitization.

Priority 5 (new module — no existing files modified).

Synchronous — called directly from api.py methods before chat/RAG operations.
No FastAPI, no async, no WebSocket. Results emitted via caller-provided callback.

Scanners
--------
PromptGuardScanner  — Meta PromptGuard 2 via llamafirewall.
                      Detects prompt injection, jailbreak, role-override.
                      Used on ALL user messages when firewall is enabled.
DocumentScanner     — PromptGuard applied to document content chunks
                      before RAG ingestion. Prevents memory poisoning.

Usage in api.py
---------------
    # In chat_send():
    scan = self._sanitizer.scan_message(user_message, session_id=conversation_id)
    if scan["blocked"]:
        self._emit("chat_error", {"error": "Message blocked by security scan.", ...})
        return

    # In rag_add_file():
    scan = self._sanitizer.scan_document(content, filename=file_path)
    if scan["blocked"]:
        return {"error": "Document blocked by security scan."}

Settings
--------
firewall_enabled stored in the `settings` SQLite table.
Default: ON when API key is set, OFF for local-only.

Requirements: llamafirewall>=0.1.0
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone

import db

log = logging.getLogger("input_sanitizer")

# ── llamafirewall install guard ───────────────────────────────────────────────

try:
    from llamafirewall import (           # type: ignore
        LlamaFirewall,
        ScannerType,
        UserMessage,
        Role,
        ScanDecision,
    )
    _LF_AVAILABLE = True
    log.info("llamafirewall loaded.")
except ImportError:
    _LF_AVAILABLE = False
    log.warning(
        "llamafirewall not installed — input sanitization disabled. "
        "Run: pip install llamafirewall"
    )

    class _Stub:                          # type: ignore
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def __getattr__(self, n): return self

    LlamaFirewall = ScannerType = UserMessage = Role = ScanDecision = _Stub


# ── Risk thresholds ───────────────────────────────────────────────────────────

BLOCK_THRESHOLD = 0.80    # score ≥ this → verdict "block"
WARN_THRESHOLD  = 0.50    # score ≥ this → verdict "warn"
CHUNK_SIZE      = 2000    # chars per document chunk for scanning

# ── Firewall settings (SQLite-backed, 30s cache) ──────────────────────────────

class _FirewallSettings:
    _TTL = 30.0

    def __init__(self):
        self._cached: bool | None = None
        self._ts: float = 0.0

    def is_enabled(self) -> bool:
        now = time.monotonic()
        if self._cached is not None and (now - self._ts) < self._TTL:
            return self._cached
        try:
            row = db.fetchone("SELECT value FROM settings WHERE key = 'firewall_enabled'")
            val = bool(int(row["value"])) if row else True
        except Exception:
            val = True
        self._cached = val
        self._ts = now
        return val

    def set_enabled(self, enabled: bool) -> None:
        val = "1" if enabled else "0"
        now = datetime.now(timezone.utc).isoformat()
        try:
            db.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES ('firewall_enabled', ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (val, now),
            )
            db.commit()
        except Exception as exc:
            log.warning("FirewallSettings write failed: %s", exc)
        self._cached = enabled
        self._ts = time.monotonic()

    def invalidate(self) -> None:
        self._cached = None
        self._ts = 0.0


# ── PromptGuard scanner ───────────────────────────────────────────────────────

class _PromptGuardScanner:
    def __init__(self):
        self._fw = None
        self._available = False
        if _LF_AVAILABLE:
            try:
                self._fw = LlamaFirewall(
                    scanners={Role.USER: [ScannerType.PROMPT_GUARD]}
                )
                self._available = True
                log.info("PromptGuardScanner ready.")
            except Exception as exc:
                log.warning("PromptGuardScanner init failed: %s", exc)

    def scan(self, text: str) -> dict:
        """
        Scan text for injection/jailbreak threats.
        Returns {"verdict": str, "score": float|None, "reason": str}.
        Verdicts: "pass" | "warn" | "block" | "skipped" | "error"
        """
        if not self._available or self._fw is None:
            return {"verdict": "skipped", "score": None, "reason": "llamafirewall not installed", "degraded": True}

        t0 = time.monotonic()
        try:
            result   = self._fw.run(UserMessage(content=text, role=Role.USER))
            score    = float(getattr(result, "score", 0.0) or 0.0)
            decision = str(getattr(result, "decision", "")).upper()
            reason   = getattr(result, "reason", "") or ""

            if "BLOCK" in decision or score >= BLOCK_THRESHOLD:
                verdict = "block"
            elif score >= WARN_THRESHOLD:
                verdict = "warn"
            else:
                verdict = "pass"

            return {
                "verdict":     verdict,
                "score":       score,
                "reason":      reason,
                "duration_ms": round((time.monotonic() - t0) * 1000, 1),
                "degraded":    False,
            }
        except Exception as exc:
            log.error("PromptGuardScanner error: %s", exc)
            return {
                "verdict":     "error",
                "score":       None,
                "reason":      f"Scanner error: {exc}",
                "duration_ms": round((time.monotonic() - t0) * 1000, 1),
                "degraded":    False,
            }


# ── Module-level singletons ───────────────────────────────────────────────────

_settings = _FirewallSettings()
_pg        = _PromptGuardScanner()


# ── Scan result helpers ───────────────────────────────────────────────────────

def _make_result(
    verdict:    str,
    scanner:    str,
    scan_type:  str,
    score:      float | None = None,
    reason:     str = "",
    duration_ms: float = 0.0,
    degraded:   bool = False,
) -> dict:
    scan_id = str(uuid.uuid4())
    blocked = verdict == "block"
    icon    = {"pass": "🔒", "block": "🚫", "warn": "⚠️", "skipped": "🔓", "error": "⚠️"}.get(verdict, "🔒")
    label_map = {
        "pass":    "Security scan: passed",
        "block":   "Security scan: blocked",
        "warn":    "Security scan: flagged",
        "skipped": "Security scan: skipped",
        "error":   "Security scan: error",
    }
    return {
        "scan_id":    scan_id,
        "verdict":    verdict,
        "blocked":    blocked,
        "scanner":    scanner,
        "scan_type":  scan_type,
        "score":      score,
        "reason":     reason,
        "duration_ms": duration_ms,
        "degraded":   degraded,
        "icon":       icon,
        "label":      label_map.get(verdict, "Security scan"),
        "detail":     f"{scanner} · {duration_ms:.0f}ms" + (f" · {score:.0%} risk" if score and score > 0.3 else ""),
    }


def _log_scan(result: dict, scan_type: str, session_id: str | None, content_preview: str) -> None:
    try:
        db.execute(
            """
            INSERT INTO security_scan_log
                (scan_id, scan_type, verdict, scanner, score, reason,
                 flagged_phrases_json, duration_ms, session_id, model_tier,
                 content_preview, degraded, created_at)
            VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?, '', ?, ?, ?)
            """,
            (
                result["scan_id"], scan_type, result["verdict"],
                result["scanner"], result["score"],
                result.get("reason", "")[:500],
                result.get("duration_ms", 0.0),
                session_id,
                content_preview[:200],
                1 if result.get("degraded") else 0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        db.commit()
    except Exception as exc:
        log.warning("security_scan_log write failed: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def scan_message(
    text:       str,
    session_id: str | None = None,
    on_result=None,  # callable | None
) -> dict:
    """
    Scan a user message before sending to Claude.

    Parameters
    ----------
    text       : The user's raw message
    session_id : For log correlation (conversation_id)
    on_result  : Optional callback(result_dict) — use to emit events to JS
                 (api.py passes self._emit as a wrapped callback)

    Returns
    -------
    dict with keys: verdict, blocked, scan_id, score, reason, icon, label, detail
    """
    if not _settings.is_enabled():
        result = _make_result("skipped", "none", "user_message", reason="Firewall disabled")
        if on_result:
            try:
                on_result(result)
            except Exception:
                pass
        return result

    raw    = _pg.scan(text)
    result = _make_result(
        verdict    = raw["verdict"],
        scanner    = "promptguard",
        scan_type  = "user_message",
        score      = raw.get("score"),
        reason     = raw.get("reason", ""),
        duration_ms = raw.get("duration_ms", 0.0),
        degraded   = raw.get("degraded", False),
    )

    _log_scan(result, "user_message", session_id, text[:200])

    if on_result:
        try:
            on_result(result)
        except Exception:
            pass

    return result


def scan_document(
    content:    str,
    filename:   str = "",
    session_id: str | None = None,
    on_result=None,  # callable | None
) -> dict:
    """
    Scan document content before RAG ingestion.

    Chunks long documents and scans each chunk. Returns the worst result.
    WARN verdict → flag for user review (not outright block).
    BLOCK verdict → reject the document.
    """
    if not _settings.is_enabled():
        result = _make_result("skipped", "none", "document", reason="Firewall disabled")
        if on_result:
            try:
                on_result(result)
            except Exception:
                pass
        return result

    if not content.strip():
        result = _make_result("pass", "promptguard", "document")
        return result

    # Chunk and scan
    chunks = _chunk_text(content, CHUNK_SIZE)
    t0     = time.monotonic()
    worst  = {"verdict": "pass", "score": 0.0, "reason": "", "degraded": False}

    for chunk in chunks:
        raw = _pg.scan(chunk)
        if raw.get("score") and (raw["score"] > (worst["score"] or 0.0)):
            worst = raw
        if raw["verdict"] == "block":
            worst = raw
            break
        if raw["verdict"] == "warn" and worst["verdict"] == "pass":
            worst = raw

    duration_ms = (time.monotonic() - t0) * 1000

    result = _make_result(
        verdict    = worst["verdict"],
        scanner    = "promptguard",
        scan_type  = "document",
        score      = worst.get("score"),
        reason     = worst.get("reason", ""),
        duration_ms = round(duration_ms, 1),
        degraded   = worst.get("degraded", False),
    )

    _log_scan(result, "document", session_id, filename or content[:200])

    if on_result:
        try:
            on_result(result)
        except Exception:
            pass

    return result


def _chunk_text(text: str, size: int) -> list[str]:
    """Split text into overlapping chunks (20% overlap)."""
    if len(text) <= size:
        return [text]
    step   = int(size * 0.8)
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i: i + size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── Settings helpers (called from api.py) ─────────────────────────────────────

def is_firewall_enabled() -> bool:
    return _settings.is_enabled()


def set_firewall_enabled(enabled: bool) -> None:
    _settings.set_enabled(enabled)
    log.info("Firewall %s", "enabled" if enabled else "disabled")


def get_firewall_status() -> dict:
    """Return full status dict for the Settings panel."""
    today = datetime.now(timezone.utc).date().isoformat()
    scans_today = blocks_today = warns_today = 0
    try:
        row = db.fetchone(
            "SELECT COUNT(*) as n FROM security_scan_log WHERE created_at LIKE ?",
            (f"{today}%",),
        )
        scans_today = row["n"] if row else 0
        row = db.fetchone(
            "SELECT COUNT(*) as n FROM security_scan_log WHERE created_at LIKE ? AND verdict='block'",
            (f"{today}%",),
        )
        blocks_today = row["n"] if row else 0
        row = db.fetchone(
            "SELECT COUNT(*) as n FROM security_scan_log WHERE created_at LIKE ? AND verdict='warn'",
            (f"{today}%",),
        )
        warns_today = row["n"] if row else 0
    except Exception:
        pass

    return {
        "llamafirewall_installed": _LF_AVAILABLE,
        "promptguard_available":   _pg._available,
        "firewall_enabled":        _settings.is_enabled(),
        "scans_today":             scans_today,
        "blocks_today":            blocks_today,
        "warns_today":             warns_today,
    }


def get_scan_log(limit: int = 50, verdict_filter: str = "") -> list[dict]:
    """Return recent scan log entries for the Settings audit panel."""
    try:
        if verdict_filter:
            rows = db.fetchall(
                "SELECT * FROM security_scan_log WHERE verdict=? ORDER BY created_at DESC LIMIT ?",
                (verdict_filter, limit),
            )
        else:
            rows = db.fetchall(
                "SELECT * FROM security_scan_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning("get_scan_log failed: %s", exc)
        return []
