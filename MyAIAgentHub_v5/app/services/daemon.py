"""
services/daemon.py — Background daemon for heartbeat, idle compaction, and Dream consolidation.

Runs as a daemon thread started by main.py alongside the ChannelManager.

Three background services:
  1. Heartbeat (every 60s): local model reachable, API key valid, disk space
  2. Idle compaction (checks every 5 min): if a session is idle > 15 min,
     proactively summarize the conversation buffer
  3. Dream consolidation (every 6 hours): scan session_facts across all
     sessions, deduplicate, promote durable facts to long-term memory,
     update .myai/MEMORY.md

Inspired by nanobot's Gateway + Dream consolidation pattern (MIT).
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("MyAIAgentHub.daemon")

# ── Intervals ────────────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL   = 60      # seconds
IDLE_CHECK_INTERVAL  = 300     # 5 minutes
IDLE_THRESHOLD       = 900     # 15 minutes
DREAM_INTERVAL       = 21600   # 6 hours


class BackgroundDaemon:
    """
    Background service manager for health, compaction, and memory consolidation.

    Started by main.py after GUI loads. Runs until app shutdown.
    """

    def __init__(
        self,
        settings,
        local_client=None,
        claude_client=None,
        memory_manager=None,
        project_root: Path | None = None,
    ) -> None:
        self._settings = settings
        self._local = local_client
        self._claude = claude_client
        self._memory = memory_manager
        self._root = project_root or Path.cwd()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_heartbeat: dict = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all background services."""
        self._stop.clear()
        services = [
            ("heartbeat", self._heartbeat_loop),
            ("idle-compact", self._idle_compaction_loop),
            ("dream", self._dream_loop),
        ]
        for name, target in services:
            t = threading.Thread(target=target, name=f"daemon-{name}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("Background daemon started (%d services)", len(self._threads))

    def stop(self) -> None:
        """Signal all services to stop."""
        self._stop.set()
        log.info("Background daemon stopping")

    def get_health(self) -> dict:
        """Return the last heartbeat result."""
        return dict(self._last_heartbeat)

    # ── Heartbeat ─────────────────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            try:
                self._run_heartbeat()
            except Exception as exc:
                log.debug("Heartbeat error: %s", exc)

    def _run_heartbeat(self) -> None:
        """
        Check system health + run user-defined HEARTBEAT.md checklist.
        Inspired by OpenClaw: HEARTBEAT_OK = silent. Only report issues.
        """
        health = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "local_model": False,
            "api_key_set": False,
            "disk_ok": True,
            "heartbeat_tasks": [],
        }

        # Local model reachable?
        if self._local:
            try:
                health["local_model"] = self._local.is_available()
            except Exception:
                pass

        # API key configured?
        key = self._settings.get("claude_api_key", "")
        health["api_key_set"] = bool(key and key.strip())

        # Disk space (warn if < 500MB free)
        try:
            stat = os.statvfs(str(self._root))
            free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            health["disk_free_mb"] = round(free_mb)
            health["disk_ok"] = free_mb > 500
        except (OSError, AttributeError):
            pass  # statvfs not available on all platforms

        # HEARTBEAT.md checklist (OpenClaw pattern)
        # Read user-defined tasks from .myai/HEARTBEAT.md
        heartbeat_file = self._root / ".myai" / "HEARTBEAT.md"
        if heartbeat_file.exists():
            try:
                content = heartbeat_file.read_text(encoding="utf-8")
                tasks = [
                    line.strip().lstrip("- ").strip()
                    for line in content.splitlines()
                    if line.strip() and line.strip().startswith("- ")
                ]
                health["heartbeat_tasks"] = tasks
                # If any tasks require attention, log them
                issues = [t for t in tasks if "check" in t.lower() or "warn" in t.lower()]
                if issues:
                    log.info("Heartbeat: %d tasks need attention", len(issues))
            except Exception:
                pass

        self._last_heartbeat = health

    # ── Idle compaction ───────────────────────────────────────────────────

    def _idle_compaction_loop(self) -> None:
        while not self._stop.wait(IDLE_CHECK_INTERVAL):
            if self._memory is None:
                continue
            try:
                self._check_idle_sessions()
            except Exception as exc:
                log.debug("Idle compaction error: %s", exc)

    def _check_idle_sessions(self) -> None:
        """Check for idle conversation buffers and compact them."""
        if not hasattr(self._memory, "_buffers"):
            return

        now = time.time()
        threshold = now - IDLE_THRESHOLD

        # Check each buffered conversation
        for conv_id in list(self._memory._buffers.keys()):
            buf = self._memory._buffers.get(conv_id)
            if not buf or len(buf) < 10:
                continue

            # Check if the last message is old enough
            last_msg = buf[-1] if buf else None
            if not last_msg:
                continue

            # Try to get timestamp from the message
            msg_time = last_msg.get("_timestamp", 0)
            if msg_time and msg_time < threshold:
                log.info("Idle compaction triggered for conversation %s", conv_id[:8])
                try:
                    self._memory.summarize_buffer(conv_id)
                except Exception as exc:
                    log.debug("Idle compaction failed for %s: %s", conv_id[:8], exc)

    # ── Dream consolidation ───────────────────────────────────────────────

    def _dream_loop(self) -> None:
        # Wait a bit before first Dream run
        if self._stop.wait(300):  # 5 min initial delay
            return
        while not self._stop.wait(DREAM_INTERVAL):
            try:
                self._run_dream()
            except Exception as exc:
                log.debug("Dream consolidation error: %s", exc)

    def _run_dream(self) -> None:
        """
        Scan session facts across all conversations, deduplicate,
        and promote durable patterns to long-term memory + MEMORY.md.
        """
        try:
            import db as _db
        except ImportError:
            return

        log.info("Dream consolidation starting...")

        # 1. Gather all session facts from the last 24 hours
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        rows = _db.fetchall(
            "SELECT DISTINCT fact FROM session_facts WHERE created_at > ? ORDER BY created_at DESC LIMIT 100",
            (cutoff,),
        )
        if not rows:
            log.info("Dream: no recent facts to consolidate")
            return

        facts = [r["fact"] for r in rows]
        log.info("Dream: found %d recent facts to consolidate", len(facts))

        # 2. Deduplicate using local model (free)
        if self._local and self._local.is_available():
            try:
                prompt = (
                    "Given these facts extracted from conversations, remove duplicates "
                    "and group related facts. Return a JSON array of unique, consolidated facts.\n\n"
                    "Facts:\n" + "\n".join(f"- {f}" for f in facts[:50]) + "\n\n"
                    "Return ONLY a JSON array of strings."
                )
                raw = self._local.chat(
                    "You deduplicate and consolidate facts. Return ONLY valid JSON.",
                    prompt, max_tokens=500,
                )
                if raw:
                    start = raw.find("[")
                    end = raw.rfind("]")
                    if start != -1 and end != -1:
                        consolidated = json.loads(raw[start:end + 1])
                        if isinstance(consolidated, list):
                            facts = [str(f) for f in consolidated if f]
            except Exception as exc:
                log.debug("Dream dedup failed: %s — using raw facts", exc)

        # 3. Promote to long-term memory entries
        now = datetime.now(timezone.utc).isoformat()
        promoted = 0
        for fact in facts[:20]:  # cap at 20 per Dream cycle
            try:
                import uuid
                existing = _db.fetchone(
                    "SELECT id FROM memory_entries WHERE content = ?", (fact,)
                )
                if not existing:
                    _db.execute(
                        "INSERT INTO memory_entries (id, content, category, source, created_at, "
                        "last_accessed, embedding_status) VALUES (?, ?, 'fact', 'dream', ?, ?, 'dirty')",
                        (str(uuid.uuid4()), fact, now, now),
                    )
                    promoted += 1
            except Exception:
                continue
        if promoted:
            _db.commit()
            log.info("Dream: promoted %d facts to long-term memory", promoted)

        # 4. Update MEMORY.md
        self._update_memory_md(facts)

    def _update_memory_md(self, facts: list[str]) -> None:
        """Write consolidated facts to .myai/MEMORY.md for project memory."""
        memory_dir = self._root / ".myai"
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_file = memory_dir / "MEMORY.md"

        # Read existing content
        existing_content = ""
        if memory_file.exists():
            try:
                existing_content = memory_file.read_text(encoding="utf-8")
            except Exception:
                pass

        # Build new section
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        new_section = f"\n## Dream Consolidation ({now})\n\n"
        for fact in facts[:20]:
            new_section += f"- {fact}\n"

        # Append to existing (keep last 5 Dream sections)
        sections = existing_content.split("## Dream Consolidation")
        if len(sections) > 5:
            # Keep header + last 4 sections
            existing_content = sections[0] + "## Dream Consolidation".join(sections[-4:])

        content = existing_content.rstrip() + "\n" + new_section
        memory_file.write_text(content, encoding="utf-8")
        log.info("Dream: updated MEMORY.md with %d facts", len(facts))
