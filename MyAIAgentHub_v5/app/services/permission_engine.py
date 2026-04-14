"""
services/permission_engine.py

Per-tool permission gates for the agentic coding loop.

Three tiers
-----------
AUTO_APPROVE  Read-only tools (file read, glob, grep, git status/diff/log).
              These run without asking the user.
CONFIRM       Write tools (file write, edit, git add/commit, git checkout).
              The agent proposes the action; the user confirms via GUI or
              a reply in the channel that triggered the task.
DENY          Explicitly blocked operations. The agent gets an error and
              must find another approach.

The permission_engine does NOT execute commands — it only classifies them
and records decisions.  Execution is handled by the individual tool classes.

Session-level allowlists
------------------------
For automated/headless use the caller can call
  engine.allow_for_session(tool_name)
to auto-approve a specific tool for the rest of the session.
This is how the GUI "Yes, always (this session)" button works.

Pending confirmations
---------------------
When CONFIRM is required, the engine stores a pending request
(keyed by a UUID) and the caller must:
  1. Show the user the proposed action.
  2. Wait for approve(request_id) or deny(request_id).
  3. Proceed or abort accordingly.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

log = logging.getLogger("MyAIAgentHub.permission_engine")


class PermissionTier(str, Enum):
    AUTO_APPROVE = "auto_approve"
    CONFIRM      = "confirm"
    DENY         = "deny"


@dataclass
class ToolCall:
    """A proposed tool call awaiting permission assessment."""
    tool:       str             # e.g. "bash", "file_write", "git_commit"
    description: str            # Human-readable summary: "Write 42 lines to main.py"
    args:       dict            = field(default_factory=dict)
    request_id: str             = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class PermissionResult:
    tier:       PermissionTier
    tool_call:  ToolCall
    reason:     str = ""


# ── Tier classification table ─────────────────────────────────────────────────
# Maps tool names to their default tier.
# "bash" is CONFIRM by default; specific safe bash patterns are not
# pre-approved here — the bash_tool blocklist handles dangerous ones, and
# the session allowlist lets the user expand permissions interactively.

_DEFAULT_TIERS: dict[str, PermissionTier] = {
    # Auto-approved (read-only)
    "file_read":      PermissionTier.AUTO_APPROVE,
    "file_glob":      PermissionTier.AUTO_APPROVE,
    "file_grep":      PermissionTier.AUTO_APPROVE,
    "git_status":     PermissionTier.AUTO_APPROVE,
    "git_diff":       PermissionTier.AUTO_APPROVE,
    "git_log":        PermissionTier.AUTO_APPROVE,
    "git_show":       PermissionTier.AUTO_APPROVE,

    # Confirm required (write)
    "file_write":     PermissionTier.CONFIRM,
    "file_edit":      PermissionTier.CONFIRM,
    "git_add":        PermissionTier.CONFIRM,
    "git_commit":     PermissionTier.CONFIRM,
    "git_checkout":   PermissionTier.CONFIRM,
    "git_branch":     PermissionTier.CONFIRM,
    "bash":           PermissionTier.CONFIRM,

    # Denied
    "shell_root":     PermissionTier.DENY,
    "rm_recursive":   PermissionTier.DENY,
}


class PermissionEngine:
    """
    Evaluate and gate tool calls for the agentic loop.

    Parameters
    ----------
    on_confirm_needed : Callable[[ToolCall], None] | None
        Callback invoked when a CONFIRM-tier tool call is pending.
        The callback should surface the request to the user
        (GUI dialog, Telegram message, etc.).
    """

    def __init__(
        self,
        on_confirm_needed: Callable[[ToolCall], None] | None = None,
    ) -> None:
        self._on_confirm = on_confirm_needed
        self._session_allowlist: set[str] = set()  # tools auto-approved this session
        self._pending: dict[str, threading.Event] = {}  # request_id -> Event
        self._decisions: dict[str, bool] = {}          # request_id -> approved?
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def check(self, tool_call: ToolCall) -> PermissionResult:
        """
        Classify a tool call and return a PermissionResult.

        Does NOT block — callers must call wait_for_confirm() separately
        for CONFIRM-tier tools.
        """
        tier = _DEFAULT_TIERS.get(tool_call.tool, PermissionTier.CONFIRM)

        # Session allowlist overrides CONFIRM → AUTO_APPROVE
        if tier == PermissionTier.CONFIRM and tool_call.tool in self._session_allowlist:
            tier = PermissionTier.AUTO_APPROVE
            reason = "allowed for this session"
        elif tier == PermissionTier.AUTO_APPROVE:
            reason = "read-only tool"
        elif tier == PermissionTier.DENY:
            reason = "tool is permanently blocked"
        else:
            reason = "write operation — user confirmation required"

        result = PermissionResult(tier=tier, tool_call=tool_call, reason=reason)

        if tier == PermissionTier.CONFIRM and self._on_confirm:
            self._on_confirm(tool_call)

        log.info(
            "Permission: tool=%s tier=%s request_id=%s",
            tool_call.tool, tier.value, tool_call.request_id,
        )
        return result

    def wait_for_confirm(self, request_id: str, timeout: float = 120.0) -> bool:
        """
        Block until the user approves or denies request_id, or timeout.

        Returns True if approved, False if denied or timed out.
        """
        evt = threading.Event()
        with self._lock:
            self._pending[request_id] = evt

        approved = evt.wait(timeout=timeout)
        if not approved:
            log.warning("Permission: request_id=%s timed out after %.0fs", request_id, timeout)

        with self._lock:
            decision = self._decisions.pop(request_id, False)
            self._pending.pop(request_id, None)

        return decision if approved else False

    def approve(self, request_id: str, allow_session: bool = False) -> None:
        """User approved a pending request."""
        with self._lock:
            self._decisions[request_id] = True
            if allow_session:
                # Find the tool name for this request and allowlist it
                # (stored by the check() call — we record it in decisions meta)
                pass
            evt = self._pending.get(request_id)
        if evt:
            evt.set()
        log.info("Permission: approved request_id=%s allow_session=%s", request_id, allow_session)

    def deny(self, request_id: str) -> None:
        """User denied a pending request."""
        with self._lock:
            self._decisions[request_id] = False
            evt = self._pending.get(request_id)
        if evt:
            evt.set()
        log.info("Permission: denied request_id=%s", request_id)

    def allow_for_session(self, tool_name: str) -> None:
        """Auto-approve a tool for the rest of this session."""
        self._session_allowlist.add(tool_name)
        log.info("Permission: tool=%s added to session allowlist", tool_name)

    def reset_session_allowlist(self) -> None:
        """Clear all session-level auto-approvals."""
        self._session_allowlist.clear()

    def get_session_allowlist(self) -> list[str]:
        return sorted(self._session_allowlist)

    @staticmethod
    def classify_tool(tool_name: str) -> PermissionTier:
        """Return the default tier for a tool name."""
        return _DEFAULT_TIERS.get(tool_name, PermissionTier.CONFIRM)
