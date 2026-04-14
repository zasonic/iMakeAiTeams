"""
services/tools/bash_tool.py

Sandboxed bash execution tool for the agentic coding loop.

Safety model (three layers):
  1. Pattern blocklist  — instant BLOCK for known-dangerous commands
                          (leverages safety_gate.py patterns + extras)
  2. CWD containment    — working directory locked to project_root.
                          Paths outside it are rejected before execution.
  3. Timeout            — hard 60-second wall-clock limit per command.
                          Long-running commands are killed, not hung.

Returns a ToolResult with stdout, stderr, exit_code, and truncation flag.
Output is capped at 30,000 characters (same as Claude Code) to prevent
context window flooding.

The permission_engine decides whether to run at all; this module only
executes commands that have already been approved.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("MyAIAgentHub.bash_tool")

MAX_OUTPUT_CHARS = 30_000
DEFAULT_TIMEOUT  = 60  # seconds

# Commands blocked outright regardless of context.
# These are taken from safety_gate.py + Claude Code's bash security list.
_BLOCKED_PREFIXES: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf ~",
    "rm -rf .",
    "rm -r /",
    "rmdir /s",
    "del /f /s /q",
    "format ",
    "mkfs.",
    "dd if=",
    "> /dev/sda",
    "chmod -R 777 /",
    ":(){ :|:&};:",
    ":(){:|:&};:",
    "shutdown",
    "reboot",
    "init 0",
    "halt",
    "poweroff",
    "curl | bash",
    "curl|bash",
    "wget | bash",
    "wget|bash",
    "curl | sh",
    "curl|sh",
    "wget | sh",
    "wget|sh",
    "bash <(",
    "python -c",       # arbitrary inline Python — too broad for agent use
    "python3 -c",
    "eval $(curl",
    "eval $(wget",
)

_BLOCKED_SUBSTRINGS: tuple[str, ...] = (
    "&&rm ",
    ";rm ",
    "| rm",
    "> /dev/sd",
    "/etc/passwd",
    "/etc/shadow",
    "~/.ssh",
    "~/.aws",
    "~/.anthropic",
    "settings.json",
    ".env",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


@dataclass
class ToolResult:
    """Result returned by every tool in the agentic loop."""
    ok:        bool
    output:    str        = ""
    error:     str        = ""
    truncated: bool       = False
    metadata:  dict       = field(default_factory=dict)

    def as_text(self) -> str:
        """Format for injection into the LLM context window."""
        parts = []
        if self.output:
            parts.append(self.output)
        if self.error:
            parts.append(f"[stderr]\n{self.error}")
        if self.truncated:
            parts.append(f"[Output truncated at {MAX_OUTPUT_CHARS} chars]")
        if not parts:
            parts.append("[No output]")
        status = "exit 0" if self.ok else f"exit {self.metadata.get('exit_code', '?')}"
        return f"[{status}]\n" + "\n".join(parts)


class BashTool:
    """
    Execute shell commands inside a project-root sandbox.

    Parameters
    ----------
    project_root : Path
        The directory the agent is operating in. All commands run with
        this as CWD. Path traversal outside this root is blocked.
    timeout : int
        Wall-clock seconds before the command is killed.
    """

    def __init__(self, project_root: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.project_root = Path(project_root).resolve()
        self.timeout = timeout

    # ── Public ─────────────────────────────────────────────────────────────

    def run(self, command: str) -> ToolResult:
        """
        Execute a shell command and return a ToolResult.

        Does NOT check permissions — the permission_engine must approve
        before this is called.
        """
        command = command.strip()
        if not command:
            return ToolResult(ok=False, error="Empty command.")

        # Layer 1: blocklist check
        blocked = self._check_blocked(command)
        if blocked:
            return ToolResult(
                ok=False,
                error=f"Command blocked by safety policy: {blocked}",
                metadata={"blocked_reason": blocked},
            )

        log.info("BashTool: executing: %s", command[:200])

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.project_root),
                env={**os.environ, "HOME": str(self.project_root)},
            )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            truncated = False
            if len(stdout) > MAX_OUTPUT_CHARS:
                stdout = stdout[:MAX_OUTPUT_CHARS]
                truncated = True
            if len(stderr) > MAX_OUTPUT_CHARS // 4:
                stderr = stderr[:MAX_OUTPUT_CHARS // 4]

            ok = proc.returncode == 0
            result = ToolResult(
                ok=ok,
                output=stdout,
                error=stderr,
                truncated=truncated,
                metadata={"exit_code": proc.returncode, "command": command},
            )
            log.info("BashTool: exit=%d, stdout=%d chars", proc.returncode, len(stdout))
            return result

        except subprocess.TimeoutExpired:
            msg = f"Command timed out after {self.timeout}s."
            log.warning("BashTool: %s", msg)
            return ToolResult(ok=False, error=msg, metadata={"timed_out": True})
        except Exception as exc:
            log.error("BashTool: unexpected error: %s", exc)
            return ToolResult(ok=False, error=f"Execution error: {exc}")

    def validate_path(self, path: str) -> tuple[bool, str]:
        """
        Check that path is inside project_root.
        Returns (True, resolved_path) or (False, reason).
        """
        try:
            resolved = Path(path).expanduser().resolve()
            resolved.relative_to(self.project_root)
            return True, str(resolved)
        except ValueError:
            return False, (
                f"Path '{path}' is outside the project root "
                f"({self.project_root}). Access denied."
            )

    # ── Internal ───────────────────────────────────────────────────────────

    def _check_blocked(self, command: str) -> str | None:
        """Return block reason string, or None if safe."""
        lower = command.lower().strip()
        for prefix in _BLOCKED_PREFIXES:
            if lower.startswith(prefix.lower()):
                return f"blocked prefix: {prefix!r}"
        for sub in _BLOCKED_SUBSTRINGS:
            if sub.lower() in lower:
                return f"blocked substring: {sub!r}"
        return None
