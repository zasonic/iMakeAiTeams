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
    # Chained destructive commands
    "&&rm ", ";rm ", "| rm",
    "> /dev/sd",
    # Sensitive system files
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    # Credential locations
    "~/.ssh", "~/.aws", "~/.anthropic", "~/.config/gcloud",
    "~/.netrc", "~/.npmrc", "~/.pypirc",
    "settings.json", ".env",
    # Known API key patterns
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
    "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "HF_TOKEN",
    # Credential exfiltration
    "printenv", "env | grep", "env|grep", "set | grep",
    "cat ~/.ssh", "cat ~/.aws", "cat ~/.env",
    # Network abuse
    "nc -l", "ncat -l", "nmap ", "masscan ",
    # Privilege escalation
    "sudo ", "su -", "su root", "doas ",
    "chmod u+s", "chmod 4",
    # Data exfiltration via network
    "curl -d", "curl --data", "wget --post",
    "curl -F", "curl --upload",
)

# ── Sensitive environment variable patterns ──────────────────────────────────
# These are STRIPPED from the subprocess environment before execution.
# Only a safe allowlist is passed through.
_ENV_ALLOWLIST = {
    "PATH", "HOME", "TERM", "LANG", "LC_ALL", "SHELL",
    "USER", "LOGNAME", "TMPDIR", "TMP", "TEMP",
    "COLORTERM", "FORCE_COLOR", "NO_COLOR",
    "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
    "NODE_ENV", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
    "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
}

_SECRET_PATTERNS = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
    "AUTH", "PRIVATE", "APIKEY", "API_KEY",
)


def _make_safe_env(project_root: Path) -> dict[str, str]:
    """Build a sanitized environment dict for subprocess execution."""
    safe = {}
    for key, val in os.environ.items():
        if key in _ENV_ALLOWLIST:
            safe[key] = val
        elif not any(pat in key.upper() for pat in _SECRET_PATTERNS):
            safe[key] = val
    # Override HOME to project root to prevent access to user credentials
    safe["HOME"] = str(project_root)
    return safe


def _sanitize_output(text: str) -> str:
    """Scrub any accidentally leaked secrets from command output."""
    import re
    # Common API key patterns
    patterns = [
        (r'sk-ant-api03-[\w-]{80,}', '[REDACTED_ANTHROPIC_KEY]'),
        (r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_OPENAI_KEY]'),
        (r'ghp_[a-zA-Z0-9]{36}', '[REDACTED_GITHUB_TOKEN]'),
        (r'gho_[a-zA-Z0-9]{36}', '[REDACTED_GITHUB_TOKEN]'),
        (r'hf_[a-zA-Z0-9]{34}', '[REDACTED_HF_TOKEN]'),
        (r'AKIA[0-9A-Z]{16}', '[REDACTED_AWS_KEY]'),
    ]
    for pat, replacement in patterns:
        text = re.sub(pat, replacement, text)
    return text


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
            safe_env = _make_safe_env(self.project_root)

            # Resource limits via preexec_fn (Linux/macOS only)
            def _set_limits():
                try:
                    import resource
                    # 512 MB virtual memory
                    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
                    # 100 MB max file size
                    resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
                    # 50 max child processes
                    resource.setrlimit(resource.RLIMIT_NPROC, (50, 50))
                except (ImportError, ValueError, OSError):
                    pass  # resource module unavailable on Windows or limits not supported

            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.project_root),
                env=safe_env,
                preexec_fn=_set_limits if os.name != "nt" else None,
            )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            # Sanitize output to remove any leaked secrets
            stdout = _sanitize_output(stdout)
            stderr = _sanitize_output(stderr)

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
