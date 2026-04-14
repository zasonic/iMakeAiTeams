"""
services/tools/git_tool.py

Git integration tool for the agentic coding loop.

Provides a curated set of git operations the agent can use safely.
Destructive operations (force push, reset --hard, clean -fd) require
explicit permission_engine approval and are not auto-approved.

Operations
----------
status      — show working tree status
diff        — show unstaged or staged changes
log         — show recent commit history
add         — stage files
commit      — create a commit
branch      — list or create branches
checkout    — switch branches (not destructive if branch exists)
show        — show a specific commit
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from services.tools.bash_tool import ToolResult

log = logging.getLogger("MyAIAgentHub.git_tool")

MAX_OUTPUT_CHARS = 20_000

# Operations that are read-only and always auto-approved.
READ_ONLY_OPS = {"status", "diff", "log", "show", "branch_list"}

# Operations that write to the repo and require write permission.
WRITE_OPS = {"add", "commit", "branch_create", "checkout"}


class GitTool:
    """
    Execute safe git commands in a project directory.

    Parameters
    ----------
    project_root : Path
        The directory containing the git repo. Commands run here.
    """

    def __init__(self, project_root: Path) -> None:
        self.root = Path(project_root).resolve()
        self._git = shutil.which("git") or "git"

    # ── Public operations ──────────────────────────────────────────────────

    def status(self) -> ToolResult:
        """Show git status."""
        return self._run(["git", "status", "--short", "--branch"])

    def diff(self, staged: bool = False, path: str = "") -> ToolResult:
        """Show diff of changes."""
        args = ["git", "diff"]
        if staged:
            args.append("--cached")
        if path:
            args += ["--", path]
        return self._run(args)

    def log(self, n: int = 10) -> ToolResult:
        """Show recent commit history."""
        n = min(max(1, n), 50)  # clamp 1-50
        return self._run([
            "git", "log",
            f"-{n}",
            "--oneline",
            "--decorate",
        ])

    def add(self, paths: list[str] | str = ".") -> ToolResult:
        """Stage files for commit."""
        if isinstance(paths, str):
            paths = [paths]
        return self._run(["git", "add", "--"] + paths)

    def commit(self, message: str) -> ToolResult:
        """Create a commit with the given message."""
        if not message or not message.strip():
            return ToolResult(ok=False, error="Commit message cannot be empty.")
        return self._run(["git", "commit", "-m", message.strip()])

    def branch(self, name: str = "") -> ToolResult:
        """List branches, or create a new branch if name is given."""
        if name:
            return self._run(["git", "checkout", "-b", name])
        return self._run(["git", "branch", "--list"])

    def checkout(self, branch: str) -> ToolResult:
        """Switch to an existing branch."""
        if not branch or not branch.strip():
            return ToolResult(ok=False, error="Branch name cannot be empty.")
        return self._run(["git", "checkout", branch.strip()])

    def show(self, ref: str = "HEAD") -> ToolResult:
        """Show details of a specific commit."""
        return self._run(["git", "show", "--stat", ref])

    def is_repo(self) -> bool:
        """Return True if project_root is inside a git repo."""
        r = self._run(["git", "rev-parse", "--git-dir"])
        return r.ok

    # ── Internal ───────────────────────────────────────────────────────────

    def _run(self, args: list[str]) -> ToolResult:
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.root),
            )
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()

            # Truncate very long output
            if len(stdout) > MAX_OUTPUT_CHARS:
                stdout = stdout[:MAX_OUTPUT_CHARS]
                truncated = True
            else:
                truncated = False

            ok = proc.returncode == 0
            output = stdout or stderr  # git often writes to stderr even on success
            if not ok and stderr:
                output = stderr

            return ToolResult(
                ok=ok,
                output=output,
                error="" if ok else stderr,
                truncated=truncated,
                metadata={
                    "exit_code": proc.returncode,
                    "command": " ".join(str(a) for a in args),
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error="Git command timed out after 30s.")
        except FileNotFoundError:
            return ToolResult(ok=False, error="git not found. Is git installed?")
        except Exception as exc:
            return ToolResult(ok=False, error=f"Git error: {exc}")
