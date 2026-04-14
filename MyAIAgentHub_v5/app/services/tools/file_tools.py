"""
services/tools/file_tools.py

File system tools for the agentic coding loop.

Tools
-----
ReadTool   — read file contents (text, with optional line range)
WriteTool  — create or overwrite a file
EditTool   — exact string replacement (old_string → new_string)
GlobTool   — find files matching a pattern
GrepTool   — search file contents with regex

All path operations are sandboxed to project_root.
Writes require permission_engine approval before reaching here.
Reads are auto-approved by the permission engine.

Output format matches ToolResult from bash_tool.py so the agent loop
can handle all tools uniformly.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path

from services.tools.bash_tool import ToolResult

log = logging.getLogger("MyAIAgentHub.file_tools")

MAX_READ_CHARS  = 50_000   # ~12K tokens — enough for most files
MAX_GREP_RESULTS = 100


class FileTools:
    """
    All file system tools in one class, bound to a project_root sandbox.

    Parameters
    ----------
    project_root : Path
        All file access is restricted to this directory tree.
    """

    def __init__(self, project_root: Path) -> None:
        self.root = Path(project_root).resolve()

    # ── Read ───────────────────────────────────────────────────────────────

    def read(
        self,
        path: str,
        start_line: int | None = None,
        end_line:   int | None = None,
    ) -> ToolResult:
        """Read a file, optionally clamped to a line range."""
        ok, resolved = self._resolve(path)
        if not ok:
            return ToolResult(ok=False, error=resolved)

        p = Path(resolved)
        if not p.exists():
            return ToolResult(ok=False, error=f"File not found: {path}")
        if not p.is_file():
            return ToolResult(ok=False, error=f"Not a file: {path}")

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(ok=False, error=f"Cannot read file: {exc}")

        lines = text.splitlines(keepends=True)
        total = len(lines)

        if start_line is not None or end_line is not None:
            s = max(0, (start_line or 1) - 1)
            e = min(total, end_line or total)
            lines = lines[s:e]
            text = "".join(lines)

        truncated = False
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS]
            truncated = True

        log.debug("ReadTool: %s (%d lines)", path, total)
        return ToolResult(
            ok=True,
            output=text,
            truncated=truncated,
            metadata={"path": resolved, "total_lines": total},
        )

    # ── Write ──────────────────────────────────────────────────────────────

    def write(self, path: str, content: str) -> ToolResult:
        """Create or overwrite a file with content."""
        ok, resolved = self._resolve(path)
        if not ok:
            return ToolResult(ok=False, error=resolved)

        p = Path(resolved)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            log.info("WriteTool: wrote %d chars to %s", len(content), path)
            return ToolResult(
                ok=True,
                output=f"Wrote {len(content)} characters to {path}",
                metadata={"path": resolved, "bytes": len(content.encode())},
            )
        except Exception as exc:
            return ToolResult(ok=False, error=f"Cannot write file: {exc}")

    # ── Edit (exact string replacement) ───────────────────────────────────

    def edit(
        self,
        path:       str,
        old_string: str,
        new_string: str,
    ) -> ToolResult:
        """
        Replace exactly one occurrence of old_string with new_string.

        Fails if old_string is not found or appears more than once
        (ambiguous edit — the agent must be more specific).
        """
        ok, resolved = self._resolve(path)
        if not ok:
            return ToolResult(ok=False, error=resolved)

        p = Path(resolved)
        if not p.exists():
            return ToolResult(ok=False, error=f"File not found: {path}")

        try:
            original = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(ok=False, error=f"Cannot read file: {exc}")

        count = original.count(old_string)
        if count == 0:
            return ToolResult(
                ok=False,
                error=(
                    f"old_string not found in {path}. "
                    "The string must match exactly (including whitespace and indentation)."
                ),
            )
        if count > 1:
            return ToolResult(
                ok=False,
                error=(
                    f"old_string appears {count} times in {path} — ambiguous. "
                    "Include more surrounding context to make it unique."
                ),
            )

        updated = original.replace(old_string, new_string, 1)
        try:
            p.write_text(updated, encoding="utf-8")
        except Exception as exc:
            return ToolResult(ok=False, error=f"Cannot write file: {exc}")

        lines_changed = new_string.count("\n") - old_string.count("\n")
        log.info("EditTool: edited %s (%+d lines)", path, lines_changed)
        return ToolResult(
            ok=True,
            output=f"Edited {path} — replaced 1 occurrence ({lines_changed:+d} lines).",
            metadata={"path": resolved, "lines_delta": lines_changed},
        )

    # ── Glob ───────────────────────────────────────────────────────────────

    def glob(self, pattern: str, limit: int = 50) -> ToolResult:
        """
        Find files matching a glob pattern relative to project_root.
        Returns a newline-separated list of relative paths.
        """
        try:
            matches = sorted(self.root.rglob(pattern))
        except Exception as exc:
            return ToolResult(ok=False, error=f"Glob error: {exc}")

        # Filter to files only, exclude common noise
        files = [
            p for p in matches
            if p.is_file()
            and "__pycache__" not in p.parts
            and ".git" not in p.parts
            and not p.name.endswith(".pyc")
        ]

        truncated = len(files) > limit
        files = files[:limit]
        rel = [str(f.relative_to(self.root)) for f in files]

        log.debug("GlobTool: pattern=%r found %d files", pattern, len(rel))
        return ToolResult(
            ok=True,
            output="\n".join(rel) if rel else "(no matches)",
            truncated=truncated,
            metadata={"count": len(rel), "pattern": pattern},
        )

    # ── Grep ───────────────────────────────────────────────────────────────

    def grep(
        self,
        pattern:   str,
        path:      str = ".",
        recursive: bool = True,
        limit:     int = MAX_GREP_RESULTS,
    ) -> ToolResult:
        """
        Search file contents for a regex pattern.
        Returns matching lines in the format: path:line_no: content
        """
        ok, resolved = self._resolve(path)
        if not ok:
            return ToolResult(ok=False, error=resolved)

        search_root = Path(resolved)
        try:
            regex = re.compile(pattern, re.MULTILINE)
        except re.error as exc:
            return ToolResult(ok=False, error=f"Invalid regex: {exc}")

        results: list[str] = []
        truncated = False

        targets: list[Path] = []
        if search_root.is_file():
            targets = [search_root]
        elif recursive:
            targets = [
                p for p in search_root.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.parts
                and ".git" not in p.parts
            ]
        else:
            targets = [p for p in search_root.iterdir() if p.is_file()]

        for fpath in sorted(targets):
            if len(results) >= limit:
                truncated = True
                break
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = str(fpath.relative_to(self.root))
                    results.append(f"{rel}:{i}: {line.rstrip()}")
                    if len(results) >= limit:
                        truncated = True
                        break

        log.debug("GrepTool: pattern=%r found %d matches", pattern, len(results))
        return ToolResult(
            ok=True,
            output="\n".join(results) if results else "(no matches)",
            truncated=truncated,
            metadata={"matches": len(results), "pattern": pattern},
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _resolve(self, path: str) -> tuple[bool, str]:
        """Resolve path inside project_root. Returns (True, resolved) or (False, error)."""
        try:
            # Allow both absolute and relative paths
            p = Path(path)
            if not p.is_absolute():
                resolved = (self.root / p).resolve()
            else:
                resolved = p.resolve()
            resolved.relative_to(self.root)  # raises ValueError if outside
            return True, str(resolved)
        except ValueError:
            return False, (
                f"Path '{path}' is outside the project root ({self.root}). Access denied."
            )
        except Exception as exc:
            return False, f"Path resolution error: {exc}"
