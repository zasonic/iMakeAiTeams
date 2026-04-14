"""
services/agent_loop.py

The agentic coding loop — while(tool_calls).

This is the core engine that makes the assistant an agent rather than a chatbot.
Inspired by Claude Code's query.ts agent loop, adapted for Python with our
existing services (router, orchestrator, safety_gate, permission_engine).

How it works
------------
1. Receive a task (natural language) and a project directory.
2. Build an initial prompt with tool definitions.
3. Call Claude API with tools available.
4. If the response contains tool_use blocks:
   a. Check permission for each tool call.
   b. Auto-execute read-only tools.
   c. Queue write/bash tools for user confirmation.
   d. Execute approved tools and feed results back.
   e. Call Claude again with tool results.
   f. Repeat until no more tool calls (stop_reason == "end_turn").
5. Emit progress events so the GUI and Telegram adapter can stream updates.
6. Return the final text response.

Built-in tools exposed to Claude
---------------------------------
  file_read, file_write, file_edit, file_glob, file_grep
  bash
  git_status, git_diff, git_log, git_add, git_commit, git_show

Token optimization
------------------
The router decides whether to use the agent loop at all.
Simple tasks (greetings, quick Q&A) never reach this loop —
they go through the normal orchestrator path.
Only Claude handles the agent loop (no local model for agentic tasks).

Safety
------
Every bash command passes through BashTool's blocklist first.
Every write operation passes through PermissionEngine — the user
must confirm before any file is modified.
The existing safety_gate.py scans all content before execution.

MAX_TURNS = 20 — prevents infinite loops. The agent gets an error
message and must conclude if it hasn't finished after 20 turns.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

log = logging.getLogger("MyAIAgentHub.agent_loop")

MAX_TURNS         = 20
MAX_OUTPUT_TOKENS = 8192

# ── Tool schema definitions (sent to Claude as tools=[]) ─────────────────────

TOOL_SCHEMAS = [
    {
        "name": "file_read",
        "description": "Read the contents of a file. Returns the full text, or a line range if specified.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":       {"type": "string", "description": "Relative file path from project root."},
                "start_line": {"type": "integer", "description": "Optional first line to read (1-indexed)."},
                "end_line":   {"type": "integer", "description": "Optional last line to read (inclusive)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_write",
        "description": "Create or overwrite a file with the given content. Requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Relative file path from project root."},
                "content": {"type": "string", "description": "Complete file contents to write."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_edit",
        "description": (
            "Replace exactly one occurrence of old_string with new_string in a file. "
            "old_string must be unique in the file and must match exactly (whitespace, indentation). "
            "Requires user confirmation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":       {"type": "string", "description": "Relative file path from project root."},
                "old_string": {"type": "string", "description": "Exact string to find and replace."},
                "new_string": {"type": "string", "description": "Replacement string."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "file_glob",
        "description": "Find files matching a glob pattern relative to the project root. Returns relative paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py', '*.txt'."},
                "limit":   {"type": "integer", "description": "Maximum results to return (default 50)."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "file_grep",
        "description": "Search file contents for a regex pattern. Returns matching lines with file:line format.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern":   {"type": "string", "description": "Python regex pattern to search for."},
                "path":      {"type": "string", "description": "File or directory to search (default: project root)."},
                "recursive": {"type": "boolean", "description": "Search recursively (default: true)."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "bash",
        "description": (
            "Execute a shell command in the project directory. "
            "Use for running tests, building, installing dependencies, etc. "
            "Requires user confirmation. Avoid destructive commands."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "git_status",
        "description": "Show the current git status of the project.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "git_diff",
        "description": "Show git diff of current changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "staged": {"type": "boolean", "description": "Show staged changes (default: false)."},
                "path":   {"type": "string",  "description": "Limit diff to this file path."},
            },
            "required": [],
        },
    },
    {
        "name": "git_log",
        "description": "Show recent git commit history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of commits to show (default 10, max 50)."},
            },
            "required": [],
        },
    },
    {
        "name": "git_add",
        "description": "Stage files for commit. Requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "File path(s) to stage, or '.' for all.",
                },
            },
            "required": ["paths"],
        },
    },
    {
        "name": "git_commit",
        "description": "Create a git commit with the staged changes. Requires user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message."},
            },
            "required": ["message"],
        },
    },
    {
        "name": "git_show",
        "description": "Show details of a specific commit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Commit ref or hash (default: HEAD)."},
            },
            "required": [],
        },
    },
]

SYSTEM_PROMPT = """You are an expert software engineer with direct access to the user's codebase.
You have tools to read files, write files, edit files, run shell commands, and interact with git.

IMPORTANT RULES:
- Always read relevant files before making changes. Never guess at file contents.
- Use file_edit for targeted changes. Use file_write only for new files or complete rewrites.
- For file_edit, old_string must match exactly — copy it from file_read output.
- Run tests after making changes to verify correctness.
- Make small, focused commits. Don't commit everything at once.
- Explain what you're doing at each step. Be concise.
- If you're unsure about something, read more files before acting.
- When finished, summarize what was done and any remaining issues.
"""


@dataclass
class AgentEvent:
    """Progress event emitted during the agent loop."""
    type:    str   # "thinking" | "tool_call" | "tool_result" | "done" | "error" | "confirm_needed"
    content: str   = ""
    meta:    dict  = field(default_factory=dict)


class AgentLoop:
    """
    Agentic coding loop.

    Parameters
    ----------
    claude_client    : The existing claude_client (for API calls with tools).
    file_tools       : FileTools instance bound to project_root.
    bash_tool        : BashTool instance bound to project_root.
    git_tool         : GitTool instance bound to project_root.
    permission_engine: PermissionEngine for gate-keeping write operations.
    on_event         : Callback for streaming progress events to UI/channel.
    safety_gate      : Optional existing safety_gate for content scanning.
    """

    def __init__(
        self,
        claude_client,
        file_tools,
        bash_tool,
        git_tool,
        permission_engine,
        on_event: Callable[[AgentEvent], None] | None = None,
        safety_gate=None,
    ) -> None:
        self._claude  = claude_client
        self._files   = file_tools
        self._bash    = bash_tool
        self._git     = git_tool
        self._perms   = permission_engine
        self._on_event = on_event or (lambda e: None)
        self._safety  = safety_gate
        self._stop    = False

    def run(self, task: str, conversation_id: str = "") -> str:
        """
        Execute the agentic loop for a task.
        Blocks until complete or MAX_TURNS reached.
        Returns the final text response.
        """
        self._stop = False
        messages: list[dict] = [{"role": "user", "content": task}]
        tools_called: list[str] = []
        files_modified: list[str] = []

        # Build system prompt with project memory + repo map
        system = SYSTEM_PROMPT
        try:
            from services.project_memory import load_project_memory
            project_mem = load_project_memory(self._files.root)
            if project_mem:
                system = system + "\n\n" + project_mem
        except Exception:
            pass
        try:
            from services.repo_map import build_repo_map_for_task
            repo_map = build_repo_map_for_task(self._files.root, task, token_budget=1024)
            if repo_map:
                system = system + "\n\n" + repo_map
        except Exception:
            pass

        self._emit("thinking", f"Starting agent loop for task: {task[:100]}")

        for turn in range(MAX_TURNS):
            if self._stop:
                return "Task stopped by user."

            log.info("Agent loop turn %d/%d", turn + 1, MAX_TURNS)

            # Write progress artifact
            self._write_progress(task, turn, tools_called, files_modified)

            # Call Claude with tools
            try:
                response = self._claude.call_with_tools(
                    system=system,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
            except Exception as exc:
                error_msg = f"API call failed: {exc}"
                log.error("Agent loop: %s", error_msg)
                self._emit("error", error_msg)
                return f"I encountered an error: {error_msg}"

            # Extract text and tool_use blocks
            text_parts: list[str] = []
            tool_calls: list[dict] = []

            for block in response.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(block)

            # Emit any text the model produced this turn
            if text_parts:
                text = "\n".join(text_parts).strip()
                if text:
                    self._emit("thinking", text)

            # No tool calls → loop ends
            if not tool_calls:
                final_text = "\n".join(text_parts).strip()

                # Self-verification: run tests if files were modified
                test_passed = self._auto_verify(files_modified)

                # Write final progress
                self._write_progress(
                    task, turn + 1, tools_called, files_modified,
                    tests_run=test_passed is not None,
                    test_passed=test_passed,
                )

                self._emit("done", final_text)
                log.info("Agent loop completed in %d turns", turn + 1)
                return final_text

            # Append assistant message to history
            messages.append({"role": "assistant", "content": response.get("content", [])})

            # Execute tools and collect results
            tool_results: list[dict] = []
            for tc in tool_calls:
                name = tc.get("name", "")
                tools_called.append(name)
                # Track file modifications
                if name in ("file_write", "file_edit"):
                    path = tc.get("input", {}).get("path", "")
                    if path:
                        files_modified.append(path)

                result = self._execute_tool(tc)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tc["id"],
                    "content":     result,
                })

            # Append tool results to history
            messages.append({"role": "user", "content": tool_results})

        # Max turns reached
        self._emit("error", f"Reached maximum of {MAX_TURNS} turns without completing.")
        return f"I reached the maximum turn limit ({MAX_TURNS}) without finishing the task. Please try breaking it into smaller steps."

    def _write_progress(self, task, turn, tools_called, files_modified,
                        tests_run=False, test_passed=None):
        """Write progress artifact to disk."""
        try:
            from services.task_artifacts import write_agent_progress
            write_agent_progress(
                self._files.root, task, turn, MAX_TURNS,
                tools_called, files_modified, tests_run, test_passed,
            )
        except Exception:
            pass  # progress tracking is best-effort

    def _auto_verify(self, files_modified):
        """
        Run project tests if files were modified. Returns True/False/None.
        None = no test command found or no files modified.
        """
        if not files_modified:
            return None
        try:
            from services.task_artifacts import discover_test_command
            test_cmd = discover_test_command(self._files.root)
            if not test_cmd:
                return None
            self._emit("thinking", f"Running tests: {test_cmd}")
            result = self._bash.run(test_cmd)
            passed = result.ok if hasattr(result, "ok") else True
            if passed:
                self._emit("thinking", "Tests passed")
            else:
                self._emit("thinking", f"Tests failed: {(result.error or '')[:200]}")
            return passed
        except Exception as exc:
            log.debug("Auto-verify failed: %s", exc)
            return None

    def stop(self) -> None:
        """Request the loop to stop after the current turn."""
        self._stop = True

    # ── Tool dispatch ──────────────────────────────────────────────────────

    def _execute_tool(self, tool_block: dict) -> str:
        """Dispatch a tool_use block to the appropriate handler. Returns result text."""
        name  = tool_block.get("name", "")
        args  = tool_block.get("input", {})
        tc_id = tool_block.get("id", "")

        self._emit("tool_call", f"{name}({self._format_args(args)})", meta={"tool": name, "id": tc_id})

        # Safety scan args before execution
        if self._safety:
            content_to_scan = json.dumps(args)
            verdict = self._safety.check_input(content_to_scan) if hasattr(self._safety, 'check_input') else None
            if verdict and getattr(verdict, 'blocked', False):
                result = f"Tool call blocked by safety gate: {verdict.reason}"
                self._emit("tool_result", result, meta={"tool": name, "blocked": True})
                return result

        # Route to handler
        handlers = {
            "file_read":  self._handle_file_read,
            "file_write": self._handle_file_write,
            "file_edit":  self._handle_file_edit,
            "file_glob":  self._handle_file_glob,
            "file_grep":  self._handle_file_grep,
            "bash":       self._handle_bash,
            "git_status": self._handle_git_status,
            "git_diff":   self._handle_git_diff,
            "git_log":    self._handle_git_log,
            "git_add":    self._handle_git_add,
            "git_commit": self._handle_git_commit,
            "git_show":   self._handle_git_show,
        }

        handler = handlers.get(name)
        if not handler:
            result = f"Unknown tool: {name}"
            self._emit("tool_result", result, meta={"tool": name, "error": True})
            return result

        try:
            from services.tools.bash_tool import ToolResult
            raw = handler(args)
            if isinstance(raw, ToolResult):
                result = raw.as_text()
            else:
                result = str(raw)
        except Exception as exc:
            result = f"Tool execution error: {exc}"
            log.exception("Tool %s raised an exception", name)

        # Emit result (truncate long outputs for event stream, not for LLM)
        preview = result[:300] + "…" if len(result) > 300 else result
        self._emit("tool_result", preview, meta={"tool": name})

        return result

    # ── Individual handlers ────────────────────────────────────────────────

    def _handle_file_read(self, args: dict):
        return self._files.read(
            args["path"],
            start_line=args.get("start_line"),
            end_line=args.get("end_line"),
        )

    def _handle_file_write(self, args: dict):
        from services.permission_engine import ToolCall, PermissionTier
        tc = ToolCall(
            tool="file_write",
            description=f"Write {len(args.get('content',''))} chars to {args['path']}",
            args=args,
        )
        result = self._perms.check(tc)
        if result.tier == PermissionTier.CONFIRM:
            self._emit("confirm_needed", tc.description, meta={"request_id": tc.request_id, "tool": "file_write"})
            approved = self._perms.wait_for_confirm(tc.request_id, timeout=120)
            if not approved:
                from services.tools.bash_tool import ToolResult
                return ToolResult(ok=False, error="User denied file write operation.")
        elif result.tier == PermissionTier.DENY:
            from services.tools.bash_tool import ToolResult
            return ToolResult(ok=False, error="File write denied by policy.")
        return self._files.write(args["path"], args["content"])

    def _handle_file_edit(self, args: dict):
        from services.permission_engine import ToolCall, PermissionTier
        tc = ToolCall(
            tool="file_edit",
            description=f"Edit {args['path']}: replace {repr(args.get('old_string','')[:40])}",
            args=args,
        )
        result = self._perms.check(tc)
        if result.tier == PermissionTier.DENY:
            from services.tools.bash_tool import ToolResult
            return ToolResult(ok=False, error="File edit denied by policy.")
        if result.tier == PermissionTier.CONFIRM:
            self._emit("confirm_needed", tc.description, meta={"request_id": tc.request_id, "tool": "file_edit"})
            approved = self._perms.wait_for_confirm(tc.request_id, timeout=120)
            if not approved:
                from services.tools.bash_tool import ToolResult
                return ToolResult(ok=False, error="User denied file edit operation.")
        return self._files.edit(args["path"], args["old_string"], args["new_string"])

    def _handle_file_glob(self, args: dict):
        return self._files.glob(args["pattern"], limit=args.get("limit", 50))

    def _handle_file_grep(self, args: dict):
        return self._files.grep(
            args["pattern"],
            path=args.get("path", "."),
            recursive=args.get("recursive", True),
        )

    def _handle_bash(self, args: dict):
        from services.permission_engine import ToolCall, PermissionTier
        command = args["command"]
        tc = ToolCall(tool="bash", description=f"Run: {command[:80]}", args=args)
        result = self._perms.check(tc)
        if result.tier == PermissionTier.DENY:
            from services.tools.bash_tool import ToolResult
            return ToolResult(ok=False, error="Bash execution denied by policy.")
        if result.tier == PermissionTier.CONFIRM:
            self._emit("confirm_needed", f"Run bash command: {command}", meta={"request_id": tc.request_id, "tool": "bash", "command": command})
            approved = self._perms.wait_for_confirm(tc.request_id, timeout=120)
            if not approved:
                from services.tools.bash_tool import ToolResult
                return ToolResult(ok=False, error="User denied bash execution.")
        return self._bash.run(command)

    def _handle_git_status(self, args: dict):
        return self._git.status()

    def _handle_git_diff(self, args: dict):
        return self._git.diff(staged=args.get("staged", False), path=args.get("path", ""))

    def _handle_git_log(self, args: dict):
        return self._git.log(n=args.get("n", 10))

    def _handle_git_add(self, args: dict):
        from services.permission_engine import ToolCall, PermissionTier
        paths = args.get("paths", ".")
        tc = ToolCall(tool="git_add", description=f"git add {paths}", args=args)
        result = self._perms.check(tc)
        if result.tier == PermissionTier.DENY:
            from services.tools.bash_tool import ToolResult
            return ToolResult(ok=False, error="git add denied by policy.")
        if result.tier == PermissionTier.CONFIRM:
            self._emit("confirm_needed", tc.description, meta={"request_id": tc.request_id, "tool": "git_add"})
            approved = self._perms.wait_for_confirm(tc.request_id, timeout=120)
            if not approved:
                from services.tools.bash_tool import ToolResult
                return ToolResult(ok=False, error="User denied git add.")
        return self._git.add(paths)

    def _handle_git_commit(self, args: dict):
        from services.permission_engine import ToolCall, PermissionTier
        msg = args.get("message", "")
        tc = ToolCall(tool="git_commit", description=f"git commit: {msg[:60]}", args=args)
        result = self._perms.check(tc)
        if result.tier == PermissionTier.DENY:
            from services.tools.bash_tool import ToolResult
            return ToolResult(ok=False, error="git commit denied by policy.")
        if result.tier == PermissionTier.CONFIRM:
            self._emit("confirm_needed", tc.description, meta={"request_id": tc.request_id, "tool": "git_commit"})
            approved = self._perms.wait_for_confirm(tc.request_id, timeout=120)
            if not approved:
                from services.tools.bash_tool import ToolResult
                return ToolResult(ok=False, error="User denied git commit.")
        return self._git.commit(msg)

    def _handle_git_show(self, args: dict):
        return self._git.show(ref=args.get("ref", "HEAD"))

    # ── Helpers ────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, content: str = "", meta: dict | None = None) -> None:
        try:
            self._on_event(AgentEvent(type=event_type, content=content, meta=meta or {}))
        except Exception:
            pass  # never let event emission crash the loop

    @staticmethod
    def _format_args(args: dict) -> str:
        """Format args dict for display in event stream."""
        parts = []
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 40:
                parts.append(f"{k}={v[:40]!r}…")
            else:
                parts.append(f"{k}={v!r}")
        return ", ".join(parts)
