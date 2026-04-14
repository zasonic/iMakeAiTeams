"""
services/tools/code_runner.py — Programmatic tool calling via sandboxed code execution.

Allows Claude to write Python code that calls tools programmatically,
processes large intermediate results in code scope (not in context),
and returns only the final summary to the conversation.

This is the highest-leverage token optimization pattern from Anthropic's
research: instead of reading large files into context and reasoning over
them, Claude writes code that processes the data and returns a digest.

Security:
  - Code runs in restricted exec() with only safe builtins + tool dispatch
  - No file I/O except through the tool dispatch function
  - No imports except a small allowlist (json, re, math, collections)
  - Output capped at MAX_OUTPUT_CHARS
  - Timeout enforced via threading
"""

import json
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger("MyAIAgentHub.code_runner")

MAX_OUTPUT_CHARS = 30_000
EXEC_TIMEOUT = 30  # seconds

# Safe builtins allowlist
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "filter": filter,
    "float": float, "frozenset": frozenset, "getattr": getattr,
    "hasattr": hasattr, "hash": hash, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next,
    "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type,
    "zip": zip, "True": True, "False": False, "None": None,
}

# Safe modules that can be imported in code
_SAFE_MODULES = {"json", "re", "math", "collections", "itertools", "functools", "textwrap"}


class CodeRunner:
    """
    Execute Python code that can call agent tools programmatically.

    The code receives a `tool(name, args_dict)` function that dispatches
    to the same tool handlers used by the agent loop.
    """

    def __init__(self, tool_dispatch: Callable[[str, dict], str]) -> None:
        """
        Parameters
        ----------
        tool_dispatch : Function that takes (tool_name, args_dict) and returns
                       the tool result as a string.
        """
        self._dispatch = tool_dispatch

    def execute(self, code: str) -> "CodeResult":
        """
        Execute Python code in a restricted sandbox.

        The code has access to:
          - tool(name, args) — call any agent tool
          - Safe builtins (no open, exec, eval, __import__ beyond allowlist)
          - Safe modules (json, re, math, collections)
          - A `result` variable — set this to return data to the agent

        Returns a CodeResult with the output.
        """
        if not code or not code.strip():
            return CodeResult(ok=False, error="Empty code.")

        # Build restricted globals
        output_capture = []

        def _safe_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            output_capture.append(text)

        def _safe_import(name, *args, **kwargs):
            if name in _SAFE_MODULES:
                return __import__(name)
            raise ImportError(f"Import of '{name}' is not allowed. Safe modules: {', '.join(sorted(_SAFE_MODULES))}")

        def _tool_call(name: str, args: dict = None) -> str:
            """Call an agent tool from code."""
            return self._dispatch(name, args or {})

        safe_globals = {
            "__builtins__": {**_SAFE_BUILTINS, "__import__": _safe_import, "print": _safe_print},
            "tool": _tool_call,
            "result": None,
        }

        # Execute with timeout
        exec_result = {"error": None, "completed": False}

        def _run():
            try:
                exec(code, safe_globals)  # noqa: S102
                exec_result["completed"] = True
            except Exception as exc:
                exec_result["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=EXEC_TIMEOUT)

        if thread.is_alive():
            return CodeResult(ok=False, error=f"Code execution timed out after {EXEC_TIMEOUT}s.")

        if exec_result["error"]:
            return CodeResult(ok=False, error=exec_result["error"])

        # Collect output
        final_result = safe_globals.get("result")
        printed = "\n".join(output_capture) if output_capture else ""

        if final_result is not None:
            try:
                output = json.dumps(final_result, indent=2, default=str)
            except (TypeError, ValueError):
                output = str(final_result)
        elif printed:
            output = printed
        else:
            output = "[Code executed successfully with no output]"

        # Truncate
        truncated = False
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS]
            truncated = True

        return CodeResult(ok=True, output=output, truncated=truncated)


@dataclass
class CodeResult:
    ok: bool
    output: str = ""
    error: str = ""
    truncated: bool = False

    def as_text(self) -> str:
        if self.ok:
            parts = [self.output]
            if self.truncated:
                parts.append(f"[Output truncated at {MAX_OUTPUT_CHARS} chars]")
            return "\n".join(parts)
        return f"[Code execution error]\n{self.error}"
