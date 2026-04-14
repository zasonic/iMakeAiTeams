"""
services/hooks.py

User-configurable hook system.

Hook points:
    pre_send        — before a message is sent to any model
    post_response   — after a response is received
    pre_route       — before the router classifies complexity
    post_route      — after routing decision is made
    pre_workflow    — before a workflow task executes
    post_workflow   — after a workflow task completes

Hooks are Python expressions stored in settings.json under "hooks".
Each hook receives a context dict and can:
    - Modify the context (e.g. append to system prompt, log, filter)
    - Return a dict to override values
    - Return None to pass through unchanged

Example hook config in settings.json:
{
    "hooks": {
        "post_response": [
            {
                "name": "auto_save_long_responses",
                "enabled": true,
                "action": "log",
                "condition": "len(ctx.get('response', '')) > 2000",
                "description": "Log when responses exceed 2000 chars"
            }
        ]
    }
}
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("MyAIEnv.hooks")


# ── Hook points ───────────────────────────────────────────────────────────────

VALID_HOOK_POINTS = {
    "pre_send",
    "post_response",
    "pre_route",
    "post_route",
    "pre_workflow",
    "post_workflow",
}


# ── Built-in hook actions ─────────────────────────────────────────────────────

def _action_log(ctx: dict, hook_cfg: dict) -> dict | None:
    """Log the hook event."""
    log.info("[hook:%s] %s — ctx keys: %s",
             hook_cfg.get("name", "unnamed"),
             hook_cfg.get("description", ""),
             list(ctx.keys()))
    return None


def _action_inject_system(ctx: dict, hook_cfg: dict) -> dict | None:
    """Append text to the system prompt."""
    extra = hook_cfg.get("inject_text", "")
    if extra and "system_prompt" in ctx:
        ctx["system_prompt"] = ctx["system_prompt"] + "\n\n" + extra
        return ctx
    return None


def _action_block(ctx: dict, hook_cfg: dict) -> dict | None:
    """Block the operation by setting a blocked flag."""
    reason = hook_cfg.get("block_reason", "Blocked by hook")
    ctx["_blocked"] = True
    ctx["_block_reason"] = reason
    return ctx


def _action_notify(ctx: dict, hook_cfg: dict) -> dict | None:
    """Set a notification flag for the frontend."""
    ctx["_notification"] = hook_cfg.get("notify_text", "Hook triggered")
    return ctx


_BUILTIN_ACTIONS: dict[str, Callable] = {
    "log":            _action_log,
    "inject_system":  _action_inject_system,
    "block":          _action_block,
    "notify":         _action_notify,
}


# ── Hook evaluation ──────────────────────────────────────────────────────────

def _evaluate_condition(condition: str, ctx: dict) -> bool:
    """
    Safely evaluate a hook condition expression.
    Only allows access to `ctx`, `len`, `str`, `int`, `bool`, `True`, `False`.
    """
    if not condition or not condition.strip():
        return True  # no condition = always fire

    safe_globals = {"__builtins__": {}}
    safe_locals  = {
        "ctx":   ctx,
        "len":   len,
        "str":   str,
        "int":   int,
        "bool":  bool,
        "True":  True,
        "False": False,
        "None":  None,
    }
    try:
        return bool(eval(condition, safe_globals, safe_locals))  # noqa: S307
    except Exception as exc:
        log.debug("Hook condition eval failed (%s): %s", condition, exc)
        return False


# ── HookManager ──────────────────────────────────────────────────────────────

class HookManager:
    """
    Manages user-configured hooks.  Reads config from Settings,
    fires hooks at the appropriate points.
    """

    def __init__(self, settings):
        self._settings = settings
        self._custom_actions: dict[str, Callable] = {}
        self._execution_log: list[dict] = []  # last 100 hook executions

    def register_action(self, name: str, fn: Callable) -> None:
        """Register a custom hook action callable."""
        self._custom_actions[name] = fn

    def get_hooks(self, hook_point: str) -> list[dict]:
        """Return all hook configs for a given hook point."""
        all_hooks = self._settings.get("hooks", {})
        return all_hooks.get(hook_point, [])

    def set_hooks(self, hook_point: str, hooks: list[dict]) -> None:
        """Save hooks for a given hook point."""
        if hook_point not in VALID_HOOK_POINTS:
            raise ValueError(f"Invalid hook point: {hook_point}")
        all_hooks = self._settings.get("hooks", {})
        all_hooks[hook_point] = hooks
        self._settings.set("hooks", all_hooks)

    def add_hook(self, hook_point: str, hook_cfg: dict) -> dict:
        """Add a single hook to a hook point. Returns the added config."""
        if hook_point not in VALID_HOOK_POINTS:
            return {"error": f"Invalid hook point: {hook_point}"}

        # Defaults
        hook_cfg.setdefault("name", f"hook_{int(time.time())}")
        hook_cfg.setdefault("enabled", True)
        hook_cfg.setdefault("action", "log")
        hook_cfg.setdefault("condition", "")
        hook_cfg.setdefault("description", "")

        hooks = self.get_hooks(hook_point)
        hooks.append(hook_cfg)
        self.set_hooks(hook_point, hooks)
        return hook_cfg

    def remove_hook(self, hook_point: str, hook_name: str) -> bool:
        """Remove a hook by name from a hook point."""
        hooks = self.get_hooks(hook_point)
        before = len(hooks)
        hooks = [h for h in hooks if h.get("name") != hook_name]
        if len(hooks) < before:
            self.set_hooks(hook_point, hooks)
            return True
        return False

    def toggle_hook(self, hook_point: str, hook_name: str, enabled: bool) -> bool:
        """Enable/disable a hook by name."""
        hooks = self.get_hooks(hook_point)
        for h in hooks:
            if h.get("name") == hook_name:
                h["enabled"] = enabled
                self.set_hooks(hook_point, hooks)
                return True
        return False

    def fire(self, hook_point: str, ctx: dict) -> dict:
        """
        Fire all enabled hooks for a hook point.
        Mutates and returns ctx.  Hooks run in order.
        """
        hooks = self.get_hooks(hook_point)
        if not hooks:
            return ctx

        for hook_cfg in hooks:
            if not hook_cfg.get("enabled", True):
                continue

            # Evaluate condition
            condition = hook_cfg.get("condition", "")
            if not _evaluate_condition(condition, ctx):
                continue

            # Execute action
            action_name = hook_cfg.get("action", "log")
            action_fn = (
                self._custom_actions.get(action_name)
                or _BUILTIN_ACTIONS.get(action_name)
            )

            if not action_fn:
                log.warning("Unknown hook action: %s", action_name)
                continue

            try:
                result = action_fn(ctx, hook_cfg)
                if isinstance(result, dict):
                    ctx.update(result)

                # Log execution
                self._log_execution(hook_point, hook_cfg, True)

            except Exception as exc:
                log.warning("Hook %s/%s failed: %s",
                            hook_point, hook_cfg.get("name"), exc)
                self._log_execution(hook_point, hook_cfg, False, str(exc))

        return ctx

    def _log_execution(self, hook_point: str, hook_cfg: dict,
                       success: bool, error: str = "") -> None:
        entry = {
            "hook_point": hook_point,
            "hook_name":  hook_cfg.get("name", ""),
            "action":     hook_cfg.get("action", ""),
            "success":    success,
            "error":      error,
            "timestamp":  time.time(),
        }
        self._execution_log.append(entry)
        # Keep only last 100
        if len(self._execution_log) > 100:
            self._execution_log = self._execution_log[-100:]

    def get_execution_log(self, limit: int = 50) -> list[dict]:
        """Return recent hook execution log entries."""
        return list(reversed(self._execution_log[-limit:]))

    def list_hook_points(self) -> list[dict]:
        """Return all valid hook points with their current hook count."""
        all_hooks = self._settings.get("hooks", {})
        return [
            {
                "hook_point": hp,
                "description": _HOOK_DESCRIPTIONS.get(hp, ""),
                "hook_count": len(all_hooks.get(hp, [])),
            }
            for hp in sorted(VALID_HOOK_POINTS)
        ]

    def list_actions(self) -> list[str]:
        """Return all available action names."""
        return sorted(set(list(_BUILTIN_ACTIONS.keys()) +
                          list(self._custom_actions.keys())))


_HOOK_DESCRIPTIONS = {
    "pre_send":       "Fires before a message is sent to any model. "
                      "Context: user_message, system_prompt, model, conversation_id.",
    "post_response":  "Fires after a model response is received. "
                      "Context: response, model, tokens_in, tokens_out, cost_usd.",
    "pre_route":      "Fires before the smart router classifies message complexity. "
                      "Context: user_message, conversation_id.",
    "post_route":     "Fires after routing decision. "
                      "Context: route_model, complexity, reasoning.",
    "pre_workflow":   "Fires before a workflow task executes. "
                      "Context: task_name, agent_role, workflow_id.",
    "post_workflow":  "Fires after a workflow task completes. "
                      "Context: task_name, agent_role, status, output_data.",
}
