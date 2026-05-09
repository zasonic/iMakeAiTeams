"""
services/camel/pipeline.py — End-to-end CaMeL turn driver.

Brings the pieces together for a single user turn:

  1. Ask the privileged LLM for a plan (P_LLM_SYSTEM_PROMPT + the user's
     raw message). The privileged client never sees retrieved chunks —
     that's the whole point of the split.
  2. Strip ```python / ``` fences if the model wrapped the plan.
  3. Build a quarantined-LLM caller closure that, on each invocation
     from inside the plan, prompts the quarantined model with
     Q_LLM_SYSTEM_PROMPT and the chunks (or a chunks-subset selected
     from the plan's source argument).
  4. Run the plan through CamelInterpreter with the supplied
     tool_executor.
  5. Return ``output_text`` + bookkeeping for camel_log persistence.

Failure modes are surfaced through the returned dict's ``error`` key
rather than re-raised — the orchestrator wants a usable ChatResult even
when the plan is malformed, so this layer logs the row and degrades
gracefully.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Iterable

from .interpreter import CamelInterpreter, InterpreterResult
from .exceptions import CapabilityViolation, PlanParseError
from .prompts import P_LLM_SYSTEM_PROMPT, Q_LLM_SYSTEM_PROMPT

log = logging.getLogger("iMakeAiTeams.camel.pipeline")


_FENCE_RE = re.compile(
    r"^\s*```(?:python|py)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE,
)


def _strip_fences(text: str) -> str:
    """Remove a single surrounding ```python ... ``` fence if present.

    The privileged model ignores instructions sometimes and wraps the
    plan in markdown. Stripping is best-effort — a partial / malformed
    fence is left to ast.parse to reject with a clear error.
    """
    if not text:
        return ""
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _render_chunks(source: Any) -> str:
    """Stringify whatever the plan passed as the ``source`` argument to
    quarantined_llm() into a single chunked text block.

    Accepts: str, list/tuple of str, or any iterable of str. Falls back
    to repr() so a misbehaving plan can't crash the quarantined call.
    """
    if source is None:
        return ""
    if isinstance(source, str):
        return source
    if isinstance(source, (list, tuple, set, frozenset)):
        return "\n\n---\n\n".join(str(x) for x in source)
    try:
        return "\n\n---\n\n".join(str(x) for x in source)
    except TypeError:
        return repr(source)


def _call_llm(client: Any, system: str, user: str, max_tokens: int = 2048) -> str:
    """Best-effort generic invocation across the LLM client surfaces this
    codebase exposes. Tries ``chat_unified`` first, falls back to
    ``chat`` then ``chat_multi_turn``. Returns "" when none are usable.

    Tests pass a MagicMock with one of these methods configured. The
    generic walk avoids forcing a single API shape on every caller.
    """
    if client is None:
        return ""
    payload_messages = [{"role": "user", "content": user}]
    if hasattr(client, "chat_unified"):
        try:
            res = client.chat_unified(system, payload_messages, max_tokens=max_tokens)
            if isinstance(res, dict):
                return str(res.get("text") or "")
            return str(res or "")
        except TypeError:
            try:
                res = client.chat_unified(system, payload_messages)
                if isinstance(res, dict):
                    return str(res.get("text") or "")
                return str(res or "")
            except Exception:
                pass
        except Exception as exc:
            log.debug("chat_unified failed: %s", exc)
    if hasattr(client, "chat"):
        try:
            return str(client.chat(system, user, max_tokens=max_tokens) or "")
        except TypeError:
            try:
                return str(client.chat(system, user) or "")
            except Exception:
                pass
        except Exception as exc:
            log.debug("chat failed: %s", exc)
    if hasattr(client, "chat_multi_turn"):
        try:
            res = client.chat_multi_turn(system, payload_messages, max_tokens=max_tokens)
            if isinstance(res, dict):
                return str(res.get("text") or "")
            return str(res or "")
        except Exception as exc:
            log.debug("chat_multi_turn failed: %s", exc)
    return ""


def camel_plan_and_execute(
    user_message: str,
    retrieved_chunks: Iterable[str] | None,
    privileged_client: Any,
    quarantined_client: Any,
    tool_executor: Callable[..., Any],
    *,
    max_steps: int = 50,
) -> dict:
    """Run one turn through the CaMeL pipeline.

    Returns a dict with the keys::

        plan_source            — the raw plan text the privileged LLM emitted
        output_text            — the plan's final ``output`` rendered as text
        executed_steps         — InterpreterResult.executed_steps
        capability_violations  — InterpreterResult.capability_violations
        blocked_calls          — list[dict] of governance denials
        error                  — "" on success, else short reason string

    The orchestrator persists every field except ``error`` into the
    camel_log table, and uses ``output_text`` as the assistant reply.
    """
    chunks = list(retrieved_chunks or [])

    # ── 1. Privileged LLM emits a plan ──────────────────────────────────────
    plan_raw = _call_llm(
        privileged_client,
        P_LLM_SYSTEM_PROMPT,
        user_message,
        max_tokens=1024,
    )
    plan_source = _strip_fences(plan_raw)

    # ── 2. Build quarantined-LLM closure ───────────────────────────────────
    def _q_llm(question: str, source: Any) -> str:
        # When the plan didn't pass anything for source, default to all
        # retrieved chunks. When it passed a name like ``retrieved_chunks``,
        # the interpreter has already resolved that to its UNTRUSTED value.
        body = _render_chunks(source) or _render_chunks(chunks)
        user = f"QUESTION: {question}\n\nDATA:\n{body}".strip()
        return _call_llm(
            quarantined_client,
            Q_LLM_SYSTEM_PROMPT,
            user,
            max_tokens=1024,
        )

    # ── 3. Run the interpreter ──────────────────────────────────────────────
    interpreter = CamelInterpreter(
        tool_executor=tool_executor,
        q_llm_caller=_q_llm,
        max_steps=max_steps,
    )

    error = ""
    output_text = ""
    executed_steps = 0
    capability_violations = 0
    blocked_calls: list[dict] = []

    if not plan_source.strip():
        error = "privileged LLM returned an empty plan"
    else:
        try:
            result: InterpreterResult = interpreter.run(plan_source)
            output_text = result.output_text
            executed_steps = result.executed_steps
            capability_violations = result.capability_violations
            blocked_calls = list(result.blocked_calls)
            if not output_text:
                # The plan didn't bind ``output`` — fall back to the
                # quarantined narrative on the full chunk set so the user
                # still gets a textual answer instead of an empty bubble.
                output_text = _q_llm("Summarise the available context", chunks)
                error = "plan did not set 'output'; fell back to quarantined summary"
        except PlanParseError as exc:
            error = f"plan parse error: {exc.message}"
            log.warning("CaMeL plan parse error: %s", exc.message)
        except CapabilityViolation as exc:
            capability_violations += 1
            error = f"capability violation: {exc.message}"
            log.warning("CaMeL capability violation: %s", exc.message)

    return {
        "plan_source": plan_source,
        "output_text": output_text,
        "executed_steps": executed_steps,
        "capability_violations": capability_violations,
        "blocked_calls": blocked_calls,
        "error": error,
    }
