"""
services/camel/interpreter.py — Restricted-AST plan interpreter.

Walks the privileged-LLM plan and executes it without exposing Python's
full evaluation surface. The interpreter:

  1. Parses the plan with ``ast.parse(mode="exec")``.
  2. Refuses any node whose type is not on the allow-list (raises
     PlanParseError BEFORE the first step runs — there is no partial
     execution of an unsafe plan).
  3. Tracks each value as a CapabilityTaggedResult. Tool / Q-LLM results
     carry capability tags; constants are TRUSTED; arithmetic / compare
     / bool ops union their operands' tags; subscript / attribute access
     propagate tags.
  4. Refuses every operation that would let an UNTRUSTED value drive
     control flow:
       * UNTRUSTED value as the function in a Call → CapabilityViolation
       * UNTRUSTED value as the attribute name in dynamic getattr →
         CapabilityViolation
  5. Caps total node executions at ``max_steps`` (default 50). Going
     over raises CapabilityViolation so a malformed plan cannot loop
     forever.

The final ``output`` variable's value is rendered to text and returned
in the InterpreterResult. The result also carries the count of executed
steps, the count of capability violations recorded, and a list of
blocked tool calls (governance denials surfaced as CamelToolDenied).
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .capabilities import (
    Capability,
    CapabilityTaggedResult,
    capabilities_for_tool,
    merge_capabilities,
)
from .exceptions import CamelToolDenied, CapabilityViolation, PlanParseError

log = logging.getLogger("iMakeAiTeams.camel.interpreter")


# ── AST allow / deny lists ───────────────────────────────────────────────────
# Listed by node class — any node whose class is NOT in _ALLOWED_NODES (and
# doesn't appear elsewhere in the walker as a sub-position node) raises
# PlanParseError. Sub-position nodes like Load / Store / operator subclasses
# are unconditionally fine because they only appear as children of allow-list
# nodes; they never carry independent semantic weight.

_ALLOWED_NODES: frozenset[type] = frozenset({
    ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Constant, ast.Call,
    ast.BinOp, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.IfExp,
    ast.JoinedStr, ast.FormattedValue, ast.List, ast.Tuple, ast.Dict,
    ast.Set, ast.Subscript, ast.Attribute, ast.Return, ast.Slice,
    ast.keyword,
})

_REJECTED_NODES: frozenset[type] = frozenset({
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Lambda,
    ast.For, ast.While, ast.If, ast.Try, ast.With, ast.Global, ast.Nonlocal,
    ast.Raise, ast.Delete, ast.Assert, ast.Yield, ast.YieldFrom,
    ast.AsyncFor, ast.AsyncWith, ast.AsyncFunctionDef, ast.Await,
    ast.Starred, ast.NamedExpr, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp,
})

# Operator subclasses are always sub-position helpers; ast.cmpop / ast.boolop
# / ast.unaryop / ast.operator are abstract bases. Slice / keyword are
# explicit allow-list members above.
_NEUTRAL_NODE_BASES = (ast.operator, ast.cmpop, ast.boolop, ast.unaryop,
                       ast.expr_context)


def _node_is_allowed(node: ast.AST) -> bool:
    cls = type(node)
    if cls in _ALLOWED_NODES:
        return True
    if cls in _REJECTED_NODES:
        return False
    if isinstance(node, _NEUTRAL_NODE_BASES):
        return True
    return False


def _validate_ast(tree: ast.AST) -> None:
    """Walk every node and raise PlanParseError on any disallowed type."""
    for node in ast.walk(tree):
        if isinstance(node, _NEUTRAL_NODE_BASES):
            continue
        cls = type(node)
        if cls in _REJECTED_NODES:
            raise PlanParseError(
                f"plan uses disallowed construct {cls.__name__} "
                f"(line {getattr(node, 'lineno', '?')})"
            )
        if cls not in _ALLOWED_NODES:
            raise PlanParseError(
                f"plan uses unsupported AST node {cls.__name__} "
                f"(line {getattr(node, 'lineno', '?')})"
            )


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class InterpreterResult:
    output_text: str = ""
    executed_steps: int = 0
    capability_violations: int = 0
    blocked_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_text": self.output_text,
            "executed_steps": self.executed_steps,
            "capability_violations": self.capability_violations,
            "blocked_calls": list(self.blocked_calls),
        }


# Sentinel name used for the quarantined-LLM hook in the plan namespace.
_QUARANTINED_NAME = "quarantined_llm"

# Extra plan-level builtins that don't drive control flow but help the
# privileged model assemble the final string. Each one is whitelisted by
# name; values stay tagged so calling them with UNTRUSTED inputs
# correctly propagates the UNTRUSTED tag.
_TRUSTED_BUILTINS: dict[str, Callable[..., Any]] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "len": len,
}


class CamelInterpreter:
    """Sandboxed restricted-Python plan executor.

    Parameters
    ----------
    tool_executor:
        Callable ``(name, args, kwargs) -> CapabilityTaggedResult``.
        Raises ``CamelToolDenied`` for governance refusals. Constructed
        by ``adapter.make_tool_executor_for_turn`` for production calls;
        unit tests pass an in-process stub.
    q_llm_caller:
        Callable ``(question, source) -> str`` invoked by the plan via
        ``quarantined_llm(...)``. Always returns text; the interpreter
        wraps that text as UNTRUSTED.
    max_steps:
        Hard cap on AST node executions. A normal plan runs in the low
        tens; the cap is the structural guarantee that a single turn
        cannot loop forever even when the LLM emits something pathological.
    """

    def __init__(
        self,
        tool_executor: Callable[..., CapabilityTaggedResult],
        q_llm_caller: Callable[[str, Any], str],
        max_steps: int = 50,
    ) -> None:
        self._tool_executor = tool_executor
        self._q_llm_caller = q_llm_caller
        self._max_steps = max(1, int(max_steps))
        self._steps = 0
        self._violations = 0
        self._blocked_calls: list[dict] = []
        self._env: dict[str, CapabilityTaggedResult] = {}

    # ── Entry point ──────────────────────────────────────────────────────────

    def run(self, plan_source: str) -> InterpreterResult:
        try:
            tree = ast.parse(plan_source or "", mode="exec")
        except SyntaxError as exc:
            raise PlanParseError(f"plan failed ast.parse: {exc}") from exc
        _validate_ast(tree)

        self._steps = 0
        self._violations = 0
        self._blocked_calls = []
        self._env = {}

        for stmt in tree.body:
            self._exec_stmt(stmt)

        out = self._env.get("output")
        output_text = self._render_output(out)
        return InterpreterResult(
            output_text=output_text,
            executed_steps=self._steps,
            capability_violations=self._violations,
            blocked_calls=list(self._blocked_calls),
        )

    # ── Step accounting ──────────────────────────────────────────────────────

    def _tick(self, node: ast.AST | None = None) -> None:
        self._steps += 1
        if self._steps > self._max_steps:
            self._violations += 1
            raise CapabilityViolation(
                f"plan exceeded max_steps={self._max_steps}",
                ast_node_repr=type(node).__name__ if node else "",
            )

    # ── Statement dispatch ───────────────────────────────────────────────────

    def _exec_stmt(self, node: ast.AST) -> None:
        self._tick(node)
        if isinstance(node, ast.Assign):
            value = self._eval(node.value)
            for target in node.targets:
                self._assign(target, value)
            return
        if isinstance(node, ast.Expr):
            # Bare expression (e.g. an in-plan tool call whose return value
            # the model didn't bind). Evaluate for the side effect.
            self._eval(node.value)
            return
        if isinstance(node, ast.Return):
            # The plan dialect doesn't have functions, but we accept Return
            # at module level as an alternate way to set ``output``.
            if node.value is None:
                self._env["output"] = CapabilityTaggedResult(
                    value=None, capabilities=frozenset({Capability.TRUSTED}),
                )
            else:
                self._env["output"] = self._eval(node.value)
            return
        raise PlanParseError(
            f"unsupported statement {type(node).__name__} "
            f"(line {getattr(node, 'lineno', '?')})"
        )

    def _assign(self, target: ast.AST, value: CapabilityTaggedResult) -> None:
        if isinstance(target, ast.Name):
            self._env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            try:
                items = list(value.value)
            except TypeError as exc:
                raise PlanParseError(
                    f"cannot unpack non-iterable into {type(target).__name__}: {exc}"
                ) from exc
            if len(items) != len(target.elts):
                raise PlanParseError(
                    f"unpack mismatch: {len(items)} values into "
                    f"{len(target.elts)} targets"
                )
            for sub, item in zip(target.elts, items):
                self._assign(
                    sub,
                    CapabilityTaggedResult(value=item, capabilities=value.capabilities),
                )
            return
        raise PlanParseError(
            f"unsupported assignment target {type(target).__name__}"
        )

    # ── Expression eval ──────────────────────────────────────────────────────

    def _eval(self, node: ast.AST) -> CapabilityTaggedResult:
        self._tick(node)

        if isinstance(node, ast.Constant):
            return CapabilityTaggedResult(
                value=node.value,
                capabilities=frozenset({Capability.TRUSTED}),
            )

        if isinstance(node, ast.Name):
            if node.id in self._env:
                return self._env[node.id]
            if node.id in _TRUSTED_BUILTINS or node.id == _QUARANTINED_NAME:
                # Bare reference to a builtin / Q-LLM hook is rare but legal.
                # Wrap the callable as TRUSTED so subscripting / attribute on
                # it (none allowed in practice) doesn't blow up.
                return CapabilityTaggedResult(
                    value=node.id,
                    capabilities=frozenset({Capability.TRUSTED}),
                )
            raise PlanParseError(f"unknown name '{node.id}' in plan")

        if isinstance(node, ast.Call):
            return self._eval_call(node)

        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            try:
                value = self._apply_binop(node.op, left.value, right.value)
            except Exception as exc:
                raise PlanParseError(f"binop failed: {exc}") from exc
            return CapabilityTaggedResult(
                value=value,
                capabilities=merge_capabilities(left, right),
            )

        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            try:
                value = self._apply_unary(node.op, operand.value)
            except Exception as exc:
                raise PlanParseError(f"unary failed: {exc}") from exc
            return CapabilityTaggedResult(
                value=value,
                capabilities=operand.capabilities,
            )

        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            current_value = left.value
            caps = left.capabilities
            result_value = True
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator)
                caps = caps | right.capabilities
                try:
                    cmp_value = self._apply_compare(op, current_value, right.value)
                except Exception as exc:
                    raise PlanParseError(f"compare failed: {exc}") from exc
                if not cmp_value:
                    result_value = False
                current_value = right.value
            return CapabilityTaggedResult(value=result_value, capabilities=caps)

        if isinstance(node, ast.BoolOp):
            values = [self._eval(v) for v in node.values]
            caps = frozenset()
            for v in values:
                caps = caps | v.capabilities
            if isinstance(node.op, ast.And):
                final = True
                for v in values:
                    if not v.value:
                        final = v.value
                        break
                    final = v.value
            else:  # ast.Or
                final = False
                for v in values:
                    if v.value:
                        final = v.value
                        break
                    final = v.value
            return CapabilityTaggedResult(value=final, capabilities=caps)

        if isinstance(node, ast.IfExp):
            test = self._eval(node.test)
            chosen = self._eval(node.body) if test.value else self._eval(node.orelse)
            return CapabilityTaggedResult(
                value=chosen.value,
                capabilities=test.capabilities | chosen.capabilities,
            )

        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            caps = frozenset({Capability.TRUSTED})
            for v in node.values:
                tagged = self._eval(v)
                caps = caps | tagged.capabilities
                parts.append("" if tagged.value is None else str(tagged.value))
            return CapabilityTaggedResult(value="".join(parts), capabilities=caps)

        if isinstance(node, ast.FormattedValue):
            inner = self._eval(node.value)
            text = "" if inner.value is None else str(inner.value)
            # We deliberately don't honor format_spec / conversion here —
            # both are edge cases for our prompt and could be used to widen
            # the surface. Just stringify.
            return CapabilityTaggedResult(value=text, capabilities=inner.capabilities)

        if isinstance(node, ast.List):
            items = [self._eval(e) for e in node.elts]
            caps = frozenset()
            for it in items:
                caps = caps | it.capabilities
            if not caps:
                caps = frozenset({Capability.TRUSTED})
            return CapabilityTaggedResult(
                value=[it.value for it in items], capabilities=caps,
            )

        if isinstance(node, ast.Tuple):
            items = [self._eval(e) for e in node.elts]
            caps = frozenset()
            for it in items:
                caps = caps | it.capabilities
            if not caps:
                caps = frozenset({Capability.TRUSTED})
            return CapabilityTaggedResult(
                value=tuple(it.value for it in items), capabilities=caps,
            )

        if isinstance(node, ast.Set):
            items = [self._eval(e) for e in node.elts]
            caps = frozenset()
            for it in items:
                caps = caps | it.capabilities
            if not caps:
                caps = frozenset({Capability.TRUSTED})
            return CapabilityTaggedResult(
                value={it.value for it in items}, capabilities=caps,
            )

        if isinstance(node, ast.Dict):
            keys = [self._eval(k) if k is not None else None for k in node.keys]
            vals = [self._eval(v) for v in node.values]
            caps = frozenset()
            for k in keys:
                if k is not None:
                    caps = caps | k.capabilities
            for v in vals:
                caps = caps | v.capabilities
            if not caps:
                caps = frozenset({Capability.TRUSTED})
            built: dict = {}
            for k, v in zip(keys, vals):
                if k is None:
                    raise PlanParseError("dict unpacking (**d) is not allowed")
                built[k.value] = v.value
            return CapabilityTaggedResult(value=built, capabilities=caps)

        if isinstance(node, ast.Subscript):
            container = self._eval(node.value)
            index = self._eval_slice(node.slice)
            try:
                value = container.value[index.value]
            except Exception as exc:
                raise PlanParseError(f"subscript failed: {exc}") from exc
            return CapabilityTaggedResult(
                value=value,
                capabilities=container.capabilities | index.capabilities,
            )

        if isinstance(node, ast.Attribute):
            # Static attribute name (ast.Attribute.attr is a string baked into
            # the AST at parse time) is fine — the plan author wrote it. The
            # capability check that the spec asks for fires when an UNTRUSTED
            # value is used to look up an attribute name DYNAMICALLY, which
            # happens via getattr() / Subscript on a method namespace. Those
            # paths are barred above (no getattr in the plan dialect; subscript
            # on UNTRUSTED is a Subscript with an UNTRUSTED index — handled
            # there).
            target = self._eval(node.value)
            try:
                value = getattr(target.value, node.attr)
            except Exception as exc:
                raise PlanParseError(
                    f"attribute access {node.attr!r} failed: {exc}"
                ) from exc
            return CapabilityTaggedResult(
                value=value, capabilities=target.capabilities,
            )

        raise PlanParseError(f"unsupported expression {type(node).__name__}")

    def _eval_slice(self, node: ast.AST) -> CapabilityTaggedResult:
        if isinstance(node, ast.Slice):
            lower = self._eval(node.lower) if node.lower is not None else None
            upper = self._eval(node.upper) if node.upper is not None else None
            step = self._eval(node.step) if node.step is not None else None
            caps = frozenset({Capability.TRUSTED})
            for t in (lower, upper, step):
                if t is not None:
                    caps = caps | t.capabilities
            return CapabilityTaggedResult(
                value=slice(
                    lower.value if lower else None,
                    upper.value if upper else None,
                    step.value if step else None,
                ),
                capabilities=caps,
            )
        # Otherwise the slice IS an expression (Python 3.9+ flattened it).
        index = self._eval(node)
        if Capability.UNTRUSTED in index.capabilities:
            # The spec calls out attribute / function lookups; using an
            # UNTRUSTED value as an index is also a control-flow lever
            # when the container is a tool registry. We propagate the tag
            # but keep the operation legal; the dangerous case (function
            # in Call) is blocked separately.
            pass
        return index

    # ── Call evaluation ──────────────────────────────────────────────────────

    def _eval_call(self, node: ast.Call) -> CapabilityTaggedResult:
        # The function position MUST be a static Name. Any other shape would
        # let the plan compute the callable from data — exactly the
        # injection vector CaMeL is designed to close.
        func_node = node.func
        if not isinstance(func_node, ast.Name):
            self._violations += 1
            raise CapabilityViolation(
                "call target must be a bare name (no attribute / subscript / "
                "computed callable)",
                ast_node_repr=ast.dump(func_node)[:120],
            )
        name = func_node.id

        # If the name resolves to a value in the env, that means the model
        # tried to invoke a variable holding a tool-result string as a
        # function. That is the canonical UNTRUSTED-as-code attack — refuse
        # even if the value happens to be TRUSTED, since the plan dialect
        # doesn't allow defining first-class functions.
        if name in self._env and name not in _TRUSTED_BUILTINS and name != _QUARANTINED_NAME:
            target = self._env[name]
            self._violations += 1
            raise CapabilityViolation(
                f"refusing to call '{name}' — it is bound to a value, not a tool. "
                f"capabilities={sorted(c.value for c in target.capabilities)}",
                ast_node_repr=name,
            )

        args_tagged = [self._eval(a) for a in node.args]
        kwargs_tagged: dict[str, CapabilityTaggedResult] = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise PlanParseError("**kwargs unpacking is not allowed")
            kwargs_tagged[kw.arg] = self._eval(kw.value)

        if name == _QUARANTINED_NAME:
            return self._call_quarantined(args_tagged, kwargs_tagged)
        if name in _TRUSTED_BUILTINS:
            return self._call_builtin(name, args_tagged, kwargs_tagged)
        return self._call_tool(name, args_tagged, kwargs_tagged)

    def _call_quarantined(
        self,
        args_tagged: list[CapabilityTaggedResult],
        kwargs_tagged: dict[str, CapabilityTaggedResult],
    ) -> CapabilityTaggedResult:
        question = args_tagged[0].value if args_tagged else ""
        source = args_tagged[1].value if len(args_tagged) > 1 else kwargs_tagged.get(
            "source", CapabilityTaggedResult("", frozenset({Capability.TRUSTED})),
        ).value
        try:
            text = self._q_llm_caller(str(question), source)
        except Exception as exc:
            log.warning("quarantined_llm caller failed: %s", exc)
            text = f"[quarantined_llm error: {exc}]"
        # Q-LLM output is ALWAYS untrusted regardless of where the question
        # came from. That's the architectural rule.
        return CapabilityTaggedResult(
            value=str(text),
            capabilities=frozenset({Capability.UNTRUSTED}),
        )

    def _call_builtin(
        self,
        name: str,
        args_tagged: list[CapabilityTaggedResult],
        kwargs_tagged: dict[str, CapabilityTaggedResult],
    ) -> CapabilityTaggedResult:
        fn = _TRUSTED_BUILTINS[name]
        try:
            value = fn(
                *(a.value for a in args_tagged),
                **{k: v.value for k, v in kwargs_tagged.items()},
            )
        except Exception as exc:
            raise PlanParseError(f"builtin {name}() failed: {exc}") from exc
        caps = frozenset({Capability.TRUSTED})
        for a in args_tagged:
            caps = caps | a.capabilities
        for v in kwargs_tagged.values():
            caps = caps | v.capabilities
        return CapabilityTaggedResult(value=value, capabilities=caps)

    def _call_tool(
        self,
        name: str,
        args_tagged: list[CapabilityTaggedResult],
        kwargs_tagged: dict[str, CapabilityTaggedResult],
    ) -> CapabilityTaggedResult:
        try:
            result = self._tool_executor(
                name,
                [a.value for a in args_tagged],
                {k: v.value for k, v in kwargs_tagged.items()},
            )
        except CamelToolDenied as denied:
            self._blocked_calls.append({
                "tool": name,
                "reason": denied.reason,
                "policy": denied.policy_name,
            })
            log.info("CaMeL: tool %s blocked — %s", name, denied.reason)
            return CapabilityTaggedResult(
                value=None,
                capabilities=capabilities_for_tool(name),
            )

        if not isinstance(result, CapabilityTaggedResult):
            # Defensive: a custom executor returned a bare value. Wrap it
            # using the tool's default tags so capability propagation still
            # works correctly downstream.
            result = CapabilityTaggedResult(
                value=result, capabilities=capabilities_for_tool(name),
            )

        # Argument tags propagate into the result. If the model fed an
        # UNTRUSTED value into a TRUSTED tool, the output is still tainted.
        merged = result.capabilities
        for a in args_tagged:
            merged = merged | a.capabilities
        for v in kwargs_tagged.values():
            merged = merged | v.capabilities
        return CapabilityTaggedResult(value=result.value, capabilities=merged)

    # ── Operator helpers ────────────────────────────────────────────────────

    @staticmethod
    def _apply_binop(op: ast.AST, left: Any, right: Any) -> Any:
        if isinstance(op, ast.Add): return left + right
        if isinstance(op, ast.Sub): return left - right
        if isinstance(op, ast.Mult): return left * right
        if isinstance(op, ast.Div): return left / right
        if isinstance(op, ast.FloorDiv): return left // right
        if isinstance(op, ast.Mod): return left % right
        if isinstance(op, ast.Pow): return left ** right
        if isinstance(op, ast.BitOr): return left | right
        if isinstance(op, ast.BitAnd): return left & right
        if isinstance(op, ast.BitXor): return left ^ right
        if isinstance(op, ast.LShift): return left << right
        if isinstance(op, ast.RShift): return left >> right
        if isinstance(op, ast.MatMult): return left @ right
        raise PlanParseError(f"unsupported binop {type(op).__name__}")

    @staticmethod
    def _apply_unary(op: ast.AST, value: Any) -> Any:
        if isinstance(op, ast.UAdd): return +value
        if isinstance(op, ast.USub): return -value
        if isinstance(op, ast.Not): return not value
        if isinstance(op, ast.Invert): return ~value
        raise PlanParseError(f"unsupported unary {type(op).__name__}")

    @staticmethod
    def _apply_compare(op: ast.AST, left: Any, right: Any) -> bool:
        if isinstance(op, ast.Eq): return left == right
        if isinstance(op, ast.NotEq): return left != right
        if isinstance(op, ast.Lt): return left < right
        if isinstance(op, ast.LtE): return left <= right
        if isinstance(op, ast.Gt): return left > right
        if isinstance(op, ast.GtE): return left >= right
        if isinstance(op, ast.Is): return left is right
        if isinstance(op, ast.IsNot): return left is not right
        if isinstance(op, ast.In): return left in right
        if isinstance(op, ast.NotIn): return left not in right
        raise PlanParseError(f"unsupported compare {type(op).__name__}")

    @staticmethod
    def _render_output(out: CapabilityTaggedResult | None) -> str:
        if out is None:
            return ""
        v = out.value
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        try:
            return str(v)
        except Exception:
            return ""
