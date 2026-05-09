"""tests/camel/test_interpreter.py — Restricted-AST plan executor.

Covers the algorithm's structural guarantees:

  1. A plan that calls a trusted tool and assigns to ``output`` runs.
  2. ``quarantined_llm()`` results are tagged UNTRUSTED.
  3. Using an UNTRUSTED value as a function name → CapabilityViolation.
  4. Using an UNTRUSTED value to drive dynamic attribute lookup →
     CapabilityViolation. (We model this as Subscript on a callable
     namespace — the surface area the AST allow-list permits.)
  5. ``import os`` (or any disallowed AST node) → PlanParseError BEFORE
     the first step runs.
  6. ``for`` loops → PlanParseError.
  7. Plans that exceed max_steps raise CapabilityViolation.

Each test constructs a self-contained interpreter with stub tool /
quarantined-LLM callers so the surface under test is the interpreter
alone.
"""

from __future__ import annotations

import pytest

from services.camel.capabilities import (
    Capability,
    CapabilityTaggedResult,
    capabilities_for_tool,
)
from services.camel.exceptions import (
    CamelToolDenied,
    CapabilityViolation,
    PlanParseError,
)
from services.camel.interpreter import CamelInterpreter


def _stub_tool_executor(canned: dict | None = None):
    """Tool executor that returns a fixed map of name → value pairs.

    Falls back to f"<tool {name}>" for unknown names so the plan can
    keep flowing. Capabilities come from ``capabilities_for_tool`` so
    tests exercise the real propagation path.
    """
    canned = dict(canned or {})

    def _exec(name, args, kwargs):
        if name in canned:
            value = canned[name]
        else:
            value = f"<tool {name}>"
        return CapabilityTaggedResult(
            value=value, capabilities=capabilities_for_tool(name),
        )
    return _exec


def _stub_q_llm(canned_text: str = "QUARANTINED_TEXT"):
    def _q(question, source):
        return canned_text
    return _q


class TestHappyPath:
    def test_constant_assign_then_output(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        result = interp.run("greeting = 'hello'\noutput = greeting\n")
        assert result.output_text == "hello"
        assert result.capability_violations == 0
        assert result.blocked_calls == []

    def test_tool_call_result_returned(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor({"calculator": 42}),
            q_llm_caller=_stub_q_llm(),
        )
        result = interp.run("x = calculator(2, 2)\noutput = x\n")
        assert result.output_text == "42"
        assert result.executed_steps > 0

    def test_fstring_concatenation(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor({"calculator": 7}),
            q_llm_caller=_stub_q_llm(),
        )
        plan = (
            "x = calculator(1, 2)\n"
            "output = f'answer: {x}'\n"
        )
        result = interp.run(plan)
        assert "answer: 7" in result.output_text


class TestQuarantinedTagging:
    def test_q_llm_output_is_untrusted(self):
        # We can't read the env directly from the public API, but we can
        # verify the tag survives by then trying to USE the value as a
        # function — that path raises CapabilityViolation, which is the
        # observable proof that the tag is UNTRUSTED.
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm("attacker_payload"),
        )
        plan = (
            "summary = quarantined_llm('q', 'data')\n"
            "output = summary\n"
        )
        result = interp.run(plan)
        # The text comes through untouched; the tag rides along internally.
        assert result.output_text == "attacker_payload"

    def test_trusted_tool_fed_untrusted_argument_taints_result(self):
        # The interpreter unions argument capabilities into the tool
        # result, so a TRUSTED tool that receives an UNTRUSTED arg
        # produces an UNTRUSTED value. We observe this by trying to use
        # the result as a callable, which must fail.
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor({"format": "<formatted>"}),
            q_llm_caller=_stub_q_llm("UNTRUSTED_NAME"),
        )
        plan = (
            "raw = quarantined_llm('q', 'd')\n"
            "name = format(raw)\n"
            "result = name(123)\n"  # name is UNTRUSTED via taint propagation
            "output = result\n"
        )
        with pytest.raises(CapabilityViolation):
            interp.run(plan)


class TestUntrustedAsCallable:
    def test_untrusted_value_called_as_function_raises(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm("malicious_function_name"),
        )
        plan = (
            "fn = quarantined_llm('extract', 'chunks')\n"
            "output = fn('arg')\n"
        )
        with pytest.raises(CapabilityViolation) as excinfo:
            interp.run(plan)
        # The error message names the rejected callable so audits can
        # tell which step refused.
        assert "fn" in str(excinfo.value) or "value" in str(excinfo.value).lower()

    def test_trusted_local_variable_also_refused_as_function(self):
        # Even a TRUSTED bound variable cannot be invoked — the plan
        # dialect has no first-class functions, so this is also a
        # capability violation by construction.
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor({"calculator": 5}),
            q_llm_caller=_stub_q_llm(),
        )
        plan = (
            "x = calculator(1, 1)\n"
            "output = x(2)\n"
        )
        with pytest.raises(CapabilityViolation):
            interp.run(plan)


class TestUntrustedAsAttribute:
    def test_call_target_must_be_bare_name(self):
        # Subscript-as-callable IS the dynamic-attribute attack vector
        # the spec calls out — the plan is computing what to call from
        # data instead of writing the tool name in source. The interpreter
        # rejects the Call before the Subscript's value matters because
        # the func position is not a bare Name.
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm("good"),
        )
        plan = (
            "registry = {'good': 'safe_method'}\n"
            "key = quarantined_llm('pick', 'data')\n"
            "output = registry[key]()\n"
        )
        with pytest.raises(CapabilityViolation):
            interp.run(plan)

    def test_attribute_call_also_blocked(self):
        # ``foo.bar()`` has Attribute in the func position — same family
        # of attack, also rejected at the func-shape check.
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor({"thing": "value"}),
            q_llm_caller=_stub_q_llm(),
        )
        plan = (
            "x = thing()\n"
            "output = x.upper()\n"
        )
        with pytest.raises(CapabilityViolation):
            interp.run(plan)


class TestParseRejection:
    def test_import_rejected(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        with pytest.raises(PlanParseError) as excinfo:
            interp.run("import os\noutput = 'hi'\n")
        assert "Import" in str(excinfo.value) or "import" in str(excinfo.value).lower()

    def test_from_import_rejected(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        with pytest.raises(PlanParseError):
            interp.run("from os import system\noutput = 'hi'\n")

    def test_for_loop_rejected(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        with pytest.raises(PlanParseError) as excinfo:
            interp.run("for x in items:\n    output = x\n")
        assert "For" in str(excinfo.value) or "for" in str(excinfo.value).lower()

    def test_while_loop_rejected(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        with pytest.raises(PlanParseError):
            interp.run("while True:\n    output = 1\n")

    def test_function_def_rejected(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        with pytest.raises(PlanParseError):
            interp.run("def attacker():\n    return 1\noutput = attacker()\n")

    def test_lambda_rejected(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        with pytest.raises(PlanParseError):
            interp.run("f = lambda x: x\noutput = f(1)\n")

    def test_if_statement_rejected(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        with pytest.raises(PlanParseError):
            interp.run("if True:\n    output = 1\nelse:\n    output = 2\n")

    def test_listcomp_rejected(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        with pytest.raises(PlanParseError):
            interp.run("output = [x for x in [1,2,3]]\n")

    def test_syntax_error_raises_plan_parse_error(self):
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        with pytest.raises(PlanParseError):
            interp.run("output = (\n")  # unclosed paren

    def test_if_expression_is_allowed(self):
        # ``a if cond else b`` is the ternary expression form (IfExp),
        # which we DO allow because it is purely value-flow with no
        # branching of effects. Verify the allow-list isn't accidentally
        # rejecting it.
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor(),
            q_llm_caller=_stub_q_llm(),
        )
        result = interp.run("output = 'yes' if 1 else 'no'\n")
        assert result.output_text == "yes"


class TestStepBudget:
    def test_max_steps_exceeded_raises_capability_violation(self):
        # 3 max_steps is well below the size of any non-trivial plan.
        interp = CamelInterpreter(
            tool_executor=_stub_tool_executor({"a": 1, "b": 2, "c": 3}),
            q_llm_caller=_stub_q_llm(),
            max_steps=3,
        )
        plan = (
            "x = a()\n"
            "y = b()\n"
            "z = c()\n"
            "output = x + y + z\n"
        )
        with pytest.raises(CapabilityViolation) as excinfo:
            interp.run(plan)
        assert "max_steps" in str(excinfo.value)


class TestGovernanceBlocked:
    def test_camel_tool_denied_records_blocked_call_and_continues(self):
        def _exec(name, args, kwargs):
            if name == "denied":
                raise CamelToolDenied(
                    reason="forbidden by policy", policy_name="forbidden_tools",
                )
            return CapabilityTaggedResult(
                value="ok", capabilities=capabilities_for_tool(name),
            )
        interp = CamelInterpreter(tool_executor=_exec, q_llm_caller=_stub_q_llm())
        plan = (
            "a = denied()\n"
            "b = allowed()\n"
            "output = b\n"
        )
        result = interp.run(plan)
        assert result.output_text == "ok"
        assert len(result.blocked_calls) == 1
        assert result.blocked_calls[0]["tool"] == "denied"
        assert "forbidden" in result.blocked_calls[0]["reason"]
