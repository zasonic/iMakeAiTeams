"""tests/camel/test_capabilities.py — Capability tag table + helpers."""

import pytest

from services.camel.capabilities import (
    Capability,
    CapabilityTaggedResult,
    UNTRUSTED_TOOLS,
    capabilities_for_tool,
    merge_capabilities,
)


class TestCapabilitiesForTool:
    def test_known_untrusted_tools_return_untrusted(self):
        for name in ("web_fetch", "rag_search", "file_read"):
            caps = capabilities_for_tool(name)
            assert caps == frozenset({Capability.UNTRUSTED}), (
                f"{name} should be UNTRUSTED, got {caps}"
            )

    def test_mcp_prefix_is_untrusted(self):
        for name in ("mcp_slack", "mcp_github", "mcp_anything"):
            assert capabilities_for_tool(name) == frozenset({Capability.UNTRUSTED})

    def test_unknown_tool_defaults_to_trusted(self):
        for name in ("calculator", "format_date", "internal_helper"):
            assert capabilities_for_tool(name) == frozenset({Capability.TRUSTED})

    def test_empty_name_defaults_to_trusted(self):
        # Defensive: an empty name shouldn't crash; default policy.
        assert capabilities_for_tool("") == frozenset({Capability.TRUSTED})

    def test_untrusted_tools_set_membership(self):
        # The constant is exposed for callers (security review tooling)
        # and must contain the three documented names.
        assert "web_fetch" in UNTRUSTED_TOOLS
        assert "rag_search" in UNTRUSTED_TOOLS
        assert "file_read" in UNTRUSTED_TOOLS


class TestCapabilityTaggedResult:
    def test_normalises_capabilities_to_frozenset(self):
        r = CapabilityTaggedResult(value=1, capabilities={Capability.TRUSTED})
        assert isinstance(r.capabilities, frozenset)

    def test_is_trusted_predicate(self):
        trusted = CapabilityTaggedResult(
            value="ok", capabilities=frozenset({Capability.TRUSTED}),
        )
        untrusted = CapabilityTaggedResult(
            value="bad", capabilities=frozenset({Capability.UNTRUSTED}),
        )
        assert trusted.is_trusted is True
        assert trusted.is_untrusted is False
        assert untrusted.is_untrusted is True
        assert untrusted.is_trusted is False


class TestMergeCapabilities:
    def test_two_trusted_stays_trusted(self):
        a = CapabilityTaggedResult(1, frozenset({Capability.TRUSTED}))
        b = CapabilityTaggedResult(2, frozenset({Capability.TRUSTED}))
        assert merge_capabilities(a, b) == frozenset({Capability.TRUSTED})

    def test_one_untrusted_taints_result(self):
        a = CapabilityTaggedResult(1, frozenset({Capability.TRUSTED}))
        b = CapabilityTaggedResult(2, frozenset({Capability.UNTRUSTED}))
        merged = merge_capabilities(a, b)
        assert Capability.UNTRUSTED in merged

    def test_no_args_defaults_to_trusted(self):
        # Empty merge is the identity for an OR lattice with TRUSTED bottom.
        assert merge_capabilities() == frozenset({Capability.TRUSTED})
