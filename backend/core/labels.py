"""core/labels.py — internal-name → user-facing-label mapping.

Source of truth for user-facing strings that correspond to internal table
names, service names, and module names. The internal names stay frozen for
backwards compatibility (existing user installs); only the display strings
travel out to the UI.
"""

from __future__ import annotations

TECHNICAL_TO_DISPLAY: dict[str, str] = {
    "agent_performance": "Quality Tracking",
    "router_log": "Smart Routing History",
    "mcp_registry": "Tool Connections",
    "governance_log": "Safety Decisions",
    "security_engine": "Safety Layer",
    "chat_orchestrator": "Conversation Engine",
    "hub_router": "Smart Router",
    "qwen_thinking": "Local AI Reasoning",
    "execution_bridge": "Tool Runner",
    "memory_buffer": "Recent Context",
    "memory_facts": "Saved Facts",
    "memory_rag": "Document Memory",
}


def display_name(internal: str) -> str:
    """Return the user-facing label for ``internal``, or ``internal`` unchanged."""
    return TECHNICAL_TO_DISPLAY.get(internal, internal)
