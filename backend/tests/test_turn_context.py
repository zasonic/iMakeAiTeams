"""
tests/test_turn_context.py — Layer C1 turn_id correlation invariants.

The TurnContext.emit() helper auto-stamps ``turn_id`` onto outgoing SSE
payloads when ``ctx.turn_id`` is set. Pinning that here means a future
edit to emit() can't silently drop the correlation id and re-fragment
the renderer-side debugging timeline.
"""

from __future__ import annotations

from services.turn_context import TurnContext


def test_emit_adds_turn_id_when_set():
    received: list = []
    ctx = TurnContext(
        conversation_id="c1", user_message="hi",
        on_event=lambda et, data: received.append((et, data)),
    )
    ctx.turn_id = "t-abc"
    ctx.emit("route_decided", {"model": "claude"})
    assert received == [("route_decided", {"model": "claude", "turn_id": "t-abc"})]


def test_emit_does_not_overwrite_caller_supplied_turn_id():
    """A call site that hands its own turn_id (e.g. a sub-turn or test
    fixture) wins — emit() only fills the slot when it's empty."""
    received: list = []
    ctx = TurnContext(
        conversation_id="c1", user_message="hi",
        on_event=lambda et, data: received.append((et, data)),
    )
    ctx.turn_id = "t-outer"
    ctx.emit("sub_event", {"turn_id": "t-inner", "n": 1})
    assert received == [("sub_event", {"turn_id": "t-inner", "n": 1})]


def test_emit_skips_stamping_when_turn_id_empty():
    """Before TurnLifecycle.open() runs, ctx.turn_id == "" — emit() must
    not add a turn_id="" key in that window."""
    received: list = []
    ctx = TurnContext(
        conversation_id="c1", user_message="hi",
        on_event=lambda et, data: received.append((et, data)),
    )
    assert ctx.turn_id == ""
    ctx.emit("pre_open_event", {"foo": 1})
    assert received == [("pre_open_event", {"foo": 1})]


def test_emit_handles_non_dict_data_without_crashing():
    """Defensive: legacy call sites that pass a non-dict payload (a
    string, an int) should not raise just because turn_id is set."""
    received: list = []
    ctx = TurnContext(
        conversation_id="c1", user_message="hi",
        on_event=lambda et, data: received.append((et, data)),
    )
    ctx.turn_id = "t-abc"
    # Should not crash on non-dict payload.
    ctx.emit("legacy_event", "raw-string-payload")  # type: ignore[arg-type]
    assert received == [("legacy_event", "raw-string-payload")]


def test_emit_no_op_without_on_event():
    """An emit on a context without on_event is silent — used by the
    background CaMeL path that doesn't have a renderer subscriber."""
    ctx = TurnContext(conversation_id="c1", user_message="hi")
    ctx.turn_id = "t-abc"
    ctx.emit("anything", {"k": "v"})  # must not raise
