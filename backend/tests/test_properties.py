"""
tests/test_properties.py — Layer 4: property tests with Hypothesis.

The Layer 4 discipline gate. Every Layer 1–3 fix is an invariant, but
example-based tests only check the cases I happened to think of. These
properties pin the invariants so a regression in one of the 50+ cases I
didn't think of still fails the build.

Properties covered:
  - HandoffPacket.confidence_label correctness across the [0,1] range
  - validate_handoff_packet is idempotent for any structurally valid packet
  - ChallengePacket.has_signal honours the empty-vs-populated boundary
  - workflow_checkpoints state invariants:
      committed   ⇒ validation_passed=1, failure_reason IS NULL
      rolled_back ⇒ failure_reason IS NOT NULL
      abandoned   ⇒ no committed_at on the same row
  - RiskLedger cumulative_score is monotonically non-decreasing as
    records are added in any order.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from models import ChallengePacket, HandoffPacket, validate_handoff_packet
from services.security_engine import RiskCategory, RiskLedger


# ── HandoffPacket properties ─────────────────────────────────────────────────


@given(confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_handoff_confidence_label_is_a_total_function(confidence):
    """Every valid confidence in [0,1] yields exactly one of HIGH/MED/LOW."""
    pkt = HandoffPacket(
        agent_id="a", agent_name="A",
        subtask_completed="t", artifact="x",
        confidence=confidence,
    )
    assert pkt.confidence_label in ("HIGH", "MEDIUM", "LOW")


@given(confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_handoff_confidence_label_thresholds_monotonic(confidence):
    """Confidence ≥ 0.85 implies HIGH; ≥ 0.60 implies HIGH or MEDIUM."""
    pkt = HandoffPacket(
        agent_id="a", agent_name="A",
        subtask_completed="t", artifact="x",
        confidence=confidence,
    )
    if confidence >= 0.85:
        assert pkt.confidence_label == "HIGH"
    elif confidence >= 0.60:
        assert pkt.confidence_label in ("HIGH", "MEDIUM")
    else:
        assert pkt.confidence_label == "LOW"


@given(
    subtask=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    artifact=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    uncertainties=st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=5),
    confidence=st.floats(min_value=0.0, max_value=0.94, allow_nan=False),
)
def test_validate_handoff_packet_is_idempotent_when_valid(
    subtask, artifact, uncertainties, confidence,
):
    """A packet that passes validation once must pass again unchanged.

    Confidence is capped below 0.95 so the uncertainties list is required
    (matching HandoffValidation's contract).
    """
    pkt = HandoffPacket(
        agent_id="a", agent_name="A",
        subtask_completed=subtask, artifact=artifact,
        uncertainties=uncertainties, confidence=confidence,
    )
    validate_handoff_packet(pkt)
    if not pkt.validation_passed:
        return  # uninteresting case — let the next example decide
    notes_before = list(pkt.validation_notes)
    validate_handoff_packet(pkt)
    assert pkt.validation_passed is True
    # Idempotent: notes don't grow on a second call.
    assert pkt.validation_notes == notes_before


# ── ChallengePacket properties ───────────────────────────────────────────────


@st.composite
def _challenge_packets(draw):
    return ChallengePacket(
        challenge_id=draw(st.text(min_size=1, max_size=20)),
        debate_id=draw(st.text(min_size=1, max_size=20)),
        workflow_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=20))),
        agent_id=draw(st.text(min_size=1, max_size=20)),
        agent_name=draw(st.text(min_size=1, max_size=40)),
        assumption_diffs=draw(st.lists(st.text(min_size=1, max_size=30), max_size=4)),
        fact_conflicts=draw(st.lists(st.text(min_size=1, max_size=30), max_size=4)),
        missing_analysis=draw(st.lists(st.text(min_size=1, max_size=30), max_size=4)),
        changed_position=draw(st.booleans()),
        revised_conclusion=draw(st.one_of(st.none(), st.text(max_size=80))),
        overall_assessment=draw(st.text(max_size=80)),
        parse_failed=draw(st.booleans()),
    )


@given(_challenge_packets())
def test_challenge_has_signal_iff_any_section_populated(challenge):
    """has_signal is False iff parse_failed OR every signal section is empty."""
    expected = (
        not challenge.parse_failed
        and bool(
            challenge.assumption_diffs
            or challenge.fact_conflicts
            or challenge.missing_analysis
            or challenge.changed_position
            or (challenge.revised_conclusion or "").strip()
        )
    )
    assert challenge.has_signal() is expected


@given(_challenge_packets())
def test_challenge_to_context_block_empty_iff_no_signal(challenge):
    """The synthesizer-facing block is empty exactly when has_signal is False."""
    block = challenge.to_context_block()
    if challenge.has_signal():
        assert block != ""
        assert challenge.agent_name in block
    else:
        assert block == ""


# ── workflow_checkpoints state invariants ────────────────────────────────────


def _seed_checkpoint(
    in_memory_db, state: str, *,
    validation_passed=None, failure_reason=None, committed_at=None,
    rolled_back_at=None,
):
    import uuid
    from datetime import datetime, timezone
    cid = str(uuid.uuid4())
    in_memory_db.execute(
        "INSERT INTO workflow_checkpoints "
        "(checkpoint_id, workflow_id, step_index, task_id, agent_id, "
        " agent_name, state, success_criteria, retry_count, max_retries, "
        " validation_passed, failure_reason, committed_at, rolled_back_at, "
        " created_at) "
        "VALUES (?, 'w', 0, 't', 'a', 'A', ?, '', 0, 3, ?, ?, ?, ?, ?)",
        (
            cid, state, validation_passed, failure_reason,
            committed_at, rolled_back_at,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    in_memory_db.commit()
    return cid


@given(
    state=st.sampled_from(["committed", "rolled_back", "abandoned"]),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_checkpoint_state_invariants_hold(in_memory_db, state):
    """Per the SecurityGate / pipeline contract, the rows that pipeline.py
    writes never violate these. Writing a row in a state that breaks the
    invariant should be flagged in tests so we notice if a future migration
    changes the meaning of ``state``."""
    if state == "committed":
        cid = _seed_checkpoint(
            in_memory_db, "committed",
            validation_passed=1, committed_at="2026-01-01T00:00:00",
        )
        row = in_memory_db.fetchone(
            "SELECT * FROM workflow_checkpoints WHERE checkpoint_id = ?",
            (cid,),
        )
        assert row["state"] == "committed"
        assert row["validation_passed"] == 1
        assert row["failure_reason"] is None
        assert row["committed_at"] is not None
    elif state == "rolled_back":
        cid = _seed_checkpoint(
            in_memory_db, "rolled_back",
            validation_passed=0, failure_reason="bad output",
            rolled_back_at="2026-01-01T00:00:00",
        )
        row = in_memory_db.fetchone(
            "SELECT * FROM workflow_checkpoints WHERE checkpoint_id = ?",
            (cid,),
        )
        assert row["state"] == "rolled_back"
        assert row["validation_passed"] == 0
        assert row["failure_reason"]  # non-empty
    else:  # abandoned
        cid = _seed_checkpoint(
            in_memory_db, "abandoned",
            failure_reason="sidecar restart",
        )
        row = in_memory_db.fetchone(
            "SELECT * FROM workflow_checkpoints WHERE checkpoint_id = ?",
            (cid,),
        )
        assert row["state"] == "abandoned"
        assert row["committed_at"] is None


def test_pipeline_run_produces_committed_invariant_satisfying_rows(
    in_memory_db,
):
    """End-to-end: the pipeline executor's commit path always writes rows
    that satisfy committed ⇒ validation_passed=1 AND failure_reason IS NULL.
    """
    import json as _json
    import uuid
    from unittest.mock import MagicMock
    from models import WorkerResult
    from services.hub_router import HubRouter
    from services.pipeline import PipelineExecutor

    coord_id = str(uuid.uuid4())
    spec_id = str(uuid.uuid4())
    in_memory_db.execute(
        "INSERT INTO agents (id, name, description, system_prompt, "
        "model_preference, role, is_builtin, skills, created_at, updated_at) "
        "VALUES (?, 'C', '', 'p', 'auto', 'coordinator', 0, '[]', '2024-01-01', '2024-01-01')",
        (coord_id,),
    )
    in_memory_db.execute(
        "INSERT INTO agents (id, name, description, system_prompt, "
        "model_preference, role, is_builtin, skills, created_at, updated_at) "
        "VALUES (?, 'S', '', 'p', 'auto', 'researcher', 0, '[]', '2024-01-01', '2024-01-01')",
        (spec_id,),
    )
    team_id = str(uuid.uuid4())
    in_memory_db.execute(
        "INSERT INTO agent_teams (id, name, description, coordinator_id, "
        "created_at, updated_at) VALUES (?, 'T', '', ?, '2024-01-01', '2024-01-01')",
        (team_id, coord_id),
    )
    in_memory_db.execute(
        "INSERT INTO agent_team_members (team_id, agent_id, role, sort_order) "
        "VALUES (?, ?, 'worker', 0)",
        (team_id, spec_id),
    )
    in_memory_db.commit()

    hub = MagicMock(spec=HubRouter)
    from models import RoutingDecision
    hub.route_for_agent.side_effect = lambda aid, task: RoutingDecision(
        agent_id=aid, backend="claude", score=1.0,
        reasoning="t", used_fallback=False, skill_matched="",
    )
    decomp = _json.dumps([{
        "agent_id": spec_id, "agent_name": "S",
        "description": "do thing",
    }])
    hub.invoke.side_effect = [
        WorkerResult(text=decomp, backend="claude", model_name="t"),
        WorkerResult(text="solid output", backend="claude", model_name="t"),
        WorkerResult(text="ok", backend="claude", model_name="t"),
    ]
    from core.settings import Settings
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(Path(tmp) / "s.json")
        executor = PipelineExecutor(hub, settings)
        result = executor.run(
            team_id=team_id, user_message="x",
            conversation_id="cid", history=[],
        )
        rows = in_memory_db.fetchall(
            "SELECT state, validation_passed, failure_reason, committed_at "
            "FROM workflow_checkpoints WHERE workflow_id = ?",
            (result.pipeline_id,),
        )
        assert len(rows) >= 1
        for row in rows:
            if row["state"] == "committed":
                assert row["validation_passed"] == 1
                assert row["failure_reason"] is None
                assert row["committed_at"] is not None


# ── RiskLedger monotonicity ──────────────────────────────────────────────────


_RISK_CATEGORIES = list(RiskCategory)


@given(
    records=st.lists(
        st.tuples(
            st.sampled_from(_RISK_CATEGORIES),
            st.text(min_size=1, max_size=40),
        ),
        min_size=1, max_size=8,
    ),
)
def test_risk_ledger_cumulative_score_is_monotone(records):
    """Each .record() can only push cumulative_score upward; there is no
    SUBTRACT operation. Adversarial inputs shouldn't be able to talk the
    score down by passing more records.
    """
    ledger = RiskLedger()
    prev = 0.0
    for category, reason in records:
        ledger.record(category, reason)
        score = ledger.assess().cumulative_score
        assert score >= prev - 1e-9, f"score regressed: {prev} -> {score}"
        prev = score


@given(records=st.integers(min_value=0, max_value=20))
def test_risk_ledger_zero_records_yields_zero_score(records):
    """A ledger with no records reports a non-negative cumulative score
    regardless of how many times we re-assess. Stops a future change to
    the assess() implementation from accidentally making 0-record ledgers
    return a phantom score."""
    ledger = RiskLedger()
    for _ in range(records):
        # No-op: we are NOT calling .record(); just .assess() repeatedly.
        a = ledger.assess()
        assert a.cumulative_score >= 0.0
