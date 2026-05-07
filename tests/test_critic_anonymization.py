"""
tests/test_critic_anonymization.py — Phase 4: peer-identity redaction.

Covers the spec's "no critic-agent prompt contains a peer model name"
success criterion. Tests run against a freshly-seeded DB.
"""

from __future__ import annotations

import uuid

import pytest

from services.agent_registry import (
    _opaque_label,
    anonymize_existing_critic_prompts,
    create_team,
    add_team_member,
    get_agent,
    is_critic_role,
    list_agents,
    seed_agents,
)


@pytest.fixture
def seeded_db(in_memory_db):
    seed_agents()
    anonymize_existing_critic_prompts()
    return in_memory_db


# ── Opaque label generator ──────────────────────────────────────────────────


class TestOpaqueLabel:
    def test_first_few(self):
        assert _opaque_label(0) == "Author A"
        assert _opaque_label(1) == "Author B"
        assert _opaque_label(25) == "Author Z"

    def test_rolls_over(self):
        assert _opaque_label(26) == "Author AA"
        assert _opaque_label(27) == "Author AB"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _opaque_label(-1)


# ── Critic-role detection ──────────────────────────────────────────────────


class TestRoleDetection:
    def test_reviewer_is_critic(self):
        assert is_critic_role("reviewer") is True
        assert is_critic_role("Reviewer") is True
        assert is_critic_role("  reviewer  ") is True

    def test_others_are_not_critic(self):
        for role in ("coordinator", "researcher", "writer", "coder",
                     "analyst", "custom", "", None):
            assert is_critic_role(role) is False, role


# ── Seeded reviewer prompts must be identifier-clean ─────────────────────────


class TestSeededCriticPrompts:
    def test_no_peer_name_appears_in_any_reviewer_prompt(self, seeded_db):
        agents = list_agents()
        reviewers = [a for a in agents if is_critic_role(a.get("role"))]
        non_reviewers = [a for a in agents if not is_critic_role(a.get("role"))]
        assert reviewers, "no reviewer-role agents seeded"
        assert non_reviewers, "no non-reviewer agents seeded"

        for reviewer in reviewers:
            prompt = reviewer["system_prompt"] or ""
            for peer in non_reviewers:
                peer_name = peer["name"]
                assert peer_name not in prompt, (
                    f"reviewer {reviewer['name']!r} prompt leaks peer name "
                    f"{peer_name!r}"
                )

    def test_anonymized_tokens_appear(self, seeded_db):
        agents = list_agents()
        reviewers = [a for a in agents if is_critic_role(a.get("role"))]
        for reviewer in reviewers:
            prompt = reviewer["system_prompt"] or ""
            # At least the first opaque label must appear when there are peers.
            assert _opaque_label(0) in prompt, (
                f"reviewer {reviewer['name']!r} prompt missing opaque label"
            )

    def test_non_reviewer_prompts_still_name_teammates(self, seeded_db):
        """Sanity: anonymization MUST be scoped to critic agents only."""
        agents = list_agents()
        non_reviewers = [a for a in agents if not is_critic_role(a.get("role"))
                         and a.get("tom_enabled", 1)]
        # Pick any non-reviewer with peers — its ToM block should still name them.
        any_named = False
        for a in non_reviewers:
            prompt = a["system_prompt"] or ""
            for peer in agents:
                if peer["id"] == a["id"]:
                    continue
                if peer["name"] in prompt:
                    any_named = True
                    break
            if any_named:
                break
        assert any_named, (
            "anonymization unintentionally applied to non-reviewer agents"
        )


# ── Backfill is idempotent ──────────────────────────────────────────────────


class TestBackfill:
    def test_running_twice_is_a_noop(self, seeded_db):
        first = anonymize_existing_critic_prompts()
        second = anonymize_existing_critic_prompts()
        # First call may or may not update (depends on whether seed already
        # produced anonymized prompts). Second call MUST be a no-op.
        assert second == 0


# ── refresh_team_tom anonymizes critic on team join ────────────────────────


class TestRefreshTeamTom:
    def test_team_membership_anonymizes_critic(self, seeded_db):
        agents = list_agents()
        reviewer = next(a for a in agents if is_critic_role(a.get("role")))
        teammates = [a for a in agents
                     if a["id"] != reviewer["id"]
                     and not is_critic_role(a.get("role"))][:2]
        assert len(teammates) == 2

        team = create_team("T", "test team", coordinator_id=teammates[0]["id"])
        for m in [reviewer, *teammates]:
            add_team_member(team["id"], m["id"])

        refreshed = get_agent(reviewer["id"])
        prompt = refreshed["system_prompt"] or ""
        for peer in teammates:
            assert peer["name"] not in prompt, (
                f"reviewer prompt after team join leaks peer name {peer['name']!r}"
            )
        # And the opaque tokens must appear.
        assert "Author A" in prompt
