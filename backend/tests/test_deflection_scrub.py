"""
tests/test_deflection_scrub.py — Upgrade 2: deflection-sentence scrub.

Covers the spec's success criteria for ``_scrub_deflections``:
  - "the assistant was unable" sentences are removed, surrounding facts kept
  - Pure self-referential limitation sentences scrub to empty string
  - Plain factual content passes through unchanged
  - save_explicit_memory drops a fully-deflected payload before persisting
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from services.memory import _scrub_deflections, MemoryManager


class TestScrubDeflections:
    def test_keeps_real_fact_drops_deflection(self):
        text = (
            "The user likes pizza. "
            "The assistant was unable to find specific nutritional data."
        )
        assert _scrub_deflections(text) == "The user likes pizza."

    def test_pure_deflection_scrubs_to_empty(self):
        out = _scrub_deflections(
            "The AI does not have access to real-time stock prices."
        )
        assert out == ""

    def test_capability_denial_scrubs(self):
        out = _scrub_deflections(
            "The assistant lacks access to private financial data."
        )
        assert out == ""

    def test_assistant_suggested_checking_scrubs(self):
        out = _scrub_deflections(
            "The user wants weather data. "
            "The assistant suggested checking a weather website."
        )
        # First sentence stays, second goes
        assert "user wants weather" in out
        assert "suggested checking" not in out

    def test_plain_fact_unchanged(self):
        text = "The meeting is scheduled for Tuesday at 3pm."
        assert _scrub_deflections(text) == text

    def test_empty_input(self):
        assert _scrub_deflections("") == ""

    def test_multiple_deflections_all_removed(self):
        text = (
            "The user is a developer. "
            "The assistant was unable to verify their employer. "
            "The AI does not have access to LinkedIn."
        )
        out = _scrub_deflections(text)
        assert "user is a developer" in out
        assert "unable" not in out
        assert "AI does not have access" not in out


class TestSaveExplicitMemoryDeflections:
    def _seed_conv(self, db, cid):
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, 'test', ?, ?)", (cid, now, now),
        )
        db.commit()

    def test_fully_deflected_payload_is_not_persisted(self, in_memory_db):
        mem = MemoryManager(rag_index=None, semantic_search_mod=None,
                             local_client=None)
        result = mem.save_explicit_memory(
            "The AI does not have access to your medical records."
        )
        assert "Nothing substantive" in result
        rows = in_memory_db.fetchall("SELECT * FROM memory_entries")
        assert len(rows) == 0

    def test_partial_deflection_keeps_substantive_content(self, in_memory_db):
        mem = MemoryManager(rag_index=None, semantic_search_mod=None,
                             local_client=None)
        mem_id = mem.save_explicit_memory(
            "The user lives in Paris. "
            "The assistant was unable to look up their postal code."
        )
        assert mem_id and not mem_id.startswith("Nothing")
        row = in_memory_db.fetchone(
            "SELECT content FROM memory_entries WHERE id = ?", (mem_id,)
        )
        assert row is not None
        assert "Paris" in row["content"]
        assert "unable" not in row["content"]
