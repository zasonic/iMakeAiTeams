"""
tests/test_audit_log.py — Phase 4: append-only JSONL audit log.

Covers the spec's "audit log survives app restart" success criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.audit_log import AuditLog


@pytest.fixture
def log(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "lifecycle_audit.jsonl")


class TestAppend:
    def test_append_creates_file(self, log: AuditLog):
        log.append("agent_started", agent_id="ag-1")
        assert log.path.exists()
        events = log.read_all()
        assert len(events) == 1
        assert events[0]["event"] == "agent_started"
        assert events[0]["agent_id"] == "ag-1"
        assert "ts" in events[0]

    def test_append_skips_none_fields(self, log: AuditLog):
        log.append("agent_started", agent_id="ag-1", reason=None)
        rec = log.read_all()[0]
        assert "reason" not in rec

    def test_append_rejects_empty_event(self, log: AuditLog):
        with pytest.raises(ValueError):
            log.append("")
        with pytest.raises(ValueError):
            log.append("   ")

    def test_records_are_jsonl_one_per_line(self, log: AuditLog):
        for i in range(5):
            log.append("agent_started", agent_id=f"ag-{i}")
        raw = log.path.read_text(encoding="utf-8")
        lines = [l for l in raw.split("\n") if l]
        assert len(lines) == 5
        for line in lines:
            json.loads(line)  # each line is valid JSON


class TestSurvivesRestart:
    def test_write_close_reopen_read_back(self, tmp_path):
        path = tmp_path / "lifecycle_audit.jsonl"
        # First "session" — write 100 events through a fresh handle.
        a = AuditLog(path)
        for i in range(100):
            a.append("agent_shutdown_requested",
                     agent_id=f"ag-{i:03d}", reason=f"reason {i}")
        # Drop the handle (simulate process exit).
        del a
        # Second "session" — fresh AuditLog instance, same path.
        b = AuditLog(path)
        events = b.read_all()
        assert len(events) == 100
        # Order preserved.
        for i, rec in enumerate(events):
            assert rec["agent_id"] == f"ag-{i:03d}"
            assert rec["event"] == "agent_shutdown_requested"


class TestQueries:
    def test_tail(self, log: AuditLog):
        for i in range(10):
            log.append("agent_started", agent_id=f"ag-{i}")
        last3 = log.tail(3)
        assert [r["agent_id"] for r in last3] == ["ag-7", "ag-8", "ag-9"]

    def test_filter_by_event(self, log: AuditLog):
        log.append("agent_started", agent_id="ag-1")
        log.append("agent_shutdown_confirmed", agent_id="ag-1")
        log.append("agent_started", agent_id="ag-2")
        starts = list(log.filter(event="agent_started"))
        assert len(starts) == 2
        assert {r["agent_id"] for r in starts} == {"ag-1", "ag-2"}

    def test_read_all_empty_when_missing(self, tmp_path):
        a = AuditLog(tmp_path / "nope.jsonl")
        assert a.read_all() == []
        assert a.tail(5) == []

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        path.write_text(
            '{"event":"ok","ts":"2026-01-01T00:00:00.000Z"}\n'
            'garbage line\n'
            '{"event":"ok2","ts":"2026-01-01T00:00:01.000Z"}\n',
            encoding="utf-8",
        )
        events = AuditLog(path).read_all()
        assert [e["event"] for e in events] == ["ok", "ok2"]
