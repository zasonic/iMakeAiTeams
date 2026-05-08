"""
routes/usage.py — read-only aggregation panel over token_usage.

GET /api/usage/summary?days=30&group_by=day|model|agent
  Aggregates rows from the existing ``token_usage`` table (and joins
  ``conversations`` for the agent grouping) into a payload the renderer
  can drop straight into stat cards + a bar chart + the two top-5 side
  tables.

  Each section is best-effort: a missing table or a SQL error returns
  the zero-shaped payload rather than failing the whole endpoint, so a
  sparse install still renders an empty panel rather than a 500.

Pure read aggregation — never mutates the underlying table.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Literal, get_args

from fastapi import APIRouter

import db as _db

router = APIRouter()
log = logging.getLogger("usage_routes")

GroupBy = Literal["day", "model", "agent"]
_GROUP_BY_VALUES: tuple[GroupBy, ...] = get_args(GroupBy)


def _cutoff_iso(days: int) -> str:
    """Return an ISO-8601 cutoff timestamp `days` ago in UTC."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _safe_fetchone(sql: str, params: tuple) -> sqlite3.Row | None:
    """fetchone that swallows missing-table errors and returns None."""
    try:
        return _db.fetchone(sql, params)
    except sqlite3.OperationalError as exc:
        log.debug("usage summary query (one) failed: %s", exc)
        return None
    except Exception as exc:
        log.warning("usage summary query (one) raised: %s", exc)
        return None


def _safe_fetchall(sql: str, params: tuple) -> list[sqlite3.Row]:
    """fetchall that swallows missing-table errors and returns []."""
    try:
        return _db.fetchall(sql, params)
    except sqlite3.OperationalError as exc:
        log.debug("usage summary query (all) failed: %s", exc)
        return []
    except Exception as exc:
        log.warning("usage summary query (all) raised: %s", exc)
        return []


def _total_section(cutoff: str) -> dict:
    row = _safe_fetchone(
        """
        SELECT
            COALESCE(SUM(tokens_in),  0) AS input_tokens,
            COALESCE(SUM(tokens_out), 0) AS output_tokens,
            COALESCE(SUM(cost_usd),   0) AS cost_usd,
            COUNT(*)                     AS turns
        FROM token_usage
        WHERE created_at >= ?
        """,
        (cutoff,),
    )
    if row is None:
        return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "turns": 0}
    return {
        "input_tokens":  int(row["input_tokens"]  or 0),
        "output_tokens": int(row["output_tokens"] or 0),
        "cost_usd":      float(row["cost_usd"]    or 0.0),
        "turns":         int(row["turns"]         or 0),
    }


def _rows_by_day(cutoff: str) -> list[dict]:
    rows = _safe_fetchall(
        """
        SELECT
            substr(created_at, 1, 10)    AS key,
            COALESCE(SUM(tokens_in),  0) AS input_tokens,
            COALESCE(SUM(tokens_out), 0) AS output_tokens,
            COALESCE(SUM(cost_usd),   0) AS cost_usd,
            COUNT(*)                     AS turns
        FROM token_usage
        WHERE created_at >= ?
        GROUP BY substr(created_at, 1, 10)
        ORDER BY key ASC
        """,
        (cutoff,),
    )
    return [
        {
            "key":           r["key"] or "",
            "input_tokens":  int(r["input_tokens"]  or 0),
            "output_tokens": int(r["output_tokens"] or 0),
            "cost_usd":      float(r["cost_usd"]    or 0.0),
            "turns":         int(r["turns"]         or 0),
        }
        for r in rows
    ]


def _rows_by_model(cutoff: str) -> list[dict]:
    rows = _safe_fetchall(
        """
        SELECT
            model                        AS key,
            COALESCE(SUM(tokens_in),  0) AS input_tokens,
            COALESCE(SUM(tokens_out), 0) AS output_tokens,
            COALESCE(SUM(cost_usd),   0) AS cost_usd,
            COUNT(*)                     AS turns
        FROM token_usage
        WHERE created_at >= ?
        GROUP BY model
        ORDER BY cost_usd DESC, key ASC
        """,
        (cutoff,),
    )
    return [
        {
            "key":           r["key"] or "",
            "input_tokens":  int(r["input_tokens"]  or 0),
            "output_tokens": int(r["output_tokens"] or 0),
            "cost_usd":      float(r["cost_usd"]    or 0.0),
            "turns":         int(r["turns"]         or 0),
        }
        for r in rows
    ]


def _rows_by_agent(cutoff: str) -> list[dict]:
    """Group token_usage by the conversation's agent_id.

    token_usage has no agent_id column — it lives on the parent conversation.
    A LEFT JOIN keeps usage rows that have no conversation row (orphaned
    after a deletion); their agent key surfaces as ``""`` so the renderer
    can still render the bar.
    """
    rows = _safe_fetchall(
        """
        SELECT
            COALESCE(c.agent_id, '')     AS key,
            COALESCE(SUM(t.tokens_in),  0) AS input_tokens,
            COALESCE(SUM(t.tokens_out), 0) AS output_tokens,
            COALESCE(SUM(t.cost_usd),   0) AS cost_usd,
            COUNT(*)                       AS turns
        FROM token_usage t
        LEFT JOIN conversations c ON c.id = t.conversation_id
        WHERE t.created_at >= ?
        GROUP BY COALESCE(c.agent_id, '')
        ORDER BY cost_usd DESC, key ASC
        """,
        (cutoff,),
    )
    return [
        {
            "key":           r["key"] or "",
            "input_tokens":  int(r["input_tokens"]  or 0),
            "output_tokens": int(r["output_tokens"] or 0),
            "cost_usd":      float(r["cost_usd"]    or 0.0),
            "turns":         int(r["turns"]         or 0),
        }
        for r in rows
    ]


def _by_model_top(cutoff: str) -> list[dict]:
    rows = _safe_fetchall(
        """
        SELECT
            model                      AS model,
            COALESCE(SUM(cost_usd), 0) AS cost_usd,
            COUNT(*)                   AS turns
        FROM token_usage
        WHERE created_at >= ?
        GROUP BY model
        ORDER BY cost_usd DESC, model ASC
        LIMIT 5
        """,
        (cutoff,),
    )
    return [
        {
            "model":    r["model"] or "",
            "cost_usd": float(r["cost_usd"] or 0.0),
            "turns":    int(r["turns"] or 0),
        }
        for r in rows
    ]


def _by_agent_top(cutoff: str) -> list[dict]:
    rows = _safe_fetchall(
        """
        SELECT
            COALESCE(c.agent_id, '')     AS agent_id,
            COALESCE(SUM(t.cost_usd), 0) AS cost_usd,
            COUNT(*)                     AS turns
        FROM token_usage t
        LEFT JOIN conversations c ON c.id = t.conversation_id
        WHERE t.created_at >= ?
          AND c.agent_id IS NOT NULL
          AND c.agent_id != ''
        GROUP BY c.agent_id
        ORDER BY cost_usd DESC, agent_id ASC
        LIMIT 5
        """,
        (cutoff,),
    )
    return [
        {
            "agent_id": r["agent_id"] or "",
            "cost_usd": float(r["cost_usd"] or 0.0),
            "turns":    int(r["turns"] or 0),
        }
        for r in rows
    ]


@router.get("/summary")
async def summary(days: int = 30, group_by: str = "day") -> dict:
    """Aggregate token_usage for the last `days` days, grouped by `group_by`.

    `days` is clamped to [1, 365]; `group_by` falls back to ``day`` for any
    unrecognised value so a misconfigured caller never crashes the panel.
    """
    days = max(1, min(int(days or 30), 365))
    group: GroupBy = group_by if group_by in _GROUP_BY_VALUES else "day"  # type: ignore[assignment]
    cutoff = _cutoff_iso(days)

    if group == "model":
        rows = _rows_by_model(cutoff)
    elif group == "agent":
        rows = _rows_by_agent(cutoff)
    else:
        rows = _rows_by_day(cutoff)

    return {
        "window_days": days,
        "group_by":    group,
        "total":       _total_section(cutoff),
        "rows":        rows,
        "by_model":    _by_model_top(cutoff),
        "by_agent":    _by_agent_top(cutoff),
    }
