"""Conversation-scoped routes that don't fit the chat orchestrator surface.

Currently hosts the cross-conversation message search backed by the
``messages_fts`` FTS5 table created in migration ``phase11.message_fts``.
The search route reads straight from SQLite via ``db.fetchall`` so it
sidesteps the ``core.api`` facade and stays trivially testable.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import db as _db
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


_MAX_LIMIT = 200
_DEFAULT_LIMIT = 50
_MAX_QUERY_LEN = 1000


# Snippet column is the 4th FTS5 column (0-indexed 3): content. The other
# columns are UNINDEXED so they can't be the snippet target. 32 tokens of
# context + ellipsis matches what existing chat exports use as a preview
# length.
_SEARCH_SQL = (
    "SELECT "
    "  m.id AS message_id, "
    "  m.conversation_id AS conversation_id, "
    "  c.title AS conversation_title, "
    "  m.role AS role, "
    "  snippet(messages_fts, 3, '<mark>', '</mark>', '...', 32) AS snippet, "
    "  m.created_at AS created_at, "
    "  bm25(messages_fts) AS rank "
    "FROM messages_fts "
    "JOIN messages m ON m.id = messages_fts.message_id "
    "JOIN conversations c ON c.id = m.conversation_id "
    "WHERE messages_fts MATCH ? "
    "  AND (? IS NULL OR m.created_at >= ?) "
    "ORDER BY bm25(messages_fts) "
    "LIMIT ?"
)


def _threshold_iso(days: int | None) -> str | None:
    """Map ``days`` to a created_at lower bound, or None for no filter."""
    if days is None:
        return None
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


@router.get("/search")
async def search_messages(
    q: str = Query(..., description="FTS5 search query"),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    days: int | None = Query(None, ge=1),
) -> list[dict]:
    """Cross-conversation message search via the FTS5 ``messages_fts`` index.

    ``q`` accepts FTS5 syntax (phrases in double quotes, AND/OR/NOT,
    NEAR, prefix matching with ``*``). Parser errors surface as 400s
    rather than 500s so the renderer can show "Invalid search query".
    """
    cleaned = (q or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid search query")
    if len(cleaned) > _MAX_QUERY_LEN:
        raise HTTPException(status_code=400, detail="Invalid search query")

    threshold = _threshold_iso(days)

    try:
        rows = _db.fetchall(_SEARCH_SQL, (cleaned, threshold, threshold, limit))
    except sqlite3.OperationalError:
        # FTS5 surfaces parser failures (unbalanced quotes, dangling
        # operators, reserved tokens) as OperationalError. Translate to a
        # 400 with a friendly message — leaking the sqlite text would be
        # noisy and unhelpful for end users.
        raise HTTPException(status_code=400, detail="Invalid search query")

    return [
        {
            "message_id": r["message_id"],
            "conversation_id": r["conversation_id"],
            "conversation_title": r["conversation_title"] or "",
            "role": r["role"],
            "snippet": r["snippet"] or "",
            "created_at": r["created_at"],
            "rank": float(r["rank"]) if r["rank"] is not None else 0.0,
        }
        for r in rows
    ]
