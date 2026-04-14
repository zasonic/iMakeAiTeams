"""
services/knowledge_graph.py — Knowledge Graph Memory (#3)

SQLite-backed triple store layered on top of the existing vector memory.
Extracts entity-relationship triples from conversations and provides
multi-hop reasoning during context retrieval.

Schema: knowledge_triples table (managed by db.py migrations)
  id, subject, predicate, object, source_conversation_id,
  confidence, created_at, last_accessed_at

Integration:
  - After each exchange: extract_triples() mines entity-relationship pairs
  - During memory.get_context(): query_for_context() finds relevant triples
  - Multi-hop: follows chains like User→works_at→Acme, Acme→uses→Salesforce
  - Output injected into MemoryContext.to_system_suffix() as "## Known relationships"
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import db as _db

log = logging.getLogger("MyAIEnv.kg")

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_HOPS = 2          # how deep to follow entity chains
MAX_TRIPLES_PER_QUERY = 8  # max triples to inject into context
STALENESS_DAYS = 90   # triples older than this get a staleness warning

_EXTRACT_SYSTEM = """Extract entity-relationship triples from this conversation exchange.
Return ONLY a JSON array of triples. Each triple is a 3-element array: [subject, predicate, object].

Rules:
- subject and object are named entities (people, companies, products, places, concepts)
- predicate is a short relationship verb phrase (works_at, uses, prefers, is_located_in, owns, etc.)
- Only extract relationships that are factual statements from the conversation
- Extract at most 5 triples
- If no meaningful triples exist, return []
- Use consistent lowercase for predicates with underscores
- Return ONLY the JSON array, no markdown, no explanation

Example:
User says they work at Acme Corp and use Salesforce.
Output: [["User", "works_at", "Acme Corp"], ["Acme Corp", "uses", "Salesforce"]]"""


# ── DB helpers (called from db.py migration) ──────────────────────────────────

def ensure_schema() -> None:
    """Create the knowledge_triples table if it doesn't exist. Called from db.py."""
    try:
        _db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_triples (
                id                     TEXT PRIMARY KEY,
                subject                TEXT NOT NULL,
                predicate              TEXT NOT NULL,
                object                 TEXT NOT NULL,
                source_conversation_id TEXT DEFAULT '',
                confidence             REAL DEFAULT 1.0,
                created_at             TEXT NOT NULL,
                last_accessed_at       TEXT NOT NULL
            )
        """)
        _db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kt_subject ON knowledge_triples(subject COLLATE NOCASE)"
        )
        _db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kt_object ON knowledge_triples(object COLLATE NOCASE)"
        )
        _db.execute(
            "CREATE INDEX IF NOT EXISTS idx_kt_conversation ON knowledge_triples(source_conversation_id)"
        )
        _db.commit()
    except Exception as exc:
        log.warning("knowledge_triples schema setup failed: %s", exc)


# ── Entity extraction ─────────────────────────────────────────────────────────

def _parse_triples(raw: str) -> list[list[str]]:
    """Parse JSON triple array from model output."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        parsed = json.loads(raw[start:end + 1])
        if not isinstance(parsed, list):
            return []
        result = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                s, p, o = [str(x).strip() for x in item]
                if s and p and o and len(s) < 100 and len(o) < 100:
                    result.append([s, p, o])
        return result[:5]  # max 5 triples
    except (json.JSONDecodeError, ValueError):
        return []


def extract_triples(
    conversation_id: str,
    user_msg: str,
    assistant_msg: str,
    local_client,
) -> int:
    """
    Extract entity-relationship triples from a conversation exchange
    and persist them to the knowledge_triples table.

    Returns the number of triples inserted.
    """
    if not local_client or not local_client.is_available():
        return 0

    try:
        prompt = (
            f"User said: {user_msg[:400]}\n"
            f"Assistant said: {assistant_msg[:400]}"
        )
        raw = local_client.chat(_EXTRACT_SYSTEM, prompt, max_tokens=300)
        triples = _parse_triples(raw)
    except Exception as exc:
        log.debug("Triple extraction failed: %s", exc)
        return 0

    if not triples:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    inserted = 0

    for subj, pred, obj in triples:
        # Deduplicate: check for existing (subject, predicate, object) combo
        existing = _db.fetchone(
            "SELECT id FROM knowledge_triples WHERE subject=? AND predicate=? AND object=? COLLATE NOCASE",
            (subj, pred, obj),
        )
        if existing:
            # Update last_accessed_at
            _db.execute(
                "UPDATE knowledge_triples SET last_accessed_at=?, confidence=MIN(confidence+0.05, 1.0) WHERE id=?",
                (now, existing["id"]),
            )
            _db.commit()
            continue

        triple_id = str(uuid.uuid4())
        try:
            _db.execute(
                """
                INSERT INTO knowledge_triples
                    (id, subject, predicate, object, source_conversation_id,
                     confidence, created_at, last_accessed_at)
                VALUES (?, ?, ?, ?, ?, 1.0, ?, ?)
                """,
                (triple_id, subj, pred, obj, conversation_id, now, now),
            )
            inserted += 1
        except Exception as exc:
            log.debug("Triple insert failed: %s", exc)

    if inserted:
        _db.commit()
        log.debug("Knowledge graph: inserted %d triples for conv %s", inserted, conversation_id[:8])

    return inserted


# ── Context query ─────────────────────────────────────────────────────────────

def _extract_query_entities(message: str) -> list[str]:
    """
    Simple entity extraction from a user query for graph lookup.
    Uses capitalized words and known patterns.
    """
    entities = []

    # Capitalized words (likely named entities)
    for word in message.split():
        clean = re.sub(r'[^\w\s]', '', word)
        if clean and clean[0].isupper() and len(clean) > 2:
            entities.append(clean)

    # Multi-word capitalized phrases (up to 3 words)
    words = message.split()
    for i in range(len(words) - 1):
        pair = " ".join(words[i:i + 2])
        if words[i] and words[i][0].isupper() and words[i + 1] and words[i + 1][0].isupper():
            entities.append(pair)

    # Also add "User" as a standing entity
    entities.append("User")

    # Deduplicate while preserving order
    seen = set()
    result = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            result.append(e)

    return result[:10]


def _fetch_triples_for_entity(entity: str, limit: int = 5) -> list[dict]:
    """Fetch triples where entity is subject or object."""
    now = datetime.now(timezone.utc).isoformat()
    rows = _db.fetchall(
        """
        SELECT id, subject, predicate, object, confidence, created_at, last_accessed_at
        FROM knowledge_triples
        WHERE subject=? OR object=? COLLATE NOCASE
        ORDER BY last_accessed_at DESC
        LIMIT ?
        """,
        (entity, entity, limit),
    )
    return [dict(r) for r in rows]


def _is_stale(created_at: str) -> bool:
    """Check if a triple is older than STALENESS_DAYS."""
    try:
        created = datetime.fromisoformat(created_at)
        now = datetime.now(timezone.utc)
        delta = now - created.replace(tzinfo=timezone.utc)
        return delta.days > STALENESS_DAYS
    except Exception:
        return False


def query_for_context(
    message: str,
    existing_session_facts: list[str],
    max_triples: int = MAX_TRIPLES_PER_QUERY,
) -> list[str]:
    """
    Query the knowledge graph for triples relevant to the user's message.
    Performs multi-hop reasoning up to MAX_HOPS.
    Deduplicates against existing session_facts.
    Returns a list of human-readable relationship strings.
    """
    try:
        entities = _extract_query_entities(message)
        if not entities:
            return []

        seen_triple_ids: set[str] = set()
        found_triples: list[dict] = []
        frontier = list(entities)

        for _hop in range(MAX_HOPS):
            if not frontier:
                break
            next_frontier = []

            for entity in frontier:
                triples = _fetch_triples_for_entity(entity, limit=5)
                for t in triples:
                    if t["id"] in seen_triple_ids:
                        continue
                    seen_triple_ids.add(t["id"])
                    found_triples.append(t)

                    # Add connected entity to next hop frontier
                    connected = t["object"] if t["subject"].lower() == entity.lower() else t["subject"]
                    if connected.lower() not in {e.lower() for e in frontier + next_frontier}:
                        next_frontier.append(connected)

            frontier = next_frontier

        if not found_triples:
            return []

        # Update last_accessed_at for retrieved triples
        now = datetime.now(timezone.utc).isoformat()
        for t in found_triples:
            try:
                _db.execute(
                    "UPDATE knowledge_triples SET last_accessed_at=? WHERE id=?",
                    (now, t["id"]),
                )
            except Exception:
                pass
        try:
            _db.commit()
        except Exception:
            pass

        # Format as human-readable strings, deduplicate against session_facts
        existing_lower = {f.lower() for f in existing_session_facts}
        result = []
        for t in found_triples[:max_triples]:
            rel_str = f"{t['subject']} {t['predicate'].replace('_', ' ')} {t['object']}"
            if t.get("confidence", 1.0) < 0.5:
                rel_str += " (uncertain)"
            if _is_stale(t.get("created_at", "")):
                rel_str += " (may be outdated)"
            if rel_str.lower() not in existing_lower:
                result.append(rel_str)

        return result

    except Exception as exc:
        log.debug("Knowledge graph query failed: %s", exc)
        return []


# ── Stats / inspection ────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Return basic stats about the knowledge graph."""
    try:
        total_row = _db.fetchone("SELECT COUNT(*) as n FROM knowledge_triples")
        total = total_row["n"] if total_row else 0

        conv_row = _db.fetchone(
            "SELECT COUNT(DISTINCT source_conversation_id) as n FROM knowledge_triples"
        )
        conversations = conv_row["n"] if conv_row else 0

        recent = _db.fetchall(
            "SELECT subject, predicate, object, confidence, created_at "
            "FROM knowledge_triples ORDER BY created_at DESC LIMIT 10"
        )

        return {
            "total_triples": total,
            "conversations_with_triples": conversations,
            "recent_triples": [dict(r) for r in recent],
        }
    except Exception as exc:
        log.debug("Knowledge graph stats failed: %s", exc)
        return {"total_triples": 0, "conversations_with_triples": 0, "recent_triples": []}


def delete_triples_for_conversation(conversation_id: str) -> int:
    """Delete all triples sourced from a given conversation."""
    try:
        _db.execute(
            "DELETE FROM knowledge_triples WHERE source_conversation_id=?",
            (conversation_id,),
        )
        _db.commit()
        return 1
    except Exception:
        return 0


def search_triples(query: str, limit: int = 20) -> list[dict]:
    """Full-text search across subjects, predicates, and objects."""
    try:
        q = f"%{query.lower()}%"
        rows = _db.fetchall(
            """
            SELECT subject, predicate, object, confidence, created_at
            FROM knowledge_triples
            WHERE lower(subject) LIKE ? OR lower(predicate) LIKE ? OR lower(object) LIKE ?
            ORDER BY last_accessed_at DESC
            LIMIT ?
            """,
            (q, q, q, limit),
        )
        return [dict(r) for r in rows]
    except Exception:
        return []
