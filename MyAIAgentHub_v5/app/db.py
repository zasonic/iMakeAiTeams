"""
db.py — SQLite database manager for iMakeAiTeams.

Single source of truth for the persistent database. All modules
import from here and never touch sqlite3 directly.

Tables (original):
  Coordination  — workflows, tasks, agent_runs
  Chat          — conversations, messages
  Documents     — documents (replaces old captions)
  Memory        — memory_entries, session_facts
  Agents        — agents, agent_teams, agent_team_members
  Analytics     — token_usage
  Prompts       — prompts, prompt_versions, prompt_experiments
  Errors        — error_logs

Tables added by Priority 1–6 upgrades:
  bm25_corpus          — BM25 document token corpus (Priority 2)
  handoff_log          — structured inter-agent HandoffPackets (Priority 3)
  workflow_checkpoints — SagaLLM transaction state (Priority 4)
  security_scan_log    — LlamaFirewall scan results (Priority 5)
  debate_log           — adversarial debate ChallengePackets (Priority 6)

Columns added to existing tables:
  agents: role, domain, scope, tom_enabled  (Priority 1)
  settings: firewall_enabled, debate_enabled, debate_tier_threshold (Priorities 5, 6)
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_db_path: Path | None = None
_conn: sqlite3.Connection | None = None


def init_db(db_file: Path) -> None:
    """Call once at startup. Creates all tables if they don't exist.

    ``db_file`` is the absolute path to the SQLite file. Callers should pass
    ``core.paths.db_path()`` rather than constructing a path themselves.
    """
    global _db_path, _conn
    _db_path = db_file
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_conn()
    _create_schema(conn)
    _run_migrations(conn)


def get_db() -> sqlite3.Connection:
    """Return the shared, thread-safe connection."""
    if _conn is None:
        raise RuntimeError("db.init_db() has not been called")
    return _conn


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    _conn = sqlite3.connect(str(_db_path), check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def _create_schema(conn: sqlite3.Connection) -> None:
    with _lock:
        cur = conn.cursor()

        # ── Multi-Agent Coordination ──────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                id          TEXT PRIMARY KEY,
                name        TEXT,
                status      TEXT DEFAULT 'pending',
                created_at  TEXT,
                updated_at  TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id             TEXT PRIMARY KEY,
                workflow_id    TEXT REFERENCES workflows(id),
                name           TEXT,
                agent_role     TEXT,
                status         TEXT DEFAULT 'pending',
                depends_on     TEXT DEFAULT '[]',
                input_data     TEXT DEFAULT '{}',
                output_data    TEXT DEFAULT '{}',
                error_message  TEXT,
                attempt_count  INTEGER DEFAULT 0,
                max_attempts   INTEGER DEFAULT 3,
                created_at     TEXT,
                updated_at     TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                id                  TEXT PRIMARY KEY,
                task_id             TEXT REFERENCES tasks(id),
                model               TEXT,
                system_prompt_hash  TEXT,
                input_tokens        INTEGER DEFAULT 0,
                output_tokens       INTEGER DEFAULT 0,
                started_at          TEXT,
                finished_at         TEXT,
                result_summary      TEXT
            )
        """)

        # ── Conversations (server-side chat history) ──────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                title       TEXT DEFAULT 'New conversation',
                agent_id    TEXT,
                created_at  TEXT,
                updated_at  TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT REFERENCES conversations(id),
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                model_used      TEXT,
                route_reason    TEXT,
                tokens_in       INTEGER DEFAULT 0,
                tokens_out      INTEGER DEFAULT 0,
                cost_usd        REAL DEFAULT 0.0,
                created_at      TEXT
            )
        """)

        # ── Documents for RAG (replaces captions table) ───────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id               TEXT PRIMARY KEY,
                content          TEXT NOT NULL,
                source           TEXT,
                doc_type         TEXT DEFAULT 'text',
                chunk_index      INTEGER DEFAULT 0,
                metadata         TEXT DEFAULT '{}',
                embedding_status TEXT DEFAULT 'dirty',
                created_at       TEXT,
                updated_at       TEXT
            )
        """)

        # ── Memory ────────────────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id               TEXT PRIMARY KEY,
                session_id       TEXT,
                content          TEXT,
                category         TEXT DEFAULT 'fact',
                source           TEXT DEFAULT 'user',
                tags             TEXT DEFAULT '[]',
                created_at       TEXT,
                last_accessed    TEXT,
                embedding_status TEXT DEFAULT 'dirty'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_facts (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT REFERENCES conversations(id),
                fact            TEXT NOT NULL,
                source          TEXT DEFAULT 'auto',
                created_at      TEXT
            )
        """)

        # ── Agents ────────────────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id               TEXT PRIMARY KEY,
                name             TEXT NOT NULL UNIQUE,
                description      TEXT,
                system_prompt    TEXT NOT NULL,
                model_preference TEXT DEFAULT 'auto',
                allowed_tools    TEXT DEFAULT '[]',
                temperature      REAL DEFAULT 0.7,
                max_tokens       INTEGER DEFAULT 4096,
                is_builtin       INTEGER DEFAULT 0,
                created_at       TEXT,
                updated_at       TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_teams (
                id             TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                description    TEXT,
                coordinator_id TEXT REFERENCES agents(id),
                created_at     TEXT,
                updated_at     TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_team_members (
                team_id    TEXT REFERENCES agent_teams(id),
                agent_id   TEXT REFERENCES agents(id),
                role       TEXT DEFAULT 'worker',
                sort_order INTEGER DEFAULT 0,
                PRIMARY KEY (team_id, agent_id)
            )
        """)

        # ── Token usage tracking ──────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT,
                model           TEXT NOT NULL,
                tokens_in       INTEGER DEFAULT 0,
                tokens_out      INTEGER DEFAULT 0,
                cost_usd        REAL DEFAULT 0.0,
                routed_reason   TEXT,
                created_at      TEXT
            )
        """)

        # ── Prompt Library ────────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id                TEXT PRIMARY KEY,
                name              TEXT UNIQUE,
                category          TEXT,
                description       TEXT,
                is_protected      INTEGER DEFAULT 0,
                active_version_id TEXT,
                created_at        TEXT,
                updated_at        TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                id               TEXT PRIMARY KEY,
                prompt_id        TEXT REFERENCES prompts(id),
                version_label    TEXT,
                text             TEXT,
                model_target     TEXT,
                estimated_tokens INTEGER DEFAULT 0,
                notes            TEXT,
                created_at       TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prompt_experiments (
                id                    TEXT PRIMARY KEY,
                prompt_a_version_id   TEXT,
                prompt_b_version_id   TEXT,
                test_input            TEXT,
                output_a              TEXT,
                output_b              TEXT,
                judge_scores          TEXT DEFAULT '{}',
                judge_rationale       TEXT,
                winner                TEXT,
                created_at            TEXT
            )
        """)

        # ── Router Feedback Log ───────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS router_log (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT,
                message_preview TEXT,
                route_taken     TEXT NOT NULL,
                complexity      TEXT,
                reasoning       TEXT,
                tokens_out      INTEGER DEFAULT 0,
                had_error       INTEGER DEFAULT 0,
                response_empty  INTEGER DEFAULT 0,
                model_used      TEXT,
                created_at      TEXT NOT NULL
            )
        """)

        # ── Structured Error Logging ──────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS error_logs (
                id                          TEXT PRIMARY KEY,
                timestamp                   TEXT,
                workflow_id                 TEXT,
                task_id                     TEXT,
                component                   TEXT,
                error_class                 TEXT,
                error_message               TEXT,
                stack_trace                 TEXT,
                input_summary               TEXT,
                error_category              TEXT,
                claude_suggestion           TEXT,
                claude_suggestion_applied   INTEGER DEFAULT 0,
                resolved_at                 TEXT
            )
        """)

        # ── Settings key-value store ──────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT
            )
        """)

        # ── Priority 2: BM25 corpus ───────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bm25_corpus (
                doc_id     TEXT PRIMARY KEY,
                tokens     TEXT NOT NULL,
                content    TEXT NOT NULL,
                metadata   TEXT DEFAULT '{}',
                updated_at TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_bm25_updated ON bm25_corpus(updated_at)"
        )

        # ── Priority 3: Handoff log ───────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS handoff_log (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                packet_id            TEXT NOT NULL UNIQUE,
                workflow_id          TEXT,
                step_index           INTEGER DEFAULT 0,
                agent_id             TEXT NOT NULL,
                agent_name           TEXT NOT NULL,
                subtask_completed    TEXT NOT NULL,
                artifact_summary     TEXT,
                assumptions_json     TEXT DEFAULT '[]',
                uncertainties_json   TEXT DEFAULT '[]',
                confidence           REAL DEFAULT 1.0,
                validation_passed    INTEGER DEFAULT 1,
                validation_notes_json TEXT DEFAULT '[]',
                duration_ms          REAL DEFAULT 0.0,
                input_tokens         INTEGER DEFAULT 0,
                output_tokens        INTEGER DEFAULT 0,
                created_at           TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_handoff_workflow ON handoff_log(workflow_id)"
        )

        # ── Priority 4: Saga checkpoints ──────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                checkpoint_id        TEXT PRIMARY KEY,
                workflow_id          TEXT NOT NULL,
                step_index           INTEGER NOT NULL DEFAULT 0,
                task_id              TEXT NOT NULL,
                agent_id             TEXT NOT NULL,
                agent_name           TEXT NOT NULL,
                state                TEXT NOT NULL DEFAULT 'provisional',
                success_criteria     TEXT NOT NULL DEFAULT '',
                artifact_summary     TEXT DEFAULT '',
                confidence_score     REAL DEFAULT NULL,
                validation_passed    INTEGER DEFAULT NULL,
                validation_reasoning TEXT DEFAULT '',
                known_gaps_json      TEXT DEFAULT '[]',
                retry_count          INTEGER NOT NULL DEFAULT 0,
                max_retries          INTEGER NOT NULL DEFAULT 3,
                failure_reason       TEXT DEFAULT NULL,
                created_at           TEXT NOT NULL,
                validated_at         TEXT DEFAULT NULL,
                committed_at         TEXT DEFAULT NULL,
                rolled_back_at       TEXT DEFAULT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_wc_workflow ON workflow_checkpoints(workflow_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_wc_task ON workflow_checkpoints(task_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_wc_state ON workflow_checkpoints(state)"
        )

        # ── Priority 5: Security scan log ─────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS security_scan_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id          TEXT NOT NULL UNIQUE,
                scan_type        TEXT NOT NULL,
                verdict          TEXT NOT NULL,
                scanner          TEXT NOT NULL,
                score            REAL DEFAULT NULL,
                reason           TEXT DEFAULT '',
                flagged_phrases_json TEXT DEFAULT '[]',
                duration_ms      REAL DEFAULT 0.0,
                session_id       TEXT DEFAULT NULL,
                model_tier       TEXT DEFAULT '',
                content_preview  TEXT DEFAULT '',
                degraded         INTEGER DEFAULT 0,
                created_at       TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ssl_created ON security_scan_log(created_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ssl_verdict ON security_scan_log(verdict)"
        )

        # ── Priority 6: Debate log ────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS debate_log (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                challenge_id          TEXT NOT NULL UNIQUE,
                debate_id             TEXT NOT NULL,
                workflow_id           TEXT,
                agent_id              TEXT NOT NULL,
                agent_name            TEXT NOT NULL,
                assumption_diffs_json TEXT DEFAULT '[]',
                fact_conflicts_json   TEXT DEFAULT '[]',
                missing_analysis_json TEXT DEFAULT '[]',
                changed_position      INTEGER DEFAULT 0,
                revised_conclusion    TEXT DEFAULT NULL,
                overall_assessment    TEXT DEFAULT '',
                input_tokens          INTEGER DEFAULT 0,
                output_tokens         INTEGER DEFAULT 0,
                duration_ms           REAL DEFAULT 0.0,
                parse_failed          INTEGER DEFAULT 0,
                created_at            TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dl_workflow ON debate_log(workflow_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dl_debate ON debate_log(debate_id)"
        )

        # ── v4.0: Knowledge Graph Triple Store ───────────────────────────────
        cur.execute("""
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
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_kt_subject ON knowledge_triples(subject COLLATE NOCASE)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_kt_object ON knowledge_triples(object COLLATE NOCASE)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_kt_conversation ON knowledge_triples(source_conversation_id)"
        )

        # ── Priority 7: Pending memory review ─────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pending_review (
                id           TEXT PRIMARY KEY,
                content      TEXT NOT NULL,
                source_type  TEXT NOT NULL,    -- session_fact | memory_entry
                context_id   TEXT DEFAULT '',  -- conversation_id or empty
                scan_verdict TEXT NOT NULL,    -- warn | block
                scan_score   REAL DEFAULT NULL,
                scan_reason  TEXT DEFAULT '',
                status       TEXT DEFAULT 'pending', -- pending | approved | rejected
                created_at   TEXT NOT NULL,
                resolved_at  TEXT DEFAULT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pr_status  ON pending_review(status)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pr_created ON pending_review(created_at)"
        )

        # ── Performance indexes on high-traffic FK and timestamp columns ─────────
        # messages is joined on conversation_id every chat turn and ordered by
        # created_at for history retrieval — without indexes these are full scans.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv    ON messages(conversation_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_ts      ON messages(created_at)"
        )
        # token_usage is summed per conversation_id on every send() for budget tracking
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_token_conv       ON token_usage(conversation_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_token_ts         ON token_usage(created_at)"
        )
        # router_log is scanned per conversation and by date for get_router_stats()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rlog_conv        ON router_log(conversation_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rlog_ts          ON router_log(created_at)"
        )
        # session_facts is queried per conversation_id on every memory recall
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_sfacts_conv      ON session_facts(conversation_id)"
        )
        # conversations is listed sorted by updated_at in the sidebar
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_updated     ON conversations(updated_at)"
        )
        # background indexer polls for embedding_status='dirty' on every cycle
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mement_status    ON memory_entries(embedding_status)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_docs_status      ON documents(embedding_status)"
        )
        # workflow/task FK chains used by orchestrator queries
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_workflow   ON tasks(workflow_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status     ON tasks(status)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_aruns_task       ON agent_runs(task_id)"
        )
        # idx_govlog_agent is created by the v5.0.0 / v5.1.indexes migrations
        # (governance_log table does not exist yet during fresh schema creation)

        conn.commit()


# ── Schema Migrations ────────────────────────────────────────────────────────

_MIGRATIONS = [
    # (version, list_of_SQL_statements)
    ("1.1.0", [
        "ALTER TABLE agents ADD COLUMN allowed_tools TEXT DEFAULT '[]'",
    ]),
    ("1.2.0", [
        # router_log already handled by CREATE TABLE IF NOT EXISTS
    ]),
    ("1.3.0", [
        # No new columns in 1.3 — but future versions go here
    ]),

    # ── Priority 1: Theory of Mind columns on agents ──────────────────────────
    ("2.1.0", [
        "ALTER TABLE agents ADD COLUMN role TEXT DEFAULT 'custom'",
        "ALTER TABLE agents ADD COLUMN domain TEXT",
        "ALTER TABLE agents ADD COLUMN scope TEXT",
        "ALTER TABLE agents ADD COLUMN tom_enabled INTEGER NOT NULL DEFAULT 1",
    ]),

    # ── Priority 2: BM25 + search_log — tables created in _create_schema ──────
    ("2.2.0", [
        # bm25_corpus created by CREATE TABLE IF NOT EXISTS above
        # search_log intentionally not added (bm25_corpus is sufficient)
    ]),

    # ── Priority 3: handoff_log — created in _create_schema ──────────────────
    ("2.3.0", [
        # handoff_log created by CREATE TABLE IF NOT EXISTS above
    ]),

    # ── Priority 4: workflow_checkpoints — created in _create_schema ──────────
    ("2.4.0", [
        # workflow_checkpoints created by CREATE TABLE IF NOT EXISTS above
    ]),

    # ── Priority 5: security_scan_log + settings defaults ─────────────────────
    ("2.5.0", [
        # security_scan_log created by CREATE TABLE IF NOT EXISTS above
        # settings table created above too — insert defaults below
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('firewall_enabled', '1', datetime('now'))",
    ]),

    # ── Priority 6: debate_log + debate settings ──────────────────────────────
    ("2.6.0", [
        # debate_log created by CREATE TABLE IF NOT EXISTS above
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('debate_enabled', '1', datetime('now'))",
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('debate_tier_threshold', 'claude', datetime('now'))",
    ]),

    # ── Priority 7: pending_review — created in _create_schema ───────────────
    ("2.7.0", [
        # pending_review created by CREATE TABLE IF NOT EXISTS above
    ]),

    # ── v4.0: Knowledge Graph + studio_mode setting ───────────────────────────
    ("4.0.0", [
        # knowledge_triples created by CREATE TABLE IF NOT EXISTS above
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('studio_mode', '0', datetime('now'))",
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('goal_decomposition_enabled', '1', datetime('now'))",
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('knowledge_graph_enabled', '1', datetime('now'))",
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('interleaved_reasoning_enabled', '1', datetime('now'))",
    ]),

    # ── v5.0: Task locking + artifact versioning + governance ──────────────
    ("5.0.0", [
        """CREATE TABLE IF NOT EXISTS governance_log (
            id             TEXT PRIMARY KEY,
            agent_id       TEXT,
            tool_name      TEXT,
            allowed        INTEGER,
            reason         TEXT,
            policy_name    TEXT,
            task_key       TEXT,
            created_at     TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_govlog_agent ON governance_log(agent_id)",
        "ALTER TABLE tasks ADD COLUMN locked_by TEXT",
        "ALTER TABLE tasks ADD COLUMN locked_until TEXT",
        """CREATE TABLE IF NOT EXISTS artifact_versions (
            id                TEXT PRIMARY KEY,
            task_id           TEXT REFERENCES tasks(id),
            version           INTEGER NOT NULL DEFAULT 1,
            parent_version    INTEGER,
            content_hash      TEXT,
            content_preview   TEXT,
            validation_status TEXT DEFAULT 'pending',
            author_agent_id   TEXT,
            created_at        TEXT
        )""",
    ]),

    # ── Phase 1: Hub routing — agents declare skills for deterministic match ─
    ("phase1.skills", [
        "ALTER TABLE agents ADD COLUMN skills TEXT DEFAULT '[]'",
    ]),

    # ── Phase 3: Per-agent thinking budget (Qwen3 hybrid /think mode) ──────
    ("phase3.thinking_budget", [
        "ALTER TABLE agents ADD COLUMN thinking_budget INTEGER DEFAULT 2048",
    ]),

    # ── v5.1: Performance indexes on high-traffic FK and timestamp columns ───
    # These are also added in _create_schema (IF NOT EXISTS) for fresh installs.
    # This migration ensures existing databases are indexed on upgrade.
    ("v5.1.indexes", [
        "CREATE INDEX IF NOT EXISTS idx_messages_conv    ON messages(conversation_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_ts      ON messages(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_token_conv       ON token_usage(conversation_id)",
        "CREATE INDEX IF NOT EXISTS idx_token_ts         ON token_usage(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_rlog_conv        ON router_log(conversation_id)",
        "CREATE INDEX IF NOT EXISTS idx_rlog_ts          ON router_log(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_sfacts_conv      ON session_facts(conversation_id)",
        "CREATE INDEX IF NOT EXISTS idx_conv_updated     ON conversations(updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_mement_status    ON memory_entries(embedding_status)",
        "CREATE INDEX IF NOT EXISTS idx_docs_status      ON documents(embedding_status)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_workflow   ON tasks(workflow_id)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_status     ON tasks(status)",
        "CREATE INDEX IF NOT EXISTS idx_aruns_task       ON agent_runs(task_id)",
        "CREATE INDEX IF NOT EXISTS idx_govlog_agent     ON governance_log(agent_id)",
    ]),
]


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Run any pending schema migrations. Idempotent — safe to call every startup."""
    with _lock:
        # Create migration tracking table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TEXT
            )
        """)
        conn.commit()

        applied = {row[0] for row in conn.execute(
            "SELECT version FROM schema_migrations"
        ).fetchall()}

        for version, statements in _MIGRATIONS:
            if version in applied:
                continue
            for sql in statements:
                if not sql.strip():
                    continue
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError as e:
                    # "duplicate column name" means migration already partially applied
                    if "duplicate column" not in str(e).lower():
                        raise
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()


def execute(sql: str, params: tuple = ()) -> None:
    """Execute a single statement, thread-safely. Does NOT return a cursor.
    Use fetchone() / fetchall() for queries that return data."""
    with _lock:
        get_db().execute(sql, params)


def execute_returning(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Execute and return all rows atomically (lock held for the full duration)."""
    with _lock:
        return get_db().execute(sql, params).fetchall()


def executemany(sql: str, params_seq) -> None:
    """Execute many statements, thread-safely."""
    with _lock:
        get_db().executemany(sql, params_seq)
        get_db().commit()


def fetchall(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with _lock:
        return get_db().execute(sql, params).fetchall()


def fetchone(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    with _lock:
        return get_db().execute(sql, params).fetchone()


def commit() -> None:
    with _lock:
        get_db().commit()
