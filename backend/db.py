"""
db.py — SQLite database manager for iMakeAiTeams.

Single source of truth for the persistent database. All modules import
from here and never touch sqlite3 directly.

Aggregate boundaries (informal — see "Why monolithic" below):

  Coordination   — workflows, tasks, agent_runs
  Chat           — conversations, messages, messages_fts
  Documents      — documents, attachments, bm25_corpus
  Memory         — memory_entries, session_facts, pending_review,
                   pending_writes, knowledge_triples
  Agents         — agents, agent_teams, agent_team_members,
                   agent_performance
  Analytics      — token_usage, router_log, camel_log, search_log,
                   handoff_log, debate_log
  Prompts        — prompts, prompt_versions, prompt_experiments,
                   prompt_templates
  Security       — security_scan_log, escalations, canary_baseline
  Saga           — workflow_checkpoints
  Voice          — voice_assets
  Errors         — error_logs
  Lifecycle      — schema_migrations

Each block in ``_create_schema()`` is comment-headered with the same
boundaries so a future maintainer can find the relevant CREATE TABLE
in O(grep) time.

Why monolithic, not one module per aggregate:

  The architectural plan (Layer C3) proposed splitting this file into
  ``db.workflows``, ``db.memory``, etc. — one module per aggregate
  above — and was explicit that the change should NOT ship: 24 caller
  import-site edits + test fallout, and no bug class disappears. The
  block-comment grouping in ``_create_schema()`` and the migration list
  already documents the boundaries for human readers; a true package
  split adds files without adding correctness.

  Concrete triggers that WOULD justify reopening the split:
    1. SQLite is replaced by Postgres / DuckDB / a separate KV store.
       Each aggregate's transaction semantics start to diverge and a
       single ``_db.transaction()`` helper can't model them all.
    2. Row-level locking semantics are needed on one aggregate (e.g.
       workflows) without paying the contention cost on the others.
    3. Read-replica or replication is introduced and per-aggregate
       routing matters.
    4. Separate connection pools per aggregate become necessary
       (memory writes contending with chat-message INSERTs on the
       same global ``_lock`` ever shows up in a profile).

  Until one of those fires, the boundaries are read by humans, not by
  the type system, and the cost-of-split is not justified.

Tables added by Priority 1–6 upgrades:
  bm25_corpus          — BM25 document token corpus (Priority 2)
  handoff_log          — structured inter-agent HandoffPackets (Priority 3)
  workflow_checkpoints — SagaLLM transaction state (Priority 4)
  security_scan_log    — LlamaFirewall scan results (Priority 5)
  debate_log           — adversarial debate ChallengePackets (Priority 6)

Columns added to existing tables:
  agents: role, domain, scope, tom_enabled  (Priority 1)
  settings: firewall_enabled, debate_enabled, debate_tier_threshold (Priorities 5, 6)
  token_usage: turn_id  (Layer C1)
  router_log:  turn_id  (Layer C1)
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
import sqlite_vec

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
    try:
        _conn.enable_load_extension(True)
        sqlite_vec.load(_conn)
        _conn.enable_load_extension(False)
    except Exception as _vec_err:
        import logging as _log_mod
        _log_mod.getLogger("iMakeAiTeams.db").warning(
            "sqlite-vec extension failed to load: %s — vector search disabled", _vec_err
        )
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
        # The ``status`` and ``last_accessed`` columns are added by the
        # migrations below — keeping column adds in one place avoids
        # duplicate-column noise on every fresh-database startup.

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
        # ``agent_performance`` is created by its own migration below
        # (``agent_performance.1.0``). Keeping the create in one place
        # avoids the duplicate ``CREATE TABLE``/``CREATE INDEX`` calls
        # that fired on every startup.

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

        # ── Vector search (sqlite-vec) ───────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vec_documents_map (
                vec_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vec_memories_map (
                vec_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT UNIQUE NOT NULL
            )
        """)
        try:
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
                    embedding float[384]
                )
            """)
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                    embedding float[384]
                )
            """)
        except Exception:
            pass  # sqlite-vec not loaded — vector search unavailable

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
    # debate_enabled defaults OFF: the challenger fires per-step and adds
    # latency to every team turn. Power users opt in. When they do,
    # debate_only_high_stakes (added below) keeps it scoped to messages that
    # warrant the cost.
    ("2.6.0", [
        # debate_log created by CREATE TABLE IF NOT EXISTS above
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('debate_enabled', '0', datetime('now'))",
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('debate_tier_threshold', 'claude', datetime('now'))",
    ]),

    # When debate is enabled, scope it to high-stakes turns by default so
    # the challenger doesn't tax every "what's 2+2" exchange.
    ("2.6.1", [
        "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES ('debate_only_high_stakes', '1', datetime('now'))",
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

    # ── sqlite-vec: vector tables + mapping tables ────────────────────────
    ("vec.1.0", [
        # Tables created by CREATE TABLE/VIRTUAL TABLE IF NOT EXISTS above.
    ]),

    # ── Session fact relevance decay: track last access for retrieval order ─
    ("session_facts.last_accessed", [
        "ALTER TABLE session_facts ADD COLUMN last_accessed TEXT",
    ]),

    # ── Deferred reflection: pending/confirmed/discarded fact lifecycle ─────
    ("session_facts.status", [
        "ALTER TABLE session_facts ADD COLUMN status TEXT DEFAULT 'confirmed'",
    ]),

    # ── Agent performance tracking (per-turn alignment + tokens) ─────────────
    ("agent_performance.1.0", [
        """CREATE TABLE IF NOT EXISTS agent_performance (
            id              TEXT PRIMARY KEY,
            agent_id        TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            aligned         INTEGER,
            quality_score   REAL,
            tokens_used     INTEGER DEFAULT 0,
            created_at      TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_ap_agent ON agent_performance(agent_id)",
    ]),

    # ── Phase 4: MAST failure-mode tagging on router_log ─────────────────────
    ("phase4.mast_category", [
        "ALTER TABLE router_log ADD COLUMN mast_category TEXT DEFAULT NULL",
    ]),

    # ── Phase 5: Wiser-Human-style escalation channel ────────────────────────
    ("phase5.escalations", [
        """CREATE TABLE IF NOT EXISTS escalations (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            triggered_at    TEXT NOT NULL,
            trigger_type    TEXT NOT NULL,
            trigger_detail  TEXT,
            model_input     TEXT,
            proposed_action TEXT,
            decision        TEXT,
            decided_at      TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_escalations_conv ON escalations(conversation_id)",
    ]),

    # ── Phase 5: MINJA-style memory injection gate ───────────────────────────
    ("phase5.pending_writes", [
        """CREATE TABLE IF NOT EXISTS pending_writes (
            id                  TEXT PRIMARY KEY,
            conversation_id     TEXT,
            write_type          TEXT NOT NULL,
            content             TEXT NOT NULL,
            contradicts_id      TEXT,
            contradicts_content TEXT,
            proposed_at         TEXT NOT NULL,
            decision            TEXT,
            decided_at          TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_pending_writes_conv ON pending_writes(conversation_id)",
    ]),

    # ── Phase 5: Local-model behavior-drift canary (arXiv 2511.15992) ────────
    ("phase5.canary_baseline", [
        """CREATE TABLE IF NOT EXISTS canary_baseline (
            id            TEXT PRIMARY KEY,
            model_id      TEXT NOT NULL,
            prompt_hash   TEXT NOT NULL,
            prompt_text   TEXT NOT NULL,
            response_text TEXT NOT NULL,
            embedding     BLOB NOT NULL,
            captured_at   TEXT NOT NULL
        )""",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_canary_model_hash ON canary_baseline(model_id, prompt_hash)",
    ]),

    # ── Phase 6: Hackett et al. (ACL 2025) Reader/Actor split ────────────────
    ("phase6.agent_role", [
        # "monolithic" (legacy), "reader", "actor"
        "ALTER TABLE router_log ADD COLUMN agent_role TEXT DEFAULT 'monolithic'",
    ]),

    # ── Phase 8: Symphony-style weighted-vote consensus samples ──────────────
    ("phase8.voting_samples", [
        "ALTER TABLE router_log ADD COLUMN voting_samples_json TEXT DEFAULT NULL",
    ]),

    # ── Phase 9: Bundled llama.cpp server lifecycle bookkeeping ──────────────
    # Each successful download writes one row. file_path is absolute and lives
    # under userData/models/. sha256 lets the wizard re-validate on next start.
    ("phase9.local_backend_mode", [
        """CREATE TABLE IF NOT EXISTS bundled_models (
            model_id        TEXT PRIMARY KEY,
            file_path       TEXT NOT NULL,
            size_bytes      INTEGER NOT NULL,
            sha256          TEXT NOT NULL,
            downloaded_at   TEXT NOT NULL,
            last_loaded_at  TEXT
        )""",
    ]),

    # ── Phase 10: Chat-input file attachments ────────────────────────────────
    # Two flavors share the same row shape: ephemeral (persist=0) attachments
    # are deleted after the next successful chat send; persistent (persist=1)
    # ones are also indexed into the RAG store via rag_doc_id. content_extract
    # is always populated so the orchestrator can prepend it as quarantined
    # context regardless of the persist flag.
    ("phase10.attachments", [
        """CREATE TABLE IF NOT EXISTS attachments (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            filename        TEXT NOT NULL,
            mime_type       TEXT,
            size_bytes      INTEGER NOT NULL,
            persist         INTEGER NOT NULL,
            rag_doc_id      TEXT,
            content_extract TEXT,
            created_at      TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_attachments_conv ON attachments(conversation_id)",
    ]),

    # ── Phase 11: Cross-conversation message search via FTS5 ─────────────────
    # All statements in this migration share the same connection-level
    # transaction; _run_migrations only commits after the loop completes, so
    # the CREATE / backfill / triggers either all land or all roll back. The
    # backfill uses INSERT OR IGNORE for defense-in-depth on partial reruns.
    ("phase11.message_fts", [
        """CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            role UNINDEXED,
            content,
            created_at UNINDEXED,
            tokenize = 'porter unicode61 remove_diacritics 2'
        )""",
        "INSERT OR IGNORE INTO messages_fts(message_id, conversation_id, role, content, created_at) "
        "SELECT id, conversation_id, role, content, created_at FROM messages",
        """CREATE TRIGGER IF NOT EXISTS messages_fts_insert
            AFTER INSERT ON messages BEGIN
              INSERT INTO messages_fts(message_id, conversation_id, role, content, created_at)
              VALUES (new.id, new.conversation_id, new.role, new.content, new.created_at);
            END""",
        """CREATE TRIGGER IF NOT EXISTS messages_fts_delete
            AFTER DELETE ON messages BEGIN
              DELETE FROM messages_fts WHERE message_id = old.id;
            END""",
        """CREATE TRIGGER IF NOT EXISTS messages_fts_update
            AFTER UPDATE ON messages BEGIN
              DELETE FROM messages_fts WHERE message_id = old.id;
              INSERT INTO messages_fts(message_id, conversation_id, role, content, created_at)
              VALUES (new.id, new.conversation_id, new.role, new.content, new.created_at);
            END""",
    ]),

    # ── Phase 12: CaMeL (DeepMind/ETH arXiv 2503.18813) audit log ────────────
    # Privileged-LLM / Quarantined-LLM split. Each turn that runs through
    # the CaMeL pipeline writes one row capturing the plan source, how many
    # AST steps executed before stop, capability violations, blocked tool
    # calls, and the final output. Index on conversation_id matches the
    # other per-turn audit tables for fast lookup in the UI.
    ("phase12.camel_log", [
        """CREATE TABLE IF NOT EXISTS camel_log (
            id                    TEXT PRIMARY KEY,
            conversation_id       TEXT,
            plan_source           TEXT,
            executed_steps        INTEGER,
            capability_violations INTEGER,
            blocked_calls         TEXT,
            output_text           TEXT,
            created_at            TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_camel_log_conv ON camel_log(conversation_id)",
    ]),

    # ── Phase 13: Voice asset bookkeeping (Whisper.cpp + Piper) ──────────────
    # Mirror of phase9.local_backend_mode for the voice models. The Whisper
    # .bin and Piper .onnx + .json files live under userData/voice/ and the
    # row records the sha256 + size for re-validation on each feature use.
    # asset_type is 'stt' for Whisper models and 'tts' for Piper voices.
    ("phase13.voice_assets", [
        """CREATE TABLE IF NOT EXISTS voice_assets (
            asset_id      TEXT PRIMARY KEY,
            asset_type    TEXT NOT NULL,
            file_path     TEXT NOT NULL,
            sha256        TEXT NOT NULL,
            size_bytes    INTEGER NOT NULL,
            downloaded_at TEXT NOT NULL
        )""",
    ]),

    # ── Phase 14: User-saved prompt templates (snippets + system prompts) ────
    # Distinct from the legacy `prompts` table (which holds versioned
    # orchestrator system prompts). Templates here are user-authored, free
    # to edit/delete, and surfaced through the slash-command picker in chat
    # plus the Settings "Set as default system prompt" action.
    ("phase14.prompt_templates", [
        """CREATE TABLE IF NOT EXISTS prompt_templates (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            body        TEXT NOT NULL,
            kind        TEXT NOT NULL,
            tags        TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            use_count   INTEGER NOT NULL DEFAULT 0
        )""",
        "CREATE INDEX IF NOT EXISTS idx_prompt_templates_kind ON prompt_templates(kind)",
    ]),

    # ── Layer C1: per-turn correlation id ────────────────────────────────────
    # Adds turn_id to token_usage + router_log so analytics can group all
    # per-phase rows for a single chat turn (reader + actor + voting
    # samples + escalation rescue). Existing rows keep NULL; new rows
    # always populated from ctx.turn_id (set in TurnLifecycle.open).
    ("layer_c1.turn_id", [
        "ALTER TABLE token_usage ADD COLUMN turn_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_token_usage_turn_id ON token_usage(turn_id)",
        "ALTER TABLE router_log ADD COLUMN turn_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_router_log_turn_id ON router_log(turn_id)",
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
    """Execute a single write statement, thread-safely.

    NOTE: this helper does NOT commit — callers must call ``commit()`` (or
    use ``transaction()`` for multi-statement atomicity). The asymmetry with
    ``executemany()`` (which DOES commit) is historical; new code should
    prefer ``transaction()`` whenever more than one statement participates
    in the same logical change.
    """
    with _lock:
        get_db().execute(sql, params)


def execute_returning(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Execute and return all rows atomically (lock held for the full duration)."""
    with _lock:
        return get_db().execute(sql, params).fetchall()


def executemany(sql: str, params_seq) -> None:
    """Execute many statements, thread-safely. Auto-commits."""
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


# ── Transactional helper ──────────────────────────────────────────────────────

import contextlib  # noqa: E402  (imported here to keep the lock declarations clean)


@contextlib.contextmanager
def transaction():
    """Run several statements atomically under the shared db lock.

    Use this whenever a logical change spans more than one execute. On
    normal completion the connection is committed; on exception the work
    is rolled back so the caller never sees a half-applied state::

        with db.transaction() as conn:
            conn.execute("UPDATE x SET y = ? WHERE id = ?", (y, x_id))
            conn.execute("INSERT INTO audit (...) VALUES (...)", (...))

    Holding the lock for the whole block also blocks competing writers
    in this process, so no other thread can read a half-applied state.
    """
    with _lock:
        conn = get_db()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
