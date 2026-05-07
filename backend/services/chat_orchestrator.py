"""
services/chat_orchestrator.py

Unified chat orchestrator. Drives the per-turn chat loop: persists user
messages, recalls memory, decides routing, runs the security engine,
dispatches to a worker via the HubRouter, persists the assistant reply,
and returns a ChatResult.

Public surface:
  - ChatOrchestrator(claude_client, local_client, router, memory,
                     settings, hub_router=None)
  - create_conversation / list_conversations / get_conversation_messages /
    update_conversation_title / delete_conversation / branch_conversation /
    export_conversation
  - send(conversation_id, user_message, agent_id=None, on_token=None,
         on_event=None) -> ChatResult
  - get_token_stats(), get_router_stats()

ChatResult carries the assistant text, the model and reasoning that
produced it, token counts, USD cost, the persisted message_id, and any
budget warning. Routing is delegated to HubRouter.route_for_agent /
route_direct + HubRouter.invoke; this module never calls model clients
directly. Per-turn input goes through the security_engine hooks
(quarantine_chunks, render_quarantined_context, enforce_context_rules,
RiskLedger) before any worker invocation, and a hard abort is raised
through SecurityAssessment when the cumulative risk score exceeds the
configured threshold.
"""

import json
import logging
import re
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

import db as _db
from models import ChatResult, ExecutionTarget, RoutingDecision, TaskDescriptor
from services.governance import GovernanceEngine
from services.hub_router import HubRouter
from services import qwen_thinking
from services.redact import redact
from services.security_engine import (
    quarantine_chunks, render_quarantined_context, enforce_context_rules,
    validate_fact_for_storage, RiskLedger, RiskCategory, SecurityAssessment,
    RISK_ABORT_THRESHOLD,
)

log = logging.getLogger("iMakeAiTeams.chat")


def _list_routable_agents() -> list[dict]:
    """Provider for the HubRouter's Qwen /no_think fallback (Phase 3).

    Returns the minimal columns the routing prompt needs. Lives at module
    scope so the closure captured at orchestrator init does not pin a stale
    DB snapshot — every fallback call re-queries.
    """
    rows = _db.fetchall(
        "SELECT id, name, role, skills, model_preference FROM agents"
    )
    return [dict(r) for r in rows]

MAX_HISTORY_MESSAGES = 40  # 20 user/assistant turns
MAX_CONTEXT_CHARS = 80_000  # ~20K tokens — safe for 128K context models
                             # Leaves room for system prompt + RAG + response

_COMPOUND_SIGNALS = re.compile(
    r"\b(and also|and then|after that|additionally|plus can you|"
    r"also please|second(?:ly)|third(?:ly)|finally|one more thing|"
    r"on top of that|separately|another thing)\b",
    re.IGNORECASE,
)


def _detect_compound(msg: str) -> bool:
    """Detect messages containing multiple independent requests."""
    return len(_COMPOUND_SIGNALS.findall(msg)) >= 2 or msg.count("?") >= 3

# Per-million-token pricing defaults. Users can override in Settings
# to keep cost tracking accurate when Anthropic changes prices.
_DEFAULT_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "haiku":  (0.80,  4.0),
    "sonnet": (3.0,  15.0),
    "opus":  (15.0,  75.0),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int,
                   settings=None) -> float:
    if not model or "claude" not in model.lower():
        return 0.0

    # Allow user-configured price overrides
    prices = dict(_DEFAULT_MODEL_PRICES)
    if settings:
        custom = settings.get("model_prices", None)
        if custom and isinstance(custom, dict):
            for key, val in custom.items():
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    prices[key] = (float(val[0]), float(val[1]))

    # Deterministic family detection. The previous code did a substring
    # search over `prices.items()` and took the first match, which depended
    # on dict iteration order — a model named e.g. `claude-haiku-with-opus-
    # fallback` could resolve to `opus` (75x output cost) or `haiku`
    # depending on Python version. Pick the family explicitly.
    m = model.lower()
    family: str | None = None
    for candidate in ("opus", "sonnet", "haiku"):
        if candidate in m:
            family = candidate
            break
    if family and family in prices:
        price_in, price_out = prices[family]
    else:
        price_in, price_out = (3.0, 15.0)
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def _log_router_event(
    conversation_id: str,
    message_preview: str,
    route_taken: str,
    complexity: str,
    reasoning: str,
    tokens_out: int,
    had_error: bool,
    response_empty: bool,
    model_used: str,
    mast_category: str | None = None,
) -> None:
    """Append one row to the router_log table. Non-fatal — never raises."""
    try:
        with _db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO router_log
                    (id, conversation_id, message_preview, route_taken, complexity,
                     reasoning, tokens_out, had_error, response_empty, model_used,
                     mast_category, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    conversation_id,
                    message_preview[:120],
                    route_taken,
                    complexity,
                    reasoning,
                    tokens_out,
                    1 if had_error else 0,
                    1 if response_empty else 0,
                    model_used,
                    mast_category,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except Exception as exc:
        log.debug("router_log write failed: %s", exc)


class ChatOrchestrator:
    def __init__(self, claude_client, local_client, router, memory, settings,
                 hub_router: HubRouter | None = None, mcp_registry=None):
        self.claude = claude_client
        self.local = local_client
        self.router = router
        self.memory = memory
        self._settings = settings
        self._mcp_registry = mcp_registry
        self._governance = GovernanceEngine(settings)
        # DiLoCo blast-radius containment: instead of a fresh-per-turn ledger
        # (which forgets) or a perpetual ledger (which locks out after ~9
        # messages), keep a sliding window of the last N turn-level risk
        # scores. Sustained high risk trips the abort; transient spikes don't.
        # An OrderedDict bounds the per-conversation dict so quiet-but-never-
        # deleted conversations cannot accumulate entries forever.
        self._risk_history: "OrderedDict[str, list[float]]" = OrderedDict()
        self._risk_history_max_conversations = 256
        # Single boundary for worker invocation (Phase 1) with Phase 3 LLM
        # fallback wired through Qwen3 /no_think for routing decisions that
        # have no deterministic skill match.
        if hub_router is None:
            fallback = qwen_thinking.make_no_think_router(
                local_client, _list_routable_agents,
            )
            hub_router = HubRouter(
                claude_client, local_client, settings, llm_fallback=fallback,
            )
        self.hub_router = hub_router

    # ── Conversation management ──────────────────────────────────────────────

    def create_conversation(self, agent_id: str | None = None,
                            title: str = "New conversation") -> str:
        cid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        _db.execute(
            "INSERT INTO conversations (id, title, agent_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cid, title, agent_id, now, now),
        )
        _db.commit()
        return cid

    def get_conversation_messages(self, conversation_id: str,
                                  limit: int = 50) -> list[dict]:
        rows = _db.fetchall(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (conversation_id, limit),
        )
        return [dict(r) for r in rows]

    def list_conversations(self, limit: int = 30) -> list[dict]:
        rows = _db.fetchall(
            "SELECT id, title, agent_id, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        _db.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conversation_id),
        )
        _db.commit()

    def delete_conversation(self, conversation_id: str) -> None:
        # Hold the db lock for the whole cascade so a crash, signal, or
        # interleaving writer can't leave the DB half-deleted. Also clean up
        # the tables that reference conversation_id but were never declared
        # with a FK in db.py (token_usage, router_log) — those used to leak
        # rows after every delete.
        with _db._lock:
            conn = _db.get_db()
            for table in (
                "messages",
                "session_facts",
                "token_usage",
                "router_log",
            ):
                conn.execute(
                    f"DELETE FROM {table} WHERE conversation_id = ?",
                    (conversation_id,),
                )
            conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            conn.commit()
        # Drop the in-memory per-conversation risk history too. Without
        # this, the dict accumulated entries forever — every send to a
        # new conversation_id added one and nothing ever removed them.
        self._risk_history.pop(conversation_id, None)

    def branch_conversation(self, conversation_id: str,
                            from_message_id: str) -> dict:
        """
        Create a new conversation that is a copy of conversation_id up to
        and including from_message_id.

        Returns {"id": new_conversation_id, "title": new_title} on success,
        or {"error": "..."} if the source conversation / message is not found.
        """
        import uuid as _uuid
        source = _db.fetchone(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        )
        if not source:
            return {"error": "Source conversation not found."}

        # Find the cutoff message and validate it belongs to this conversation
        cutoff_msg = _db.fetchone(
            "SELECT * FROM messages WHERE id = ? AND conversation_id = ?",
            (from_message_id, conversation_id),
        )
        if not cutoff_msg:
            return {"error": "Message not found in this conversation."}

        # Fetch all messages in order, then slice at the cutoff message
        all_messages = _db.fetchall(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at ASC, rowid ASC",
            (conversation_id,),
        )
        # Collect messages up to and including from_message_id
        messages = []
        found = False
        for row in all_messages:
            messages.append(row)
            if row["id"] == from_message_id:
                found = True
                break
        if not found:
            return {"error": "Message not found in this conversation."}

        now = datetime.now(timezone.utc).isoformat()
        new_id = str(_uuid.uuid4())
        branch_title = f"Branch of: {source['title'] or 'conversation'}"

        _db.execute(
            "INSERT INTO conversations (id, title, agent_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (new_id, branch_title, source["agent_id"], now, now),
        )

        for msg in messages:
            _db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, model_used, "
                "route_reason, tokens_in, tokens_out, cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(_uuid.uuid4()), new_id,
                    msg["role"], msg["content"],
                    msg["model_used"], msg["route_reason"],
                    msg["tokens_in"] or 0, msg["tokens_out"] or 0,
                    msg["cost_usd"] or 0.0, msg["created_at"],
                ),
            )
        _db.commit()
        log.info("Branched conversation %s → %s at message %s",
                 conversation_id[:8], new_id[:8], from_message_id[:8])
        return {"id": new_id, "title": branch_title}

    def export_conversation(self, conversation_id: str,
                            fmt: str = "markdown") -> dict:
        """
        Export a conversation as markdown or JSON.

        Returns {"content": str, "filename": str} on success,
        or {"error": "..."} on failure.

        fmt must be "markdown" or "json".
        """
        conv = _db.fetchone(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        )
        if not conv:
            return {"error": "Conversation not found."}

        messages = _db.fetchall(
            "SELECT role, content, model_used, cost_usd, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )

        title = conv["title"] or "conversation"
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:60]

        if fmt == "json":
            import json as _json
            payload = {
                "conversation_id": conversation_id,
                "title": title,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "messages": [dict(m) for m in messages],
            }
            return {
                "content": _json.dumps(payload, indent=2, ensure_ascii=False),
                "filename": f"{safe_title}.json",
            }

        # markdown (default)
        lines = [f"# {title}", ""]
        for msg in messages:
            role_label = "**You**" if msg["role"] == "user" else "**Assistant**"
            ts = ""
            if msg["created_at"]:
                try:
                    ts = f" _{datetime.fromisoformat(msg['created_at']).strftime('%Y-%m-%d %H:%M')}_"
                except Exception:
                    pass
            model_note = f" · {msg['model_used']}" if msg["model_used"] else ""
            lines.append(f"{role_label}{model_note}{ts}")
            lines.append("")
            lines.append(msg["content"] or "")
            lines.append("")
            lines.append("---")
            lines.append("")
        return {
            "content": "\n".join(lines),
            "filename": f"{safe_title}.md",
        }

    # ── Token-aware history trimming (Fix 7) ────────────────────────────────

    def _trim_history_to_budget(self, messages: list,
                                budget_chars: int = MAX_CONTEXT_CHARS) -> list:
        """
        Trim oldest messages first until total chars fit within budget.
        Always keeps at least the most recent user message.
        Prevents context window overflow from long conversations with large messages.
        """
        if not messages:
            return messages

        total = sum(len(m.get("content", "")) for m in messages)
        if total <= budget_chars:
            return messages

        trimmed = list(messages)
        while len(trimmed) > 1 and sum(len(m.get("content", "")) for m in trimmed) > budget_chars:
            trimmed.pop(0)

        log.info("History trimmed: %d → %d messages (%d → %d chars, budget %d)",
                 len(messages), len(trimmed), total,
                 sum(len(m.get("content", "")) for m in trimmed), budget_chars)
        return trimmed

    # ── Execution target resolution (Improvement 6) ──────────────────────────

    def _resolve_target(self, route_model: str, agent: dict | None) -> ExecutionTarget:
        """Resolve the execution target from the route decision and agent config."""
        agent_max_tokens = int(agent.get("max_tokens", 4096)) if agent else 4096
        if route_model == "claude":
            return ExecutionTarget(
                backend="claude",
                model_name=self.claude._model,
                max_tokens=agent_max_tokens,
            )
        else:
            return ExecutionTarget(
                backend="local",
                model_name=self._settings.get("default_local_model", "local"),
                max_tokens=min(agent_max_tokens, 2048),
            )

    # ── Send message (core loop) ─────────────────────────────────────────────

    def send(self, conversation_id: str, user_message: str,
             agent_id: str | None = None,
             on_token=None, on_event=None) -> ChatResult:
        """
        The main chat loop. Routes to the right model, injects memory,
        streams back, saves everything to SQLite, returns a ChatResult.

        on_event(event_type, data_dict) — optional callback for structured
        progress events (route_decided, memory_recalled). Non-fatal.
        """
        def _emit_event(event_type: str, data: dict) -> None:
            if on_event:
                try:
                    on_event(event_type, data)
                except Exception:
                    pass

        now = datetime.now(timezone.utc).isoformat()

        # ── Improvement 2: Token budget enforcement ──────────────────────────
        # Hold the db lock across the SUM and the INSERT so two concurrent
        # chat_send calls on the same conversation can't both pass the cap
        # before either has recorded its user message.
        budget = self._settings.get("max_conversation_budget_usd", 5.0)
        warn_pct = self._settings.get("budget_warning_threshold_pct", 80.0)
        user_msg_id = str(uuid.uuid4())
        budget_exceeded = False
        spent = 0.0
        with _db._lock:
            conn = _db.get_db()
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) as total FROM token_usage WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            spent = row["total"] if row else 0.0
            if budget > 0 and spent >= budget:
                budget_exceeded = True
            else:
                conn.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                    "VALUES (?, ?, 'user', ?, ?)",
                    (user_msg_id, conversation_id, user_message, now),
                )
                conn.commit()

        if budget_exceeded:
            return ChatResult(
                text=f"\u26a0\ufe0f This conversation has reached the ${budget:.2f} budget limit. "
                     f"Start a new conversation or increase the limit in Settings.",
                model="",
                route_reason="budget_exceeded",
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                message_id=str(uuid.uuid4()),
            )

        # Load agent config — convert sqlite3.Row to dict so .get() works
        agent = None
        if agent_id:
            row = _db.fetchone("SELECT * FROM agents WHERE id = ?", (agent_id,))
            if row:
                agent = dict(row)
        system_prompt = (
            agent.get("system_prompt", "You are a helpful AI assistant.") if agent
            else self._settings.get("system_prompt", "You are a helpful AI assistant.")
        )

        # ── Team pipeline: activate when the selected agent coordinates a team ──
        # When the user is chatting with an agent that's a team coordinator,
        # decompose the request, dispatch sub-tasks to specialists via the
        # HubRouter, chain HandoffPackets, and synthesise. Single-agent chat
        # (no team active) falls through to the existing path below.
        team_row = None
        if agent_id:
            team_row = _db.fetchone(
                "SELECT id FROM agent_teams WHERE coordinator_id = ?",
                (agent_id,),
            )
        if team_row:
            return self._run_team_pipeline(
                team_id=team_row["id"],
                conversation_id=conversation_id,
                user_message=user_message,
                spent=spent,
                budget=budget,
                warn_pct=warn_pct,
                on_event=on_event,
                on_token=on_token,
            )

        # ── Improvement 4: ToolPermissionContext enforcement ─────────────────
        _allowed_tools = None
        if agent and agent.get("allowed_tools") and agent["allowed_tools"] != "[]":
            try:
                parsed = json.loads(agent["allowed_tools"])
                if parsed and isinstance(parsed, list):
                    _allowed_tools = parsed
                    log.info("Agent %s restricted to tools: %s", agent["name"], _allowed_tools)
            except (json.JSONDecodeError, TypeError):
                pass

        # History — capped to prevent context window overflow
        history_rows = _db.fetchall(
            "SELECT role, content FROM messages WHERE conversation_id = ? "
            "AND role IN ('user', 'assistant') "
            "ORDER BY created_at DESC LIMIT ?",
            (conversation_id, MAX_HISTORY_MESSAGES),
        )
        messages = [
            {"role": r["role"], "content": r["content"]}
            for r in reversed(history_rows)
        ]

        # ── Fix 7: Token-aware trimming ──────────────────────────────────────
        messages = self._trim_history_to_budget(messages)

        # Recall memory and build system context
        mem = self.memory.get_context(conversation_id, user_message)
        mem_suffix = mem.to_system_suffix()

        full_system = system_prompt
        if mem_suffix:
            full_system = system_prompt + "\n\n" + mem_suffix

        # ── Fix 9: Inject tool restrictions into system prompt ───────────────
        if _allowed_tools:
            tool_names = ", ".join(_allowed_tools)
            full_system += (
                "\n\n## Tool Restrictions\n"
                f"You may ONLY use these tools: {tool_names}. "
                "Do not attempt to use any other tools or capabilities "
                "outside this list."
            )

        # Emit structured event so the frontend can show memory indicator
        _emit_event("memory_recalled", {
            "facts_count": len(mem.session_facts),
            "rag_chunks": len(mem.rag_chunks),
            "memories": len(mem.memories),
        })

        if self.memory.should_summarize(conversation_id):
            self.memory.summarize_buffer(conversation_id)

        # Route: Claude or local?
        model_pref = agent.get("model_preference", "auto") if agent else "auto"
        complexity = "complex"
        route_confidence = 1.0
        route_needs_context = False
        if model_pref == "claude":
            route_model = "claude"
            route_reason = "agent prefers claude"
        elif model_pref == "local":
            route_model = "local"
            route_reason = "agent prefers local"
        else:
            route = self.router.classify(user_message, messages, mem)
            route_model = route.model
            route_reason = route.reasoning
            complexity = route.complexity
            route_confidence = route.confidence
            route_needs_context = route.needs_context

        # Emit structured event so the frontend can show which model is being used
        _emit_event("route_decided", {
            "model": route_model, "complexity": complexity,
            "reasoning": route_reason,
            "confidence": route_confidence,
            "needs_context": route_needs_context,
        })

        if _detect_compound(user_message):
            _emit_event("compound_query_detected", {
                "message": "This looks like multiple requests. A team of agents might handle this better.",
                "suggestion": "Try selecting a team coordinator for complex multi-part requests.",
            })

        # ── v4.1: Adaptive memory injection budget (Engram-inspired) ─────────
        # The Engram U-shaped finding says ~25% memory, ~75% reasoning is
        # optimal. For simple queries, we cap injected context aggressively
        # to avoid RAG noise overwhelming the model. For complex queries,
        # we allow more context. This prevents the common failure mode where
        # irrelevant retrieved chunks confuse a simple Q&A response.
        max_context_items = {"simple": 2, "medium": 4, "complex": 8}.get(
            complexity, 4
        )
        if len(mem.rag_chunks) > max_context_items:
            log.debug("Memory budget: trimming RAG from %d to %d chunks (%s)",
                      len(mem.rag_chunks), max_context_items, complexity)
            mem.rag_chunks = mem.rag_chunks[:max_context_items]
            # Rebuild system prompt with trimmed context
            mem_suffix = mem.to_system_suffix()
            full_system = system_prompt
            if mem_suffix:
                full_system = system_prompt + "\n\n" + mem_suffix
            if _allowed_tools:
                tool_names = ", ".join(_allowed_tools)
                full_system += (
                    "\n\n## Tool Restrictions\n"
                    f"You may ONLY use these tools: {tool_names}. "
                    "Do not attempt to use any other tools or capabilities "
                    "outside this list."
                )

        # Inject MCP tool descriptions for this agent's skills
        if agent and agent.get("skills") and self._mcp_registry:
            try:
                agent_skills = (
                    json.loads(agent["skills"]) if isinstance(agent["skills"], str)
                    else agent["skills"]
                )
                skill_names = [
                    s.get("name", "") for s in agent_skills
                    if isinstance(s, dict)
                ]
                mcp_tools = self._mcp_registry.get_tools_for_tags(skill_names)
                if mcp_tools:
                    tool_lines = "\n".join(
                        f"- **{t['name']}**: {t['description']}" for t in mcp_tools[:10]
                    )
                    full_system += (
                        "\n\n## Available External Tools\n"
                        "(These tools are available via MCP. Mention them if relevant.)\n\n"
                        + tool_lines
                    )
            except Exception:
                pass  # MCP injection is best-effort

        # ── Phase 1: Build routing decision through the HubRouter ────────────
        # The TaskRouter above decided which *backend* to use; the HubRouter
        # decides which *worker* and authorizes the dispatch. When the caller
        # specified an agent_id we go through ``route_for_agent`` (which can
        # raise AuthorizationError); when no agent is specified we synthesize
        # a hub-direct decision so the chat path keeps working without forcing
        # every caller to declare a worker.
        task = TaskDescriptor(
            text=user_message,
            preferred_agent_id=agent_id,
            backend_hint=route_model,
        )
        if agent_id:
            decision = self.hub_router.route_for_agent(agent_id, task)
        else:
            decision = RoutingDecision(
                agent_id="",
                backend=route_model,
                score=1.0,
                reasoning=route_reason,
                used_fallback=False,
                skill_matched="",
            )

        # ── Improvement 6: Resolve execution target ──────────────────────────
        target = self._resolve_target(decision.backend, agent)

        # ══════════════════════════════════════════════════════════════════════
        # SECURITY ENGINE: Structural enforcement before model inference
        # Runs AFTER context assembly, AFTER hooks, BEFORE any model call.
        # Uses deterministic rules (not classifiers) — can't be prompt-injected.
        # ══════════════════════════════════════════════════════════════════════
        security = SecurityAssessment()
        try:
            # --- Context Quarantine: wrap RAG chunks with provenance tags ---
            if mem.rag_chunks:
                quarantined = quarantine_chunks(
                    mem.rag_chunks,
                    source_type="user_document",
                    source_id=conversation_id,
                )
                security.quarantined_chunks = len(quarantined)
                quarantined_section = render_quarantined_context(quarantined)
                if quarantined_section:
                    # Replace raw RAG injection in system prompt with
                    # provenance-tagged, structurally isolated version
                    raw_rag = mem.to_system_suffix()
                    if raw_rag and "## Reference documents the user has provided" in full_system:
                        # Swap the raw documents section for quarantined version
                        full_system = full_system.replace(
                            "## Reference documents the user has provided",
                            "## Retrieved Context (Quarantined)",
                        )

            # --- Deterministic Rule Engine: strip structural attacks ---
            full_system, violations = enforce_context_rules(
                full_system, source_label=conversation_id[:8]
            )
            security.context_violations = violations

            # --- Risk Ledger: track cumulative risk for THIS turn only ---
            # A fresh ledger is created each turn because DATA_READ +
            # EXTERNAL_API accumulate to 0.35 per message; persisting across
            # turns causes the conversation to hit the 3.0 abort threshold
            # after ~9 messages.
            #
            # DiLoCo blast-radius containment: replace the per-turn amnesia
            # with a sliding window of the last 5 turn-level cumulative
            # scores. A sustained injection campaign (5+ turns averaging
            # 0.6+) trips the abort; a single spike followed by normal
            # turns does not.
            ledger = RiskLedger()
            ledger.record(
                RiskCategory.DATA_READ,
                f"Context assembled: {len(mem.rag_chunks)} RAG chunks, "
                f"{len(mem.session_facts)} facts, {len(mem.memories)} memories",
            )
            if target.backend == "claude":
                ledger.record(
                    RiskCategory.EXTERNAL_API,
                    f"Sending to external API: {target.model_name}",
                    weight_override=0.15,  # low weight for standard chat
                )
            security.risk_assessment = ledger.assess()

            history = self._risk_history.setdefault(conversation_id, [])
            history.append(security.risk_assessment.cumulative_score)
            if len(history) > 5:
                del history[:-5]
            # Mark this conversation as most-recently-used and evict the
            # least-recently-used once we exceed the bound. Without this
            # the dict accumulated one entry per conversation forever.
            self._risk_history.move_to_end(conversation_id)
            while len(self._risk_history) > self._risk_history_max_conversations:
                self._risk_history.popitem(last=False)
            if len(history) >= 5:
                window_avg = sum(history) / len(history)
                if window_avg > RISK_ABORT_THRESHOLD / 5:
                    security.risk_assessment.should_abort = True

            # --- Hard abort if risk threshold exceeded ---
            if security.risk_assessment.should_abort:
                security.blocked = True
                security.block_reason = (
                    f"Cumulative risk score {security.risk_assessment.cumulative_score:.1f} "
                    f"exceeds threshold {3.0}. Requires human approval."
                )
                _emit_event("security_assessment", security.to_event())
                return ChatResult(
                    text=(
                        f"🛡️ This workflow has been paused because the cumulative "
                        f"risk score ({security.risk_assessment.cumulative_score:.1f}) "
                        f"exceeds the safety threshold. This happens when a conversation "
                        f"involves many high-risk operations. Start a new conversation "
                        f"or adjust the risk threshold in Settings."
                    ),
                    model="", route_reason="security_abort",
                    tokens_in=0, tokens_out=0, cost_usd=0.0,
                    message_id=str(uuid.uuid4()),
                )

            # Emit security assessment to frontend thinking timeline
            _emit_event("security_assessment", security.to_event())

        except Exception as exc:
            log.debug("Security engine non-fatal error: %s", exc)

        # ══════════════════════════════════════════════════════════════════════

        # ── Phase 5: Wiser-Human escalation channel ──────────────────────────
        # Detects Lynch et al.'s 3 triggers (replacement_threat,
        # autonomy_reduction, goal_conflict) on the pending send. When fired,
        # the orchestrator returns a placeholder ChatResult with route_reason
        # "escalation_pending" — the worker is NOT invoked. The user must
        # approve via POST /api/escalation/{id}/approve before the action
        # would be retried.
        escalation_verdict = self._governance.escalation_channel.check_escalation(
            conversation_id=conversation_id,
            user_message=user_message,
            system_prompt=full_system,
            proposed_action=None,
        )
        if escalation_verdict.requires_review:
            _emit_event("escalation_required", {
                "escalation_id": escalation_verdict.escalation_id,
                "trigger_type": escalation_verdict.trigger_type,
                "trigger_detail": escalation_verdict.trigger_detail,
                "conversation_id": conversation_id,
            })
            return ChatResult(
                text="Awaiting your review for this action.",
                model="", route_reason="escalation_pending",
                tokens_in=0, tokens_out=0, cost_usd=0.0,
                message_id=str(uuid.uuid4()),
            )

        # ── Governance: enforce per-agent policies before invocation ─────────
        if agent_id:
            tool_verdict = self._governance.check_tool_call(
                tool_name="chat_invoke",
                agent_id=agent_id,
                task_key=conversation_id,
            )
            if not tool_verdict.allowed:
                _emit_event("governance_blocked", {
                    "agent_id": agent_id,
                    "reason": tool_verdict.reason,
                    "policy": tool_verdict.policy_name,
                })
                return ChatResult(
                    text=f"⚠️ Governance policy blocked this request: {tool_verdict.reason}",
                    model="", route_reason="governance_blocked",
                    tokens_in=0, tokens_out=0, cost_usd=0.0,
                    message_id=str(uuid.uuid4()),
                )

            budget_verdict = self._governance.check_token_budget(
                tokens_used=target.max_tokens,
                agent_id=agent_id,
                task_key=conversation_id,
            )
            if not budget_verdict.allowed:
                _emit_event("governance_blocked", {
                    "agent_id": agent_id,
                    "reason": budget_verdict.reason,
                    "policy": budget_verdict.policy_name,
                })
                return ChatResult(
                    text=f"⚠️ Token budget exceeded: {budget_verdict.reason}",
                    model="", route_reason="governance_budget",
                    tokens_in=0, tokens_out=0, cost_usd=0.0,
                    message_id=str(uuid.uuid4()),
                )

        response_text = ""
        tokens_in = 0
        tokens_out = 0
        model_name = target.model_name
        had_error = False

        # ── v4.0 #4: Interleaved Reasoning Visibility ────────────────────────
        # When routing to Claude and extended thinking is available, emit
        # a reasoning step event before generating the final response.
        reasoning_enabled = self._settings.get("interleaved_reasoning_enabled", True)
        if (
            reasoning_enabled
            and target.backend == "claude"
            and complexity == "complex"
            and not on_token  # only in non-streaming path (thinking is blocking)
        ):
            try:
                _emit_event("reasoning_started", {
                    "label": "Extended reasoning…",
                    "detail": "Claude is thinking through your request",
                })
                thinking_result = self.claude.extended_thinking_chat(
                    system=full_system,
                    user_message=user_message,
                    budget_tokens=5000,
                )
                if thinking_result.get("thinking"):
                    _emit_event("reasoning_complete", {
                        "label": "Reasoning complete",
                        "thinking_preview": thinking_result["thinking"][:200],
                        "detail": f"{len(thinking_result['thinking'])} chars of reasoning",
                    })
                    # Use the answer from extended thinking as our response
                    response_text = thinking_result.get("answer", "")
                    if response_text:
                        # Emit tokens one-by-one for the streaming feel
                        # (thinking used non-streaming path intentionally)
                        pass  # response_text already set
            except Exception as exc:
                log.debug("Extended thinking skipped: %s", exc)

        # ── Execute (normal path if decomposition/reasoning didn't produce output) ─
        # Phase 1: All worker invocations route through HubRouter.invoke().
        # The orchestrator no longer calls model clients directly here.
        if not response_text:
            worker_result = self.hub_router.invoke(
                decision, full_system, messages,
                max_tokens=target.max_tokens, on_token=on_token,
            )
            response_text = worker_result.text
            tokens_in = worker_result.input_tokens
            tokens_out = worker_result.output_tokens
            if worker_result.had_error:
                had_error = True

        # ── Post-assembly alignment check (informational) ───────────────────
        # When an agent was involved, ask the local model whether the worker's
        # response actually addresses the user's request. Best-effort, never
        # blocks or replaces the response — only emits an alignment_warning
        # event when the local model says the response drifted.
        if (
            not had_error
            and agent_id is not None
            and response_text
            and len(user_message.split()) >= 8
        ):
            try:
                from services.task_artifacts import local_first_call
                align_raw = local_first_call(
                    self.local, None,
                    "Does this response address the user's original request? "
                    "Return ONLY JSON: {\"aligned\": true/false, \"reason\": \"one sentence\"}",
                    f"REQUEST: {user_message[:300]}\nRESPONSE: {response_text[:500]}",
                    max_tokens=100,
                )
                if align_raw:
                    import json as _json
                    _astart = align_raw.find("{")
                    _aend = align_raw.rfind("}")
                    if _astart != -1 and _aend != -1 and _aend > _astart:
                        try:
                            parsed = _json.loads(align_raw[_astart:_aend + 1])
                        except (ValueError, TypeError):
                            parsed = {}
                        if parsed.get("aligned") is False:
                            _emit_event("alignment_warning", {
                                "reason": parsed.get("reason", "Response may not address your request"),
                            })
                        # Persist agent performance data
                        try:
                            _db.execute(
                                "INSERT INTO agent_performance "
                                "(id, agent_id, conversation_id, aligned, quality_score, tokens_used, created_at) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (str(uuid.uuid4()), agent_id, conversation_id,
                                 1 if parsed.get("aligned", True) else 0,
                                 None,  # quality_score filled by quality gate below if it runs
                                 tokens_in + tokens_out,
                                 datetime.now(timezone.utc).isoformat()),
                            )
                            _db.commit()
                        except Exception:
                            pass  # performance logging is best-effort
            except Exception:
                pass  # alignment check is best-effort, never block response

        # ── Local response quality gate ─────────────────────────────────────
        # If response came from local and looks weak, escalate to Claude.
        # An empty local response is the strongest possible signal of failure,
        # so it bypasses the quality scorer (which can't grade an empty input)
        # and escalates directly.
        response_empty = len((response_text or "").strip()) < 20

        def _escalate_to_claude(reason: str) -> bool:
            nonlocal response_text, tokens_in, tokens_out, route_model, model_name
            try:
                escalation = RoutingDecision(
                    agent_id=decision.agent_id,
                    backend="claude",
                    score=decision.score,
                    reasoning=reason,
                    used_fallback=False,
                    skill_matched=decision.skill_matched,
                )
                esc_result = self.hub_router.invoke(
                    escalation, full_system, messages,
                    max_tokens=target.max_tokens, on_token=on_token,
                )
                response_text = esc_result.text
                tokens_in = esc_result.input_tokens
                tokens_out = esc_result.output_tokens
                route_model = "claude"
                model_name = esc_result.model_name
                return True
            except Exception as esc_exc:
                log.debug("Escalation to Claude failed: %s", esc_exc)
                return False

        if (
            not had_error
            and target.backend == "local"
            and self.local and self.local.is_available()
            and len(user_message.split()) >= 5  # skip for trivial messages
        ):
            if response_empty:
                log.info("Local response empty — escalating to Claude")
                _escalate_to_claude("local response empty; escalated")
            else:
                try:
                    from services.task_artifacts import local_first_call
                    quality_raw = local_first_call(
                        self.local, None,  # local only, no Claude fallback
                        "Rate this response's relevance and completeness for the given question. "
                        "Respond with ONLY a JSON: {\"score\": 0-10, \"reason\": \"...\"}",
                        f"QUESTION: {user_message[:300]}\nRESPONSE: {(response_text or '')[:500]}",
                        max_tokens=100,
                    )
                    if quality_raw:
                        import json as _json
                        _qstart = quality_raw.find("{")
                        _qend = quality_raw.rfind("}")
                        if _qstart != -1 and _qend != -1:
                            try:
                                quality = _json.loads(quality_raw[_qstart:_qend + 1])
                            except (ValueError, TypeError):
                                quality = {}
                            # Coerce score to a number; a model emitting
                            # {"score": "low"} would otherwise raise TypeError
                            # on the comparison and silently disable escalation
                            # via the outer `except Exception: pass` swallow.
                            try:
                                score = float(quality.get("score", 10))
                            except (TypeError, ValueError):
                                score = 10.0
                            if score < 4:
                                log.info("Local response scored %s — escalating to Claude", score)
                                _escalate_to_claude("local response failed quality gate; escalated")
                except Exception:
                    pass  # quality check is best-effort, never block response

        # Recompute after possible escalation so router_log records the
        # post-escalation state, not the stale pre-escalation reading.
        response_empty = len((response_text or "").strip()) < 20

        # Persist router feedback
        turn_failed = had_error or response_text.startswith("[Error")
        mast_category: str | None = None
        if turn_failed:
            try:
                mast_category = self.hub_router.classify_failure(
                    user_message,
                    response_text,
                    response_text if response_text.startswith("[Error") else "",
                )
            except Exception as exc:
                log.debug("MAST classify_failure skipped: %s", exc)
        _log_router_event(
            conversation_id=conversation_id,
            message_preview=user_message,
            route_taken=route_model,
            complexity=complexity,
            reasoning=route_reason,
            tokens_out=tokens_out,
            had_error=turn_failed,
            response_empty=response_empty,
            model_used=model_name,
            mast_category=mast_category,
        )

        # Save assistant message — redact the persisted copy so credentials
        # never land on disk. The streaming UI already received the original
        # text via on_token, and ChatResult.text below stays un-redacted for
        # the in-flight return value.
        cost = _estimate_cost(model_name, tokens_in, tokens_out, self._settings)
        reply_text_for_storage = redact(response_text)
        asst_msg_id = str(uuid.uuid4())
        resp_now = datetime.now(timezone.utc).isoformat()
        # Persist assistant message + conversation update + token_usage as a
        # single transaction. Splitting these used to leave the DB in a torn
        # state on a crash (message saved but token_usage missing — budget
        # under-counted). _db.transaction() rolls back on exception so a
        # mid-write SQLite error leaves both rows absent, never just one.
        # Re-reading the running total inside the same transaction also
        # closes the race where two concurrent sends both used a stale
        # ``spent`` and skipped the budget warning.
        # Auto-title runs AFTER this transaction (not inside) to preserve
        # current behavior — it makes a blocking LLM call we don't hold locks across.
        budget_warning = ""
        with _db.transaction() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, model_used, "
                "route_reason, tokens_in, tokens_out, cost_usd, created_at) "
                "VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?)",
                (asst_msg_id, conversation_id, reply_text_for_storage, model_name,
                 route_reason, tokens_in, tokens_out, cost, resp_now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ?, "
                "title = CASE WHEN title = 'New conversation' THEN ? ELSE title END "
                "WHERE id = ?",
                (resp_now, user_message[:60], conversation_id),
            )
            conn.execute(
                "INSERT INTO token_usage (id, conversation_id, model, tokens_in, "
                "tokens_out, cost_usd, routed_reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conversation_id, model_name,
                 tokens_in, tokens_out, cost, route_reason, resp_now),
            )
            if budget > 0:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) as total FROM token_usage "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                new_spent = row["total"] if row else (spent + cost)
                pct = (new_spent / budget) * 100
                if pct >= warn_pct:
                    budget_warning = (
                        f"⚠️ Approaching conversation budget limit "
                        f"(${new_spent:.2f}/${budget:.2f})"
                    )

        # Auto-title: generate a concise title from the first exchange.
        # Only fires once — when the title is still the raw truncation.
        conv_row = _db.fetchone(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        )
        if conv_row and conv_row["title"] == user_message[:60]:
            if self.local and self.local.is_available():
                try:
                    title_raw = self.local.chat(
                        "Generate a 3-6 word title for this conversation. "
                        "Return ONLY the title text, no quotes, no explanation.",
                        f"User: {user_message[:200]}\nAssistant: {response_text[:200]}",
                        max_tokens=20,
                    )
                    if title_raw and 2 < len(title_raw.strip()) <= 80:
                        clean_title = title_raw.strip().strip('"\'').strip()
                        _db.execute(
                            "UPDATE conversations SET title = ? WHERE id = ?",
                            (clean_title, conversation_id),
                        )
                        _db.commit()
                except Exception:
                    pass  # auto-title is best-effort, never block

        # Update memory
        self.memory.add_to_buffer(conversation_id, "user", user_message)
        self.memory.add_to_buffer(conversation_id, "assistant", response_text)
        self.memory.extract_facts(conversation_id, user_message, response_text)

        return ChatResult(
            text=response_text,
            model=model_name,
            route_reason=route_reason,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            message_id=asst_msg_id,
            budget_warning=budget_warning,
        )

    # ── Team pipeline ────────────────────────────────────────────────────────

    def _run_team_pipeline(
        self, team_id: str, conversation_id: str, user_message: str,
        spent: float, budget: float, warn_pct: float,
        on_event=None, on_token=None,
    ) -> ChatResult:
        """Dispatch a turn to the team PipelineExecutor and persist its result.

        The pipeline owns decomposition, specialist dispatch, HandoffPacket
        chaining, and synthesis. This wrapper persists the synthesised reply
        as a normal assistant message, updates token_usage, and refreshes
        memory buffers so the team turn looks identical to a single-agent
        turn from the rest of the system's point of view.
        """
        from services.pipeline import PipelineExecutor

        history_rows = _db.fetchall(
            "SELECT role, content FROM messages WHERE conversation_id = ? "
            "AND role IN ('user', 'assistant') "
            "ORDER BY created_at DESC LIMIT ?",
            (conversation_id, MAX_HISTORY_MESSAGES),
        )
        history = [
            {"role": r["role"], "content": r["content"]}
            for r in reversed(history_rows)
        ]
        history = self._trim_history_to_budget(history)

        executor = PipelineExecutor(self.hub_router, self._settings)
        try:
            result = executor.run(
                team_id=team_id,
                user_message=user_message,
                conversation_id=conversation_id,
                history=history,
                on_event=on_event,
                on_token=on_token,
            )
        except Exception as exc:
            log.exception("Pipeline execution failed: %s", exc)
            return ChatResult(
                text=f"[Team pipeline error: {exc}]",
                model="pipeline",
                route_reason="pipeline_error",
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                message_id=str(uuid.uuid4()),
            )

        synthesis = result.synthesis or ""
        cost = _estimate_cost(
            result.synthesis_model, result.total_tokens_in,
            result.total_tokens_out, self._settings,
        )
        route_reason = f"team pipeline ({len(result.steps)} steps)"
        asst_msg_id = str(uuid.uuid4())
        resp_now = datetime.now(timezone.utc).isoformat()

        _db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, model_used, "
            "route_reason, tokens_in, tokens_out, cost_usd, created_at) "
            "VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?)",
            (
                asst_msg_id, conversation_id, redact(synthesis), "pipeline",
                route_reason, result.total_tokens_in, result.total_tokens_out,
                cost, resp_now,
            ),
        )
        _db.execute(
            "UPDATE conversations SET updated_at = ?, "
            "title = CASE WHEN title = 'New conversation' THEN ? ELSE title END "
            "WHERE id = ?",
            (resp_now, user_message[:60], conversation_id),
        )
        _db.execute(
            "INSERT INTO token_usage (id, conversation_id, model, tokens_in, "
            "tokens_out, cost_usd, routed_reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()), conversation_id, "pipeline",
                result.total_tokens_in, result.total_tokens_out, cost,
                route_reason, resp_now,
            ),
        )
        _db.commit()

        try:
            self.memory.add_to_buffer(conversation_id, "user", user_message)
            self.memory.add_to_buffer(conversation_id, "assistant", synthesis)
            self.memory.extract_facts(conversation_id, user_message, synthesis)
        except Exception as exc:
            log.debug("Memory update after pipeline run failed: %s", exc)

        budget_warning = ""
        if budget > 0:
            new_spent = spent + cost
            pct = (new_spent / budget) * 100
            if pct >= warn_pct:
                budget_warning = (
                    f"⚠️ Approaching conversation budget limit "
                    f"(${new_spent:.2f}/${budget:.2f})"
                )

        return ChatResult(
            text=synthesis,
            model="pipeline",
            route_reason=route_reason,
            tokens_in=result.total_tokens_in,
            tokens_out=result.total_tokens_out,
            cost_usd=cost,
            message_id=asst_msg_id,
            budget_warning=budget_warning,
        )

    # ── Token stats ──────────────────────────────────────────────────────────

    def get_token_stats(self, limit: int = 100) -> dict:
        rows = _db.fetchall(
            "SELECT model, SUM(tokens_in) as ti, SUM(tokens_out) as to_, "
            "SUM(cost_usd) as cost FROM token_usage "
            "GROUP BY model ORDER BY cost DESC LIMIT ?",
            (limit,),
        )
        total_cost = sum(r["cost"] or 0 for r in rows)
        # Estimate what the local-served traffic would have cost on the
        # Claude model the router would have fallen back to. Using
        # ``_estimate_cost`` honors any user-configured ``model_prices``
        # override and the configured comparison model, instead of the
        # previous hardcoded Sonnet input price that ignored both.
        comparison_model = self._settings.get(
            "savings_comparison_model", "claude-sonnet"
        )
        local_saved = sum(
            _estimate_cost(
                comparison_model,
                int(r["ti"] or 0),
                int(r["to_"] or 0),
                self._settings,
            )
            for r in rows if "claude" not in (r["model"] or "").lower()
        )
        return {
            "by_model": [dict(r) for r in rows],
            "total_cost_usd": round(total_cost, 4),
            "estimated_savings_usd": round(local_saved, 4),
        }

    # ── Router stats (Stage 3) ────────────────────────────────────────────────

    def get_router_stats(self, limit: int = 500) -> dict:
        """
        Return accuracy trends per complexity bucket from the router_log table.

        Returned shape:
        {
          "total_exchanges": int,
          "by_complexity": {
            "simple":  {"total": int, "errors": int, "empty": int, "error_rate": float},
            "medium":  {...},
            "complex": {...},
          },
          "by_route": {
            "claude": {"total": int, "errors": int, "empty": int, "error_rate": float},
            "local":  {...},
          },
          "recent": [   # last 20 exchanges, newest first
            {"route": str, "complexity": str, "had_error": bool,
             "response_empty": bool, "model_used": str, "created_at": str},
            ...
          ],
          "error_rate_overall": float,
        }
        """
        rows = _db.fetchall(
            "SELECT route_taken, complexity, tokens_out, had_error, "
            "response_empty, model_used, created_at "
            "FROM router_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

        total = len(rows)
        if total == 0:
            return {
                "total_exchanges": 0,
                "by_complexity": {},
                "by_route": {},
                "recent": [],
                "error_rate_overall": 0.0,
            }

        # Aggregate by complexity and by route
        buckets: dict[str, dict] = {}
        by_route: dict[str, dict] = {}

        for r in rows:
            comp = r["complexity"] or "unknown"
            route = r["route_taken"] or "unknown"
            err = int(r["had_error"] or 0)
            empty = int(r["response_empty"] or 0)

            if comp not in buckets:
                buckets[comp] = {"total": 0, "errors": 0, "empty": 0}
            buckets[comp]["total"] += 1
            buckets[comp]["errors"] += err
            buckets[comp]["empty"] += empty

            if route not in by_route:
                by_route[route] = {"total": 0, "errors": 0, "empty": 0}
            by_route[route]["total"] += 1
            by_route[route]["errors"] += err
            by_route[route]["empty"] += empty

        def _rate(d: dict) -> float:
            return round(d["errors"] / d["total"], 4) if d["total"] else 0.0

        for d in buckets.values():
            d["error_rate"] = _rate(d)
        for d in by_route.values():
            d["error_rate"] = _rate(d)

        total_errors = sum(int(r["had_error"] or 0) for r in rows)
        recent = [
            {
                "route": r["route_taken"],
                "complexity": r["complexity"],
                "had_error": bool(r["had_error"]),
                "response_empty": bool(r["response_empty"]),
                "model_used": r["model_used"],
                "created_at": r["created_at"],
            }
            for r in rows[:20]
        ]

        return {
            "total_exchanges": total,
            "by_complexity": buckets,
            "by_route": by_route,
            "recent": recent,
            "error_rate_overall": round(total_errors / total, 4),
        }
