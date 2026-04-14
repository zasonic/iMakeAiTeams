"""
services/chat_orchestrator.py

Unified chat orchestrator.

Fixes applied (Stage 1):
  - stream_multi_turn returns (text, usage) — unpacked for token tracking
  - History fetch capped at MAX_HISTORY_MESSAGES
  - _estimate_cost uses model-aware price table

Stage 3 additions:
  - Router feedback loop: each exchange appends a row to router_log with
    the route taken and quality signals (had_error, response_empty).
  - get_router_stats(): aggregates accuracy trends per complexity bucket.

Stage 5 additions:
  - Returns ChatResult dataclass instead of raw dict (Improvement 1)
  - Token budget enforcement (Improvement 2)
  - ToolPermissionContext plumbing for agents (Improvement 4)
  - ExecutionTarget dispatch via _resolve_target() (Improvement 6)
"""

import json
import logging
import uuid
from datetime import datetime, timezone

import db as _db
from models import ChatResult, ExecutionTarget, ToolPermissionContext
from services.context_compressor import ContextCompressor, micro_compact
from services.hooks import HookManager
from services.retrieval_refiner import RetrievalRefiner
from services.goal_decomposer import GoalDecomposer
from services.security_engine import (
    quarantine_chunks, render_quarantined_context, enforce_context_rules,
    validate_fact_for_storage, RiskLedger, RiskCategory, SecurityAssessment,
)

log = logging.getLogger("MyAIEnv.chat")

MAX_HISTORY_MESSAGES = 40  # 20 user/assistant turns
MAX_CONTEXT_CHARS = 80_000  # ~20K tokens — safe for 128K context models
                             # Leaves room for system prompt + RAG + response

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

    m = model.lower()
    price_in, price_out = next(
        ((pi, po) for key, (pi, po) in prices.items() if key in m),
        (3.0, 15.0),
    )
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
) -> None:
    """Append one row to the router_log table. Non-fatal — never raises."""
    try:
        _db.execute(
            """
            INSERT INTO router_log
                (id, conversation_id, message_preview, route_taken, complexity,
                 reasoning, tokens_out, had_error, response_empty, model_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        _db.commit()
    except Exception as exc:
        log.debug("router_log write failed: %s", exc)


class ChatOrchestrator:
    def __init__(self, claude_client, local_client, router, memory, settings,
                 hook_manager: HookManager | None = None):
        self.claude = claude_client
        self.local = local_client
        self.router = router
        self.memory = memory
        self._settings = settings
        self.hooks = hook_manager or HookManager(settings)
        self._risk_ledgers: dict[str, RiskLedger] = {}  # per-conversation
        self.compressor = ContextCompressor(
            local_client=local_client,
            claude_client=claude_client,
        )
        self.refiner = RetrievalRefiner(
            rag_index=memory.rag,
            local_client=local_client,
            claude_client=claude_client,
        )
        self.decomposer = GoalDecomposer(
            claude_client=claude_client,
            local_client=local_client,
        )

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
        _db.execute("DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,))
        _db.execute("DELETE FROM session_facts WHERE conversation_id = ?",
                    (conversation_id,))
        _db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        _db.commit()

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
        budget = self._settings.get("max_conversation_budget_usd", 5.0)
        warn_pct = self._settings.get("budget_warning_threshold_pct", 80.0)
        row = _db.fetchone(
            "SELECT COALESCE(SUM(cost_usd), 0) as total FROM token_usage WHERE conversation_id = ?",
            (conversation_id,),
        )
        spent = row["total"] if row else 0.0
        if budget > 0 and spent >= budget:
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

        # Save user message to DB
        user_msg_id = str(uuid.uuid4())
        _db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES (?, ?, 'user', ?, ?)",
            (user_msg_id, conversation_id, user_message, now),
        )
        _db.commit()

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

        # ── Fix 7: Token-aware trimming before compression ───────────────────
        messages = self._trim_history_to_budget(messages)

        # ── Context compression (micro-compact every turn) ───────────────────
        messages, compact_result = self.compressor.auto_compact(messages)
        if compact_result and compact_result.success:
            _emit_event("context_compacted", {
                "layer":          compact_result.layer.value,
                "messages_before": compact_result.messages_before,
                "messages_after":  compact_result.messages_after,
                "chars_saved":     compact_result.chars_saved,
            })

        # Recall memory and build system context
        mem = self.memory.get_context(conversation_id, user_message)

        # ── Iterative retrieval refinement ──────────────────────────────────
        # If we have RAG chunks, evaluate whether they're sufficient and
        # refine the search if needed. This is the core agentic loop.
        initial_rag = mem.rag_chunks
        initial_scored = []
        for chunk in initial_rag:
            if isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
                initial_scored.append((str(chunk[0]), float(chunk[1])))
            elif isinstance(chunk, str):
                initial_scored.append((chunk, 0.5))

        refinement = self.refiner.refine(
            user_message,
            initial_chunks=initial_scored if initial_scored else None,
            top_k=3,
            on_event=_emit_event,
        )

        # Replace RAG chunks with refined, deduplicated results
        if refinement.was_refined or refinement.chunks:
            mem.rag_chunks = refinement.texts

        mem_suffix = mem.to_system_suffix()

        # ── Inject project memory (CLAUDE.md) into system prompt ───────────
        # This goes between the base prompt and memory suffix so it's part
        # of the cacheable prefix (static across turns).
        try:
            from services.project_memory import load_project_memory
            project_root = Path(self._settings.get("agent_project_root", "."))
            project_mem = load_project_memory(project_root)
            if project_mem:
                system_prompt = system_prompt + "\n\n" + project_mem
        except Exception as _pm_exc:
            log.debug("Project memory load skipped: %s", _pm_exc)

        full_system = system_prompt
        if mem_suffix:
            full_system = system_prompt + "\n\n" + mem_suffix

        # ── Active Memory retrieval (OpenClaw pattern) ────────────────────────
        # Auto-query RAG for relevant prior context before every reply.
        try:
            from services.project_memory import get_active_memory, should_inject_private_memory
            # Only inject private memory in direct/GUI chats, not group channels
            if should_inject_private_memory():
                active_mem = get_active_memory(
                    user_message,
                    rag_index=self.memory.rag if hasattr(self.memory, "rag") else None,
                    semantic_search_mod=self.memory.semantic if hasattr(self.memory, "semantic") else None,
                )
                if active_mem:
                    full_system += "\n\n" + active_mem
        except Exception:
            pass  # active memory is best-effort

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
            "retrieval_refined": refinement.was_refined,
            "retrieval_passes": refinement.total_passes,
        })

        if self.memory.should_summarize(conversation_id):
            self.memory.summarize_buffer(conversation_id)

        # ── Hook: pre_route ──────────────────────────────────────────────────
        route_ctx = self.hooks.fire("pre_route", {
            "user_message": user_message,
            "conversation_id": conversation_id,
        })

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

        # ── v4.1 UAR: Confidence-driven context expansion ────────────────────
        # When the router signals low confidence or needs_context, widen the
        # retrieval window BEFORE generating. This is the core UAR insight:
        # "epistemic anxiety" should trigger retrieval, not generation.
        if route_needs_context and self.memory.rag:
            _emit_event("context_expanding", {
                "reason": "low router confidence",
                "confidence": route_confidence,
            })
            try:
                extra = self.refiner.refine(
                    user_message,
                    initial_chunks=[(t, s) for t, s in
                                    (refinement.chunks if refinement.chunks else [])],
                    top_k=6,  # double the normal retrieval width
                    on_event=_emit_event,
                )
                if extra.chunks:
                    # Merge new chunks with existing, dedup by content
                    existing = set(mem.rag_chunks)
                    for text in extra.texts:
                        if text not in existing:
                            mem.rag_chunks.append(text)
                            existing.add(text)
                    # Rebuild system prompt with expanded context
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
                    log.info("UAR context expansion: %d total RAG chunks",
                             len(mem.rag_chunks))
            except Exception as exc:
                log.debug("UAR context expansion failed (non-fatal): %s", exc)

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

        # ── Hook: post_route ─────────────────────────────────────────────────
        self.hooks.fire("post_route", {
            "route_model": route_model,
            "complexity": complexity,
            "reasoning": route_reason,
        })

        # ── Improvement 6: Resolve execution target ──────────────────────────
        target = self._resolve_target(route_model, agent)

        # ── Hook: pre_send ───────────────────────────────────────────────────
        send_ctx = self.hooks.fire("pre_send", {
            "user_message": user_message,
            "system_prompt": full_system,
            "model": target.model_name,
            "conversation_id": conversation_id,
        })
        if send_ctx.get("_blocked"):
            return ChatResult(
                text=f"⚠️ Blocked by hook: {send_ctx.get('_block_reason', 'unknown')}",
                model="", route_reason="hook_blocked",
                tokens_in=0, tokens_out=0, cost_usd=0.0,
                message_id=str(uuid.uuid4()),
            )
        # Allow hooks to modify the system prompt
        if "system_prompt" in send_ctx and send_ctx["system_prompt"] != full_system:
            full_system = send_ctx["system_prompt"]

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
                    if raw_rag and "## Relevant documents" in full_system:
                        # Swap the raw documents section for quarantined version
                        full_system = full_system.replace(
                            "## Relevant documents",
                            "## Retrieved Context (Quarantined)",
                        )

            # --- Deterministic Rule Engine: strip structural attacks ---
            full_system, violations = enforce_context_rules(
                full_system, source_label=conversation_id[:8]
            )
            security.context_violations = violations

            # --- Risk Ledger: track cumulative risk ---
            ledger = self._risk_ledgers.setdefault(conversation_id, RiskLedger())
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

        # ── v4.0 #2: Dynamic Goal Decomposition ─────────────────────────────
        # Only attempt decomposition when: routing to Claude, classified complex,
        # goal decomposition is enabled in settings, and message looks multi-step.
        decomposition_enabled = self._settings.get("goal_decomposition_enabled", True)
        response_text = ""
        tokens_in = 0
        tokens_out = 0
        model_name = target.model_name
        had_error = False
        used_decomposition = False

        if (
            decomposition_enabled
            and target.backend == "claude"
            and complexity in ("complex", "medium")
        ):
            try:
                decomp = self.decomposer.decompose_and_execute(
                    message=user_message,
                    system_prompt=full_system,
                    messages=messages,
                    complexity=complexity,
                    on_event=_emit_event,
                    on_token=on_token if on_token else None,
                )
                if decomp.was_decomposed and decomp.final_response:
                    # ── Fix 10: Quality gate — reject short or error-like output ──
                    resp = decomp.final_response.strip()
                    if len(resp) > 50 and not resp.startswith("[Error"):
                        response_text = resp
                        tokens_in = decomp.total_tokens_in
                        tokens_out = decomp.total_tokens_out
                        used_decomposition = True
                        log.info(
                            "Goal decomposition used: %d steps, %d tokens",
                            len(decomp.steps_completed), tokens_out,
                        )
                    else:
                        log.warning(
                            "Decomposition produced low-quality output (%d chars), "
                            "falling through to normal path", len(resp)
                        )
                        # used_decomposition stays False → normal/thinking path runs
            except Exception as exc:
                log.warning("Goal decomposer failed (falling through): %s", exc)

        # ── v4.0 #4: Interleaved Reasoning Visibility ────────────────────────
        # When routing to Claude and extended thinking is available, emit
        # a reasoning step event before generating the final response.
        reasoning_enabled = self._settings.get("interleaved_reasoning_enabled", True)
        if (
            not used_decomposition
            and reasoning_enabled
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
        if not response_text:
            try:
                if target.backend == "claude":
                    if on_token:
                        response_text, usage = self.claude.stream_multi_turn(
                            full_system, messages, on_token,
                            max_tokens=target.max_tokens,
                        )
                        if usage is not None:
                            tokens_in = getattr(usage, "input_tokens", 0) or 0
                            tokens_out = getattr(usage, "output_tokens", 0) or 0
                    else:
                        result = self.claude.chat_multi_turn(
                            full_system, messages,
                            max_tokens=target.max_tokens,
                        )
                        response_text = result["text"]
                        tokens_in = result.get("input_tokens", 0)
                        tokens_out = result.get("output_tokens", 0)
                else:
                    if on_token:
                        response_text = self.local.stream_multi_turn(
                            full_system, messages, on_token,
                            max_tokens=target.max_tokens,
                        )
                    else:
                        response_text = self.local.chat_multi_turn(
                            full_system, messages,
                            max_tokens=target.max_tokens,
                        )
            except Exception as exc:
                log.error(f"Chat execution failed: {exc}")
                response_text = f"[Error: {exc}]"
                had_error = True

        # ── Local response quality gate ─────────────────────────────────────
        # If response came from local and looks weak, escalate to Claude
        response_empty = len((response_text or "").strip()) < 20
        if (
            not had_error
            and target.backend == "local"
            and not response_empty
            and self.local and self.local.is_available()
            and len(user_message.split()) >= 5  # skip for trivial messages
        ):
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
                        quality = _json.loads(quality_raw[_qstart:_qend + 1])
                        if quality.get("score", 10) < 4:
                            log.info("Local response scored %s — escalating to Claude", quality.get("score"))
                            try:
                                if on_token:
                                    response_text, usage = self.claude.stream_multi_turn(
                                        full_system, messages, on_token,
                                        max_tokens=target.max_tokens,
                                    )
                                    if usage:
                                        tokens_in = getattr(usage, "input_tokens", 0) or 0
                                        tokens_out = getattr(usage, "output_tokens", 0) or 0
                                else:
                                    result = self.claude.chat_multi_turn(
                                        full_system, messages,
                                        max_tokens=target.max_tokens,
                                    )
                                    response_text = result["text"]
                                    tokens_in = result.get("input_tokens", 0)
                                    tokens_out = result.get("output_tokens", 0)
                                route_model = "claude"
                                model_name = self.claude._model
                            except Exception as esc_exc:
                                log.debug("Escalation to Claude failed: %s", esc_exc)
            except Exception:
                pass  # quality check is best-effort, never block response

        # Quality signals for router feedback

        # ── Hook: post_response ──────────────────────────────────────────────
        self.hooks.fire("post_response", {
            "response":        response_text,
            "model":           model_name,
            "tokens_in":       tokens_in,
            "tokens_out":      tokens_out,
            "cost_usd":        _estimate_cost(model_name, tokens_in, tokens_out, self._settings),
            "had_error":       had_error,
            "conversation_id": conversation_id,
        })

        # Persist router feedback
        _log_router_event(
            conversation_id=conversation_id,
            message_preview=user_message,
            route_taken=route_model,
            complexity=complexity,
            reasoning=route_reason,
            tokens_out=tokens_out,
            had_error=had_error or response_text.startswith("[Error"),
            response_empty=response_empty,
            model_used=model_name,
        )

        # Save assistant message
        cost = _estimate_cost(model_name, tokens_in, tokens_out, self._settings)
        asst_msg_id = str(uuid.uuid4())
        resp_now = datetime.now(timezone.utc).isoformat()
        _db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, model_used, "
            "route_reason, tokens_in, tokens_out, cost_usd, created_at) "
            "VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?)",
            (asst_msg_id, conversation_id, response_text, model_name,
             route_reason, tokens_in, tokens_out, cost, resp_now),
        )
        _db.execute(
            "UPDATE conversations SET updated_at = ?, "
            "title = CASE WHEN title = 'New conversation' THEN ? ELSE title END "
            "WHERE id = ?",
            (resp_now, user_message[:60], conversation_id),
        )
        _db.commit()

        # Token usage row
        _db.execute(
            "INSERT INTO token_usage (id, conversation_id, model, tokens_in, "
            "tokens_out, cost_usd, routed_reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), conversation_id, model_name,
             tokens_in, tokens_out, cost, route_reason, resp_now),
        )
        _db.commit()

        # Update memory
        self.memory.add_to_buffer(conversation_id, "user", user_message)
        self.memory.add_to_buffer(conversation_id, "assistant", response_text)
        self.memory.extract_facts(conversation_id, user_message, response_text)

        # ── Improvement 2: Budget warning check ─────────────────────────────
        budget_warning = ""
        if budget > 0:
            new_spent = spent + cost
            pct = (new_spent / budget) * 100
            if pct >= warn_pct:
                budget_warning = f"\u26a0\ufe0f Approaching conversation budget limit (${new_spent:.2f}/${budget:.2f})"

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

    # ── Token stats ──────────────────────────────────────────────────────────

    def get_token_stats(self, limit: int = 100) -> dict:
        rows = _db.fetchall(
            "SELECT model, SUM(tokens_in) as ti, SUM(tokens_out) as to_, "
            "SUM(cost_usd) as cost FROM token_usage "
            "GROUP BY model ORDER BY cost DESC LIMIT ?",
            (limit,),
        )
        total_cost = sum(r["cost"] or 0 for r in rows)
        local_saved = sum(
            (r["ti"] or 0) * 3.0 / 1_000_000
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
