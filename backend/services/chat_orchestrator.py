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

import base64
import concurrent.futures
import difflib
import json
import logging
import re
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

import db as _db
from models import (
    ChatResult, ExecutionTarget, ReaderOutput, RoutingDecision, TaskDescriptor,
    WorkerResult,
)
from services.governance import GovernanceEngine, is_high_stakes_message
from services.hub_router import HubRouter
from services.local_client import LocalVisionUnavailable
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

# Phase 8: parse 'CONFIDENCE: NN' from the tail of a CoT sample.
_CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*:\s*(\d{1,3})", re.IGNORECASE)


def _parse_confidence(text: str) -> int:
    """Extract a 0-100 confidence from a CoT sample. Missing/invalid → 50."""
    if not text:
        return 50
    m = _CONFIDENCE_RE.search(text)
    if not m:
        return 50
    try:
        v = int(m.group(1))
    except ValueError:
        return 50
    return max(0, min(100, v))


def _strip_confidence(text: str) -> str:
    """Drop the trailing 'CONFIDENCE: NN' so the chosen answer renders cleanly."""
    if not text:
        return ""
    return _CONFIDENCE_RE.sub("", text).rstrip()


def _detect_compound(msg: str) -> bool:
    """Detect messages containing multiple independent requests."""
    return len(_COMPOUND_SIGNALS.findall(msg)) >= 2 or msg.count("?") >= 3

def _estimate_cost(model: str, tokens_in: int, tokens_out: int,
                   settings=None) -> float:
    """Estimate per-turn USD cost.

    Delegates to ``core.model_catalog`` so the catalog file is the
    single source of truth — adding a model requires editing
    ``backend/config/models.json`` and nothing else. Users can still
    override prices via the ``model_prices`` setting; ModelCatalog
    threads that dict through before falling back to catalog defaults.
    """
    if not model or "claude" not in model.lower():
        return 0.0

    from core.model_catalog import get_catalog

    user_overrides: dict[str, tuple[float, float]] | None = None
    if settings:
        custom = settings.get("model_prices", None)
        if custom and isinstance(custom, dict):
            user_overrides = {}
            for key, val in custom.items():
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    user_overrides[key] = (float(val[0]), float(val[1]))

    price_in, price_out = get_catalog().prices_for_model(model, user_overrides)
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
    agent_role: str = "monolithic",
    voting_samples_json: str | None = None,
    turn_id: str = "",
) -> None:
    """Append one row to the router_log table. Non-fatal — never raises.

    Layer C1: ``turn_id`` is the per-turn correlation id from
    ``TurnContext.turn_id``. Every per-phase row (reader, actor,
    voting samples, escalation rescue, monolithic summary, CaMeL turn)
    carries it so analytics can reassemble the timeline of a single
    chat turn with one indexed query.
    """
    try:
        with _db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO router_log
                    (id, conversation_id, message_preview, route_taken, complexity,
                     reasoning, tokens_out, had_error, response_empty, model_used,
                     mast_category, agent_role, voting_samples_json, turn_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    agent_role,
                    voting_samples_json,
                    turn_id or None,
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
        # Layer 3 extraction: memory recall + system-prompt assembly. The
        # RAG-trim path used to duplicate the assembly logic; now both the
        # initial recall and the post-trim rebuild route through the same
        # MemoryRecall._assemble() so the two can never drift apart.
        from services.memory_recall import MemoryRecall
        from services.turn_lifecycle import TurnLifecycle
        from services.turn_router import TurnRouter
        from services.security_gate import SecurityGate
        from services.escalation_ladder import EscalationLadder
        from services.worker_dispatch import WorkerDispatch
        self._memory_recall = MemoryRecall(memory, settings, mcp_registry)
        # TurnLifecycle owns the open/close transaction boundary that
        # Layer 1's Bug 5 + Bug 6 fixes hardened. Auto-title uses the local
        # client OUTSIDE the transaction (see TurnLifecycle.maybe_auto_title).
        self._turn_lifecycle = TurnLifecycle(settings, local_client)
        # TurnRouter wraps the agent.model_preference override around
        # TaskRouter.classify(); see services/turn_router.py.
        self._turn_router = TurnRouter(router)
        # SecurityGate owns the quarantine + rule-engine + sliding-window
        # risk ledger. Bug 7's LRU cap on the per-conversation history
        # dict lives inside it; delete_conversation calls .forget() to
        # evict on archival.
        self._security_gate = SecurityGate()
        # EscalationLadder is constructed below, after hub_router is wired.
        self._EscalationLadder = EscalationLadder
        self._local_client_ref = local_client
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
        # Now that hub_router is wired, finish building EscalationLadder.
        self._escalation_ladder = self._EscalationLadder(
            self.hub_router, self._local_client_ref,
        )
        # WorkerDispatch wraps RoutingDecision construction + hub_router.invoke
        # so the orchestrator stops re-implementing the same shell at every
        # call site (per-turn dispatch, reader phase, actor phase).
        self._worker_dispatch = WorkerDispatch(self.hub_router)

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
        self._security_gate.forget(conversation_id)

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

    # ── PR 8: file attachments ───────────────────────────────────────────────

    @staticmethod
    def _fetch_ephemeral_attachment_chunks(conversation_id: str) -> list[str]:
        """Return the ``content_extract`` of every ephemeral attachment for
        this conversation, in upload order. Persistent attachments live in
        RAG and are reached via the normal ``mem.rag_chunks`` path.

        Skips image rows — images are fed to the model via vision blocks,
        not as quarantined text context.
        """
        rows = _db.fetchall(
            "SELECT filename, mime_type, content_extract FROM attachments "
            "WHERE conversation_id = ? AND persist = 0 "
            "ORDER BY created_at ASC",
            (conversation_id,),
        )
        chunks: list[str] = []
        for r in rows:
            mime = (r["mime_type"] or "").lower()
            if mime.startswith("image/"):
                continue
            extract = r["content_extract"] or ""
            if not extract.strip():
                continue
            chunks.append(f"[{r['filename']}]\n{extract}")
        return chunks

    @staticmethod
    def _fetch_image_attachments(conversation_id: str) -> list[dict]:
        """Return image attachments for this conversation as a list of
        ``{"id", "filename", "mime_type", "data": <base64 str>}`` dicts.

        Reads each image file from ``userData/attachments/`` and base64-
        encodes it. Files that fail to read are skipped and logged at
        warning level so a single bad row doesn't poison the turn.
        """
        rows = _db.fetchall(
            "SELECT id, filename, mime_type FROM attachments "
            "WHERE conversation_id = ? AND mime_type LIKE 'image/%' "
            "ORDER BY created_at ASC",
            (conversation_id,),
        )
        if not rows:
            return []
        from core import paths as _paths
        adir = _paths.attachments_dir()
        out: list[dict] = []
        for r in rows:
            name = r["filename"] or ""
            ext = ""
            dot = name.rfind(".")
            if dot >= 0:
                ext = name[dot:].lower()
            disk_path = adir / f"{r['id']}{ext}"
            try:
                raw = disk_path.read_bytes()
            except OSError as exc:
                log.warning(
                    "image attachment %s missing on disk: %s", r["id"], exc,
                )
                continue
            try:
                data = base64.b64encode(raw).decode("ascii")
            except Exception as exc:  # noqa: BLE001
                log.warning("image attachment %s encode failed: %s", r["id"], exc)
                continue
            out.append({
                "id": r["id"],
                "filename": name,
                "mime_type": (r["mime_type"] or "image/png").lower(),
                "data": data,
            })
        return out

    @staticmethod
    def _attach_images_to_messages(
        messages: list, images: list[dict],
    ) -> list:
        """Return a copy of ``messages`` with Anthropic image blocks
        prepended to the last user message's content.

        The shape matches Anthropic's spec:
          {"type": "image", "source": {"type": "base64",
            "media_type": "image/png", "data": "<base64>"}}

        When the last user message has string content, it's converted to
        a content-block list with the original text as a trailing text
        block. When already a list, image blocks are prepended verbatim.
        """
        if not images:
            return messages
        out = list(messages)
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") != "user":
                continue
            content = out[i].get("content")
            blocks: list[dict] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["mime_type"],
                        "data": img["data"],
                    },
                }
                for img in images
            ]
            if isinstance(content, list):
                blocks.extend(content)
            else:
                blocks.append({"type": "text", "text": str(content or "")})
            out[i] = {**out[i], "content": blocks}
            break
        return out

    @staticmethod
    def _purge_ephemeral_attachments(conversation_id: str) -> None:
        """Delete ephemeral attachments + their on-disk files after a turn
        completes. Persistent rows are left alone — they live in RAG.
        """
        rows = _db.fetchall(
            "SELECT id, filename FROM attachments "
            "WHERE conversation_id = ? AND persist = 0",
            (conversation_id,),
        )
        if not rows:
            return
        from core import paths as _paths
        adir = _paths.attachments_dir()
        for r in rows:
            ext = ""
            name = r["filename"] or ""
            dot = name.rfind(".")
            if dot >= 0:
                ext = name[dot:].lower()
            disk_path = adir / f"{r['id']}{ext}"
            try:
                disk_path.unlink(missing_ok=True)
            except OSError as exc:
                log.debug("attachment unlink failed: %s", exc)
        _db.execute(
            "DELETE FROM attachments WHERE conversation_id = ? AND persist = 0",
            (conversation_id,),
        )
        _db.commit()

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

    # ── Phase 6: Hackett et al. (ACL 2025) Reader/Actor split ────────────────

    @staticmethod
    def _load_prompt_template(name: str, fallback: str) -> str:
        """Load a prompt template from backend/templates/.

        Falls back to the inline string when the file is missing (which can
        happen if the PyInstaller bundle drops the templates directory or in
        unusual test layouts). The fallback keeps the architectural wall —
        same intent, just terser.
        """
        try:
            from pathlib import Path
            here = Path(__file__).resolve().parent.parent
            text = (here / "templates" / name).read_text(encoding="utf-8")
            if text.strip():
                return text
        except Exception:
            pass
        return fallback

    def _read_phase(
        self,
        conversation_id: str,
        user_message: str,
        agent_id: str | None,
        history: list,
        mem,
        turn_id: str = "",
    ) -> ReaderOutput:
        """Reader: analyze the request and propose tools. No tool execution."""
        reader_system = self._load_prompt_template(
            "reader_system.txt",
            "You are the Reader. Output JSON only with keys "
            "intent, constraints, relevant_facts, proposed_tools, red_flags. "
            "Never call tools.",
        )

        # Quarantined retrieval surface for the Reader. The Reader is the
        # ONLY phase that sees retrieved data; the Actor never does.
        retrieval_block = ""
        if mem.rag_chunks:
            quarantined = quarantine_chunks(
                mem.rag_chunks,
                source_type="user_document",
                source_id=conversation_id,
            )
            retrieval_block = render_quarantined_context(quarantined)

        memory_block = ""
        if mem.session_facts or mem.memories:
            mem_lines: list[str] = []
            if mem.session_facts:
                mem_lines.append("## Session facts")
                mem_lines.extend(f"- {f}" for f in mem.session_facts)
            if mem.memories:
                mem_lines.append("## Long-term memories")
                mem_lines.extend(f"- {m}" for m in mem.memories)
            memory_block = "\n".join(mem_lines)

        reader_user = (
            f"USER MESSAGE:\n{user_message}\n\n"
            + (f"{retrieval_block}\n\n" if retrieval_block else "")
            + (f"{memory_block}\n\n" if memory_block else "")
            + "Return JSON now."
        )

        decision = self._worker_dispatch.build_phase_decision(agent_id, user_message)
        worker = self._worker_dispatch.dispatch(
            decision,
            reader_system,
            [{"role": "user", "content": reader_user}],
            max_tokens=1024,
            on_token=None,  # Reader output is JSON; never stream to user
            agent_role="reader",
        )
        self._log_phase_router_event(
            conversation_id=conversation_id,
            preview=user_message,
            decision=decision,
            worker=worker,
            agent_role="reader",
            turn_id=turn_id,
        )
        return ReaderOutput.from_raw(worker.text)

    def _act_phase(
        self,
        conversation_id: str,
        reader_output: ReaderOutput,
        history: list,
        full_system: str,
        agent_id: str | None,
        on_token=None,
        max_tokens: int = 4096,
        vote: bool = False,
        voting_message_id: str = "",
        voting_emit=None,
        turn_id: str = "",
    ) -> tuple[WorkerResult, list[dict] | None]:
        """Actor: execute against the Reader's plan. Never sees raw user text.

        ``full_system`` here is the agent persona ONLY — the orchestrator
        passes the pre-memory ``system_prompt`` so the Actor does not receive
        retrieved RAG, session facts, or memories through its system prompt.
        The Reader is the single phase that touches retrieved data.

        Phase 8: when ``vote`` is True AND the actor's resolved decision
        targets Claude, the single hub_router.invoke is replaced with a
        3-sample weighted-vote consensus. The voting_message_id and
        voting_emit are forwarded so the frontend can show the consensus
        spinner around the right message.
        """
        actor_system_template = self._load_prompt_template(
            "actor_system.txt",
            "You are the Actor. Use only tools listed in proposed_tools. "
            "You receive a JSON plan; the user's raw message is hidden.",
        )
        # Compose: actor instructions on top, then the bare persona.
        actor_system = actor_system_template + "\n\n" + (full_system or "")

        # Populate the per-task ledger BEFORE the Actor runs so any tool call
        # the Actor proposes is gated against the Reader's allowlist.
        self._governance.set_proposed_tools(
            conversation_id, reader_output.proposed_tools
        )

        # The Reader's relevant_facts are forwarded through the existing
        # security_engine quarantine path as user_document so any
        # instructions that survived the Reader's filtering still hit the
        # structural-isolation delimiters before reaching the Actor.
        plan_payload = {
            "intent": reader_output.intent,
            "constraints": list(reader_output.constraints),
            "proposed_tools": list(reader_output.proposed_tools),
            "red_flags": list(reader_output.red_flags),
        }
        quarantine_block = ""
        if reader_output.relevant_facts:
            quarantined = quarantine_chunks(
                list(reader_output.relevant_facts),
                source_type="user_document",
                source_id=conversation_id,
            )
            quarantine_block = render_quarantined_context(quarantined)

        actor_user = (
            "Planner output (your only view of the request):\n"
            f"{json.dumps(plan_payload, ensure_ascii=False)}"
            + (f"\n\n{quarantine_block}" if quarantine_block else "")
        )

        decision = self._worker_dispatch.build_phase_decision(agent_id, actor_user)
        actor_messages = [{"role": "user", "content": actor_user}]
        voting_samples: list[dict] | None = None
        if vote and decision.backend == "claude":
            if voting_emit is not None:
                try:
                    voting_emit("chat_event", {
                        "type": "high_stakes_voting_started",
                        "message_id": voting_message_id,
                    })
                except Exception:
                    pass
            worker, voting_samples = self._high_stakes_consensus(
                decision, actor_system, actor_messages,
                max_tokens=max_tokens, on_token=on_token,
                agent_role="actor",
            )
            if voting_emit is not None:
                try:
                    voting_emit("chat_event", {
                        "type": "high_stakes_voting_complete",
                        "message_id": voting_message_id,
                    })
                except Exception:
                    pass
        else:
            worker = self._worker_dispatch.dispatch(
                decision,
                actor_system,
                actor_messages,
                max_tokens=max_tokens,
                on_token=on_token,
                agent_role="actor",
            )
        self._log_phase_router_event(
            conversation_id=conversation_id,
            preview=reader_output.intent or "actor",
            decision=decision,
            worker=worker,
            agent_role="actor",
            voting_samples_json=(
                json.dumps(voting_samples) if voting_samples is not None
                else None
            ),
            turn_id=turn_id,
        )
        return worker, voting_samples

    @staticmethod
    def _synthesize_phase(
        reader_output: ReaderOutput, actor_result: WorkerResult,
    ) -> WorkerResult:
        """Combine Reader + Actor into the final assistant turn.

        The final text is the Actor's text. Tokens are summed from the Actor
        (the Reader's tokens are tracked separately via router_log). The
        Reader's red_flags are surfaced to the orchestrator via the returned
        WorkerResult's text only when the Actor produced nothing usable.
        """
        text = actor_result.text or ""
        if not text.strip() and reader_output.red_flags:
            text = (
                "I could not produce a response: the Reader flagged "
                f"{len(reader_output.red_flags)} suspicious pattern(s) in the "
                "retrieved context. Please review the input."
            )
        return WorkerResult(
            text=text,
            backend=actor_result.backend,
            model_name=actor_result.model_name,
            input_tokens=actor_result.input_tokens,
            output_tokens=actor_result.output_tokens,
            had_error=actor_result.had_error,
        )

    def _log_phase_router_event(
        self,
        *,
        conversation_id: str,
        preview: str,
        decision: RoutingDecision,
        worker: WorkerResult,
        agent_role: str,
        voting_samples_json: str | None = None,
        turn_id: str = "",
    ) -> None:
        text = worker.text or ""
        _log_router_event(
            conversation_id=conversation_id,
            message_preview=preview,
            route_taken=decision.backend,
            complexity="phase",
            reasoning=f"reader_actor split: {agent_role}",
            tokens_out=worker.output_tokens,
            had_error=worker.had_error,
            response_empty=len(text.strip()) < 20,
            model_used=worker.model_name,
            agent_role=agent_role,
            voting_samples_json=voting_samples_json,
            turn_id=turn_id,
        )

    # ── Phase 8: Symphony-style weighted-vote consensus ─────────────────────

    def _high_stakes_consensus(
        self,
        decision: RoutingDecision,
        full_system: str,
        messages: list,
        max_tokens: int,
        on_token,
        agent_role: str = "monolithic",
    ) -> tuple[WorkerResult, list[dict]]:
        """Run hub_router.invoke 3 times in parallel with a CoT prompt
        addendum, then return the weighted-majority winner.

        The 3 calls don't stream — collecting token-by-token output from
        three concurrent runs would interleave nonsensically. After voting
        picks a winner, the chosen text is replayed through ``on_token``
        in one shot so the UI still receives content.

        Returns ``(WorkerResult, voting_samples)`` where ``voting_samples``
        is a JSON-serializable list capturing each sample's text preview,
        confidence, weighted score, and which one was chosen.
        """
        cot_system = (full_system or "") + (
            "\n\n## Consensus instructions\n"
            "First reason through the request step-by-step, then state your "
            "final answer. End with 'CONFIDENCE: X' where X is 0-100 "
            "representing your confidence in your answer."
        )
        per_call_max = max(256, int(max_tokens * 0.7))

        samples: list[WorkerResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(
                    self.hub_router.invoke,
                    decision,
                    cot_system,
                    list(messages),
                    max_tokens=per_call_max,
                    on_token=None,
                    agent_role=agent_role,
                )
                for _ in range(3)
            ]
            for fut in futures:
                try:
                    samples.append(fut.result())
                except Exception as exc:
                    log.warning("voting sample failed: %s", exc)
                    samples.append(WorkerResult(
                        text="", backend=decision.backend, model_name="",
                        input_tokens=0, output_tokens=0, had_error=True,
                    ))

        confidences = [_parse_confidence(s.text) for s in samples]
        texts = [_strip_confidence(s.text) for s in samples]

        scores: list[float] = []
        for i in range(len(samples)):
            sim_sum = sum(
                difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
                for j in range(len(samples)) if j != i
            )
            scores.append((confidences[i] / 100.0) * sim_sum)

        # Detect "all three diverge" — every pairwise similarity below 0.4.
        # In that case we fall back to highest raw confidence and tag the
        # samples payload so router_log preserves the divergence.
        all_diverged = True
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                if difflib.SequenceMatcher(None, texts[i], texts[j]).ratio() >= 0.4:
                    all_diverged = False
                    break
            if not all_diverged:
                break

        if all_diverged:
            winner_idx = max(
                range(len(samples)),
                key=lambda i: (confidences[i], len(texts[i] or "")),
            )
        else:
            winner_idx = max(
                range(len(samples)),
                key=lambda i: (scores[i], confidences[i]),
            )

        winner = samples[winner_idx]
        chosen_text = texts[winner_idx]

        if on_token and chosen_text:
            try:
                on_token(chosen_text)
            except Exception:
                pass

        tokens_in = sum(s.input_tokens for s in samples)
        tokens_out = sum(s.output_tokens for s in samples)

        voting_samples = [
            {
                "text": (texts[i] or "")[:1000],
                "confidence": confidences[i],
                "score": round(scores[i], 4),
                "chosen": (i == winner_idx),
                "all_diverged": all_diverged,
            }
            for i in range(len(samples))
        ]

        return WorkerResult(
            text=chosen_text,
            backend=winner.backend,
            model_name=winner.model_name,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            had_error=all(s.had_error for s in samples),
        ), voting_samples

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

        # Layer 3 extraction: TurnLifecycle owns the budget check + user
        # message INSERT under the same db lock. Mutates ctx with budget,
        # warn_pct, spent, budget_exceeded, user_msg_id; we keep local
        # aliases to minimise downstream churn for the rest of send().
        from services.turn_context import TurnContext
        ctx = TurnContext(
            conversation_id=conversation_id,
            user_message=user_message,
            agent_id=agent_id,
            on_event=on_event,
            on_token=on_token,
        )
        budget_exceeded = not self._turn_lifecycle.open(ctx)
        budget = ctx.budget
        warn_pct = ctx.warn_pct
        spent = ctx.spent
        user_msg_id = ctx.user_msg_id
        now = ctx.started_at

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

        # ── PR 8: ephemeral attachment context ───────────────────────────────
        # Attachments dropped onto the chat input live in their own table —
        # ephemeral rows (persist=0) are out-of-band context for the next
        # send only. We splice their extracted text into the last user
        # message inside the existing quarantine envelope so the model
        # treats them as data, not instructions, and we never write the
        # combined string back to ``messages`` table (the user's typed text
        # is the persisted record). Persistent rows (persist=1) live in
        # RAG already, so they flow through ``mem.rag_chunks`` like any
        # other indexed document.
        ephemeral_chunks = self._fetch_ephemeral_attachment_chunks(conversation_id)
        if ephemeral_chunks and messages:
            quarantined = quarantine_chunks(
                ephemeral_chunks,
                source_type="user_document",
                source_id=f"attach:{conversation_id}",
            )
            attachment_block = render_quarantined_context(quarantined)
            if attachment_block:
                last = messages[-1]
                if last.get("role") == "user":
                    last["content"] = (
                        attachment_block + "\n\n" + last.get("content", "")
                    )

        # ── PR 11: image attachments (vision input) ──────────────────────────
        # Images are fetched separately from text — they don't go through the
        # quarantine envelope (binary, not data-as-text), they ride as
        # provider-specific blocks instead. When the vision_enabled setting
        # is off, images are silently ignored to preserve existing behavior.
        image_attachments: list[dict] = []
        if self._settings.get("vision_enabled", True):
            try:
                image_attachments = self._fetch_image_attachments(conversation_id)
            except Exception as exc:
                log.debug("image attachment fetch failed: %s", exc)

        # Recall memory + build system context. Layer 3: MemoryRecall
        # owns the get_context call, the mem_suffix stitching, the tool
        # restriction block, and the MCP tool description block. Both
        # initial recall and the post-trim rebuild route through it, so
        # the two paths can't drift the way they used to.
        mem_result = self._memory_recall.recall(
            conversation_id=conversation_id,
            user_message=user_message,
            system_prompt=system_prompt,
            allowed_tools=_allowed_tools,
            agent=agent,
        )
        mem = mem_result.mem
        mem_suffix = mem_result.mem_suffix
        full_system = mem_result.full_system

        _emit_event("memory_recalled", self._memory_recall.memory_recalled_event(mem))
        self._memory_recall.maybe_summarize(conversation_id)

        # Layer 3 extraction: TurnRouter owns the agent.model_preference
        # override + delegation to TaskRouter.classify(). The five fields
        # downstream code uses (route_model, route_reason, complexity,
        # route_confidence, route_needs_context) come back as one
        # RouteOutcome and are unpacked into locals to keep the rest of
        # send() touching as little as possible.
        ctx.agent = agent
        route_outcome = self._turn_router.decide(ctx, messages, mem)
        route_model = route_outcome.model
        route_reason = route_outcome.reasoning
        complexity = route_outcome.complexity
        route_confidence = route_outcome.confidence
        route_needs_context = route_outcome.needs_context
        self._turn_router.emit_decision(ctx, route_outcome)

        if _detect_compound(user_message):
            _emit_event("compound_query_detected", {
                "message": "This looks like multiple requests. A team of agents might handle this better.",
                "suggestion": "Try selecting a team coordinator for complex multi-part requests.",
            })

        # Adaptive RAG trim (Engram-inspired). MemoryRecall owns the
        # complexity → max-items mapping and the system-prompt rebuild, so
        # the trim and the initial assembly always produce identical
        # prompts for the same inputs.
        mem_result = self._memory_recall.trim_for_complexity(
            mem_result, complexity, system_prompt,
            allowed_tools=_allowed_tools, agent=agent,
        )
        mem = mem_result.mem
        mem_suffix = mem_result.mem_suffix
        full_system = mem_result.full_system

        # ── Phase 1: Build routing decision through the HubRouter ────────────
        # The TaskRouter above decided which *backend* to use; the HubRouter
        # decides which *worker* and authorizes the dispatch. WorkerDispatch
        # handles both branches: route_for_agent when agent_id is set
        # (AuthorizationError still propagates) and hub-direct synthesis
        # driven by the TurnRouter's RouteOutcome when it isn't.
        task = TaskDescriptor(
            text=user_message,
            preferred_agent_id=agent_id,
            backend_hint=route_model,
        )
        decision = self._worker_dispatch.build_turn_decision(
            agent_id, task, route_outcome,
        )

        # ── Improvement 6: Resolve execution target ──────────────────────────
        target = self._resolve_target(decision.backend, agent)

        # ── PR 11: vision dispatch decision ──────────────────────────────────
        # Decide once, here, whether image attachments will ride along on
        # this turn. For Claude, we splice image blocks into the messages
        # list so any downstream path (voting, monolithic, reader/actor)
        # passes them through transparently. For local, we need a
        # vision-capable model — otherwise raise LocalVisionUnavailable now
        # so the user sees a friendly hint instead of a silent text-only
        # response. The list is reused below.
        if image_attachments and target.backend == "local":
            local_model = self._settings.get("default_local_model", "")
            if not (
                hasattr(self.local, "is_vision_model")
                and self.local.is_vision_model(local_model)
            ):
                families = self._settings.get("vision_local_models", []) or []
                fams = ", ".join(str(f) for f in families) or "(none configured)"
                msg = (
                    f"🖼️ Your active local model ({local_model or 'none'}) "
                    f"can't see images. Switch to a vision-capable model "
                    f"such as: {fams}, then resend."
                )
                _emit_event("vision_unavailable", {
                    "active_model": local_model,
                    "families": list(families),
                })
                # Drop the ephemeral image rows so they don't linger to the
                # next send and trigger the same error again.
                try:
                    self._purge_ephemeral_attachments(conversation_id)
                except Exception:
                    pass
                return ChatResult(
                    text=msg, model=local_model,
                    route_reason="vision_unavailable_local",
                    tokens_in=0, tokens_out=0, cost_usd=0.0,
                    message_id=str(uuid.uuid4()),
                )
        if image_attachments and target.backend == "claude":
            messages = self._attach_images_to_messages(messages, image_attachments)

        # ══════════════════════════════════════════════════════════════════════
        # SECURITY ENGINE: Structural enforcement before model inference
        # Runs AFTER context assembly, AFTER hooks, BEFORE any model call.
        # Uses deterministic rules (not classifiers) — can't be prompt-injected.
        # ══════════════════════════════════════════════════════════════════════
        # Layer 3 extraction: SecurityGate owns quarantine + rule engine +
        # sliding-window risk ledger (Bug 7 LRU lives inside it). Returns
        # an updated full_system and a SecurityResult with the abort flag.
        security_result = self._security_gate.evaluate(ctx, full_system, mem, target)
        full_system = security_result.full_system
        security = security_result.assessment
        if security_result.blocked:
            return ChatResult(
                text=(
                    f"\U0001f6e1️ This workflow has been paused because the cumulative "
                    f"risk score ({security.risk_assessment.cumulative_score:.1f}) "
                    f"exceeds the safety threshold. This happens when a conversation "
                    f"involves many high-risk operations. Start a new conversation "
                    f"or adjust the risk threshold in Settings."
                ),
                model="", route_reason="security_abort",
                tokens_in=0, tokens_out=0, cost_usd=0.0,
                message_id=str(uuid.uuid4()),
            )

        # ══════════════════════════════════════════════════════════════════════

        # ── Governance: enforce per-agent policies before invocation ─────────
        # Runs before voting/escalation so a blocked agent doesn't burn 3
        # consensus calls or fire an escalation modal that won't ever be
        # acted on.
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

        # ── Phase 6 split flag (computed early — Phase 8 voting needs it) ────
        split_enabled = bool(
            self._settings.get("reader_actor_split_enabled", False)
        )

        # ── Phase 12: CaMeL — Privileged/Quarantined LLM split ──────────────
        # CaMeL (DeepMind/ETH, arXiv 2503.18813) is mutually exclusive with
        # the Reader/Actor split for the same turn. When both flags are on,
        # CaMeL wins because it is a strictly stricter superset: it also
        # quarantines retrieved data, but additionally enforces capability
        # tags through a sandboxed plan interpreter. Voting is also skipped
        # on a CaMeL turn — the plan IS the model's reasoning step, and
        # running 3 plans concurrently would burn 3x the privileged budget
        # for no measurable consensus signal on a structurally-bounded
        # output.
        camel_enabled = bool(self._settings.get("camel_enabled", False))
        camel_active = camel_enabled and bool(mem.rag_chunks)
        if camel_active:
            split_enabled = False  # CaMeL takes precedence on this turn.

        # ── Phase 8: Symphony-style weighted-vote consensus ──────────────────
        # On high-stakes turns run 3 parallel CoT samples and pick a weighted
        # majority. Composes with the Wiser-Human escalation channel: voting
        # runs FIRST, then check_escalation below; if the modal still fires,
        # the consensus result is preserved in router_log.voting_samples_json
        # for audit. CaMeL — when added — would take precedence over voting
        # for RAG-context turns; the two are mutually exclusive paths.
        voting_enabled = bool(
            self._settings.get("high_stakes_voting_enabled", True)
        )
        risk_score = (
            security.risk_assessment.cumulative_score
            if security.risk_assessment else 0.0
        )
        escalation_will_trigger = (
            self._governance.escalation_channel.would_trigger(
                user_message, full_system,
            )
        )
        is_high_stakes = (
            escalation_will_trigger
            or is_high_stakes_message(user_message)
            or risk_score > 0.7
        )
        # Voting only fires when the resolved target is Claude. Local-only
        # turns skip it (3x latency on a local model is too painful).
        # CaMeL replaces the worker invocation entirely, so voting is
        # mutually exclusive on a CaMeL turn (see comment above).
        should_vote = (
            is_high_stakes
            and voting_enabled
            and target.backend == "claude"
            and not camel_active
        )
        # Pre-allocate the assistant message id so the voting_complete event
        # carries it; the same id is reused when persisting the assistant
        # message below so the frontend can attach a small "consensus" badge.
        asst_msg_id = str(uuid.uuid4())

        response_text = ""
        tokens_in = 0
        tokens_out = 0
        model_name = target.model_name
        had_error = False

        voting_samples: list[dict] | None = None
        if should_vote and not split_enabled:
            _emit_event("chat_event", {
                "type": "high_stakes_voting_started",
                "message_id": asst_msg_id,
            })
            voting_result, voting_samples = self._high_stakes_consensus(
                decision, full_system, messages,
                max_tokens=target.max_tokens,
                on_token=on_token,
            )
            _emit_event("chat_event", {
                "type": "high_stakes_voting_complete",
                "message_id": asst_msg_id,
            })
            response_text = voting_result.text
            tokens_in = voting_result.input_tokens
            tokens_out = voting_result.output_tokens
            model_name = voting_result.model_name or target.model_name
            if voting_result.had_error:
                had_error = True

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
            # Preserve the consensus samples in router_log even when the
            # placeholder is returned — useful for audit when the user later
            # approves and we want to know what the voting layer saw.
            if voting_samples is not None:
                _log_router_event(
                    conversation_id=conversation_id,
                    message_preview=user_message,
                    route_taken=route_model,
                    complexity=complexity,
                    reasoning="voting before escalation_pending",
                    tokens_out=tokens_out,
                    had_error=had_error,
                    response_empty=True,
                    model_used=model_name,
                    agent_role="monolithic",
                    voting_samples_json=json.dumps(voting_samples),
                    turn_id=ctx.turn_id,
                )
            return ChatResult(
                text="Awaiting your review for this action.",
                model="", route_reason="escalation_pending",
                tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=0.0,
                message_id=str(uuid.uuid4()),
            )

        # ── Phase 12: CaMeL plan + execute ───────────────────────────────────
        # The privileged client gets the user message; the quarantined
        # client only sees the retrieved chunks. The interpreter walks the
        # restricted-Python plan and refuses any control flow driven by an
        # UNTRUSTED value. CaMeL is mutually exclusive with the
        # Reader/Actor split — see comment by ``camel_active`` above.
        if camel_active:
            try:
                from services.camel import (
                    camel_plan_and_execute, make_tool_executor_for_turn,
                )
                tool_executor = make_tool_executor_for_turn(
                    agent_id=agent_id or "",
                    conversation_id=conversation_id,
                    governance=self._governance,
                    execution_bridge=getattr(self, "_execution_bridge", None),
                )
                _emit_event("camel_started", {
                    "rag_chunks": len(mem.rag_chunks),
                })
                camel_result = camel_plan_and_execute(
                    user_message=user_message,
                    retrieved_chunks=list(mem.rag_chunks),
                    privileged_client=self.claude,
                    quarantined_client=self.local if (
                        self.local and getattr(self.local, "is_available", lambda: False)()
                    ) else self.claude,
                    tool_executor=tool_executor,
                )
                response_text = camel_result.get("output_text", "") or ""
                if on_token and response_text:
                    try:
                        on_token(response_text)
                    except Exception:
                        pass
                # Persist a camel_log row capturing the plan + audit counters.
                try:
                    _db.execute(
                        "INSERT INTO camel_log "
                        "(id, conversation_id, plan_source, executed_steps, "
                        "capability_violations, blocked_calls, output_text, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            conversation_id,
                            camel_result.get("plan_source", "") or "",
                            int(camel_result.get("executed_steps", 0) or 0),
                            int(camel_result.get("capability_violations", 0) or 0),
                            json.dumps(camel_result.get("blocked_calls", []) or []),
                            response_text,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    _db.commit()
                except Exception as exc:
                    log.debug("camel_log insert failed: %s", exc)
                _emit_event("camel_complete", {
                    "executed_steps": camel_result.get("executed_steps", 0),
                    "capability_violations": camel_result.get(
                        "capability_violations", 0,
                    ),
                    "blocked_calls": len(
                        camel_result.get("blocked_calls", []) or []
                    ),
                    "error": camel_result.get("error", "") or "",
                })
                model_name = target.model_name
                # The privileged + quarantined calls each consumed tokens we
                # can't count without instrumenting every client implementation;
                # leave tokens at 0 so the existing UI and budget logic
                # continue to function on a CaMeL turn rather than miscounting.
            except Exception as exc:
                log.warning("CaMeL pipeline crashed, falling back to monolithic: %s", exc)
                response_text = ""
                # Re-enable downstream paths so the turn still produces an
                # answer instead of a hang. CaMeL is opt-in; a crash should
                # not eat the user's message.
                camel_active = False

        # ── Phase 6: Hackett et al. (ACL 2025) Reader/Actor split ────────────
        # When the flag is on we run a 3-phase pipeline: the Reader analyzes
        # the request and proposes tools, the Actor executes against the
        # Reader's structured plan without ever seeing the raw user message
        # or raw retrieved data, and the synthesizer combines them. The
        # extended-thinking branch is skipped because both phases own their
        # own reasoning prompts.
        if split_enabled:
            try:
                reader_output = self._read_phase(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    agent_id=agent_id,
                    history=messages,
                    mem=mem,
                    turn_id=ctx.turn_id,
                )
                _emit_event("reader_complete", {
                    "intent": reader_output.intent[:200],
                    "proposed_tools": list(reader_output.proposed_tools),
                    "red_flags": list(reader_output.red_flags),
                })
                actor_result, actor_voting_samples = self._act_phase(
                    conversation_id=conversation_id,
                    reader_output=reader_output,
                    history=messages,
                    # Pass the bare persona prompt — NOT full_system (which
                    # carries the memory/RAG suffix). The Actor must not see
                    # raw retrieved data via its system prompt either.
                    full_system=system_prompt,
                    agent_id=agent_id,
                    on_token=on_token,
                    max_tokens=target.max_tokens,
                    vote=should_vote,
                    voting_message_id=asst_msg_id,
                    voting_emit=_emit_event,
                    turn_id=ctx.turn_id,
                )
                if actor_voting_samples is not None:
                    voting_samples = actor_voting_samples
                final = self._synthesize_phase(reader_output, actor_result)
                response_text = final.text
                tokens_in = final.input_tokens
                tokens_out = final.output_tokens
                model_name = final.model_name or target.model_name
                if final.had_error:
                    had_error = True
            finally:
                # Per-turn ledger: clear so the next turn re-establishes from
                # its own Reader output (no stale carry-over between turns).
                self._governance.clear_proposed_tools(conversation_id)

        # ── v4.0 #4: Interleaved Reasoning Visibility ────────────────────────
        # When routing to Claude and extended thinking is available, emit
        # a reasoning step event before generating the final response.
        # Phase 8: skip when voting already produced response_text — the
        # consensus answer is final.
        reasoning_enabled = self._settings.get("interleaved_reasoning_enabled", True)
        if (
            not split_enabled
            and not response_text
            and reasoning_enabled
            and target.backend == "claude"
            and complexity == "complex"
            and not on_token  # only in non-streaming path (thinking is blocking)
            and not image_attachments  # extended-thinking takes a str user
                                       # message; falling through to the
                                       # standard path keeps image blocks
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
        # Phase 6: when the split ran, do NOT fall back to a monolithic
        # invocation even if the Actor produced an empty reply — that would
        # leak the user's raw message + retrieved data past the wall.
        # Phase 11: when the route is local + the user attached images,
        # bypass hub_router and call chat_with_images so the images ride
        # via Ollama's payload field instead of being ignored.
        if not response_text and not split_enabled:
            if image_attachments and target.backend == "local":
                try:
                    text = self.local.chat_with_images(
                        full_system, messages,
                        [img["data"] for img in image_attachments],
                        max_tokens=target.max_tokens,
                    )
                    response_text = text or ""
                    tokens_in = 0
                    tokens_out = 0
                    if on_token and response_text:
                        try:
                            on_token(response_text)
                        except Exception:
                            pass
                except LocalVisionUnavailable as exc:
                    response_text = (
                        f"🖼️ {exc}. Switch to a vision-capable model and resend."
                    )
                    had_error = True
                except Exception as exc:
                    log.warning("local vision invocation failed: %s", exc)
                    response_text = f"[Error: {exc}]"
                    had_error = True
            else:
                worker_result = self._worker_dispatch.dispatch(
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

        # Layer 3 extraction: EscalationLadder owns the empty-response
        # rung + quality-gate rung. Returns an EscalationOutcome whose
        # response_empty already reflects the post-escalation state
        # (Bug 4 fix), so the router_log write below sees the true value.
        esc_outcome = self._escalation_ladder.maybe_escalate(
            ctx=ctx,
            decision=decision,
            target=target,
            full_system=full_system,
            messages=messages,
            response_text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            route_model=route_model,
            model_name=model_name,
            had_error=had_error,
            split_enabled=split_enabled,
        )
        response_text = esc_outcome.response_text
        tokens_in = esc_outcome.tokens_in
        tokens_out = esc_outcome.tokens_out
        route_model = esc_outcome.route_model
        model_name = esc_outcome.model_name
        response_empty = esc_outcome.response_empty

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
        # Phase 6: in split mode the per-phase router_log rows already cover
        # this turn (reader + actor). Skip the legacy turn-summary write so
        # we don't double-count or produce a misleading "monolithic" row.
        # Phase 12: CaMeL writes its own row to camel_log. Tag the
        # router_log entry as ``camel`` so analytics queries can tell
        # which path produced this turn.
        if not split_enabled:
            _log_router_event(
                conversation_id=conversation_id,
                message_preview=user_message,
                route_taken=route_model,
                complexity=complexity,
                reasoning=("camel plan+execute" if camel_active else route_reason),
                tokens_out=tokens_out,
                had_error=turn_failed,
                response_empty=response_empty,
                model_used=model_name,
                mast_category=mast_category,
                agent_role=("camel" if camel_active else "monolithic"),
                voting_samples_json=(
                    json.dumps(voting_samples) if voting_samples is not None
                    else None
                ),
                turn_id=ctx.turn_id,
            )

        # Layer 3 extraction: TurnLifecycle.close owns the three-INSERT-plus-
        # UPDATE atomicity (Bug 6) and the in-transaction SUM that closes
        # Bug 5's race on stale ``spent``. Auto-title fires post-commit so
        # its blocking LLM call doesn't sit inside the db lock.
        # Phase 8: asst_msg_id was pre-allocated above so the
        # high_stakes_voting_complete event could carry it for the
        # frontend badge; reuse the same id when persisting.
        cost = _estimate_cost(model_name, tokens_in, tokens_out, self._settings)
        close_result = self._turn_lifecycle.close(
            ctx,
            asst_msg_id=asst_msg_id,
            response_text=response_text,
            route_reason=route_reason,
            model_name=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
        )
        budget_warning = close_result.budget_warning
        self._turn_lifecycle.maybe_auto_title(ctx, response_text)

        # Update memory
        self.memory.add_to_buffer(conversation_id, "user", user_message)
        self.memory.add_to_buffer(conversation_id, "assistant", response_text)
        self.memory.extract_facts(conversation_id, user_message, response_text)

        # PR 8: drop ephemeral attachments — they only exist for one turn.
        # Persistent (persist=1) rows stay because they're already in RAG.
        try:
            self._purge_ephemeral_attachments(conversation_id)
        except Exception as exc:
            log.debug("ephemeral attachment purge failed: %s", exc)

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

        executor = PipelineExecutor(
            self.hub_router, self._settings,
            claude_client=self.claude, local_client=self.local,
        )
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

        # PR 8: drop ephemeral attachments — same lifecycle as single-agent.
        try:
            self._purge_ephemeral_attachments(conversation_id)
        except Exception as exc:
            log.debug("ephemeral attachment purge (pipeline) failed: %s", exc)

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
