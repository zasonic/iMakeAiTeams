"""
channels/channel_manager.py

Channel manager — the glue between adapters and the core pipeline.

Responsibilities
----------------
1. Start/stop all configured channel adapters (Telegram, future: Discord).
2. Receive InboundMessages from adapters via on_message().
3. Route each message through:
   a. Guardrails check (input safety)
   b. Normal chat orchestrator  (non-agent messages)
   c. Agent loop                (agent-mode messages)
4. Send responses back through the originating adapter.
5. Handle permission confirmations: when agent_loop needs user approval,
   channel_manager routes the confirm request to the correct adapter
   and waits for the reply.
6. Emit status events to the GUI via EventBus.

The channel_manager is instantiated in main.py and started on boot.
It is accessible via api.py for status queries and settings changes.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from core.settings import Settings
    from core.events import EventBus
    from channels.inbound_message import InboundMessage, Channel
    from channels.access_control import AccessControl

log = logging.getLogger("MyAIAgentHub.channel_manager")


class ChannelManager:
    """
    Manages all channel adapters and routes messages to the pipeline.

    Parameters
    ----------
    settings          : App settings (for bot tokens, allowlists, etc.)
    bus               : EventBus for GUI notifications.
    chat_orchestrator : The existing ChatOrchestrator instance.
    claude_client     : For agent loop (calls with tools).
    local_client      : For routing classification.
    memory            : Memory system.
    safety_gate       : Existing safety_gate instance.
    guardrails_gate   : New guardrails_gate instance.
    project_root      : Default project directory for agent runs.
    """

    def __init__(
        self,
        settings: "Settings",
        bus: "EventBus",
        chat_orchestrator,
        claude_client,
        local_client,
        memory,
        safety_gate=None,
        guardrails_gate=None,
        project_root: Path | None = None,
    ) -> None:
        self._settings     = settings
        self._bus          = bus
        self._orchestrator = chat_orchestrator
        self._claude       = claude_client
        self._local        = local_client
        self._memory       = memory
        self._safety       = safety_gate
        self._guardrails   = guardrails_gate
        self._root         = project_root or Path.cwd()

        self._adapters: dict[str, object] = {}
        self._active_loops: dict[str, object] = {}  # conv_id -> AgentLoop
        self._lock = threading.Lock()

        # Access control shared across all adapters
        from channels.access_control import AccessControl
        self._ac = AccessControl(settings)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all configured channel adapters."""
        log.info("ChannelManager starting…")

        telegram_token = self._settings.get("telegram_bot_token", "")
        if telegram_token:
            self._start_telegram(telegram_token)
        else:
            log.info("Telegram: no token configured — adapter not started")

        log.info("ChannelManager started. Active adapters: %s", list(self._adapters.keys()))
        self._emit_status()

    def stop(self) -> None:
        """Stop all adapters cleanly."""
        for name, adapter in list(self._adapters.items()):
            try:
                adapter.stop()
                log.info("Stopped adapter: %s", name)
            except Exception as exc:
                log.warning("Error stopping adapter %s: %s", name, exc)
        self._adapters.clear()

    def status(self) -> dict:
        """Return status dict for GUI display."""
        result = {}
        for name, adapter in self._adapters.items():
            try:
                result[name] = adapter.status()
            except Exception:
                result[name] = {"running": False, "error": True}

        result["configured"] = {
            "telegram": bool(self._settings.get("telegram_bot_token", "")),
        }
        result["active_agent_tasks"] = len(self._active_loops)
        return result

    def stop_agent_task(self, conversation_id: str) -> bool:
        """Stop a running agent loop for a conversation."""
        with self._lock:
            loop = self._active_loops.get(conversation_id)
        if loop:
            loop.stop()
            log.info("Stopped agent task for conv_id=%s", conversation_id)
            return True
        return False

    # ── Message routing ────────────────────────────────────────────────────

    def on_message(self, msg: "InboundMessage") -> None:
        """
        Entry point for all inbound messages from any adapter.
        Runs in the adapter's thread — spawns worker threads for processing.
        """
        log.info("ChannelManager: received %r", msg)

        if msg.is_agent_trigger:
            threading.Thread(
                target=self._handle_agent_message,
                args=(msg,),
                daemon=True,
                name=f"agent-{msg.conversation_id}",
            ).start()
        else:
            threading.Thread(
                target=self._handle_chat_message,
                args=(msg,),
                daemon=True,
                name=f"chat-{msg.conversation_id}",
            ).start()

    # ── Chat message handler ───────────────────────────────────────────────

    def _handle_chat_message(self, msg: "InboundMessage") -> None:
        """Route a normal chat message through the existing orchestrator."""
        try:
            # Guardrails check
            if self._guardrails:
                verdict = self._guardrails.check_input(msg.text)
                if verdict.blocked:
                    self._send_reply(msg, f"⛔ Message blocked: {verdict.reason}")
                    return

            # Use existing orchestrator — it handles routing, memory, etc.
            # We create a minimal send_fn that replies via the correct adapter.
            tokens_collected = []

            def on_token(token: str):
                tokens_collected.append(token)

            result = self._orchestrator.send(
                conversation_id=msg.conversation_id,
                user_message=msg.text,
                on_token=on_token,
            )

            # Get final response text
            if hasattr(result, 'text'):
                response = result.text
            elif isinstance(result, dict):
                response = result.get("text", result.get("response", str(result)))
            else:
                response = str(result) if result else "".join(tokens_collected)

            # Guardrails check on output
            if self._guardrails and response:
                out_verdict = self._guardrails.check_output(response)
                if out_verdict.blocked:
                    response = "⚠️ My response was blocked by the safety filter. Please rephrase your question."

            if response:
                self._send_reply(msg, response)

        except Exception as exc:
            log.exception("Error handling chat message from %s", msg.username)
            self._send_reply(msg, f"⚠️ An error occurred: {exc}")

    # ── Agent message handler ──────────────────────────────────────────────

    def _handle_agent_message(self, msg: "InboundMessage") -> None:
        """Route an agent-mode message through the agentic coding loop."""
        task = msg.clean_text
        conv_id = msg.conversation_id

        try:
            # Guardrails check on task
            if self._guardrails:
                verdict = self._guardrails.check_input(task)
                if verdict.blocked:
                    self._send_reply(msg, f"⛔ Task blocked: {verdict.reason}")
                    return

            self._send_reply(msg, f"🤖 Starting agent task: _{task[:100]}_")

            # Build tools for this session
            from pathlib import Path as _Path
            from services.tools.bash_tool import BashTool
            from services.tools.file_tools import FileTools
            from services.tools.git_tool import GitTool
            from services.permission_engine import PermissionEngine, ToolCall
            from services.agent_loop import AgentLoop, AgentEvent

            project_root = _Path(self._settings.get("agent_project_root", str(self._root)))

            def on_confirm_needed(tool_call: ToolCall):
                """Send permission request to the user via the originating adapter."""
                adapter = self._adapters.get(msg.channel.value)
                if adapter and hasattr(adapter, 'ask_confirm'):
                    chat_id = msg.raw_meta.get("chat_id", msg.user_id)
                    adapter.ask_confirm(
                        chat_id=int(chat_id),
                        request_id=tool_call.request_id,
                        description=tool_call.description,
                        callback=lambda approved, rid=tool_call.request_id: (
                        perms.approve(rid) if approved else perms.deny(rid)
                    ),
                    )
                else:
                    # GUI mode or no adapter — auto-approve for now, GUI handles it
                    perms.approve(tool_call.request_id)

            perms = PermissionEngine(on_confirm_needed=on_confirm_needed)

            def on_event(event: AgentEvent):
                """Forward agent progress to the user."""
                if event.type == "thinking" and event.content:
                    # Don't flood with every thinking step — only key ones
                    if len(event.content) > 50:
                        self._send_reply(msg, f"💭 {event.content[:200]}")
                elif event.type == "tool_call":
                    self._send_reply(msg, f"🔧 `{event.content}`")
                elif event.type == "tool_result":
                    # Only send non-trivial results
                    if event.content and len(event.content) > 10:
                        preview = event.content[:300]
                        self._send_reply(msg, f"📋 {preview}")
                elif event.type == "confirm_needed":
                    pass  # handled by on_confirm_needed
                elif event.type == "error":
                    self._send_reply(msg, f"❌ {event.content}")

            loop = AgentLoop(
                claude_client=self._claude,
                file_tools=FileTools(project_root),
                bash_tool=BashTool(project_root),
                git_tool=GitTool(project_root),
                permission_engine=perms,
                on_event=on_event,
                safety_gate=self._guardrails,
            )

            # Track active loop for /stop support
            with self._lock:
                self._active_loops[conv_id] = loop

            try:
                final = loop.run(task, conversation_id=conv_id)
                self._send_reply(msg, f"✅ *Done*\n\n{final}")
            finally:
                with self._lock:
                    self._active_loops.pop(conv_id, None)

        except Exception as exc:
            log.exception("Error in agent loop for %s", msg.username)
            self._send_reply(msg, f"❌ Agent loop error: {exc}")

    # ── Reply routing ──────────────────────────────────────────────────────

    def _send_reply(self, msg: "InboundMessage", text: str) -> None:
        """Send a response back through the correct adapter."""
        if not text:
            return

        channel_name = msg.channel.value
        adapter = self._adapters.get(channel_name)

        if adapter and hasattr(adapter, 'send_message'):
            chat_id = msg.raw_meta.get("chat_id", msg.user_id)
            try:
                adapter.send_message(chat_id, text)
            except Exception as exc:
                log.error("Failed to send reply via %s: %s", channel_name, exc)
        else:
            # GUI messages go through the EventBus
            self._bus.emit("channel_response", {"text": text, "conversation_id": msg.conversation_id})

    # ── Adapter setup ──────────────────────────────────────────────────────

    def _start_telegram(self, token: str) -> None:
        """Initialise and start the Telegram adapter."""
        try:
            from channels.telegram_adapter import TelegramAdapter
            adapter = TelegramAdapter(
                token=token,
                access_control=self._ac,
                on_message=self.on_message,
                project_root=self._root,
            )
            adapter.start()
            self._adapters["telegram"] = adapter
            log.info("Telegram adapter started")
        except Exception as exc:
            log.error("Failed to start Telegram adapter: %s", exc)

    def _emit_status(self) -> None:
        """Emit channel status to GUI."""
        try:
            self._bus.emit("channel_status", self.status())
        except Exception:
            pass
