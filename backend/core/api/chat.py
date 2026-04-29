"""
core/api/chat.py — Chat and extended-thinking bridge methods.
"""

from __future__ import annotations

import threading

from core.service_guard import requires as _requires
from core.worker import run_in_thread

from services import input_sanitizer
from services.rate_limiter import rate_limit_chat

from models import StreamEvent

from ._base import BaseAPI


class ChatAPI(BaseAPI):

    def _stop_signal_for(self, conversation_id: str) -> threading.Event:
        """Get-or-create the stop event for one conversation."""
        with self._stop_signals_lock:
            ev = self._stop_signals.get(conversation_id)
            if ev is None:
                ev = threading.Event()
                self._stop_signals[conversation_id] = ev
            return ev

    @rate_limit_chat
    @_requires("chat_orchestrator")
    def chat_send(self, conversation_id: str, user_message: str,
                  agent_id: str = "") -> None:
        """
        Send a message in a conversation. Streams tokens back via chat_token events,
        then emits chat_done with the complete result.

        Stage 5: Also emits structured 'chat_event' events for message_start,
        route_decided, and memory_recalled (Improvement 3).
        """
        stop_event = self._stop_signal_for(conversation_id)
        stop_event.clear()

        def _on_token(token: str):
            if stop_event.is_set():
                raise InterruptedError("chat stopped")
            self._emit("chat_token", {"token": token, "conversation_id": conversation_id})

        def _work():
            try:
                try:
                    self._emit("chat_event", StreamEvent(
                        "message_start", conversation_id,
                        {"agent_id": agent_id or ""},
                    ).to_dict())
                except Exception:
                    pass

                def _on_event(event_type, data):
                    self._emit("chat_event", StreamEvent(
                        event_type, conversation_id, data,
                    ).to_dict())

                try:
                    def _on_scan_result(r):
                        self._emit("chat_event", StreamEvent(
                            "security_scan", conversation_id,
                            {"icon": r["icon"], "label": r["label"],
                             "detail": r["detail"], "verdict": r["verdict"],
                             "score": r["score"], "scan_id": r["scan_id"]},
                        ).to_dict())
                    _scan = input_sanitizer.scan_message(
                        user_message, session_id=conversation_id,
                        on_result=_on_scan_result,
                    )
                    if _scan.get("blocked"):
                        self._emit("chat_error", {
                            "error": "Your message was blocked by the security scanner. Please rephrase.",
                            "conversation_id": conversation_id,
                        })
                        return
                except Exception as _fe:
                    self._log.debug(f"Firewall scan skipped: {_fe}")

                result = self._chat.send(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    agent_id=agent_id or None,
                    on_token=_on_token,
                    on_event=_on_event,
                )
                self._emit("chat_done", {**result.to_dict(), "conversation_id": conversation_id})
            except InterruptedError:
                self._emit("chat_stopped", {"conversation_id": conversation_id})
            except Exception as e:
                self._log.error(f"chat_send error: {e}", exc_info=True)
                err_msg = str(e).lower()
                if "authentication" in err_msg or "api key" in err_msg or "401" in err_msg:
                    friendly = "Invalid API key — update it in Settings."
                elif "rate" in err_msg or "429" in err_msg:
                    friendly = "Claude is busy right now — wait a moment and try again."
                elif "context" in err_msg or "too long" in err_msg:
                    friendly = "This conversation is too long for Claude to process. Start a new one."
                elif "connection" in err_msg or "timeout" in err_msg or "network" in err_msg:
                    friendly = "Connection lost — check your internet and try again."
                elif "local model unavailable" in err_msg or "no response" in err_msg:
                    friendly = "Local model didn't respond — is it still running? Check Settings."
                else:
                    friendly = f"Something went wrong: {type(e).__name__}. Check the error log in Settings for details."
                self._emit("chat_error", {"error": friendly, "conversation_id": conversation_id})

        run_in_thread(_work)

    def chat_stop(self, conversation_id: str = "") -> None:
        """Stop a streaming response.

        With a conversation_id: stop only that conversation. Without one:
        stop every active stream (back-compat for older renderers that
        didn't track which conversation was active).
        """
        with self._stop_signals_lock:
            if conversation_id:
                ev = self._stop_signals.get(conversation_id)
                if ev is not None:
                    ev.set()
            else:
                for ev in self._stop_signals.values():
                    ev.set()

    @_requires("chat_orchestrator", default={"error": "chat unavailable"})
    def chat_new_conversation(self, agent_id: str = "",
                              title: str = "New conversation") -> dict:
        """Create a new conversation and return its id."""
        cid = self._chat.create_conversation(
            agent_id=agent_id or None, title=title
        )
        return {"id": cid}

    @_requires("chat_orchestrator", default=[])
    def chat_list_conversations(self, limit: int = 30) -> list:
        return self._chat.list_conversations(limit=limit)

    @_requires("chat_orchestrator", default=[])
    def chat_get_messages(self, conversation_id: str, limit: int = 100) -> list:
        return self._chat.get_conversation_messages(conversation_id, limit=limit)

    @_requires("chat_orchestrator", default={"error": "chat unavailable"})
    def chat_rename_conversation(self, conversation_id: str, title: str) -> dict:
        self._chat.update_conversation_title(conversation_id, title)
        return {"ok": True}

    @_requires("chat_orchestrator", default={"error": "chat unavailable"})
    def chat_delete_conversation(self, conversation_id: str) -> dict:
        self._chat.delete_conversation(conversation_id)
        # Drop the per-conversation stop signal too. Without this, the
        # _stop_signals dict grew forever — every send_chat ever made
        # left an Event behind, and delete_conversation only swept DB
        # tables.
        with self._stop_signals_lock:
            self._stop_signals.pop(conversation_id, None)
        return {"ok": True}

    @_requires("chat_orchestrator", default={"error": "chat unavailable"})
    def chat_branch_conversation(self, conversation_id: str,
                                  from_message_id: str) -> dict:
        return self._chat.branch_conversation(conversation_id, from_message_id)

    @_requires("chat_orchestrator", default={"error": "chat unavailable"})
    def chat_export_conversation(self, conversation_id: str,
                                  fmt: str = "markdown") -> dict:
        return self._chat.export_conversation(conversation_id, fmt)

    @_requires("chat_orchestrator", default={})
    def chat_token_stats(self) -> dict:
        return self._chat.get_token_stats()

    @_requires("chat_orchestrator", default={})
    def get_router_stats(self) -> dict:
        return self._chat.get_router_stats()

    def ask_with_thinking(self, user_message: str,
                          budget_tokens: int = 10000) -> None:
        def _work():
            try:
                model = self._settings.get("claude_model")
                if "haiku" in model.lower():
                    self._emit("thinking_error", {
                        "error": "Extended thinking isn't available on Haiku. "
                                 "Switch to Sonnet or Opus in Settings first."
                    })
                    return

                system = self._settings.get(
                    "system_prompt", "You are a helpful AI assistant."
                )
                result = self._claude.extended_thinking_chat(
                    system, user_message, budget_tokens=budget_tokens
                )
                self._emit("thinking_done", result)
            except Exception as e:
                err_msg = str(e).lower()
                if "model" in err_msg or "not support" in err_msg:
                    friendly = "Extended thinking isn't available for the selected model. Switch to Sonnet or Opus in Settings."
                elif "authentication" in err_msg:
                    friendly = "Invalid API key — update it in Settings."
                else:
                    friendly = "Extended thinking failed — try again or switch models in Settings."
                self._emit("thinking_error", {"error": friendly})
        run_in_thread(_work)
