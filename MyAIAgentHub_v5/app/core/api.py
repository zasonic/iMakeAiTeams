"""
core/api.py — PyWebView JS API bridge.

Every public method is exposed to the frontend via window.pywebview.api.
Methods that do I/O run in a worker thread and push results back with
self._emit(event_name, payload_dict).

Architecture:
  - ChatOrchestrator  — unified conversation loop (routing, memory, tokens)
  - TaskRouter        — classifies messages, picks Claude vs local
  - MemoryManager     — three-tier memory (buffer, facts, RAG/semantic)
  - AgentRegistry     — CRUD for agents and teams
  - ClaudeClient      — Anthropic SDK wrapper
  - LocalClient       — Ollama / LM Studio client
  - RAGIndex          — sentence-transformer semantic search over files
"""

import base64
import json
import logging
import os
import subprocess
import threading
import uuid as _uuid
import webbrowser
import zipfile
import csv
import io
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from core import paths
from core.service_guard import requires as _requires
from core.settings import Settings
from core.events import EventBus
from core.worker import run_in_thread

from services.claude_client import ClaudeClient
from services.local_client import LocalClient
from services.rag_index import RAGIndex
from services import semantic_search, error_classifier, health_monitor
from services import prompt_library
from services.router import TaskRouter
from services.memory import MemoryManager
from services.chat_orchestrator import ChatOrchestrator
from services.agent_registry import (
    seed_agents, update_builtin_tom,
    generate_agent_tom, refresh_team_tom,
    add_team_member as _registry_add_member,
    remove_team_member as _registry_remove_member,
)
from services import input_sanitizer
from services.rate_limiter import rate_limit_chat, rate_limit

from models import StreamEvent

import db as _db_module


class API:
    def __init__(self, settings: Settings, bus: EventBus, app_root: Path,
                 log: logging.Logger):
        self._settings = settings
        self._bus = bus
        self._app_root = app_root
        self._log = log
        self._window = None
        self._stop_chat = threading.Event()

        # Each service records its init status here. Writers only mutate this
        # dict during __init__ on the main thread; downstream readers (channel
        # manager on a background thread) read it snapshot-only after init
        # completes, so no lock is required.
        self._status: dict[str, dict] = {}

        # ── Claude client (required — no useful app without it) ───────────────
        self._claude = self._safe_init(
            "claude_client",
            lambda: ClaudeClient(
                api_key=self._settings.get("claude_api_key", ""),
                model=self._settings.get("claude_model", "claude-sonnet-4-6"),
                use_caching=self._settings.get("claude_prompt_caching", True),
            ),
            required=True,
        )

        # ── Local model client ────────────────────────────────────────────────
        self._local = self._safe_init(
            "local_client",
            lambda: LocalClient(self._settings),
        )

        # ── RAG index (fast — constructs empty, embedder attached later) ──────
        # The SentenceTransformer model is NOT loaded here: that can block up
        # to 60s on first run while HuggingFace downloads 90MB. Deferred to
        # _start_deferred_init() which runs after the window paints.
        self._rag = self._safe_init(
            "rag_index",
            lambda: RAGIndex(model=None),
        )

        # ── Database (required — chat/memory can't degrade without it) ────────
        self._safe_init(
            "database",
            lambda: _db_module.init_db(paths.db_path()),
            required=True,
        )

        # ── Prompts ───────────────────────────────────────────────────────────
        self._safe_init("prompts_seed", prompt_library.seed_prompts)

        # ── Agents ────────────────────────────────────────────────────────────
        self._safe_init("agents_seed", seed_agents)

        # ── Priority 1: refresh built-in ToM on every startup ─────────────────
        self._safe_init("theory_of_mind", update_builtin_tom)

        # ── Priority 5: set firewall default based on API key ─────────────────
        def _firewall_init():
            has_key = bool(self._settings.get("claude_api_key", "").strip())
            input_sanitizer.set_firewall_enabled(has_key)
            return has_key
        self._safe_init("firewall", _firewall_init)

        # ── Memory manager (fast — stores refs; rag/semantic lazy-check) ──────
        self._memory = self._safe_init(
            "memory_manager",
            lambda: MemoryManager(
                rag_index=self._rag,
                semantic_search_mod=semantic_search,
                local_client=self._local,
            ),
        )

        # ── Task router ───────────────────────────────────────────────────────
        self._router = self._safe_init(
            "router",
            lambda: TaskRouter(self._local, self._settings),
        )

        # ── Chat orchestrator ─────────────────────────────────────────────────
        self._chat = self._safe_init(
            "chat_orchestrator",
            lambda: ChatOrchestrator(
                claude_client=self._claude,
                local_client=self._local,
                router=self._router,
                memory=self._memory,
                settings=self._settings,
            ),
        )

        # ── Deferred services — mark pending so UI renders a spinner row ──────
        # These come up after the window paints, driven by start_deferred_init.
        for _name in ("embedder", "rag_load", "semantic_search",
                      "semantic_search_indexer"):
            self._status[_name] = {"ok": False, "error": None, "pending": True}

    # ── Deferred initialization ───────────────────────────────────────────────

    def start_deferred_init(self) -> None:
        """Start the slow services in a background thread.

        Call this from main.py after the window's `loaded` event fires. Heavy
        work — sentence-transformers model load (~2-5s, or 60s+ on first-run
        download), ChromaDB client construction (500ms-2s), background indexer
        thread start — all run here so the user sees a painted window within
        a second of launch instead of a blank PyWebView frame.
        """
        threading.Thread(
            target=self._run_deferred_init,
            daemon=True,
            name="api-deferred-init",
        ).start()

    def _run_deferred_init(self) -> None:
        # 1. Embedding model (the hang risk)
        model = self._safe_init("embedder", self._load_shared_embedder)
        self._emit_service_update("embedder")

        # Attach the model to the already-constructed RAG index. RAGIndex
        # tolerates a None model for construction and uses this attribute
        # lazily in search/index operations.
        if self._rag is not None and model is not None:
            self._rag._model = model

        # 2. RAG cache load — only meaningful once embedder is available
        _rag_path = paths.rag_cache_dir() / "index.npz"
        if self._rag is not None and model is not None and _rag_path.exists():
            self._safe_init(
                "rag_load",
                lambda: (self._rag.load(_rag_path), self._rag.chunk_count())[1],
            )
        else:
            self._status["rag_load"] = {
                "ok": model is not None,
                "error": None if model is not None else "embedder unavailable",
            }
        self._emit_service_update("rag_load")

        # 3. ChromaDB vector store
        self._safe_init(
            "semantic_search",
            lambda: semantic_search.init_vector_store(
                paths.vector_store_dir(), shared_model=model,
            ),
        )
        self._emit_service_update("semantic_search")

        # 4. Background indexer thread
        self._safe_init(
            "semantic_search_indexer",
            lambda: semantic_search.start_background_indexer(interval_seconds=60),
        )
        self._emit_service_update("semantic_search_indexer")

        self._log.info("Deferred init complete.")

    def _emit_service_update(self, name: str) -> None:
        """Push a live status update to the frontend so the UI can refresh."""
        entry = self._status.get(name, {})
        self._emit("service_status_update", {"service": name, **entry})

    # ── Fail-soft service init ────────────────────────────────────────────────

    def _safe_init(self, name, factory, *, required=False, fallback=None):
        """Run ``factory()`` and record the outcome in self._status[name].

        If ``required`` is True, re-raise on failure — the caller treats this
        service as a ship-blocker and the app should fail loudly. Otherwise
        log a warning, mark the service unavailable, and return ``fallback``.
        """
        try:
            result = factory()
            self._status[name] = {"ok": True, "error": None}
            return result
        except Exception as exc:
            self._status[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._log.warning("Service %s failed to initialise: %s", name, exc,
                              exc_info=True)
            if required:
                raise
            return fallback

    def _load_shared_embedder(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        self._log.info("Shared SentenceTransformer model loaded.")
        return model

    def service_status(self) -> dict:
        """Return a snapshot of per-service init status for the UI."""
        return {name: dict(entry) for name, entry in self._status.items()}

    # ── Window reference ─────────────────────────────────────────────────────

    def set_window(self, window) -> None:
        self._window = window

    # ── Event emission ───────────────────────────────────────────────────────

    def _emit(self, event: str, payload: Any = None) -> None:
        """Send an event to the frontend via base64-encoded JSON (no escaping bugs)."""
        if self._window is None:
            return
        try:
            raw = json.dumps(payload if payload is not None else {})
            b64 = base64.b64encode(raw.encode()).decode()
            self._window.evaluate_js(f"window.__emit('{event}', atob('{b64}'))")
        except Exception as e:
            self._log.debug(f"_emit failed for '{event}': {e}")

    # ── OS notification ──────────────────────────────────────────────────────

    def _os_notify(self, title: str, message: str) -> None:
        try:
            if os.name == "nt":
                try:
                    import ctypes
                    try:
                        import winrt.windows.ui.notifications as win_notif
                        import winrt.windows.data.xml.dom as xml_dom
                        mgr = win_notif.ToastNotificationManager
                        tmpl = mgr.get_template_content(
                            win_notif.ToastTemplateType.TOAST_TEXT02)
                        nodes = tmpl.get_elements_by_tag_name("text")
                        nodes[0].append_child(tmpl.create_text_node(title))
                        nodes[1].append_child(tmpl.create_text_node(message))
                        notifier = mgr.create_toast_notifier("iMakeAiTeams")
                        notifier.show(win_notif.ToastNotification(tmpl))
                    except Exception:
                        ctypes.windll.user32.MessageBeep(0)
                except Exception:
                    pass
            elif hasattr(os, "uname") and os.uname().sysname == "Darwin":
                # Escape backslashes first, then double-quotes, to prevent
                # breaking out of the AppleScript string literal.
                def _esc(s: str) -> str:
                    return s.replace("\\", "\\\\").replace('"', '\\"')
                subprocess.Popen(
                    ["osascript", "-e",
                     f'display notification "{_esc(message)}" with title "{_esc(title)}"'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    ["notify-send", title, message],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    # ── Settings ─────────────────────────────────────────────────────────────

    def get_settings(self) -> dict:
        raw_key = self._settings.get("claude_api_key", "")
        # Return a masked key for display — never expose the full secret to the frontend
        if raw_key and len(raw_key) > 8:
            masked_key = raw_key[:7] + "•" * (len(raw_key) - 11) + raw_key[-4:]
        elif raw_key:
            masked_key = "•" * len(raw_key)
        else:
            masked_key = ""
        return {
            "lm_studio_url":         self._settings.get("lm_studio_url",         "http://localhost:1234"),
            "ollama_url":            self._settings.get("ollama_url",            "http://localhost:11434"),
            "claude_api_key":        masked_key,
            "claude_api_key_set":    bool(raw_key),
            "claude_model":          self._settings.get("claude_model",          "claude-sonnet-4-6"),
            "claude_prompt_caching": self._settings.get("claude_prompt_caching", True),
            "default_local_backend": self._settings.get("default_local_backend", "ollama"),
            "default_local_model":   self._settings.get("default_local_model",   ""),
            "system_prompt":         self._settings.get("system_prompt",         "You are a helpful AI assistant."),
            "start_tab":             self._settings.get("start_tab",             "chat"),
            "routing_enabled":               self._settings.get("routing_enabled",               True),
            "smart_routing_enabled":         self._settings.get("routing_enabled",               True),
            "interleaved_reasoning_enabled": self._settings.get("interleaved_reasoning_enabled", True),
            "firewall_enabled":              self._settings.get("firewall_enabled",              True),
            "is_first_run":                  not self._settings.get("first_run_complete",        False),
            "first_run_complete":            self._settings.get("first_run_complete",            False),
            "max_conversation_budget_usd":   self._settings.get("max_conversation_budget_usd",  5.0),
            "budget_warning_threshold_pct":  self._settings.get("budget_warning_threshold_pct", 80.0),
        }

    def save_setting(self, key: str, value: Any) -> None:
        # Normalise routing key so both names write to the same slot
        if key == "smart_routing_enabled":
            key = "routing_enabled"
        self._settings.set(key, value)
        # Keep live services in sync — router may be None if init failed.
        if key == "routing_enabled" and self._router is not None:
            self._router.set_enabled(bool(value))
        if key == "firewall_enabled":
            try:
                from services import input_sanitizer as _san
                _san.set_firewall_enabled(bool(value))
            except Exception:
                pass

    def set_setting(self, key: str, value: Any) -> dict:
        self._settings.set(key, value)
        _claude_keys = {"claude_api_key", "claude_model", "claude_prompt_caching"}
        if key in _claude_keys:
            self._claude.update_config(
                api_key=self._settings.get("claude_api_key", "") if key == "claude_api_key" else None,
                model=self._settings.get("claude_model", "claude-sonnet-4-6") if key == "claude_model" else None,
                use_caching=self._settings.get("claude_prompt_caching", True) if key == "claude_prompt_caching" else None,
            )
        if key in ("routing_enabled", "smart_routing_enabled"):
            self._settings.set("routing_enabled", bool(value))
            if self._router is not None:
                self._router.set_enabled(bool(value))
        if key == "firewall_enabled":
            try:
                from services import input_sanitizer as _san
                _san.set_firewall_enabled(bool(value))
            except Exception:
                pass
        return {"ok": True}

    def get_setting(self, key: str) -> dict:
        return {"value": self._settings.get(key, None)}

    def complete_first_run(self, start_tab: str) -> None:
        self._settings.set("first_run_complete", True)
        self._settings.set("start_tab", start_tab)

    def verify_api_key(self, key: str) -> dict:
        """Synchronously verify an Anthropic API key. Used by the setup wizard."""
        key = (key or "").strip()
        if not key:
            return {"ok": False, "message": "Please enter your API key."}
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=key)
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            # Key is valid — persist it now
            self._settings.set("claude_api_key", key)
            self._claude.update_config(api_key=key)
            return {"ok": True, "message": "Connected to Claude ✓"}
        except Exception as exc:
            name = type(exc).__name__
            msg = str(exc).lower()
            if "authentication" in name.lower() or "auth" in msg or "invalid" in msg:
                return {"ok": False, "message": "Invalid API key — double-check it at console.anthropic.com"}
            if any(w in name.lower() for w in ("connection", "timeout", "network")):
                return {"ok": False, "message": "Can't reach Anthropic — check your internet connection"}
            return {"ok": False, "message": f"Unexpected error: {exc}"}

    def detect_local_setup(self) -> dict:
        """
        Probe for local model backends and suggest a model based on RAM.
        Called synchronously by the setup wizard.
        """
        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except Exception:
            ram_gb = 0.0

        if ram_gb >= 32:
            recommended, rec_reason = "llama3:8b", f"{ram_gb} GB RAM — 8B model runs comfortably"
        elif ram_gb >= 16:
            recommended, rec_reason = "llama3:8b", f"{ram_gb} GB RAM — 8B model should work well"
        elif ram_gb >= 8:
            recommended, rec_reason = "phi3:mini", f"{ram_gb} GB RAM — smaller model recommended"
        else:
            recommended, rec_reason = "phi3:mini", f"{ram_gb} GB RAM — lightweight model recommended"

        if self._local is not None:
            ollama_running = self._local.is_available(backend="ollama")
            lmstudio_running = self._local.is_available(backend="lmstudio")
            ollama_models = self._local.list_models(backend="ollama") if ollama_running else []
            lmstudio_models = self._local.list_models(backend="lmstudio") if lmstudio_running else []
        else:
            ollama_running = lmstudio_running = False
            ollama_models = lmstudio_models = []

        return {
            "ram_gb": ram_gb,
            "recommended_model": recommended,
            "recommendation_reason": rec_reason,
            "ollama_running": ollama_running,
            "ollama_models": ollama_models,
            "lmstudio_running": lmstudio_running,
            "lmstudio_models": lmstudio_models,
        }

    # ── Chat ─────────────────────────────────────────────────────────────────

    @rate_limit_chat
    def chat_send(self, conversation_id: str, user_message: str,
                  agent_id: str = "") -> None:
        """
        Send a message in a conversation. Streams tokens back via chat_token events,
        then emits chat_done with the complete result.

        Stage 5: Also emits structured 'chat_event' events for message_start,
        route_decided, and memory_recalled (Improvement 3).
        """
        self._stop_chat.clear()

        def _on_token(token: str):
            if self._stop_chat.is_set():
                raise InterruptedError("chat stopped")
            self._emit("chat_token", {"token": token, "conversation_id": conversation_id})

        def _work():
            completed = False
            try:
                # ── Structured events (non-fatal: failures here must not block chat) ──
                try:
                    self._emit("chat_event", StreamEvent(
                        "message_start", conversation_id,
                        {"agent_id": agent_id or ""},
                    ).to_dict())
                except Exception:
                    pass

                # ── Send the message (ChatOrchestrator handles routing + memory) ──
                def _on_event(event_type, data):
                    self._emit("chat_event", StreamEvent(
                        event_type, conversation_id, data,
                    ).to_dict())

                # ── Priority 5: firewall scan ────────────────────────────────
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
                # Improvement 1: Convert ChatResult to dict at the JS boundary
                self._emit("chat_done", {**result.to_dict(), "conversation_id": conversation_id})
                completed = True
            except InterruptedError:
                self._emit("chat_stopped", {"conversation_id": conversation_id})
                completed = True
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
                completed = True
            finally:
                # Safety net: if no terminal event was emitted (e.g. thread died
                # unexpectedly), emit chat_error so the frontend can unlock the
                # send button and not stay stuck in streaming mode forever.
                if not completed:
                    self._emit("chat_error", {
                        "error": "Unexpected error — please try again.",
                        "conversation_id": conversation_id,
                    })

        run_in_thread(_work)

    def chat_stop(self) -> None:
        """Stop the current streaming response."""
        self._stop_chat.set()

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
        return {"ok": True}

    @_requires("chat_orchestrator", default={"error": "chat unavailable"})
    def chat_branch_conversation(self, conversation_id: str,
                                  from_message_id: str) -> dict:
        """
        Branch a conversation from a specific message.
        Returns {id, title} for the new conversation on success, or {error}.
        """
        return self._chat.branch_conversation(conversation_id, from_message_id)

    @_requires("chat_orchestrator", default={"error": "chat unavailable"})
    def chat_export_conversation(self, conversation_id: str,
                                  fmt: str = "markdown") -> dict:
        """
        Export a conversation as 'markdown' or 'json'.
        Returns {content: str, filename: str} or {error: str}.
        """
        return self._chat.export_conversation(conversation_id, fmt)

    @_requires("chat_orchestrator", default={})
    def chat_token_stats(self) -> dict:
        return self._chat.get_token_stats()

    @_requires("chat_orchestrator", default={})
    def get_router_stats(self) -> dict:
        """
        Return accuracy trends per complexity bucket from the router feedback log.
        Exposed to the frontend for diagnostics / Settings view.
        """
        return self._chat.get_router_stats()

    # ── Extended thinking ─────────────────────────────────────────────────────

    def ask_with_thinking(self, user_message: str,
                          budget_tokens: int = 10000) -> None:
        def _work():
            try:
                # Extended thinking is not supported on Haiku
                model = self._settings.get("claude_model", "claude-sonnet-4-6")
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
                    friendly = "Extended thinking isn't available for the selected model. Try claude-sonnet-4-6."
                elif "authentication" in err_msg:
                    friendly = "Invalid API key — update it in Settings."
                else:
                    friendly = "Extended thinking failed — try again or switch models in Settings."
                self._emit("thinking_error", {"error": friendly})
        run_in_thread(_work)

    # ── RAG / Documents ───────────────────────────────────────────────────────

    @_requires("embedder", default=None)
    def build_rag_index(self, folder_path: str) -> None:
        """Build/rebuild the RAG index from a folder."""
        def _work():
            try:
                self._emit("rag_progress", {"status": "Scanning files…", "pct": 5})

                def _on_progress(status, pct):
                    self._emit("rag_progress", {"status": status, "pct": pct})

                self._rag.build_from_folder(Path(folder_path), on_progress=_on_progress)
                cache_path = paths.rag_cache_dir() / "index.npz"
                self._rag.save(cache_path)
                count = self._rag.chunk_count()
                self._emit("rag_done", {"chunks": count, "folder": folder_path})
                self._os_notify("RAG Index Built",
                                f"Indexed {count} chunks from {folder_path}")
            except Exception as e:
                self._log.error(f"RAG build error: {e}")
                err_msg = str(e).lower()
                if "permission" in err_msg or "access" in err_msg:
                    friendly = "Can't read that folder — check that the app has permission to access it."
                elif "not found" in err_msg or "no such" in err_msg:
                    friendly = "Folder not found — make sure the path still exists."
                elif "memory" in err_msg or "oom" in err_msg:
                    friendly = "Not enough memory to index that folder. Try a smaller one, or index individual files."
                else:
                    friendly = "Indexing failed. Check the folder path and try again."
                self._emit("rag_error", {"error": friendly})
        run_in_thread(_work)

    @_requires("embedder", default={"error": "RAG unavailable"})
    def rag_add_file(self, file_path: str) -> dict:
        """Add a single file to the existing RAG index."""
        # ── Priority 5: scan document content before indexing ─────────────────
        try:
            _content = Path(file_path).read_text(errors="replace")[:50000]
            _scan = input_sanitizer.scan_document(_content, filename=file_path)
            if _scan.get("blocked"):
                return {"error": f"Document blocked by security scan — possible injection content detected.",
                        "scan_id": _scan.get("scan_id")}
        except Exception as _fe:
            self._log.debug(f"Document scan skipped: {_fe}")
        try:
            p = Path(file_path)
            n = self._rag.add_file(p)
            if n:
                cache_path = paths.rag_cache_dir() / "index.npz"
                self._rag.save(cache_path)
            return {"chunks_added": n, "total_chunks": self._rag.chunk_count()}
        except Exception as e:
            return {"error": str(e)}

    @_requires("embedder", default={"error": "RAG unavailable"})
    def rag_add_text(self, text: str, source: str = "manual") -> dict:
        """Add raw text to the RAG index."""
        try:
            n = self._rag.add_text(text, source=source)
            if n:
                cache_path = paths.rag_cache_dir() / "index.npz"
                self._rag.save(cache_path)
            return {"chunks_added": n, "total_chunks": self._rag.chunk_count()}
        except Exception as e:
            return {"error": str(e)}

    @_requires("rag_index", default={"error": "RAG unavailable"})
    def rag_clear(self) -> dict:
        """Clear the entire RAG index."""
        self._rag.clear()
        cache_path = paths.rag_cache_dir() / "index.npz"
        if cache_path.exists():
            cache_path.unlink()
        chunks_path = paths.rag_cache_dir() / "index_chunks.json"
        if chunks_path.exists():
            chunks_path.unlink()
        return {"ok": True}

    def rag_status(self) -> dict:
        status = self._status.get("rag_load", {}) if hasattr(self, "_status") else {}
        return {
            "chunk_count": self._rag.chunk_count() if self._rag is not None else 0,
            "index_exists": (paths.rag_cache_dir() / "index.npz").exists(),
            "available": bool(status.get("ok", self._rag is not None)),
            "error": status.get("error"),
        }

    @_requires("embedder", default=[])
    def rag_search(self, query: str, top_k: int = 5) -> list:
        results = self._rag.search(query, top_k=top_k)
        # Unwrap (text, score) tuples — the frontend only needs the text strings
        return [r[0] if isinstance(r, (list, tuple)) else r for r in results]

    def pick_folder(self) -> str | None:
        """Open a native folder picker dialog."""
        import webview as _wv
        result = self._window.create_file_dialog(
            _wv.FOLDER_DIALOG, allow_multiple=False
        )
        return result[0] if result else None

    def save_file_dialog(self, content: str, suggested_filename: str = "export.md") -> dict:
        """Open a native save dialog and write content to the chosen path."""
        import webview as _wv
        try:
            result = self._window.create_file_dialog(
                _wv.SAVE_DIALOG,
                save_filename=suggested_filename,
            )
            path = result[0] if isinstance(result, (list, tuple)) else result
            if not path:
                return {"ok": False, "cancelled": True}
            Path(path).write_text(content, encoding="utf-8")
            return {"ok": True, "path": str(path)}
        except Exception as exc:
            self._log.warning(f"save_file_dialog failed: {exc}")
            return {"ok": False, "error": str(exc)}

    def pick_files(self) -> list[str]:
        """Open a native file picker dialog. Returns list of absolute paths."""
        import webview as _wv
        result = self._window.create_file_dialog(
            _wv.OPEN_DIALOG, allow_multiple=True,
            file_types=(
                'Document Files (*.txt;*.md;*.pdf;*.py;*.js;*.json;*.csv;*.html;*.css;*.ts;*.jsx;*.tsx;*.yaml;*.yml;*.toml;*.xml;*.sql;*.sh;*.bat;*.ps1;*.r;*.rs;*.go;*.java;*.c;*.cpp;*.h;*.rb)',
                'All Files (*.*)',
            ),
        )
        return list(result) if result else []

    # ── Hardware / connection probing ─────────────────────────────────────────

    def probe_hardware(self) -> None:
        def _work():
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            ram = psutil.virtual_memory()
            ram_free_gb = round(ram.available / (1024 ** 3), 1)
            ram_total_gb = round(ram.total / (1024 ** 3), 1)
            gpu = "not detected"
            vram_free_gb = vram_total_gb = 0.0
            memory_type = "RAM"
            try:
                nr = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=3)
                mr = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3)
                if nr.returncode == 0 and mr.returncode == 0:
                    gpu = nr.stdout.strip().split("\n")[0]
                    parts = mr.stdout.strip().split("\n")[0].split(",")
                    vram_free_gb = round(int(parts[0].strip()) / 1024, 1)
                    vram_total_gb = round(int(parts[1].strip()) / 1024, 1)
                    memory_type = "VRAM"
            except Exception:
                pass
            if gpu == "not detected" and hasattr(os, "uname"):
                try:
                    r = subprocess.run(
                        ["system_profiler", "SPDisplaysDataType", "-json"],
                        capture_output=True, text=True, timeout=5)
                    if r.returncode == 0:
                        data = json.loads(r.stdout)
                        for d in data.get("SPDisplaysDataType", []):
                            info = d.get("spdisplays_vendor", "") + d.get("sppci_model", "")
                            if any(k in info.lower() for k in ("apple", " m1", " m2", " m3", " m4")):
                                gpu = d.get("sppci_model", "Apple Silicon")
                                vram_total_gb = ram_total_gb
                                vram_free_gb = ram_free_gb
                                memory_type = "Unified"
                                break
                except Exception:
                    pass
            ollama_ok = lmstudio_ok = False
            try:
                ollama_ok = requests.get(
                    self._settings.get("ollama_url", "http://localhost:11434") + "/api/tags",
                    timeout=2).status_code == 200
            except Exception:
                pass
            try:
                lmstudio_ok = requests.get(
                    self._settings.get("lm_studio_url", "http://localhost:1234") + "/v1/models",
                    timeout=2).status_code == 200
            except Exception:
                pass
            self._emit("hardware", {
                "cpu": cpu, "ram_free": ram_free_gb, "ram_total": ram_total_gb,
                "gpu": gpu, "vram_free": vram_free_gb, "vram_total": vram_total_gb,
                "memory_type": memory_type, "ollama": ollama_ok, "lmstudio": lmstudio_ok,
            })
        run_in_thread(_work)

    def test_connection(self, backend: str) -> None:
        def _work():
            url = (
                self._settings.get("ollama_url", "http://localhost:11434") + "/api/tags"
                if backend == "ollama"
                else self._settings.get("lm_studio_url", "http://localhost:1234") + "/v1/models"
            )
            try:
                ok = requests.get(url, timeout=3).status_code == 200
            except Exception:
                ok = False
            self._emit("connection_result", {"backend": backend, "ok": ok})
        run_in_thread(_work)

    def fetch_chat_models(self, backend: str) -> None:
        def _work():
            if self._local is None:
                self._emit("chat_models", {"backend": backend, "models": [],
                                            "error": "local client unavailable"})
                return
            models = self._local.list_models(backend=backend)
            self._emit("chat_models", {"backend": backend, "models": models})
        run_in_thread(_work)

    # ── Agent management ──────────────────────────────────────────────────────

    def agent_list(self) -> list:
        from services.agent_registry import list_agents
        return list_agents()

    def agent_get(self, agent_id: str) -> dict | None:
        from services.agent_registry import get_agent
        return get_agent(agent_id)

    def agent_create(self, name: str, description: str, system_prompt: str,
                     model_preference: str = "auto", temperature: float = 0.7,
                     max_tokens: int = 4096) -> dict:
        from services.agent_registry import create_agent
        return create_agent(name=name, description=description,
                            system_prompt=system_prompt,
                            model_preference=model_preference,
                            temperature=temperature, max_tokens=max_tokens)

    def agent_update(self, agent_id: str, fields: dict = None, **kwargs) -> dict:
        from services.agent_registry import update_agent
        try:
            # JS bridge passes a plain object as the second positional argument.
            # Python callers may use **kwargs directly.
            update_fields = {**(fields or {}), **kwargs}
            if not update_fields:
                return {"error": "No fields to update"}
            update_agent(agent_id, **update_fields)
            return {"ok": True}
        except ValueError as e:
            return {"error": str(e)}

    def agent_duplicate(self, agent_id: str, new_name: str) -> dict:
        from services.agent_registry import duplicate_agent
        try:
            return duplicate_agent(agent_id, new_name)
        except ValueError as e:
            return {"error": str(e)}

    def agent_delete(self, agent_id: str) -> dict:
        from services.agent_registry import delete_agent
        try:
            delete_agent(agent_id)
            return {"ok": True}
        except ValueError as e:
            return {"error": str(e)}

    # ── Team management ───────────────────────────────────────────────────────

    def team_list(self) -> list:
        from services.agent_registry import list_teams
        return list_teams()

    def team_get(self, team_id: str) -> dict | None:
        from services.agent_registry import get_team_with_members
        return get_team_with_members(team_id)

    def team_create(self, name: str, description: str,
                    coordinator_id: str) -> dict:
        from services.agent_registry import create_team
        return create_team(name=name, description=description,
                           coordinator_id=coordinator_id)

    def team_add_member(self, team_id: str, agent_id: str,
                        role: str = "worker") -> dict:
        try:
            updated = _registry_add_member(team_id, agent_id, role)
            return {"ok": True, "tom_refreshed_count": len(updated)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def team_remove_member(self, team_id: str, agent_id: str) -> dict:
        try:
            updated = _registry_remove_member(team_id, agent_id)
            return {"ok": True, "tom_refreshed_count": len(updated)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def team_delete(self, team_id: str) -> dict:
        from services.agent_registry import delete_team
        delete_team(team_id)
        return {"ok": True}

    # ── Prompt library ────────────────────────────────────────────────────────

    def prompt_list(self) -> list:
        return prompt_library.list_prompts()

    def prompt_versions(self, prompt_id: str) -> list:
        return prompt_library.get_prompt_versions(prompt_id)

    def prompt_save(self, prompt_id: str, text: str, notes: str = "") -> dict:
        return prompt_library.save_prompt_version(prompt_id, text, notes=notes)

    def prompt_create(self, name: str, category: str, description: str,
                      text: str, model_target: str = "auto") -> dict:
        return prompt_library.create_prompt(name, category, description,
                                            text, model_target=model_target)

    def prompt_duplicate(self, source_id: str, new_name: str) -> dict:
        return prompt_library.duplicate_prompt(source_id, new_name)

    def prompt_restore_version(self, version_id: str) -> dict:
        return prompt_library.restore_version(version_id)

    def prompt_delete(self, prompt_id: str) -> dict:
        return prompt_library.delete_prompt(prompt_id)

    def prompt_export(self, prompt_id: str) -> dict:
        return prompt_library.export_prompt(prompt_id)

    def prompt_import(self, data: dict) -> dict:
        return prompt_library.import_prompt(data)

    # ── Health check ─────────────────────────────────────────────────────────

    def run_health_check(self, skip_api: bool = False) -> None:
        def _work():
            results = health_monitor.check_all(
                api_key=self._settings.get("claude_api_key", ""),
                app_root=str(self._app_root),
                ollama_url=self._settings.get("ollama_url", "http://localhost:11434"),
                lmstudio_url=self._settings.get("lm_studio_url", "http://localhost:1234"),
                skip_api=skip_api,
            )
            self._emit("health_check_done", {
                "results": results,
                "has_failures": health_monitor.has_blocking_failures(results),
            })
        run_in_thread(_work)

    # ── Error logs ────────────────────────────────────────────────────────────

    def get_error_logs(self, limit: int = 50) -> list:
        return error_classifier.get_recent_errors(limit)

    def mark_error_resolved(self, record_id: str) -> dict:
        error_classifier.mark_resolved(record_id)
        return {"ok": True}

    # ── Semantic search ───────────────────────────────────────────────────────

    def search_memories_semantic(self, query: str, top_k: int = 5) -> list:
        return semantic_search.search_memories(query, top_k=top_k)

    def search_documents_semantic(self, query: str, top_k: int = 10,
                                  doc_type: str = "") -> list:
        return semantic_search.search_documents(
            query, top_k=top_k, doc_type=doc_type or None
        )

    def semantic_search_available(self) -> bool:
        status = self._status.get("semantic_search", {})
        return bool(status.get("ok")) and semantic_search.is_available()

    @_requires("memory_manager", default={"error": "memory unavailable"})
    def save_memory(self, content: str, category: str = "fact") -> dict:
        mem_id = self._memory.save_explicit_memory(content, category)
        return {"id": mem_id}

    def get_stale_memories(self, days: int = 30) -> list:
        """
        Return memory entries not accessed in the last `days` days.
        Used by the frontend Stale Memories panel so users can review and delete.
        """
        return semantic_search.get_stale_memories(days=days)

    def delete_memory_entry(self, entry_id: str) -> dict:
        """Delete a specific memory entry from both SQLite and ChromaDB."""
        ok = semantic_search.delete_memory_entry(entry_id)
        return {"ok": ok}

    # ── Diagnostics export ────────────────────────────────────────────────────

    def export_diagnostics(self) -> None:
        def _work():
            try:
                out_dir = self._app_root / "diagnostics"
                out_dir.mkdir(exist_ok=True)

                # Auto-clean diagnostic zips older than 7 days to prevent unbounded growth
                import time as _time
                cutoff = _time.time() - 7 * 86400
                for old_zip in out_dir.glob("myai_diagnostics_*.zip"):
                    try:
                        if old_zip.stat().st_mtime < cutoff:
                            old_zip.unlink()
                    except Exception:
                        pass

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_path = out_dir / f"myai_diagnostics_{stamp}.zip"

                with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
                    errors = error_classifier.get_recent_errors(500)
                    buf = io.StringIO()
                    if errors:
                        writer = csv.DictWriter(buf, fieldnames=list(errors[0].keys()))
                        writer.writeheader()
                        writer.writerows(errors)
                    zf.writestr("error_log.csv", buf.getvalue())

                    settings = self._settings.all()
                    redacted = {
                        k: "[REDACTED]" if any(s in k.lower()
                                               for s in ("key", "token", "secret", "password"))
                        else v
                        for k, v in settings.items()
                    }
                    zf.writestr("settings.json", json.dumps(redacted, indent=2))

                    env_info = {
                        "python_version": sys.version,
                        "platform": platform.platform(),
                        "app_version": "iMakeAiTeams v5.0.2",
                    }
                    try:
                        pip_out = subprocess.run(
                            [sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, timeout=10
                        )
                        env_info["pip_freeze"] = pip_out.stdout
                    except Exception:
                        env_info["pip_freeze"] = "unavailable"
                    zf.writestr("environment.json", json.dumps(env_info, indent=2))

                self._emit("diagnostics_ready", {"path": str(zip_path)})
            except Exception as exc:
                self._emit("diagnostics_error", {"error": str(exc)})
        run_in_thread(_work)

    # ── Changelog / What's new ────────────────────────────────────────────────

    _CURRENT_VERSION = "1.3.0"

    def get_changelog(self) -> dict:
        """
        Return changelog data. Called on startup by the frontend.

        Returns:
          {
            "current_version": "1.3.0",
            "last_seen_version": "1.2.0",  # or "" if never seen
            "is_new": True,                # True when current > last_seen
            "entries": [                   # all parsed changelog entries
              {"version": "1.3.0", "body": "...markdown..."},
              ...
            ],
            "new_entries": [               # only entries since last_seen
              {"version": "1.3.0", "body": "...markdown..."},
            ],
          }
        """
        last_seen = self._settings.get("last_seen_version", "")
        entries = self._parse_changelog()
        new_entries = [
            e for e in entries
            if not last_seen or self._version_gt(e["version"], last_seen)
        ]
        return {
            "current_version": self._CURRENT_VERSION,
            "last_seen_version": last_seen,
            "is_new": bool(new_entries),
            "entries": entries,
            "new_entries": new_entries,
        }

    def mark_changelog_seen(self) -> dict:
        """Record that the user has seen the current version's changelog."""
        self._settings.set_raw("last_seen_version", self._CURRENT_VERSION)
        return {"ok": True}

    def _parse_changelog(self) -> list[dict]:
        """
        Parse CHANGELOG.md into a list of {version, body} dicts,
        newest first. Falls back to empty list if the file isn't found.
        """
        # CHANGELOG lives next to the source tree, not in user data
        install_root = paths.install_root()
        changelog_path = install_root / "CHANGELOG.md"
        if not changelog_path.exists():
            changelog_path = install_root.parent / "CHANGELOG.md"
        if not changelog_path.exists():
            return []
        try:
            text = changelog_path.read_text(encoding="utf-8")
            entries = []
            current_version = None
            current_lines: list[str] = []
            for line in text.splitlines():
                if line.startswith("## v"):
                    if current_version and current_lines:
                        entries.append({
                            "version": current_version,
                            "body": "\n".join(current_lines).strip(),
                        })
                    current_version = line.replace("## v", "").strip()
                    current_lines = []
                elif current_version is not None:
                    current_lines.append(line)
            if current_version and current_lines:
                entries.append({
                    "version": current_version,
                    "body": "\n".join(current_lines).strip(),
                })
            return entries
        except Exception as exc:
            self._log.warning("Could not parse CHANGELOG.md: %s", exc)
            return []

    @staticmethod
    def _version_gt(a: str, b: str) -> bool:
        """Return True if version string a is strictly greater than b."""
        def _parts(v: str):
            try:
                return tuple(int(x) for x in v.strip().split("."))
            except ValueError:
                return (0,)
        return _parts(a) > _parts(b)

    # ── Utility ───────────────────────────────────────────────────────────────

    def open_url(self, url: str) -> None:
        # Only allow http/https — prevent arbitrary protocol launches (file://, etc.)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            webbrowser.open(url)
        else:
            self._log.warning(f"open_url blocked non-http URL: {url!r}")


    # ── Priority 1: Theory of Mind ────────────────────────────────────────────

    def agent_generate_tom(self, agent_name: str, agent_domain: str,
                           agent_scope: str, teammates: list | None = None) -> dict:
        """Generate a Theory of Mind preview block (does not persist)."""
        try:
            tom = generate_agent_tom(agent_name, agent_domain, agent_scope, teammates or [])
            return {"ok": True, "tom_block": tom, "teammate_count": len(teammates or [])}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "tom_block": ""}

    def agent_refresh_team_tom(self, team_id: str) -> dict:
        """Regenerate Theory of Mind for all agents in a team."""
        try:
            updated = refresh_team_tom(team_id)
            return {"ok": True, "updated_count": len(updated), "updated_ids": updated}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ── Priority 2: Hybrid Search ─────────────────────────────────────────────

    def rag_search_hybrid(self, query: str, top_k: int = 5,
                          method: str = "hybrid", doc_type: str = "") -> list:
        """Hybrid BM25 + vector + RRF document search."""
        return semantic_search.search_documents_hybrid(
            query_text=query, top_k=top_k, doc_type=doc_type or None, method=method
        )

    def bm25_corpus_size(self) -> dict:
        """Return BM25 corpus size and availability."""
        return {
            "bm25_available": getattr(semantic_search, "_bm25_available", False),
            "corpus_size":    len(getattr(semantic_search, "_bm25_doc_ids", [])),
            "chroma_docs":    semantic_search.document_count(),
        }

    # ── Priority 5: Firewall ──────────────────────────────────────────────────

    def security_get_status(self) -> dict:
        """Return firewall status for the Settings Security panel."""
        return input_sanitizer.get_firewall_status()

    def security_toggle_firewall(self, enabled: bool) -> dict:
        """Enable or disable the input sanitization firewall."""
        input_sanitizer.set_firewall_enabled(enabled)
        return {"ok": True, "firewall_enabled": enabled}

    def security_get_scan_log(self, limit: int = 50, verdict_filter: str = "") -> list:
        """Return recent scan log for the Settings Security audit panel."""
        return input_sanitizer.get_scan_log(limit=limit, verdict_filter=verdict_filter)

    # ── v4.0: Studio Mode ────────────────────────────────────────────────────

    def studio_mode_get(self) -> dict:
        """Return current studio mode state."""
        return {"enabled": bool(self._settings.get("studio_mode", False))}

    def studio_mode_set(self, enabled: bool) -> dict:
        """Enable or disable Studio Mode (shows advanced nav items)."""
        self._settings.set("studio_mode", enabled)
        return {"ok": True, "enabled": enabled}

    def shutdown(self) -> None:
        self._log.info("Shutting down services…")
        self._stop_chat.set()
        self._log.info("Shutdown complete.")

    def agent_set_project_root(self, path: str) -> dict:
        """Set the default project root for agent runs."""
        from pathlib import Path
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"error": f"Path does not exist: {path}"}
        if not p.is_dir():
            return {"error": f"Path is not a directory: {path}"}
        self._settings.set("agent_project_root", str(p))
        return {"ok": True, "path": str(p)}

    # ── Fix 8: Model pricing management ──────────────────────────────────────

    def get_model_prices(self) -> dict:
        """Return current model pricing (defaults + any user overrides)."""
        from services.chat_orchestrator import _DEFAULT_MODEL_PRICES
        defaults = {k: {"input": v[0], "output": v[1]} for k, v in _DEFAULT_MODEL_PRICES.items()}
        custom = self._settings.get("model_prices", None)
        if custom and isinstance(custom, dict):
            for k, v in custom.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    defaults[k] = {"input": float(v[0]), "output": float(v[1])}
        return defaults

    def set_model_prices(self, prices: dict) -> dict:
        """
        Update model pricing. Format: {"haiku": [0.80, 4.0], "sonnet": [3.0, 15.0]}
        Values are per-million-token [input_price, output_price].
        """
        if not isinstance(prices, dict):
            return {"error": "prices must be a dict"}
        clean = {}
        for key, val in prices.items():
            if isinstance(val, (list, tuple)) and len(val) == 2:
                clean[key] = [float(val[0]), float(val[1])]
            elif isinstance(val, dict) and "input" in val and "output" in val:
                clean[key] = [float(val["input"]), float(val["output"])]
        self._settings.set("model_prices", clean)
        return {"ok": True, "prices": clean}
