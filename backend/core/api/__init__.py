"""
core/api/__init__.py — Service container facade for the FastAPI sidecar.

Originally a PyWebView js_api bridge; now consumed by FastAPI route handlers
in backend/routes/. Public methods that previously ran on the PyWebView main
thread are still safe to call from FastAPI threadpool contexts. Methods that
fan out work to background threads push results to the renderer via
events_sse.publish() (replaces the old window.__emit JS shim).

Architecture:
  - ChatOrchestrator  — unified conversation loop (routing, memory, tokens)
  - TaskRouter        — classifies messages, picks Claude vs local
  - MemoryManager     — three-tier memory (buffer, facts, RAG/semantic)
  - AgentRegistry     — CRUD for agents and teams
  - ClaudeClient      — Anthropic SDK wrapper
  - LocalClient       — Ollama / LM Studio client
  - RAGIndex          — sentence-transformer semantic search over files

The domain methods are split across api/chat.py, api/agents.py,
api/memory.py, api/rag.py, api/settings.py. This module holds the facade
`API` class: it owns service handles and the status dict, constructs the
sub-API instances, and forwards bridge calls to them through one-line
delegators. Everything that didn't fit a domain (diagnostics, security,
health, hardware, file dialogs, prompts, errors, changelog) stays here.
"""

import json
import logging
import os
import subprocess
import threading
import webbrowser
import zipfile
import csv
import io
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

import events_sse
from core import paths
from core.settings import Settings
from core.events import EventBus

from services.claude_client import ClaudeClient
from services.local_client import LocalClient
from services.rag_index import RAGIndex
from services import semantic_search, error_classifier, health_monitor
from services import prompt_library
from services.router import TaskRouter
from services.memory import MemoryManager
from services.chat_orchestrator import ChatOrchestrator
from services.agent_registry import (
    seed_agents, update_builtin_tom, seed_default_skills,
    anonymize_existing_critic_prompts,
)
from services import input_sanitizer
from services.mcp_registry import MCPRegistry
from services.audit_log import AuditLog
from services.lifecycle import LifecycleManager

import db as _db_module

from .chat import ChatAPI
from .agents import AgentsAPI
from .memory import MemoryAPI
from .rag import RagAPI
from .settings import SettingsAPI
from .mcp import MCPAPI
from .lifecycle import LifecycleAPI


class API:
    def __init__(self, settings: Settings, bus: EventBus, app_root: Path,
                 log: logging.Logger):
        self._settings = settings
        self._bus = bus
        self._app_root = app_root
        self._log = log
        self._stop_chat = threading.Event()

        # Each service records its init status here. Writers only mutate this
        # dict during __init__ on the main thread; downstream readers read it
        # snapshot-only after init completes, so no lock is required.
        self._status: dict[str, dict] = {}

        # ── Claude client (required — no useful app without it) ───────────────
        self._claude = self._safe_init(
            "claude_client",
            lambda: ClaudeClient(
                api_key=self._settings.get("claude_api_key", ""),
                model=self._settings.get("claude_model"),
                use_caching=self._settings.get("claude_prompt_caching"),
            ),
            required=True,
        )

        # ── Local model client ────────────────────────────────────────────────
        self._local = self._safe_init(
            "local_client",
            lambda: LocalClient(self._settings),
        )

        # ── RAG index (fast — constructs empty, embedder attached later) ──────
        # SentenceTransformer load is deferred: see _run_deferred_init.
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

        self._safe_init("prompts_seed", prompt_library.seed_prompts)
        self._safe_init("agents_seed", seed_agents)
        self._safe_init("theory_of_mind", update_builtin_tom)
        self._safe_init("hub_skills_seed", seed_default_skills)
        # Phase 4: scrub any pre-existing reviewer prompts of peer identifiers.
        self._safe_init("critic_anonymization", anonymize_existing_critic_prompts)

        def _firewall_init():
            has_key = bool(self._settings.get("claude_api_key", "").strip())
            input_sanitizer.set_firewall_enabled(has_key)
            return has_key
        self._safe_init("firewall", _firewall_init)

        self._memory = self._safe_init(
            "memory_manager",
            lambda: MemoryManager(
                rag_index=self._rag,
                semantic_search_mod=semantic_search,
                local_client=self._local,
            ),
        )

        self._router = self._safe_init(
            "router",
            lambda: TaskRouter(self._local, self._settings),
        )

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

        # Phase 2: MCP tool registry (catalog only; execution deferred).
        self._mcp_registry = self._safe_init(
            "mcp_registry",
            lambda: MCPRegistry(paths.mcp_servers_dir(), self._settings),
        )

        # Phase 4: append-only audit log + human-in-loop lifecycle gate.
        self._audit_log = self._safe_init(
            "audit_log",
            lambda: AuditLog(paths.user_dir() / "lifecycle_audit.jsonl"),
        )
        self._lifecycle = self._safe_init(
            "lifecycle",
            lambda: LifecycleManager(self._audit_log, emit=self._emit),
        )

        # Deferred services — mark pending so the UI renders a spinner row.
        for _name in ("embedder", "rag_load", "semantic_search",
                      "semantic_search_indexer"):
            self._status[_name] = {"ok": False, "error": None, "pending": True}

        # ── Domain sub-APIs (composition) ─────────────────────────────────────
        self._chat_api = ChatAPI(self)
        self._agents_api = AgentsAPI(self)
        self._memory_api = MemoryAPI(self)
        self._rag_api = RagAPI(self)
        self._settings_api = SettingsAPI(self)
        self._mcp_api = MCPAPI(self)
        self._lifecycle_api = LifecycleAPI(self)

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
        model = self._safe_init("embedder", self._load_shared_embedder)
        self._emit_service_update("embedder")

        if self._rag is not None and model is not None:
            self._rag._model = model

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

        self._safe_init(
            "semantic_search",
            lambda: semantic_search.init_vector_store(
                paths.vector_store_dir(), shared_model=model,
            ),
        )
        self._emit_service_update("semantic_search")

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
        model = SentenceTransformer(str(paths.bundled_model_dir()))
        self._log.info("Shared SentenceTransformer model loaded.")
        return model

    def service_status(self) -> dict:
        """Return a snapshot of per-service init status for the UI."""
        return {name: dict(entry) for name, entry in self._status.items()}

    # ── Event emission ───────────────────────────────────────────────────────

    def _emit(self, event: str, payload: Any = None) -> None:
        """Push an event onto the SSE bus for the renderer's EventSource to drain."""
        try:
            events_sse.publish(event, payload if payload is not None else {})
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

    # ── File dialogs ─────────────────────────────────────────────────────────
    # File pickers are now an Electron concern. The renderer calls
    # window.electronAPI.selectFolder() / .selectFiles() / .saveFileDialog()
    # directly, then hands the resulting path(s) to backend routes that need
    # them (e.g. POST /api/rag/index_folder takes a `path` body field).
    # Save-from-server flows write the file path returned by the Electron
    # save dialog rather than letting the sidecar prompt for a path.

    def write_file(self, path: str, content: str) -> dict:
        """Write `content` to `path` (used by chat_export_conversation flow)."""
        try:
            Path(path).write_text(content, encoding="utf-8")
            return {"ok": True, "path": str(path)}
        except Exception as exc:
            self._log.warning(f"write_file failed: {exc}")
            return {"ok": False, "error": str(exc)}

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
                    self._settings.get("ollama_url") + "/api/tags",
                    timeout=2).status_code == 200
            except Exception:
                pass
            try:
                lmstudio_ok = requests.get(
                    self._settings.get("lm_studio_url") + "/v1/models",
                    timeout=2).status_code == 200
            except Exception:
                pass
            self._emit("hardware", {
                "cpu": cpu, "ram_free": ram_free_gb, "ram_total": ram_total_gb,
                "gpu": gpu, "vram_free": vram_free_gb, "vram_total": vram_total_gb,
                "memory_type": memory_type, "ollama": ollama_ok, "lmstudio": lmstudio_ok,
            })
        from core.worker import run_in_thread
        run_in_thread(_work)

    def test_connection(self, backend: str) -> None:
        def _work():
            url = (
                self._settings.get("ollama_url") + "/api/tags"
                if backend == "ollama"
                else self._settings.get("lm_studio_url") + "/v1/models"
            )
            try:
                ok = requests.get(url, timeout=3).status_code == 200
            except Exception:
                ok = False
            self._emit("connection_result", {"backend": backend, "ok": ok})
        from core.worker import run_in_thread
        run_in_thread(_work)

    def fetch_chat_models(self, backend: str) -> None:
        def _work():
            if self._local is None:
                self._emit("chat_models", {"backend": backend, "models": [],
                                            "error": "local client unavailable"})
                return
            models = self._local.list_models(backend=backend)
            self._emit("chat_models", {"backend": backend, "models": models})
        from core.worker import run_in_thread
        run_in_thread(_work)

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
                ollama_url=self._settings.get("ollama_url"),
                lmstudio_url=self._settings.get("lm_studio_url"),
                skip_api=skip_api,
            )
            self._emit("health_check_done", {
                "results": results,
                "has_failures": health_monitor.has_blocking_failures(results),
            })
        from core.worker import run_in_thread
        run_in_thread(_work)

    # ── Error logs ────────────────────────────────────────────────────────────

    def get_error_logs(self, limit: int = 50) -> list:
        return error_classifier.get_recent_errors(limit)

    def mark_error_resolved(self, record_id: str) -> dict:
        error_classifier.mark_resolved(record_id)
        return {"ok": True}

    # ── Diagnostics export ────────────────────────────────────────────────────

    def export_diagnostics(self) -> None:
        def _work():
            try:
                out_dir = self._app_root / "diagnostics"
                out_dir.mkdir(exist_ok=True)

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
        from core.worker import run_in_thread
        run_in_thread(_work)

    # ── Changelog / What's new ────────────────────────────────────────────────

    _CURRENT_VERSION = "1.3.0"

    def get_changelog(self) -> dict:
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

    # ── Security / firewall ───────────────────────────────────────────────────

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

    # ── Misc ──────────────────────────────────────────────────────────────────

    def open_url(self, url: str) -> None:
        # Only allow http/https — prevent arbitrary protocol launches (file://, etc.)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            webbrowser.open(url)
        else:
            self._log.warning(f"open_url blocked non-http URL: {url!r}")

    def shutdown(self) -> None:
        self._log.info("Shutting down services…")
        self._stop_chat.set()
        self._log.info("Shutdown complete.")

    # ── Domain sub-API delegators ─────────────────────────────────────────────
    # These exist so PyWebView sees every bridge method directly on API. They
    # forward to the domain-specific sub-API instances, which share facade
    # state via BaseAPI.__getattr__. Preserving the exact surface is what
    # keeps the frontend untouched.

    # Chat
    def chat_send(self, conversation_id, user_message, agent_id=""):
        return self._chat_api.chat_send(conversation_id, user_message, agent_id)

    def chat_stop(self):
        return self._chat_api.chat_stop()

    def chat_new_conversation(self, agent_id="", title="New conversation"):
        return self._chat_api.chat_new_conversation(agent_id, title)

    def chat_list_conversations(self, limit=30):
        return self._chat_api.chat_list_conversations(limit)

    def chat_get_messages(self, conversation_id, limit=100):
        return self._chat_api.chat_get_messages(conversation_id, limit)

    def chat_rename_conversation(self, conversation_id, title):
        return self._chat_api.chat_rename_conversation(conversation_id, title)

    def chat_delete_conversation(self, conversation_id):
        return self._chat_api.chat_delete_conversation(conversation_id)

    def chat_branch_conversation(self, conversation_id, from_message_id):
        return self._chat_api.chat_branch_conversation(conversation_id, from_message_id)

    def chat_export_conversation(self, conversation_id, fmt="markdown"):
        return self._chat_api.chat_export_conversation(conversation_id, fmt)

    def chat_token_stats(self):
        return self._chat_api.chat_token_stats()

    def get_router_stats(self):
        return self._chat_api.get_router_stats()

    def ask_with_thinking(self, user_message, budget_tokens=10000):
        return self._chat_api.ask_with_thinking(user_message, budget_tokens)

    # Agents
    def agent_list(self):
        return self._agents_api.agent_list()

    def agent_get(self, agent_id):
        return self._agents_api.agent_get(agent_id)

    def agent_create(self, name, description, system_prompt,
                     model_preference="auto", temperature=0.7, max_tokens=4096):
        return self._agents_api.agent_create(
            name, description, system_prompt,
            model_preference, temperature, max_tokens,
        )

    def agent_update(self, agent_id, fields=None, **kwargs):
        return self._agents_api.agent_update(agent_id, fields, **kwargs)

    def agent_duplicate(self, agent_id, new_name):
        return self._agents_api.agent_duplicate(agent_id, new_name)

    def agent_delete(self, agent_id):
        return self._agents_api.agent_delete(agent_id)

    def agent_generate_tom(self, agent_name, agent_domain, agent_scope, teammates=None):
        return self._agents_api.agent_generate_tom(
            agent_name, agent_domain, agent_scope, teammates,
        )

    def agent_refresh_team_tom(self, team_id):
        return self._agents_api.agent_refresh_team_tom(team_id)

    def agent_set_project_root(self, path):
        return self._agents_api.agent_set_project_root(path)

    def team_list(self):
        return self._agents_api.team_list()

    def team_get(self, team_id):
        return self._agents_api.team_get(team_id)

    def team_create(self, name, description, coordinator_id):
        return self._agents_api.team_create(name, description, coordinator_id)

    def team_add_member(self, team_id, agent_id, role="worker"):
        return self._agents_api.team_add_member(team_id, agent_id, role)

    def team_remove_member(self, team_id, agent_id):
        return self._agents_api.team_remove_member(team_id, agent_id)

    def team_delete(self, team_id):
        return self._agents_api.team_delete(team_id)

    # Memory
    def save_memory(self, content, category="fact"):
        return self._memory_api.save_memory(content, category)

    def search_memories_semantic(self, query, top_k=5):
        return self._memory_api.search_memories_semantic(query, top_k)

    def search_documents_semantic(self, query, top_k=10, doc_type=""):
        return self._memory_api.search_documents_semantic(query, top_k, doc_type)

    def semantic_search_available(self):
        return self._memory_api.semantic_search_available()

    def get_stale_memories(self, days=30):
        return self._memory_api.get_stale_memories(days)

    def delete_memory_entry(self, entry_id):
        return self._memory_api.delete_memory_entry(entry_id)

    # RAG
    def build_rag_index(self, folder_path):
        return self._rag_api.build_rag_index(folder_path)

    def rag_add_file(self, file_path):
        return self._rag_api.rag_add_file(file_path)

    def rag_add_text(self, text, source="manual"):
        return self._rag_api.rag_add_text(text, source)

    def rag_clear(self):
        return self._rag_api.rag_clear()

    def rag_status(self):
        return self._rag_api.rag_status()

    def rag_search(self, query, top_k=5):
        return self._rag_api.rag_search(query, top_k)

    def rag_search_hybrid(self, query, top_k=5, method="hybrid", doc_type=""):
        return self._rag_api.rag_search_hybrid(query, top_k, method, doc_type)

    def bm25_corpus_size(self):
        return self._rag_api.bm25_corpus_size()

    # Settings
    def get_settings(self):
        return self._settings_api.get_settings()

    def save_setting(self, key, value):
        return self._settings_api.save_setting(key, value)

    def set_setting(self, key, value):
        return self._settings_api.set_setting(key, value)

    def get_setting(self, key):
        return self._settings_api.get_setting(key)

    def complete_first_run(self, start_tab):
        return self._settings_api.complete_first_run(start_tab)

    def verify_api_key(self, key):
        return self._settings_api.verify_api_key(key)

    def detect_local_setup(self):
        return self._settings_api.detect_local_setup()

    def get_model_prices(self):
        return self._settings_api.get_model_prices()

    def set_model_prices(self, prices):
        return self._settings_api.set_model_prices(prices)

    def studio_mode_get(self):
        return self._settings_api.studio_mode_get()

    def studio_mode_set(self, enabled):
        return self._settings_api.studio_mode_set(enabled)

    # ── MCP servers (Phase 2) ─────────────────────────────────────────────────

    def list_mcp_servers(self):
        return self._mcp_api.list_mcp_servers()

    def pick_mcp_server_folder(self, folder_path="", overwrite=False):
        return self._mcp_api.pick_mcp_server_folder(
            folder_path=folder_path, overwrite=bool(overwrite),
        )

    def remove_mcp_server(self, server_id):
        return self._mcp_api.remove_mcp_server(server_id)

    def set_mcp_server_enabled(self, server_id, enabled):
        return self._mcp_api.set_mcp_server_enabled(server_id, bool(enabled))

    def set_mcp_secret(self, server_id, key, value):
        return self._mcp_api.set_mcp_secret(server_id, key, value)

    def clear_mcp_secret(self, server_id, key):
        return self._mcp_api.clear_mcp_secret(server_id, key)

    def refresh_mcp_registry(self):
        return self._mcp_api.refresh_mcp_registry()

    # ── Lifecycle (Phase 4) ──────────────────────────────────────────────────

    def confirm_shutdown(self, token):
        return self._lifecycle_api.confirm_shutdown(token)

    def deny_shutdown(self, token):
        return self._lifecycle_api.deny_shutdown(token)

    def list_lifecycle_audit(self, limit=100):
        return self._lifecycle_api.list_lifecycle_audit(limit)

    def request_agent_shutdown_demo(self, target_id="agent-b",
                                     requester_id="agent-a", reason="demo"):
        return self._lifecycle_api.request_agent_shutdown_demo(
            target_id, requester_id, reason,
        )
