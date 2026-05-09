"""
core/settings.py — Persistent settings backed by a JSON file on disk.

Stage 2 additions:
  - SETTINGS_DEFAULTS: typed schema with default values
  - _migrate(): fills missing keys with defaults on startup
  - set(): validates type and rejects unknown keys with a warning log
  - get_all_with_defaults(): helper for frontend introspection
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("iMakeAiTeams.settings")

# ── Secret routing via OS keyring ────────────────────────────────────────────
# Keys in this set are stored in the platform keyring (DPAPI on Windows,
# Keychain on macOS, SecretService on Linux) instead of settings.json.
# On first load, a plaintext value found in settings.json is migrated to the
# keyring and cleared from the JSON file so the secret only lives on-disk in
# the OS-native store.
SECRET_KEYS: set[str] = {"claude_api_key", "power_mode_api_key"}
KEYRING_SERVICE = "iMakeAiTeams"


def _keyring_get(key: str) -> str | None:
    # Broad BaseException catch is intentional: some backends (e.g. pyo3-based
    # SecretService on a host missing its native deps) raise PanicException,
    # which derives from BaseException. A broken keyring must never bring the
    # app down — fall back to plaintext silently.
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, key)
    except BaseException as exc:
        # Warn so the user can diagnose why their API key appears missing.
        # A broken keyring makes stored secrets inaccessible, which looks
        # like an empty API key rather than a configuration problem.
        log.warning(
            "keyring.get_password(%s) failed: %s — stored secrets may be "
            "inaccessible. Check your OS keyring/SecretService setup.", key, exc
        )
        return None


def _keyring_set(key: str, value: str) -> bool:
    try:
        import keyring
        keyring.set_password(KEYRING_SERVICE, key, value)
        return True
    except BaseException as exc:
        log.warning("keyring.set_password(%s) failed: %s — falling back to plaintext", key, exc)
        return False


def _keyring_delete(key: str) -> None:
    try:
        import keyring
        keyring.delete_password(KEYRING_SERVICE, key)
    except BaseException as exc:
        log.debug("keyring.delete_password(%s) failed: %s", key, exc)

# ── Schema ────────────────────────────────────────────────────────────────────
# Each entry: key -> (python_type_or_types, default_value)
# Use a tuple of types to allow multiple acceptable types (e.g. str and NoneType).
SETTINGS_DEFAULTS: dict[str, tuple] = {
    # API / model
    "claude_api_key":              (str,   ""),
    "claude_model":                (str,   "claude-sonnet-4-6"),
    "default_local_model":         (str,   ""),
    "system_prompt":               (str,   "You are a helpful AI assistant."),

    # Local model backends
    "ollama_url":                  (str,   "http://localhost:11434"),
    "lm_studio_url":               (str,   "http://localhost:1234"),
    "default_local_backend":       (str,   "ollama"),

    # Phase 9: Bundled llama.cpp server.
    # local_backend_mode picks which local stack the LocalClient routes to:
    #   "auto"      — preserve historical detection (Ollama → LM Studio fallback)
    #   "ollama"    — force the Ollama URL
    #   "lm_studio" — force the LM Studio URL
    #   "bundled"   — use the bundled llama-server spawned from BundledServer
    # Validation of the string values lives in the LocalClient routing layer;
    # the schema is intentionally a plain str so legacy installs don't lose
    # the field on a forward-incompatible enum check.
    "local_backend_mode":          (str,   "auto"),
    "bundled_model_id":            (str,   ""),

    # Routing
    "routing_enabled":             (bool,  True),
    "local_model_min_params":      (str,   "7B"),

    # Caching
    "claude_prompt_caching":       (bool,  True),

    # UI — start tab
    "start_tab":                   (str,   "chat"),

    # RAG / indexing
    "rag_folder":                  (str,   ""),
    "rag_chunk_size":              (int,   800),
    "rag_chunk_overlap":           (int,   200),

    # Memory
    "memory_similarity_threshold": (float, 0.5),
    "memory_history_cap":          (int,   40),
    "memory_write_gate_enabled":   (bool,  True),

    # Phase 5: Local-model behavior-drift canary (arXiv 2511.15992).
    # When enabled, the canary captures a 30-prompt baseline on the first
    # observed load of each model_id and re-checks on subsequent loads.
    # mean_drift > 0.40 emits a `model_canary_alert` SSE event.
    "model_canary_enabled":        (bool,  True),

    # Health / diagnostics
    "health_check_enabled":        (bool,  True),
    "diagnostics_retention_days":  (int,   7),

    # UI
    "theme":                       (str,   "system"),
    "show_token_counts":           (bool,  True),
    "show_cost_estimates":         (bool,  True),

    # First-run
    "first_run_complete":          (bool,  False),
    "onboarding_step":             (int,   0),
    "last_seen_version":           (str,   ""),

    # Phase 10: silent auto-update (electron-updater). When True, the Electron
    # main process polls the GitHub publish target every 6h and downloads new
    # releases in the background. The user is never force-restarted — the
    # downloaded update only applies after they click "Restart now" in the
    # UpdateBanner.
    "auto_update_enabled":         (bool,  True),

    # Token budget (Stage 5)
    "max_conversation_budget_usd":  (float, 5.0),    # stop sending if cumulative cost exceeds this
    "budget_warning_threshold_pct": (float, 80.0),    # warn frontend at this % of budget

    # Feature flags (v4.0+)
    "goal_decomposition_enabled":    (bool,  True),
    "interleaved_reasoning_enabled": (bool,  True),
    "knowledge_graph_enabled":       (bool,  True),
    "studio_mode":                   (bool,  False),
    "firewall_enabled":              (bool,  True),
    # Adversarial debate (Du 2024) defaults off — it adds an LLM call per
    # specialist step. Power users opt in. When enabled, the high-stakes
    # gate below keeps it from firing on cheap "what's 2+2" turns.
    "debate_enabled":                (bool,  False),
    "debate_only_high_stakes":       (bool,  True),
    "guardrails_enabled":            (bool,  False),

    # DiLoCo-inspired sliding-window risk ledger. Default off — the legacy
    # per-turn-reset behavior stays in place. When enabled, the per-conversation
    # risk ledger persists across turns and prunes entries older than the
    # window, catching sustained risky behavior without locking conversations
    # out after ~9 messages.
    "sliding_window_risk_enabled":   (bool,  False),
    "sliding_window_risk_minutes":   (float, 10.0),

    # Phase 5: Wiser-Human-style escalation channel. When the orchestrator
    # detects a Lynch et al. trigger (replacement_threat, autonomy_reduction,
    # goal_conflict) it pauses worker invocation and surfaces the action to
    # the human for approval. Default on for new installs; user-toggleable.
    "escalation_channel_enabled":    (bool,  True),

    # Phase 6: Hackett et al. (ACL 2025) Reader/Actor split. When True the
    # orchestrator runs a 3-phase pipeline (read → act → synthesize) where
    # the Actor never sees raw retrieved data and may only call tools the
    # Reader proposed. Default False so existing behavior is preserved.
    "reader_actor_split_enabled":    (bool,  False),

    # Phase 12: CaMeL (Defeating Prompt Injections by Design — DeepMind/ETH,
    # arXiv 2503.18813). Privileged-LLM / Quarantined-LLM split with
    # capability-tagged plan execution. Only fires when the turn has
    # retrieved RAG chunks. Mutually exclusive with the Reader/Actor split:
    # when both flags are on, CaMeL wins (it is a stricter superset).
    # Default False so existing behavior is preserved.
    "camel_enabled":                 (bool,  False),

    # Phase 8: Symphony-style weighted-vote consensus on high-stakes turns.
    # Three parallel CoT samples, majority weighted by self-reported
    # confidence + lexical similarity. Only fires when the message is
    # high-stakes (escalation trigger, governance.HIGH_STAKES_KEYWORDS, or
    # cumulative risk_score > 0.7) AND the resolved target is Claude.
    "high_stakes_voting_enabled":    (bool,  True),

    # Phase 11: image input. Claude has built-in vision; local routes need
    # a vision-capable model. ``vision_local_models`` is a prefix-match
    # family list — any local model id that starts with one of these
    # strings (case-insensitive) is treated as vision-capable.
    "vision_enabled":                (bool,  True),
    "vision_local_models":           (list,  [
        "qwen2.5-vl", "llava", "minicpm-v", "moondream",
    ]),

    # Phase 13: voice input (Whisper.cpp) + voice output (Piper). Both
    # default off so a fresh install doesn't show the mic / speaker UI
    # until the user opts in. The model ids reference the build-pipeline
    # catalog at branding/sidecar-bundle/voice_assets.json; the actual
    # binary files (Whisper .bin, Piper .onnx + .json) live under
    # userData/voice/ and download on first feature use.
    "voice_input_enabled":           (bool,  False),
    "voice_output_enabled":          (bool,  False),
    "stt_model_id":                  (str,   "whisper-base.en"),
    "tts_voice_id":                  (str,   "en_US-amy-medium"),

    # Agent / project
    "agent_project_root":            (str,   ""),

    # Phase 2: MCP servers (per-server enable list lives in this setting; the
    # registry itself reads server folders from paths.mcp_servers_dir()).
    "mcp_servers_disabled":          ((list, type(None)), []),

    # Phase 3: Qwen3 hybrid thinking. Per-agent budget overrides live on the
    # agents table; this is the global ceiling enforced for any agent.
    "qwen_thinking_global_budget_cap": (int, 8192),

    # v3 Power Mode: opt-in OpenClaw delegation for execution-class tasks.
    # All settings default to off / empty; Power Mode is dormant until the
    # user enables it from Settings → Power Mode.
    "power_mode_enabled":             (bool, False),
    "power_mode_workspace":           (str,  ""),
    "power_mode_model_provider":      (str,  "anthropic"),
    "power_mode_model_name":          (str,  "claude-sonnet-4-6"),
    "power_mode_api_key":             (str,  ""),
    "power_mode_autostart":           (bool, False),
    "power_mode_gateway_port":        (int,  18789),
    # Override the OpenClaw container image. Empty string falls back to the
    # pinned default in services/docker_manager.py — change this only when
    # intentionally moving to a new release.
    "openclaw_image":                 (str,  ""),

    # Advanced (complex types)
    "model_prices":                  ((dict, type(None)),  None),
    "hooks":                         ((list, type(None)),  None),
    "channel_allowlist":             ((dict, type(None)),  None),

    # Misc
    "default_agent_id":            ((str, type(None)),  None),
    "app_version":                 (str,   "5.0.2"),
}


def _coerce(key: str, value: Any, expected_type) -> Any:
    """
    Attempt to coerce a value to the expected type.
    Returns the coerced value, or the original value if coercion fails
    (type mismatch is logged as a warning instead of crashing).
    """
    # Handle tuple of types (Union-like)
    if isinstance(expected_type, tuple):
        if isinstance(value, expected_type):
            return value
        for t in expected_type:
            if t is type(None):
                continue
            try:
                return t(value)
            except (ValueError, TypeError):
                pass
        log.warning(
            "settings: key '%s' has value %r which could not be coerced to %s; "
            "keeping as-is.", key, value, expected_type
        )
        return value

    if isinstance(value, expected_type):
        return value

    # Special-case bool: "true"/"false" strings are common in JSON config files
    if expected_type is bool:
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
        if isinstance(value, int):
            return bool(value)

    try:
        return expected_type(value)
    except (ValueError, TypeError):
        log.warning(
            "settings: key '%s' has value %r; expected %s, keeping as-is.",
            key, value, expected_type.__name__
        )
        return value


class Settings:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict = {}
        self._load()
        self._migrate()
        self._migrate_secrets_to_keyring()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                log.warning("settings: could not load %s, starting with defaults.", self._path)
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def _migrate(self) -> None:
        """
        Fill any missing keys with their default values and coerce existing
        values to the expected type. Saves the file if any changes were made.
        Runs once at startup so all callers can rely on the full key set.
        """
        changed = False
        for key, (expected_type, default_value) in SETTINGS_DEFAULTS.items():
            if key not in self._data:
                self._data[key] = default_value
                changed = True
                log.debug("settings: migrated missing key '%s' = %r", key, default_value)
            else:
                coerced = _coerce(key, self._data[key], expected_type)
                if coerced != self._data[key]:
                    self._data[key] = coerced
                    changed = True

        if changed:
            try:
                self._save()
            except OSError as exc:
                log.warning("settings: could not save migrated settings: %s", exc)

    def _migrate_secrets_to_keyring(self) -> None:
        """
        One-time migration: move plaintext secrets from settings.json into the
        OS keyring and clear the JSON copy. Runs on every load; no-op once the
        JSON value is blank.
        """
        changed = False
        for key in SECRET_KEYS:
            plain = self._data.get(key)
            if not plain:
                continue
            if _keyring_set(key, plain):
                self._data[key] = ""
                changed = True
                log.info("settings: migrated secret '%s' into OS keyring", key)
        if changed:
            try:
                self._save()
            except OSError as exc:
                log.warning("settings: could not save after secret migration: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        if key in SECRET_KEYS:
            stored = _keyring_get(key)
            if stored:
                return stored
            # Fall through to plaintext lookup — either keyring is unavailable
            # (headless Linux without dbus, tests) or the secret was never set.
        with self._lock:
            if key in self._data:
                return self._data[key]
            if key in SETTINGS_DEFAULTS:
                return SETTINGS_DEFAULTS[key][1]
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a setting value. Unknown keys are rejected with a warning.
        Known keys are type-coerced before saving.
        """
        if key not in SETTINGS_DEFAULTS:
            log.warning(
                "settings: attempted to set unknown key '%s'; ignoring. "
                "Add it to SETTINGS_DEFAULTS if this is intentional.", key
            )
            return

        expected_type, _ = SETTINGS_DEFAULTS[key]
        value = _coerce(key, value, expected_type)

        if key in SECRET_KEYS:
            str_value = "" if value is None else str(value)
            if str_value:
                if _keyring_set(key, str_value):
                    with self._lock:
                        # Don't persist secrets to disk when keyring succeeds.
                        self._data[key] = ""
                        self._save()
                    return
                # keyring unavailable — fall through to plaintext write so the
                # app still works (with the same security properties as before
                # this change).
            else:
                _keyring_delete(key)

        with self._lock:
            self._data[key] = value
            self._save()

    def set_raw(self, key: str, value: Any) -> None:
        """
        Bypass schema validation — use only for keys that are dynamically
        generated at runtime (e.g. version strings, per-install IDs).
        Prefer set() in all other cases.
        """
        with self._lock:
            self._data[key] = value
            self._save()

    def all(self) -> dict:
        """Return a snapshot of all settings merged with defaults."""
        with self._lock:
            result = {k: v for k, (_, v) in SETTINGS_DEFAULTS.items()}
            result.update(self._data)
            return result

    def get_schema(self) -> dict:
        """
        Return the schema as {key: {"type": str, "default": value}}.
        Useful for frontend introspection.
        """
        out = {}
        for key, (expected_type, default_value) in SETTINGS_DEFAULTS.items():
            if isinstance(expected_type, tuple):
                type_name = "|".join(
                    t.__name__ for t in expected_type if t is not type(None)
                )
            else:
                type_name = expected_type.__name__
            out[key] = {"type": type_name, "default": default_value}
        return out
