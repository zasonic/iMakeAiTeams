"""
services/local_client.py

Unified client for Ollama and LM Studio.
Mirrors ClaudeClient's interface so the router can swap between them.

Fix applied:
  - All requests.post calls wrapped in explicit try/except with graceful
    fallback strings rather than letting KeyError / JSONDecodeError / network
    exceptions propagate unhandled to ChatOrchestrator.
"""

import json
import logging
import re
import requests
from core.settings import Settings
from services.llm_interface import LLMClient

# Phase 3: Qwen3-30B-A3B detection. Matches LM Studio's typical id forms
# such as "qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B-Instruct", "qwen3-30b-a3b-q4_k_m".
_QWEN3_30B_A3B_ID = re.compile(r"qwen3.*30b.*a3b", re.IGNORECASE)

log = logging.getLogger("iMakeAiTeams.local")

_FALLBACK = "[Local model unavailable — no response]"


class LocalVisionUnavailable(RuntimeError):
    """Raised when the active local model can't see images.

    The orchestrator catches this and surfaces a friendly error to the
    user that names a vision-capable model they could switch to. The
    family list comes from the ``vision_local_models`` setting.
    """

    def __init__(self, active_model: str, families: list[str]):
        self.active_model = active_model
        self.families = list(families)
        super().__init__(
            f"Local model '{active_model or '(none)'}' can't see images. "
            f"Switch to one of: {', '.join(self.families) or '(no families configured)'}"
        )


class LocalClient(LLMClient):
    def __init__(self, settings: Settings):
        self._settings = settings
        # Phase 9: BundledServer is wired in by core/api/__init__.py at startup.
        # Without it set, bundled-mode requests still try the LM Studio shape
        # at the persisted port (read from paths.bundled_server_port_file)
        # so that a sidecar restart doesn't lose the connection.
        self._bundled_server: object | None = None

    def attach_bundled_server(self, bundled_server: object) -> None:
        """Late-bind the BundledServer handle.

        Called by the API container after both services are constructed; lets
        bundled-mode requests resolve the live llama-server port without the
        LocalClient holding a forward reference at import time.
        """
        self._bundled_server = bundled_server

    def _bundled_port(self) -> int | None:
        """Return the live bundled-server port, or None if not running."""
        bs = self._bundled_server
        if bs is not None:
            try:
                if bs.is_running():
                    return bs.port()
            except Exception:  # noqa: BLE001
                pass
        # Fallback: read the persisted port file. Used when the wizard
        # started the server in a previous process and we just respawned.
        try:
            from core import paths as _paths  # noqa: PLC0415
            port_file = _paths.bundled_server_port_file()
            if port_file.exists():
                txt = port_file.read_text(encoding="utf-8").strip()
                if txt.isdigit():
                    return int(txt)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _url(self, backend: str | None = None) -> str:
        b = backend or self._effective_backend()
        if b == "ollama":
            return self._settings.get("ollama_url", "http://localhost:11434")
        if b == "bundled":
            port = self._bundled_port()
            if port is None:
                # Force callers down their existing failure path; an
                # unreachable URL surfaces the same way as a stopped backend.
                return "http://127.0.0.1:0"
            return f"http://127.0.0.1:{port}"
        return self._settings.get("lm_studio_url", "http://localhost:1234")

    def _effective_backend(self) -> str:
        """Resolve the active backend name from local_backend_mode + legacy
        default_local_backend.

        ``local_backend_mode == "auto"`` (or unset) preserves the historical
        single-knob behaviour: read ``default_local_backend``. Any explicit
        mode wins over auto so the user can pin a backend without touching
        the legacy setting.
        """
        mode = (self._settings.get("local_backend_mode", "auto") or "auto").strip()
        if mode in ("ollama", "lm_studio", "bundled"):
            return mode
        return self._settings.get("default_local_backend", "ollama") or "ollama"

    def _backend(self, backend: str | None = None) -> str:
        return backend or self._effective_backend()

    def is_available(self, backend: str | None = None) -> bool:
        """Check if a local model backend is reachable."""
        try:
            b = self._backend(backend)
            url = self._url(b)
            endpoint = "/api/tags" if b == "ollama" else "/v1/models"
            return requests.get(url + endpoint, timeout=2).status_code == 200
        except Exception:
            return False

    def list_models(self, backend: str | None = None) -> list[str]:
        """Return available model names."""
        b = self._backend(backend)
        url = self._url(b)
        try:
            if b == "ollama":
                r = requests.get(url + "/api/tags", timeout=5)
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
            else:
                r = requests.get(url + "/v1/models", timeout=5)
                r.raise_for_status()
                return [m["id"] for m in r.json().get("data", [])]
        except Exception as exc:
            log.warning(f"list_models failed for backend '{b}': {exc}")
            return []

    def list_local_models(self) -> list[dict]:
        """Return installed local models across both Ollama and LM Studio.

        Each row carries enough info for a model browser UI:
            {"id": str, "size_bytes": int|None, "context_length": int|None,
             "quantization": str|None, "backend": "ollama"|"lm_studio",
             "loaded": bool}

        Backends that are unreachable contribute zero rows; failures are
        logged at warning level and never raise.
        """
        active = self._settings.get("default_local_model", "") or ""
        out: list[dict] = []

        ollama_url = self._settings.get("ollama_url", "http://localhost:11434")
        try:
            r = requests.get(ollama_url + "/api/tags", timeout=5)
            r.raise_for_status()
            for m in r.json().get("models", []) or []:
                details = m.get("details") or {}
                mid = m.get("name", "")
                out.append({
                    "id":             mid,
                    "size_bytes":     m.get("size"),
                    "context_length": None,
                    "quantization":   details.get("quantization_level") or None,
                    "backend":        "ollama",
                    "loaded":         bool(mid) and mid == active,
                })
        except Exception as exc:
            log.warning("list_local_models: ollama probe failed: %s", exc)

        lm_url = self._settings.get("lm_studio_url", "http://localhost:1234")
        try:
            r = requests.get(lm_url + "/v1/models", timeout=5)
            r.raise_for_status()
            for m in r.json().get("data", []) or []:
                mid = m.get("id", "")
                ctx = m.get("context_length") or m.get("max_context_length")
                out.append({
                    "id":             mid,
                    "size_bytes":     m.get("size") if isinstance(m.get("size"), int) else None,
                    "context_length": ctx if isinstance(ctx, int) else None,
                    "quantization":   m.get("quantization") or None,
                    "backend":        "lm_studio",
                    "loaded":         bool(mid) and mid == active,
                })
        except Exception as exc:
            log.warning("list_local_models: lm_studio probe failed: %s", exc)

        return out

    def list_models_detailed(self, backend: str | None = None) -> list[dict]:
        """Return available models as list of {id, raw} dicts.

        Phase 3 needs structured model info for Qwen3 detection while keeping
        ``list_models()`` (string list) backward-compatible for existing
        callers that just need names.
        """
        b = self._backend(backend)
        url = self._url(b)
        try:
            if b == "ollama":
                r = requests.get(url + "/api/tags", timeout=5)
                r.raise_for_status()
                return [{"id": m["name"], "raw": m} for m in r.json().get("models", [])]
            r = requests.get(url + "/v1/models", timeout=5)
            r.raise_for_status()
            return [{"id": m["id"], "raw": m} for m in r.json().get("data", [])]
        except Exception as exc:
            log.warning(f"list_models_detailed failed for backend '{b}': {exc}")
            return []

    def detect_qwen3_30b_a3b(self, backend: str | None = None) -> dict:
        """Probe LM Studio (or Ollama) for a Qwen3-30B-A3B GGUF.

        Returns ``{"detected": bool, "model_id": str, "fallback_reason": str}``.

        - On hit: ``model_id`` is the matching id; ``fallback_reason`` is empty.
        - On miss with other models present: ``model_id`` is the first available
          model id; ``fallback_reason`` is a plain-English notice the UI can
          display verbatim.
        - On no backend reachable: empty ``model_id`` and a plain-English reason.

        Whenever a non-empty ``model_id`` is identified (detected or fallback)
        this fires the Phase 5 behavior-drift canary in a daemon thread.
        """
        models = self.list_models_detailed(backend)
        for m in models:
            if _QWEN3_30B_A3B_ID.search(str(m.get("id", ""))):
                self.signal_model_loaded(m["id"])
                return {
                    "detected":        True,
                    "model_id":        m["id"],
                    "fallback_reason": "",
                }
        if not models:
            return {
                "detected":        False,
                "model_id":        "",
                "fallback_reason": (
                    "No local model server is reachable. Start LM Studio (or "
                    "Ollama), load a model, then come back to this screen."
                ),
            }
        fallback = models[0]["id"]
        self.signal_model_loaded(fallback)
        return {
            "detected":        False,
            "model_id":        fallback,
            "fallback_reason": (
                f"Qwen3-30B-A3B not detected — falling back to '{fallback}'. "
                "Hybrid thinking will use a single budget cap; install a "
                "Qwen3-30B-A3B GGUF in LM Studio for the recommended setup."
            ),
        }

    def is_vision_model(self, model_id: str) -> bool:
        """Return True when ``model_id`` matches a vision-capable family.

        Compares the id against the ``vision_local_models`` setting using
        a case-insensitive prefix match. Empty ids and empty family lists
        return False.
        """
        if not model_id:
            return False
        families = self._settings.get("vision_local_models", []) or []
        if not isinstance(families, list):
            return False
        mid = str(model_id).strip().lower()
        for fam in families:
            if not fam:
                continue
            if mid.startswith(str(fam).strip().lower()):
                return True
        return False

    def chat_with_images(
        self,
        system: str,
        messages: list,
        images_b64: list[str],
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """Single-shot chat that attaches base64 images to the last user turn.

        Uses Ollama's ``/api/chat`` ``images`` field (a list of base64
        strings on the user message). Raises ``LocalVisionUnavailable``
        when the active model isn't vision-capable.

        ``messages`` is a list of {"role", "content"} dicts in the same
        shape as ``chat_multi_turn``. The images are attached to the last
        user message; if the list is empty or has no user message, this
        falls through to a plain text chat.
        """
        active = model or self._settings.get("default_local_model", "")
        families = self._settings.get("vision_local_models", []) or []
        if not self.is_vision_model(active):
            raise LocalVisionUnavailable(active, list(families))

        b = self._backend()
        url = self._url(b)
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        # Attach images to the last user message. Ollama's /api/chat
        # accepts ``images: [<base64>, ...]`` on a message.
        if images_b64 and msgs:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    # Strip data-uri prefixes if present.
                    cleaned = [
                        s.split(",", 1)[1] if isinstance(s, str) and s.startswith("data:")
                        else s
                        for s in images_b64
                    ]
                    msgs[i] = {**msgs[i], "images": cleaned}
                    break

        payload = {
            "model": active, "messages": msgs,
            "max_tokens": max_tokens, "stream": False,
        }
        try:
            if b == "ollama":
                r = requests.post(url + "/api/chat", json=payload, timeout=120)
                r.raise_for_status()
                return r.json().get("message", {}).get("content", _FALLBACK)
            # LM Studio / bundled don't speak Ollama's image protocol.
            # Fall back to a plain chat so the user gets *something*.
            log.warning(
                "chat_with_images: backend %s does not support images; "
                "falling back to text-only.", b,
            )
            r = requests.post(url + "/v1/chat/completions", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            log.warning("LocalClient.chat_with_images failed: %s", exc)
            return _FALLBACK

    def signal_model_loaded(self, model_id: str) -> None:
        """Notify the behavior-drift canary that ``model_id`` has been loaded.

        Best-effort: any failure (including a missing canary module on a
        partial install) is swallowed at warning level so the local-model
        load path can never raise into the request handler.
        """
        if not model_id:
            return
        try:
            from services import model_canary  # noqa: PLC0415
            model_canary.signal_model_loaded(self, model_id, self._settings)
        except Exception as exc:
            log.warning("signal_model_loaded(%s) failed: %s", model_id, exc)

    def chat(self, system: str, user_message: str, model: str | None = None,
             max_tokens: int = 2048) -> str:
        """Single-turn chat. Returns response text, or a fallback string on error."""
        b = self._backend()
        model = model or self._settings.get("default_local_model", "")
        url = self._url(b)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        payload = {"model": model, "messages": messages,
                   "max_tokens": max_tokens, "stream": False}
        try:
            if b == "ollama":
                r = requests.post(url + "/api/chat", json=payload, timeout=120)
                r.raise_for_status()
                return r.json().get("message", {}).get("content", _FALLBACK)
            else:
                r = requests.post(url + "/v1/chat/completions", json=payload, timeout=120)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            log.warning(f"LocalClient.chat failed: {exc}")
            return _FALLBACK

    def chat_multi_turn(self, system: str, messages: list, model: str | None = None,
                        max_tokens: int = 2048) -> str:
        """Multi-turn chat. Returns response text, or a fallback string on error."""
        b = self._backend()
        model = model or self._settings.get("default_local_model", "")
        url = self._url(b)
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        payload = {"model": model, "messages": msgs,
                   "max_tokens": max_tokens, "stream": False}
        try:
            if b == "ollama":
                r = requests.post(url + "/api/chat", json=payload, timeout=120)
                r.raise_for_status()
                return r.json().get("message", {}).get("content", _FALLBACK)
            else:
                r = requests.post(url + "/v1/chat/completions", json=payload, timeout=120)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            log.warning(f"LocalClient.chat_multi_turn failed: {exc}")
            return _FALLBACK

    def stream_multi_turn(self, system: str, messages: list, on_token,
                          model: str | None = None,
                          max_tokens: int = 2048) -> tuple[str, object]:
        """Streaming multi-turn chat. Calls on_token per chunk.

        Returns ``(full_text, usage)`` to mirror ``ClaudeClient.stream_multi_turn``;
        local backends do not report usage, so the second element is always None.
        Falls back to non-streaming on any error rather than crashing.
        """
        b = self._backend()
        model = model or self._settings.get("default_local_model", "")
        url = self._url(b)
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        payload = {"model": model, "messages": msgs,
                   "max_tokens": max_tokens, "stream": True}
        endpoint = "/api/chat" if b == "ollama" else "/v1/chat/completions"
        full = ""
        try:
            with requests.post(url + endpoint, json=payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    raw = line.decode("utf-8")
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        text = (
                            chunk.get("message", {}).get("content", "")
                            if b == "ollama"
                            else chunk["choices"][0]["delta"].get("content", "")
                        )
                        if text:
                            on_token(text)
                            full += text
                    except Exception:
                        continue
        except Exception as exc:
            log.warning(f"LocalClient.stream_multi_turn failed: {exc}. Falling back to non-streaming.")
            # Fallback: try non-streaming so the user still gets a response
            full = self.chat_multi_turn(system, messages, model=model, max_tokens=max_tokens)
            if full and full != _FALLBACK:
                on_token(full)
        return full, None

    # ── LLMClient interface ─────────────────────────────────────────────────

    def chat_unified(self, system, messages, max_tokens=4096):
        text = self.chat_multi_turn(system, messages, max_tokens=max_tokens)
        return {"text": text or "", "input_tokens": 0, "output_tokens": 0}

    def stream_unified(self, system, messages, on_token, max_tokens=4096):
        text, _usage = self.stream_multi_turn(
            system, messages, on_token, max_tokens=max_tokens,
        )
        return {"text": text or "", "input_tokens": 0, "output_tokens": 0}

    def client_name(self) -> str:
        return self._settings.get("default_local_model", "local")
