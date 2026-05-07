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

# Phase 3: Qwen3-30B-A3B detection. Matches LM Studio's typical id forms
# such as "qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B-Instruct", "qwen3-30b-a3b-q4_k_m".
_QWEN3_30B_A3B_ID = re.compile(r"qwen3.*30b.*a3b", re.IGNORECASE)

log = logging.getLogger("MyAIEnv.local")

_FALLBACK = "[Local model unavailable — no response]"


class LocalClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    def _url(self, backend: str | None = None) -> str:
        b = backend or self._settings.get("default_local_backend", "ollama")
        if b == "ollama":
            return self._settings.get("ollama_url", "http://localhost:11434")
        return self._settings.get("lm_studio_url", "http://localhost:1234")

    def _backend(self, backend: str | None = None) -> str:
        return backend or self._settings.get("default_local_backend", "ollama")

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
        """
        models = self.list_models_detailed(backend)
        for m in models:
            if _QWEN3_30B_A3B_ID.search(str(m.get("id", ""))):
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
        return {
            "detected":        False,
            "model_id":        fallback,
            "fallback_reason": (
                f"Qwen3-30B-A3B not detected — falling back to '{fallback}'. "
                "Hybrid thinking will use a single budget cap; install a "
                "Qwen3-30B-A3B GGUF in LM Studio for the recommended setup."
            ),
        }

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
                          model: str | None = None, max_tokens: int = 2048) -> str:
        """Streaming multi-turn chat. Calls on_token per chunk. Returns full text.
        Falls back to non-streaming on any error rather than crashing."""
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
        return full
