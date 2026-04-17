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
import requests
from core.settings import Settings

log = logging.getLogger("MyAIEnv.local")

_FALLBACK = "[Local model unavailable — no response]"


class LocalClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._avail_cache: dict[str, tuple[bool, float]] = {}
        self._avail_ttl = 10.0  # seconds

    def _url(self, backend: str | None = None) -> str:
        b = backend or self._settings.get("default_local_backend", "ollama")
        if b == "ollama":
            return self._settings.get("ollama_url", "http://localhost:11434")
        return self._settings.get("lm_studio_url", "http://localhost:1234")

    def _backend(self, backend: str | None = None) -> str:
        return backend or self._settings.get("default_local_backend", "ollama")

    def is_available(self, backend: str | None = None) -> bool:
        """Check if a local model backend is reachable, with 10-second TTL cache."""
        import time
        b = self._backend(backend)
        cached = self._avail_cache.get(b)
        now = time.monotonic()
        if cached and (now - cached[1]) < self._avail_ttl:
            return cached[0]
        try:
            url = self._url(b)
            endpoint = "/api/tags" if b == "ollama" else "/v1/models"
            result = requests.get(url + endpoint, timeout=2).status_code == 200
        except Exception:
            result = False
        self._avail_cache[b] = (result, now)
        return result

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

    def _build_payload(self, model: str, messages: list, max_tokens: int,
                        stream: bool, backend: str) -> dict:
        payload: dict = {"model": model, "messages": messages, "stream": stream}
        if backend == "ollama":
            payload["options"] = {"num_predict": max_tokens}
        else:
            payload["max_tokens"] = max_tokens
        return payload

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
        payload = self._build_payload(model, messages, max_tokens, False, b)
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
        payload = self._build_payload(model, msgs, max_tokens, False, b)
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
        payload = self._build_payload(model, msgs, max_tokens, True, b)
        endpoint = "/api/chat" if b == "ollama" else "/v1/chat/completions"
        chunks: list[str] = []
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
                            chunks.append(text)
                    except Exception:
                        continue
        except Exception as exc:
            log.warning(f"LocalClient.stream_multi_turn failed: {exc}. Falling back to non-streaming.")
            full = self.chat_multi_turn(system, messages, model=model, max_tokens=max_tokens)
            if full and full != _FALLBACK:
                on_token(full)
            return full
        return "".join(chunks)
