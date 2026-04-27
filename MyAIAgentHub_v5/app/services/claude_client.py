"""
services/claude_client.py

Sole wrapper around the Anthropic SDK.

Fixes applied:
  - stream_multi_turn returns (text, usage) tuple so callers can track tokens
  - Removed stale prompt-caching beta header (caching is now GA; cache_control
    blocks still work without it)
  - Files API beta header comment updated

v4.3 — System-prompt caching in multi-turn paths:
  - _build_system() wraps the system parameter in a cache_control block when
    use_caching=True. Multi-turn calls (chat_multi_turn, stream_multi_turn,
    call_with_tools) now benefit from the same 90%-token-discount on repeated
    system prompts that single-turn calls already got via _build_content().
  - Cache TTL is 5 minutes (Anthropic ephemeral default). Turns within that
    window share a cached prompt; longer gaps fall back to a full token read.
  - Blocks below Anthropic's 1,024-token minimum are silently passed through
    uncached, so short prompts degrade safely.
"""

import base64
from pathlib import Path
from typing import Callable

from anthropic import Anthropic


class ClaudeClient:
    """
    Wrapper for the Anthropic Messages API.

    Handles:
      - Plain chat with optional prompt caching
      - Streaming chat with per-token callback + usage tracking
      - Multi-turn conversation (full history)
      - Streaming multi-turn with usage tracking
      - Chat with a previously uploaded file (document source)
      - Extended thinking chat
    """

    def __init__(self, api_key: str, model: str, use_caching: bool = True):
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._use_caching = use_caching
        self._file_cache: dict[str, str] = {}  # file_path -> file_id

    # ── Configuration ────────────────────────────────────────────────────────────────────────────

    def update_config(
        self,
        api_key: str | None = None,
        model: str | None = None,
        use_caching: bool | None = None,
    ) -> None:
        if api_key is not None and api_key != getattr(self._client, "api_key", None):
            self._client = Anthropic(api_key=api_key)
        if model is not None:
            self._model = model
        if use_caching is not None:
            self._use_caching = use_caching

    # ── Content helpers ────────────────────────────────────────────────────────────────────────

    def _build_content(self, project_summary: str, user_message: str) -> list:
        """
        Build the content list for a standard chat request.
        If project_summary is non-empty and caching is on, it is added as a
        cached text block before the user message.
        """
        content: list = []
        if project_summary and self._use_caching:
            content.append({
                "type": "text",
                "text": project_summary,
                "cache_control": {"type": "ephemeral"},
            })
        content.append({"type": "text", "text": user_message})
        return content

    def _build_system(self, system: str) -> str | list:
        """
        Return the system parameter in the correct shape for the Messages API.

        When prompt caching is enabled and the system string is non-empty, the
        prompt is wrapped in an ephemeral cache_control block.  This lets the
        API cache the compiled token representation for up to 5 minutes, so
        consecutive turns that share the same system prompt pay only 0.1× the
        normal input-token rate on cache reads (90% savings).

        Falls back to a plain string when:
          - _use_caching is False (user toggled caching off in Settings)
          - system is empty (nothing to cache)
          - The block ends up below Anthropic's 1,024-token minimum (the API
            silently ignores cache_control in that case, so no error occurs)
        """
        if self._use_caching and system:
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return system

    # ── Single-turn chat ──────────────────────────────────────────────────────────────────────────

    def chat(self, system: str, project_summary: str, user_message: str,
             max_tokens: int = 4096) -> str:
        """Send a single-turn chat and return the response as a plain string."""
        content = self._build_content(project_summary, user_message)
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        response = self._client.messages.create(**kwargs)
        if not response.content:
            return ""
        return response.content[0].text

    # ── Single-turn streaming ─────────────────────────────────────────────────────────────────────

    def stream_chat(
        self,
        system: str,
        project_summary: str,
        user_message: str,
        on_token: Callable[[str], None],
        max_tokens: int = 4096,
    ) -> str:
        """
        Stream a chat response, calling on_token for each text token.
        Returns the full concatenated response string when done.
        """
        content = self._build_content(project_summary, user_message)
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        tokens: list[str] = []
        with self._client.messages.stream(**kwargs) as stream:
            for token in stream.text_stream:
                on_token(token)
                tokens.append(token)
        return "".join(tokens)

    # ── Multi-turn chat ──────────────────────────────────────────────────────────────────────────

    def chat_multi_turn(self, system: str, messages: list, max_tokens: int = 4096) -> dict:
        """
        Send a multi-turn conversation. messages = [{"role":..., "content":...}]
        Returns dict with "text", "input_tokens", "output_tokens".
        """
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": self._build_system(system),
            "messages": messages,
        }
        response = self._client.messages.create(**kwargs)
        return {
            "text": response.content[0].text if response.content else "",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    def stream_multi_turn(
        self,
        system: str,
        messages: list,
        on_token: Callable[[str], None],
        max_tokens: int = 4096,
    ) -> tuple[str, object]:
        """
        Stream a multi-turn conversation with per-token callback.
        Returns (full_response_text, usage) where usage has .input_tokens
        and .output_tokens attributes (or None if unavailable).

        Callers must unpack the tuple:
            text, usage = claude.stream_multi_turn(...)
        """
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": self._build_system(system),
            "messages": messages,
        }
        tokens: list[str] = []
        usage = None
        with self._client.messages.stream(**kwargs) as stream:
            for token in stream.text_stream:
                on_token(token)
                tokens.append(token)
            try:
                usage = stream.get_final_usage()
            except Exception:
                pass  # usage unavailable — caller handles gracefully
        return "".join(tokens), usage

    # ── Tool use (agentic loop) ─────────────────────────────────────────────────────

    def call_with_tools(
        self,
        system: str,
        messages: list,
        tools: list,
        max_tokens: int = 8192,
    ) -> dict:
        """
        Call the Messages API with tool definitions.
        Returns a dict matching the Anthropic response shape:
          {"content": [...], "stop_reason": "end_turn"|"tool_use", ...}
        Each content block is {"type":"text","text":...} or
        {"type":"tool_use","id":..."name":..."input":...}.
        """
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": self._build_system(system),
            "messages": messages,
            "tools": tools,
        }
        response = self._client.messages.create(**kwargs)
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return {
            "content": content,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    # ── File upload ──────────────────────────────────────────────────────────────────────────────

    def upload_file(self, file_path: Path, mime_type: str) -> str:
        """
        Upload a file to the Anthropic Files API.
        Returns the cached file_id if the same path was uploaded this session.
        """
        key = str(file_path)
        if key in self._file_cache:
            return self._file_cache[key]
        with open(file_path, "rb") as fh:
            result = self._client.beta.files.upload(
                file=(file_path.name, fh, mime_type),
            )
        file_id = result.id
        self._file_cache[key] = file_id
        return file_id

    # ── Chat with uploaded file ───────────────────────────────────────────────────────────────────

    def chat_with_file(self, system: str, file_id: str, user_message: str) -> str:
        """
        Send a chat that references a previously uploaded file by its file_id.
        Uses the Files API beta header — verify header date against Anthropic docs
        if this feature stops working after an API update.
        """
        content = [
            {"type": "document", "source": {"type": "file", "file_id": file_id}},
            {"type": "text", "text": user_message},
        ]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": content}],
            extra_headers={"anthropic-beta": "files-api-2025-04-14"},
        )
        if not response.content:
            return ""
        return response.content[0].text

    # ── Extended thinking ────────────────────────────────────────────────────────────────────────

    def extended_thinking_chat(
        self,
        system: str,
        user_message: str,
        budget_tokens: int = 10000,
        model: str | None = None,
    ) -> dict:
        """
        Run a chat with extended thinking enabled.
        Returns a dict with keys "thinking" and "answer".
        """
        thinking_model = model or self._model
        response = self._client.messages.create(
            model=thinking_model,
            max_tokens=16000,
            system=system,
            thinking={
                "type": "enabled",
                "budget_tokens": budget_tokens,
            },
            messages=[{"role": "user", "content": user_message}],
        )
        thinking_text = ""
        answer_text = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                answer_text = block.text
        return {"thinking": thinking_text, "answer": answer_text}
