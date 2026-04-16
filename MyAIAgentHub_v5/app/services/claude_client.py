"""
services/claude_client.py

Sole wrapper around the Anthropic SDK.

Fixes applied:
  - stream_multi_turn returns (text, usage) tuple so callers can track tokens
  - Removed stale prompt-caching beta header (caching is now GA; cache_control
    blocks still work without it)
  - Files API beta header comment updated

Enhancements (research-driven, April 2026):
  - Prompt caching on system messages: cache_control ephemeral on static system
    prompts cuts costs 60-80% on multi-turn sessions (Anthropic docs recommend
    placing cacheable content first, dynamic content last).
  - Retry with exponential backoff: API transient failures (rate limits, 500s)
    are retried automatically via tenacity. Non-technical users see fewer
    "connection error" messages.
  - Extended thinking budget is now caller-configurable (was hardcoded to 5000).
  - Friendly error classification: rate limits, auth, overload get distinct
    messages so users know what to do.
"""

import base64
import logging
from pathlib import Path
from typing import Callable

from anthropic import Anthropic, APIError, RateLimitError, AuthenticationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

log = logging.getLogger("MyAIEnv.claude_client")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIError) and getattr(exc, "status_code", 0) >= 500:
        return True
    return False


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "Invalid API key. Check your Anthropic API key in Settings."
    if isinstance(exc, RateLimitError):
        return "Rate limited by Anthropic. Wait a moment and try again, or reduce message frequency."
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", "")
        if code == 529:
            return "Anthropic API is temporarily overloaded. Try again in a few seconds."
        return f"Anthropic API error ({code}). Try again shortly."
    return f"Connection error: {exc}"


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

    # ── Configuration ─────────────────────────────────────────────────────────

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

    # ── Content helpers ───────────────────────────────────────────────────────

    def _build_system(self, system: str) -> list:
        """
        Build the system parameter as a list of content blocks.
        When caching is enabled and the system prompt is long enough (>=1024
        tokens ~= 4096 chars), apply cache_control so Anthropic caches it.
        Cache hits cost 90% less and reduce latency up to 85%.
        """
        if self._use_caching and len(system) >= 2048:
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return system

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

    # ── Single-turn chat ──────────────────────────────────────────────────────

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
        return response.content[0].text

    # ── Single-turn streaming ─────────────────────────────────────────────────

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
        full_text = ""
        with self._client.messages.stream(**kwargs) as stream:
            for token in stream.text_stream:
                on_token(token)
                full_text += token
        return full_text

    # ── Multi-turn chat ───────────────────────────────────────────────────────

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def chat_multi_turn(self, system: str, messages: list, max_tokens: int = 4096,
                        model: str | None = None) -> dict:
        """
        Send a multi-turn conversation. messages = [{"role":..., "content":...}]
        Returns dict with "text", "input_tokens", "output_tokens".
        Accepts optional model override for tiered routing.
        """
        try:
            kwargs = {
                "model": model or self._model,
                "max_tokens": max_tokens,
                "system": self._build_system(system),
                "messages": messages,
            }
            response = self._client.messages.create(**kwargs)
            return {
                "text": response.content[0].text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            }
        except (RateLimitError, APIError) as exc:
            if not _is_retryable(exc):
                raise RuntimeError(_friendly_error(exc)) from exc
            raise

    def stream_multi_turn(
        self,
        system: str,
        messages: list,
        on_token: Callable[[str], None],
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> tuple[str, object]:
        """
        Stream a multi-turn conversation with per-token callback.
        Returns (full_response_text, usage) where usage has .input_tokens
        and .output_tokens attributes (or None if unavailable).
        Accepts optional model override for tiered routing.

        Callers must unpack the tuple:
            text, usage = claude.stream_multi_turn(...)
        """
        try:
            kwargs = {
                "model": model or self._model,
                "max_tokens": max_tokens,
                "system": self._build_system(system),
                "messages": messages,
            }
            full_text = ""
            usage = None
            with self._client.messages.stream(**kwargs) as stream:
                for token in stream.text_stream:
                    on_token(token)
                    full_text += token
                try:
                    usage = stream.get_final_usage()
                except Exception:
                    pass
            return full_text, usage
        except (AuthenticationError, RateLimitError, APIError) as exc:
            raise RuntimeError(_friendly_error(exc)) from exc

    # ── Tool use (agentic loop) ─────────────────────────────────────────────

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
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
        {"type":"tool_use","id":...,"name":...,"input":...}.
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

    # ── File upload ───────────────────────────────────────────────────────────

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

    # ── Chat with uploaded file ───────────────────────────────────────────────

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
        return response.content[0].text

    # ── Extended thinking ─────────────────────────────────────────────────────

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(2),
           wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
    def extended_thinking_chat(
        self,
        system: str,
        user_message: str,
        budget_tokens: int = 10000,
        model: str | None = None,
    ) -> dict:
        """
        Run a chat with extended thinking enabled.
        budget_tokens is now configurable via settings (default 10000).
        Returns a dict with keys "thinking" and "answer".
        """
        thinking_model = model or self._model
        try:
            response = self._client.messages.create(
                model=thinking_model,
                max_tokens=max(16000, budget_tokens + 6000),
                system=self._build_system(system),
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
        except (AuthenticationError, RateLimitError, APIError) as exc:
            raise RuntimeError(_friendly_error(exc)) from exc
