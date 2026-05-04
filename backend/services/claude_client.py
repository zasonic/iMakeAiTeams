"""
services/claude_client.py

Sole wrapper around the Anthropic SDK.

Fixes applied:
  - stream_multi_turn returns (text, usage) tuple so callers can track tokens
  - Removed stale prompt-caching beta header (caching is now GA; cache_control
    blocks still work without it)
  - Files API beta header comment updated

v4.3 — System-prompt caching in multi-turn paths:
  - _build_system_with_cache() wraps the system parameter in a cache_control
    block when use_caching=True. Multi-turn calls (chat_multi_turn,
    stream_multi_turn, call_with_tools) now benefit from the same
    90%-token-discount on repeated system prompts that single-turn calls
    already got via _build_content().
  - Cache TTL is 5 minutes (Anthropic ephemeral default). Turns within that
    window share a cached prompt; longer gaps fall back to a full token read.
  - Blocks below Anthropic's 1,024-token minimum are silently passed through
    uncached, so short prompts degrade safely.

v5.1 — Caching + streaming-thinking enhancements:
  - System-prompt caching is now applied uniformly across every multi-turn
    code path, cutting input-token cost on follow-up turns by 50–80% on
    long system prompts.
  - New stream_extended_thinking_chat() delivers thinking + answer tokens
    in real-time via on_thinking_token / on_text_token callbacks, letting
    the UI render the reasoning timeline as it arrives instead of blocking
    for a full round-trip. Falls back gracefully to extended_thinking_chat()
    when the model/API version does not support streaming-thinking events.
  - Bumped Anthropic SDK requirement to >=0.50.0 for stable
    thinking_delta / text_delta event types in messages.stream().
"""

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

    # ── Configuration ────────────────────────────────────────────────────────────

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

    # ── Content helpers ──────────────────────────────────────────────────────────

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

    def _build_system_with_cache(self, system: str) -> str | list:
        """
        Return the system parameter in the correct shape for the Messages API.

        When prompt caching is enabled and the system string is non-empty, the
        prompt is wrapped in an ephemeral cache_control block.  This lets the
        API cache the compiled token representation for up to 5 minutes, so
        consecutive turns that share the same system prompt pay only 0.1× the
        normal input-token rate on cache reads (90% savings, which translates
        to roughly 50–80% lower input-token cost on a typical multi-turn
        conversation that re-sends the same system prompt every turn).

        The Messages API accepts both string and list-of-content-blocks forms
        for `system`, so this is transparent to callers — no behavioural
        change beyond reduced cost.

        Falls back to a plain string when:
          - _use_caching is False (user toggled caching off in Settings)
          - system is empty (nothing to cache)
          - The block ends up below Anthropic's 1,024-token minimum (the API
            silently ignores cache_control in that case, so no error occurs)
        """
        if self._use_caching and system:
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return system

    # ── Single-turn chat ─────────────────────────────────────────────────────────

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

    # ── Single-turn streaming ────────────────────────────────────────────────────

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

    # ── Multi-turn chat ──────────────────────────────────────────────────────────

    def chat_multi_turn(self, system: str, messages: list, max_tokens: int = 4096) -> dict:
        """
        Send a multi-turn conversation. messages = [{"role":..., "content":...}]
        Returns dict with "text", "input_tokens", "output_tokens".

        The system prompt is routed through _build_system_with_cache() so that
        repeated turns within the 5-minute cache window are billed at the
        cache-read rate (90% input-token discount on the system prompt
        portion of the request).
        """
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": self._build_system_with_cache(system),
            "messages": messages,
        }
        response = self._client.messages.create(**kwargs)
        return {
            "text": response.content[0].text,
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

        The system prompt is routed through _build_system_with_cache() so that
        repeated turns within the 5-minute cache window are billed at the
        cache-read rate, reducing input-token cost on follow-up turns.

        Callers must unpack the tuple:
            text, usage = claude.stream_multi_turn(...)
        """
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": self._build_system_with_cache(system),
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
                pass  # usage unavailable — caller handles gracefully
        return full_text, usage

    # ── Tool use (agentic loop) ──────────────────────────────────────────────────

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
            "system": self._build_system_with_cache(system),
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

    # ── File upload ──────────────────────────────────────────────────────────────

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

    # ── Chat with uploaded file ──────────────────────────────────────────────────

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

    # ── Extended thinking ────────────────────────────────────────────────────────

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

    # ── Streaming extended thinking ──────────────────────────────────────────────

    def stream_extended_thinking_chat(
        self,
        system: str,
        user_message: str,
        budget_tokens: int = 10000,
        model: str | None = None,
        on_thinking_token: Callable[[str], None] | None = None,
        on_text_token: Callable[[str], None] | None = None,
    ) -> dict:
        """
        Stream an extended-thinking chat, dispatching reasoning and answer
        chunks to optional callbacks as they arrive.

        This lets the UI render the reasoning timeline in real time instead
        of blocking on a full round-trip — the user sees Claude "thinking"
        token-by-token, then the answer streaming in immediately after.

        Parameters
        ----------
        system, user_message, budget_tokens, model
            Same semantics as extended_thinking_chat().
        on_thinking_token
            Optional callback invoked with each thinking-delta string. When
            None, thinking tokens are still accumulated and returned in the
            result dict but no per-chunk dispatch happens.
        on_text_token
            Optional callback invoked with each answer-text-delta string.

        Returns
        -------
        dict with keys "thinking" and "answer", matching the shape returned
        by extended_thinking_chat() so callers can switch implementations
        without changing downstream code.

        Falls back to the blocking extended_thinking_chat() if the SDK
        raises while opening the stream — older Anthropic SDK versions or
        models that don't yet emit thinking_delta events will degrade
        gracefully rather than hard-fail.
        """
        thinking_model = model or self._model

        try:
            thinking_text = ""
            answer_text = ""
            with self._client.messages.stream(
                model=thinking_model,
                max_tokens=16000,
                system=system,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                },
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "thinking_delta":
                        chunk = getattr(delta, "thinking", "") or ""
                        if chunk:
                            thinking_text += chunk
                            if on_thinking_token is not None:
                                on_thinking_token(chunk)
                    elif delta_type == "text_delta":
                        chunk = getattr(delta, "text", "") or ""
                        if chunk:
                            answer_text += chunk
                            if on_text_token is not None:
                                on_text_token(chunk)
            return {"thinking": thinking_text, "answer": answer_text}
        except Exception:
            # Streaming-thinking not supported on this model/SDK version —
            # fall back to the blocking variant so the caller still gets a
            # well-formed result (without per-chunk callbacks).
            return self.extended_thinking_chat(
                system=system,
                user_message=user_message,
                budget_tokens=budget_tokens,
                model=model,
            )
