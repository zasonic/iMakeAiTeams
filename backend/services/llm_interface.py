"""
services/llm_interface.py — Unified LLM client interface.

Both ClaudeClient and LocalClient implement this so HubRouter.invoke()
can call either without branching on backend type.
"""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Minimal interface for LLM invocation. Both ClaudeClient and LocalClient
    implement this. HubRouter calls only these methods."""

    @abstractmethod
    def chat_unified(self, system: str, messages: list,
                     max_tokens: int = 4096) -> dict:
        """Non-streaming chat.
        Returns {"text": str, "input_tokens": int, "output_tokens": int}
        """

    @abstractmethod
    def stream_unified(self, system: str, messages: list, on_token,
                       max_tokens: int = 4096) -> dict:
        """Streaming chat. Calls on_token(chunk) during generation.
        Returns {"text": str, "input_tokens": int, "output_tokens": int}
        """

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def client_name(self) -> str: ...
