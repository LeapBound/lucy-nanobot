"""LLM provider abstraction module."""

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.claude_agent_provider import ClaudeAgentProvider

try:
    from nanobot.providers.litellm_provider import LiteLLMProvider
except Exception:  # pragma: no cover - optional dependency import guard
    LiteLLMProvider = None  # type: ignore[assignment]

__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider", "ClaudeAgentProvider"]
