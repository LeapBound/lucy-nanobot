"""Claude Agent SDK provider implementation."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, is_dataclass
from typing import Any

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

try:
    from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, query as sdk_query

    _CLAUDE_SDK_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via dependency absence
    AgentDefinition = Any  # type: ignore[assignment]
    ClaudeAgentOptions = Any  # type: ignore[assignment]
    sdk_query = None
    _CLAUDE_SDK_IMPORT_ERROR = exc


_TOOL_RESULT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["content", "tool_calls"],
        "additionalProperties": False,
    },
}


class ClaudeAgentProvider(LLMProvider):
    """
    LLM provider backed by Claude Agent SDK.

    This adapter keeps nanobot's existing `AgentLoop` unchanged by converting:
    - nanobot/OpenAI-style tools -> SDK prompt schema
    - SDK assistant/tool-use messages -> `LLMResponse` + `ToolCallRequest`
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "claude-sonnet-4-5",
        agents: dict[str, Any] | None = None,
        permission_mode: str = "default",
        allowed_tools: list[str] | None = None,
        cwd: str | None = None,
        cli_path: str | None = None,
        query_func: Callable[..., AsyncIterator[Any]] | None = None,
    ):
        """
        Initialize Claude Agent SDK provider.

        Args:
            api_key: Anthropic API key (optional if Claude Code auth already exists).
            api_base: Reserved for interface compatibility; currently unused by SDK.
            default_model: Default model identifier.
            agents: Optional custom agent definitions.
            permission_mode: Claude Agent SDK permission mode.
            allowed_tools: Optional SDK allow-list for built-in tools.
            cwd: Optional working directory passed to SDK runtime.
            cli_path: Optional custom path for Claude Code CLI binary.
            query_func: Optional injected query function (used by tests).
        """
        super().__init__(api_key=api_key, api_base=api_base)

        if query_func is None and _CLAUDE_SDK_IMPORT_ERROR is not None:
            raise RuntimeError(
                "claude-agent-sdk is not installed. "
                "Install it with: pip install claude-agent-sdk"
            ) from _CLAUDE_SDK_IMPORT_ERROR

        self.default_model = default_model
        self.permission_mode = permission_mode
        self.allowed_tools = allowed_tools or []
        self.cwd = cwd
        self.cli_path = cli_path
        self._query_func = query_func or sdk_query
        self._agents = self._normalize_agents(agents)

        if api_key:
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat request via Claude Agent SDK.

        Args:
            messages: Conversation messages in nanobot/OpenAI format.
            tools: Optional tools in OpenAI function schema format.
            model: Optional model override.
            max_tokens: Maximum output tokens hint (best-effort in adapter).
            temperature: Sampling temperature hint (best-effort in adapter).

        Returns:
            Standardized `LLMResponse` for nanobot agent loop.
        """
        if self._query_func is None:
            return LLMResponse(
                content="Error calling Claude Agent SDK: query function is unavailable",
                finish_reason="error",
            )

        try:
            system_prompt, prompt = self._build_prompt(messages, tools or [])
            options = self._build_options(
                model=model or self.default_model,
                system_prompt=system_prompt,
                tools=tools or [],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            text_fragments: list[str] = []
            tool_calls: list[ToolCallRequest] = []
            usage: dict[str, int] = {}
            finish_reason = "stop"
            structured_output: dict[str, Any] | None = None

            async for sdk_message in self._query_func(prompt=prompt, options=options):
                message_kind = type(sdk_message).__name__

                if message_kind.endswith("AssistantMessage"):
                    content = getattr(sdk_message, "content", [])
                    for block in content:
                        block_kind = type(block).__name__
                        if block_kind.endswith("TextBlock"):
                            block_text = getattr(block, "text", "")
                            if block_text:
                                text_fragments.append(str(block_text))
                        elif block_kind.endswith("ToolUseBlock"):
                            call_id = str(getattr(block, "id", "") or "")
                            call_name = str(getattr(block, "name", "") or "")
                            call_input = getattr(block, "input", {})
                            tool_calls.append(
                                ToolCallRequest(
                                    id=call_id or f"call_{len(tool_calls) + 1}",
                                    name=self._normalize_tool_name(call_name),
                                    arguments=self._coerce_tool_arguments(call_input),
                                )
                            )

                elif message_kind.endswith("ResultMessage"):
                    is_error = bool(getattr(sdk_message, "is_error", False))
                    if is_error:
                        finish_reason = "error"

                    raw_usage = getattr(sdk_message, "usage", None)
                    usage = self._convert_usage(raw_usage)

                    raw_structured = getattr(sdk_message, "structured_output", None)
                    if isinstance(raw_structured, dict):
                        structured_output = raw_structured

                    result_text = getattr(sdk_message, "result", None)
                    if result_text and not text_fragments:
                        text_fragments.append(str(result_text))

            if not tool_calls and structured_output:
                structured_calls = self._extract_structured_tool_calls(structured_output)
                tool_calls.extend(structured_calls)

            content = "\n".join(fragment.strip() for fragment in text_fragments if fragment.strip())

            if not tool_calls and tools:
                parsed = self._parse_tool_payload_from_text(content)
                if parsed:
                    content = parsed.get("content") or content
                    tool_calls.extend(self._extract_structured_tool_calls(parsed))

            if tool_calls:
                finish_reason = "tool_calls" if finish_reason != "error" else finish_reason

            return LLMResponse(
                content=content or None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )
        except Exception as exc:  # pragma: no cover - network/SDK runtime failures
            return LLMResponse(
                content=f"Error calling Claude Agent SDK: {exc}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        """Get provider default model."""
        return self.default_model

    def _build_options(
        self,
        model: str,
        system_prompt: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Any:
        """Build Claude Agent SDK options for a query call."""
        sdk_model = self._normalize_model(model)
        extra_args: dict[str, str | None] = {
            "max-output-tokens": str(max_tokens),
            "temperature": str(temperature),
        }

        options_kwargs: dict[str, Any] = {
            "model": sdk_model,
            "permission_mode": self.permission_mode,
            "allowed_tools": self.allowed_tools,
            "agents": self._agents,
            "system_prompt": system_prompt or None,
            "cwd": self.cwd,
            "cli_path": self.cli_path,
            "extra_args": extra_args,
            "max_turns": 1,
        }

        if tools:
            options_kwargs["output_format"] = _TOOL_RESULT_SCHEMA

        # Remove unset values to keep SDK payload clean
        filtered_options = {k: v for k, v in options_kwargs.items() if v is not None}

        if is_dataclass(ClaudeAgentOptions):
            return ClaudeAgentOptions(**filtered_options)

        # Fallback for tests when SDK class isn't available
        class _Options:
            def __init__(self, **kwargs: Any):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        return _Options(**filtered_options)

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Convert nanobot messages + tools into SDK-friendly prompt text."""
        system_parts: list[str] = []
        convo_parts: list[str] = []

        for message in messages:
            role = str(message.get("role", "user"))
            content = self._content_to_text(message.get("content"))

            if role == "system":
                if content:
                    system_parts.append(content)
                continue

            if role == "assistant":
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    serialized_calls = json.dumps(tool_calls, ensure_ascii=False)
                    content = (
                        f"{content}\n\nTool calls emitted:\n{serialized_calls}"
                        if content
                        else f"Tool calls emitted:\n{serialized_calls}"
                    )
                convo_parts.append(f"Assistant:\n{content}".strip())
            elif role == "tool":
                tool_name = str(message.get("name", "tool"))
                tool_call_id = str(message.get("tool_call_id", ""))
                label = f"Tool Result ({tool_name}, id={tool_call_id})"
                convo_parts.append(f"{label}:\n{content}".strip())
            else:
                convo_parts.append(f"User:\n{content}".strip())

        if not convo_parts:
            convo_parts.append("User:\nPlease respond to the latest request.")

        prompt = "\n\n".join(part for part in convo_parts if part)

        if tools:
            sdk_tools = self._convert_tools_to_sdk(tools)
            tools_json = json.dumps(sdk_tools, ensure_ascii=False, indent=2)
            prompt += (
                "\n\nAvailable tools (JSON schema):\n"
                f"{tools_json}\n\n"
                "When tool use is needed, return tool_calls as structured data with "
                "id/name/arguments. Keep content concise."
            )

        system_prompt = "\n\n".join(system_parts)
        return system_prompt, prompt

    def _normalize_agents(self, agents: dict[str, Any] | None) -> dict[str, Any] | None:
        """Normalize custom agents into SDK `AgentDefinition` objects."""
        if not agents:
            return None

        normalized: dict[str, Any] = {}
        for name, definition in agents.items():
            if is_dataclass(definition):
                normalized[name] = definition
                continue

            if not isinstance(definition, dict):
                raise ValueError(f"Agent '{name}' must be a mapping")

            description = str(definition.get("description", "")).strip()
            prompt = str(definition.get("prompt", "")).strip()
            if not description or not prompt:
                raise ValueError(
                    f"Agent '{name}' must include non-empty 'description' and 'prompt'"
                )

            tools = definition.get("tools")
            tool_list = [str(item) for item in tools] if isinstance(tools, list) else None

            model = definition.get("model")
            model_name = str(model).strip() if model is not None else None
            if model_name not in {None, "sonnet", "opus", "haiku", "inherit"}:
                raise ValueError(
                    f"Agent '{name}' has unsupported model '{model_name}'. "
                    "Use sonnet, opus, haiku, or inherit."
                )

            if is_dataclass(AgentDefinition):
                normalized[name] = AgentDefinition(
                    description=description,
                    prompt=prompt,
                    tools=tool_list,
                    model=model_name,
                )
            else:
                normalized[name] = {
                    "description": description,
                    "prompt": prompt,
                    "tools": tool_list,
                    "model": model_name,
                }

        return normalized

    def _convert_tools_to_sdk(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert nanobot OpenAI-style tools into SDK-friendly metadata."""
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue

            function = tool.get("function") or {}
            name = str(function.get("name", "")).strip()
            if not name:
                continue

            converted.append(
                {
                    "name": name,
                    "description": str(function.get("description", "")).strip(),
                    "input_schema": function.get("parameters") or {"type": "object"},
                }
            )

        return converted

    def _extract_structured_tool_calls(self, payload: dict[str, Any]) -> list[ToolCallRequest]:
        """Extract tool calls from structured payload."""
        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list):
            return []

        calls: list[ToolCallRequest] = []
        for index, raw in enumerate(raw_calls, start=1):
            if not isinstance(raw, dict):
                continue

            name = str(raw.get("name", "")).strip()
            if not name:
                continue

            arguments = self._coerce_tool_arguments(raw.get("arguments"))
            call_id = str(raw.get("id") or f"call_{index}")
            calls.append(
                ToolCallRequest(
                    id=call_id,
                    name=self._normalize_tool_name(name),
                    arguments=arguments,
                )
            )

        return calls

    def _parse_tool_payload_from_text(self, content: str) -> dict[str, Any] | None:
        """Attempt to parse a JSON payload containing tool calls from assistant text."""
        if not content:
            return None

        candidate = content.strip()

        if candidate.startswith("```"):
            candidate = self._strip_code_fences(candidate)

        parsed = self._json_load(candidate)
        if isinstance(parsed, dict):
            return parsed

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = self._json_load(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed

        return None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences from a text block."""
        lines = [line for line in text.strip().splitlines()]
        if not lines:
            return ""

        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _json_load(raw: str) -> Any:
        """Safe JSON decode helper."""
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Convert message content variants to plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append(str(item.get("text", "")))
                    elif item_type == "image_url":
                        parts.append("[image attachment]")
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)

        return str(content)

    @staticmethod
    def _normalize_model(model: str) -> str:
        """Normalize model names between nanobot and Claude SDK conventions."""
        normalized = model.strip()
        for prefix in ("anthropic/", "claude-agent/"):
            if normalized.startswith(prefix):
                return normalized[len(prefix) :]
        return normalized

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        """Map SDK/MCP tool names back to nanobot tool names when possible."""
        prefix = "mcp__nanobot__"
        if name.startswith(prefix):
            return name[len(prefix) :]
        return name

    @staticmethod
    def _coerce_tool_arguments(arguments: Any) -> dict[str, Any]:
        """Coerce tool arguments into dictionary form."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"raw": arguments}
        return {"value": arguments}

    @staticmethod
    def _convert_usage(raw_usage: Any) -> dict[str, int]:
        """Normalize SDK usage payload into nanobot usage format."""
        if not isinstance(raw_usage, dict):
            return {}

        usage: dict[str, int] = {}

        prompt_tokens = raw_usage.get("input_tokens") or raw_usage.get("prompt_tokens")
        completion_tokens = raw_usage.get("output_tokens") or raw_usage.get(
            "completion_tokens"
        )

        if isinstance(prompt_tokens, int):
            usage["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            usage["completion_tokens"] = completion_tokens

        if "prompt_tokens" in usage and "completion_tokens" in usage:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        elif isinstance(raw_usage.get("total_tokens"), int):
            usage["total_tokens"] = int(raw_usage["total_tokens"])

        return usage

    def get_agents_config(self) -> dict[str, Any] | None:
        """Return configured custom agents as plain dictionaries."""
        if not self._agents:
            return None

        plain: dict[str, Any] = {}
        for name, definition in self._agents.items():
            if is_dataclass(definition):
                plain[name] = asdict(definition)
            else:
                plain[name] = definition
        return plain
