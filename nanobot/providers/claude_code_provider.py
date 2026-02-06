"""Claude Code CLI provider implementation."""

import asyncio
import json
import os
import subprocess
import uuid
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class ClaudeCodeProvider(LLMProvider):
    """LLM provider that calls the Claude Code CLI (`claude -p`)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "sonnet",
        command: str = "claude",
        timeout: int = 300,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.command = command
        self.timeout = timeout

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a chat request through `claude -p` and parse into ``LLMResponse``."""
        chosen_model = model or self.default_model
        prompt = self._build_prompt(messages, tools)

        command = [
            self.command,
            "-p",
            "--output-format",
            "json",
        ]

        if chosen_model:
            command.extend(["--model", chosen_model])

        if tools:
            command.extend([
                "--json-schema",
                json.dumps(self._build_tool_response_schema(tools), ensure_ascii=False),
            ])

        command.append(prompt)
        env = self._build_env()

        if max_tokens != 4096:
            logger.debug("ClaudeCodeProvider ignores max_tokens={}, CLI has no direct flag", max_tokens)
        if temperature != 0.7:
            logger.debug("ClaudeCodeProvider ignores temperature={}, CLI has no direct flag", temperature)

        logger.debug(
            "Calling Claude CLI: command={}, model={}, tools_count={}",
            self.command,
            chosen_model,
            len(tools or []),
        )

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
                env=env,
            )
        except FileNotFoundError:
            msg = f"Claude CLI not found: '{self.command}'"
            logger.error(msg)
            return LLMResponse(content=msg, finish_reason="error")
        except subprocess.TimeoutExpired:
            msg = f"Claude CLI timed out after {self.timeout}s"
            logger.error(msg)
            return LLMResponse(content=msg, finish_reason="error")
        except Exception as e:
            msg = f"Unexpected error calling Claude CLI: {e}"
            logger.error(msg)
            return LLMResponse(content=msg, finish_reason="error")

        if completed.returncode != 0:
            error_text = self._build_process_error(completed)
            logger.error("Claude CLI failed (exit={}): {}", completed.returncode, error_text)
            return LLMResponse(
                content=f"Error calling Claude CLI: {error_text}",
                finish_reason="error",
            )

        return self._parse_response(completed.stdout, tools_provided=bool(tools))

    def get_default_model(self) -> str:
        """Get the default model name."""
        return self.default_model

    def _build_env(self) -> dict[str, str]:
        """Build subprocess environment for Claude CLI."""
        env = os.environ.copy()
        if self.api_key and "ANTHROPIC_API_KEY" not in env:
            env["ANTHROPIC_API_KEY"] = self.api_key
        if self.api_base and "ANTHROPIC_BASE_URL" not in env:
            env["ANTHROPIC_BASE_URL"] = self.api_base
        return env

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        """Convert messages/tools into a single CLI prompt text."""
        transcript = self._messages_to_transcript(messages)

        if not tools:
            return transcript

        tools_payload = self._convert_tools_for_cli(tools)
        tool_instructions = (
            "You are acting as an LLM backend for a tool-calling agent.\n"
            "Return a JSON object with this exact shape:\n"
            '{"content": "string", "tool_calls": [{"id": "string", "name": "string", "arguments": {}}]}\n\n'
            "Rules:\n"
            "- If a tool is needed, put one or more calls in tool_calls and keep content concise.\n"
            "- If no tool is needed, return tool_calls as [] and provide final answer in content.\n"
            "- arguments must be a JSON object.\n\n"
            f"Available tools (converted from OpenAI schema):\n{json.dumps(tools_payload, ensure_ascii=False, indent=2)}\n\n"
            f"Conversation:\n{transcript}"
        )
        return tool_instructions

    def _messages_to_transcript(self, messages: list[dict[str, Any]]) -> str:
        """Render chat messages into plain text transcript."""
        rendered: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = self._normalize_content(message.get("content"))

            if message.get("role") == "tool":
                tool_name = message.get("name", "tool")
                tool_call_id = message.get("tool_call_id", "")
                header = f"[TOOL:{tool_name}:{tool_call_id}]"
            else:
                header = f"[{role}]"

            rendered.append(f"{header}\n{content}")

        return "\n\n".join(rendered)

    def _normalize_content(self, content: Any) -> str:
        """Normalize message content to string for CLI prompts."""
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
                        parts.append("[image content omitted]")
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)

        return str(content)

    def _convert_tools_for_cli(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tools into CLI-friendly JSON descriptors."""
        converted: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = function.get("name")
            if not name:
                continue

            converted.append({
                "name": name,
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            })

        return converted

    def _build_tool_response_schema(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Build JSON schema used by Claude CLI output validation."""
        tool_names = [
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        ]
        tool_names = [name for name in tool_names if name]

        name_schema: dict[str, Any] = {"type": "string"}
        if tool_names:
            name_schema["enum"] = tool_names

        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": name_schema,
                            "arguments": {"type": "object"},
                        },
                        "required": ["id", "name", "arguments"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["content", "tool_calls"],
            "additionalProperties": False,
        }

    def _parse_response(self, output: str, tools_provided: bool) -> LLMResponse:
        """Parse Claude CLI output into ``LLMResponse``."""
        raw = output.strip()
        if not raw:
            logger.error("Claude CLI returned empty output")
            return LLMResponse(content="Claude CLI returned empty output", finish_reason="error")

        parsed = self._parse_json_or_jsonl(raw)
        usage: dict[str, int] = {}
        finish_reason = "stop"
        payload: Any = raw

        if isinstance(parsed, dict):
            if parsed.get("type") == "result":
                finish_reason = parsed.get("stop_reason") or "stop"
                usage = self._extract_usage(parsed)
                if parsed.get("is_error"):
                    errors = parsed.get("errors") or []
                    error_text = "; ".join(str(err) for err in errors) if errors else "Unknown Claude error"
                    logger.error("Claude CLI result error: {}", error_text)
                    return LLMResponse(content=error_text, finish_reason="error", usage=usage)
                payload = (
                    parsed.get("result")
                    or parsed.get("content")
                    or parsed.get("output")
                    or ""
                )
            else:
                payload = parsed

        return self._payload_to_llm_response(
            payload=payload,
            tools_provided=tools_provided,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _payload_to_llm_response(
        self,
        payload: Any,
        tools_provided: bool,
        finish_reason: str,
        usage: dict[str, int],
    ) -> LLMResponse:
        """Convert parsed payload into ``LLMResponse``."""
        if isinstance(payload, str) and tools_provided:
            maybe_structured = self._safe_json_loads(payload)
            if isinstance(maybe_structured, dict):
                payload = maybe_structured

        if isinstance(payload, dict):
            content = payload.get("content", "")
            tool_calls = self._parse_tool_calls(payload.get("tool_calls"))
            return LLMResponse(
                content=content if isinstance(content, str) else self._normalize_content(content),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )

        if isinstance(payload, list):
            text = "\n".join(self._normalize_content(item) for item in payload)
            return LLMResponse(content=text, finish_reason=finish_reason, usage=usage)

        return LLMResponse(
            content=self._normalize_content(payload),
            finish_reason=finish_reason,
            usage=usage,
        )

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[ToolCallRequest]:
        """Parse tool call payload from CLI response."""
        if not isinstance(raw_tool_calls, list):
            return []

        calls: list[ToolCallRequest] = []
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue

            call_id = str(item.get("id") or f"call_{uuid.uuid4().hex}")
            name = item.get("name")
            if not name:
                continue

            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                parsed_args = self._safe_json_loads(arguments)
                arguments = parsed_args if isinstance(parsed_args, dict) else {"raw": arguments}
            elif not isinstance(arguments, dict):
                arguments = {"raw": self._normalize_content(arguments)}

            calls.append(ToolCallRequest(id=call_id, name=str(name), arguments=arguments))

        return calls

    def _parse_json_or_jsonl(self, text: str) -> Any:
        """Parse JSON text, falling back to JSONL parsing."""
        direct = self._safe_json_loads(text)
        if direct is not None:
            return direct

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        events: list[Any] = []
        for line in lines:
            parsed = self._safe_json_loads(line)
            if parsed is not None:
                events.append(parsed)

        if not events:
            return None

        for event in reversed(events):
            if isinstance(event, dict) and event.get("type") == "result":
                return event

        return events[-1]

    def _safe_json_loads(self, text: str) -> Any | None:
        """Parse JSON safely. Returns None on decode failure."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _extract_usage(self, result_obj: dict[str, Any]) -> dict[str, int]:
        """Extract token usage from Claude JSON result."""
        usage_obj = result_obj.get("usage")
        if not isinstance(usage_obj, dict):
            return {}

        prompt_tokens = int(
            usage_obj.get("input_tokens")
            or usage_obj.get("prompt_tokens")
            or 0
        )
        completion_tokens = int(
            usage_obj.get("output_tokens")
            or usage_obj.get("completion_tokens")
            or 0
        )
        total_tokens = int(usage_obj.get("total_tokens") or (prompt_tokens + completion_tokens))

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _build_process_error(self, completed: subprocess.CompletedProcess[str]) -> str:
        """Build readable error message from process output."""
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()

        if stdout:
            parsed = self._parse_json_or_jsonl(stdout)
            if isinstance(parsed, dict):
                errors = parsed.get("errors")
                if isinstance(errors, list) and errors:
                    return "; ".join(str(err) for err in errors)

                message = parsed.get("message")
                if isinstance(message, str) and message:
                    return message

        if stderr:
            return stderr.splitlines()[-1]
        if stdout:
            return stdout.splitlines()[-1]

        return "Unknown Claude CLI error"
