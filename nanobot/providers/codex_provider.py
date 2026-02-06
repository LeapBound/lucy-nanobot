"""Codex CLI provider implementation."""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CodexProvider(LLMProvider):
    """LLM provider that calls the Codex CLI (`codex exec`)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "o3",
        command: str = "codex",
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
        """Send a chat request through `codex exec` and parse into ``LLMResponse``."""
        chosen_model = model or self.default_model
        prompt = self._build_prompt(messages, tools)
        env = self._build_env()

        schema_path: Path | None = None
        output_file_path: Path | None = None

        command = [
            self.command,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--color",
            "never",
        ]

        if chosen_model:
            command.extend(["--model", chosen_model])

        if max_tokens != 4096:
            logger.debug("CodexProvider ignores max_tokens={}, CLI has no direct flag", max_tokens)
        if temperature != 0.7:
            logger.debug("CodexProvider ignores temperature={}, CLI has no direct flag", temperature)

        try:
            if tools:
                schema = self._build_tool_response_schema(tools)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix="nanobot-codex-schema-",
                    delete=False,
                    encoding="utf-8",
                ) as schema_file:
                    json.dump(schema, schema_file, ensure_ascii=False)
                    schema_path = Path(schema_file.name)

                command.extend(["--output-schema", str(schema_path)])

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                prefix="nanobot-codex-last-message-",
                delete=False,
                encoding="utf-8",
            ) as output_file:
                output_file_path = Path(output_file.name)

            command.extend(["--output-last-message", str(output_file_path)])
            command.append(prompt)

            logger.debug(
                "Calling Codex CLI: command={}, model={}, tools_count={}",
                self.command,
                chosen_model,
                len(tools or []),
            )

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
            msg = f"Codex CLI not found: '{self.command}'"
            logger.error(msg)
            return LLMResponse(content=msg, finish_reason="error")
        except subprocess.TimeoutExpired:
            msg = f"Codex CLI timed out after {self.timeout}s"
            logger.error(msg)
            return LLMResponse(content=msg, finish_reason="error")
        except Exception as e:
            msg = f"Unexpected error calling Codex CLI: {e}"
            logger.error(msg)
            return LLMResponse(content=msg, finish_reason="error")
        finally:
            if schema_path:
                self._safe_unlink(schema_path)

        return self._parse_process_result(
            completed=completed,
            tools_provided=bool(tools),
            output_file_path=output_file_path,
        )

    def get_default_model(self) -> str:
        """Get the default model name."""
        return self.default_model

    def _build_env(self) -> dict[str, str]:
        """Build subprocess environment for Codex CLI."""
        env = os.environ.copy()
        if self.api_key and "OPENAI_API_KEY" not in env:
            env["OPENAI_API_KEY"] = self.api_key
        if self.api_base and "OPENAI_BASE_URL" not in env:
            env["OPENAI_BASE_URL"] = self.api_base
        return env

    def _safe_unlink(self, path: Path) -> None:
        """Best-effort cleanup for temporary files."""
        try:
            if path.exists():
                path.unlink()
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to delete temp file {}: {}", path, e)

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
            "You must return ONLY a JSON object that matches the output schema.\n"
            "Use this exact structure:\n"
            '{"content": "string", "tool_calls": [{"id": "string", "name": "string", "arguments": {}}]}\n\n'
            "Rules:\n"
            "- If a tool is needed, add one or more entries in tool_calls.\n"
            "- If no tool is needed, set tool_calls to [] and put final answer in content.\n"
            "- arguments must be a JSON object.\n\n"
            f"Available tools (converted from OpenAI schema):\n{json.dumps(tools_payload, ensure_ascii=False, indent=2)}\n\n"
            "Do not run shell commands or edit files; only produce the structured JSON response.\n\n"
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
        """Build JSON schema used by Codex CLI structured output."""
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

    def _parse_process_result(
        self,
        completed: subprocess.CompletedProcess[str],
        tools_provided: bool,
        output_file_path: Path | None,
    ) -> LLMResponse:
        """Parse process outputs and return ``LLMResponse``."""
        try:
            events = self._parse_jsonl_events(completed.stdout)
            error_text = self._extract_error_text(events, completed.stderr)
            finish_reason = self._extract_finish_reason(events)
            usage = self._extract_usage(events)

            final_text = self._read_output_last_message(output_file_path)

            if completed.returncode != 0:
                message = error_text or "Unknown Codex CLI error"
                logger.error("Codex CLI failed (exit={}): {}", completed.returncode, message)
                return LLMResponse(content=f"Error calling Codex CLI: {message}", finish_reason="error")

            if tools_provided:
                return self._parse_structured_response(
                    final_text=final_text,
                    events=events,
                    finish_reason=finish_reason,
                    usage=usage,
                )

            content = final_text or self._extract_text_fallback(events)
            return LLMResponse(content=content, finish_reason=finish_reason, usage=usage)
        finally:
            if output_file_path:
                self._safe_unlink(output_file_path)

    def _parse_structured_response(
        self,
        final_text: str,
        events: list[dict[str, Any]],
        finish_reason: str,
        usage: dict[str, int],
    ) -> LLMResponse:
        """Parse structured JSON output for tool-calling mode."""
        payload: Any = None

        if final_text:
            payload = self._safe_json_loads(final_text)

        if payload is None:
            payload = self._extract_json_payload_from_events(events)

        if not isinstance(payload, dict):
            logger.warning("Codex structured output missing/invalid JSON; falling back to text")
            return LLMResponse(
                content=final_text or self._extract_text_fallback(events),
                finish_reason=finish_reason,
                usage=usage,
            )

        content = payload.get("content", "")
        tool_calls = self._parse_tool_calls(payload.get("tool_calls"))

        return LLMResponse(
            content=content if isinstance(content, str) else self._normalize_content(content),
            tool_calls=tool_calls,
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

    def _parse_jsonl_events(self, raw_stdout: str) -> list[dict[str, Any]]:
        """Parse `codex exec --json` JSONL events."""
        events: list[dict[str, Any]] = []
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            parsed = self._safe_json_loads(line)
            if isinstance(parsed, dict):
                events.append(parsed)

        return events

    def _extract_error_text(self, events: list[dict[str, Any]], stderr: str) -> str:
        """Extract the best human-readable error message from output."""
        for event in reversed(events):
            event_type = event.get("type")
            if event_type == "turn.failed":
                err = event.get("error")
                if isinstance(err, dict) and isinstance(err.get("message"), str):
                    return err["message"]
            if event_type == "error" and isinstance(event.get("message"), str):
                return event["message"]

        stderr_lines = [line for line in stderr.splitlines() if line.strip()]
        if stderr_lines:
            return stderr_lines[-1]

        return ""

    def _extract_finish_reason(self, events: list[dict[str, Any]]) -> str:
        """Extract finish reason from Codex events."""
        for event in reversed(events):
            event_type = event.get("type")
            if event_type == "turn.completed":
                reason = event.get("stop_reason") or event.get("reason")
                if isinstance(reason, str) and reason:
                    return reason
                return "stop"
            if event_type == "turn.failed":
                return "error"

        return "stop"

    def _extract_usage(self, events: list[dict[str, Any]]) -> dict[str, int]:
        """Extract token usage from Codex events when available."""
        for event in reversed(events):
            usage_candidate = event.get("usage")
            if not isinstance(usage_candidate, dict):
                continue

            prompt_tokens = int(
                usage_candidate.get("input_tokens")
                or usage_candidate.get("prompt_tokens")
                or 0
            )
            completion_tokens = int(
                usage_candidate.get("output_tokens")
                or usage_candidate.get("completion_tokens")
                or 0
            )
            total_tokens = int(
                usage_candidate.get("total_tokens")
                or (prompt_tokens + completion_tokens)
            )

            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        return {}

    def _extract_text_fallback(self, events: list[dict[str, Any]]) -> str:
        """Fallback text extraction from event stream."""
        candidates = ["final_output", "content", "text", "message"]

        for event in reversed(events):
            for key in candidates:
                value = event.get(key)
                if isinstance(value, str) and value.strip():
                    return value

        return ""

    def _extract_json_payload_from_events(self, events: list[dict[str, Any]]) -> Any:
        """Try extracting JSON payload from event fields."""
        candidate_keys = ["final_output", "content", "text", "message"]

        for event in reversed(events):
            for key in candidate_keys:
                value = event.get(key)
                if isinstance(value, str):
                    parsed = self._safe_json_loads(value)
                    if parsed is not None:
                        return parsed
                elif isinstance(value, dict):
                    return value

        return None

    def _read_output_last_message(self, output_file_path: Path | None) -> str:
        """Read `--output-last-message` file content."""
        if not output_file_path:
            return ""
        try:
            if output_file_path.exists():
                return output_file_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning("Failed reading output-last-message file {}: {}", output_file_path, e)
        return ""

    def _safe_json_loads(self, text: str) -> Any | None:
        """Parse JSON safely. Returns None on decode failure."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
