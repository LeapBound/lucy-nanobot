import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nanobot.cli.commands import _load_agents_config
from nanobot.providers.claude_agent_provider import ClaudeAgentProvider


@dataclass
class _TextBlock:
    text: str


@dataclass
class _ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class _AssistantMessage:
    content: list[Any]


@dataclass
class _ResultMessage:
    is_error: bool
    usage: dict[str, Any] | None = None
    structured_output: dict[str, Any] | None = None
    result: str | None = None


def _fake_query_from_messages(messages: list[Any]):
    async def _query(*, prompt: str, options: Any):
        assert isinstance(prompt, str)
        assert options is not None
        for message in messages:
            yield message

    return _query


@pytest.mark.asyncio
async def test_provider_maps_assistant_tool_use_blocks() -> None:
    provider = ClaudeAgentProvider(
        default_model="anthropic/claude-sonnet-4-5",
        query_func=_fake_query_from_messages(
            [
                _AssistantMessage(
                    content=[
                        _TextBlock(text="Working on it"),
                        _ToolUseBlock(
                            id="tool_1",
                            name="mcp__nanobot__read_file",
                            input={"path": "README.md"},
                        ),
                    ]
                ),
                _ResultMessage(is_error=False, usage={"input_tokens": 10, "output_tokens": 5}),
            ]
        ),
        agents={
            "developer": {
                "description": "Writes code",
                "prompt": "Implement feature requests.",
                "tools": ["Read", "Write"],
                "model": "sonnet",
            }
        },
    )

    response = await provider.chat(
        messages=[{"role": "user", "content": "Read the README"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
    )

    assert response.content == "Working on it"
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    agents = provider.get_agents_config()
    assert agents is not None
    assert agents["developer"]["model"] == "sonnet"


@pytest.mark.asyncio
async def test_provider_parses_structured_tool_calls_from_result() -> None:
    provider = ClaudeAgentProvider(
        default_model="claude-sonnet-4-5",
        query_func=_fake_query_from_messages(
            [
                _AssistantMessage(content=[_TextBlock(text="{"), _TextBlock(text="}")]),
                _ResultMessage(
                    is_error=False,
                    structured_output={
                        "content": "Let me execute that.",
                        "tool_calls": [
                            {
                                "id": "tool_2",
                                "name": "write_file",
                                "arguments": {"path": "out.txt", "content": "ok"},
                            }
                        ],
                    },
                ),
            ]
        ),
    )

    response = await provider.chat(
        messages=[{"role": "user", "content": "Write a file"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ],
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "write_file"
    assert response.tool_calls[0].arguments == {"path": "out.txt", "content": "ok"}


def test_load_agents_config_json_and_yaml(tmp_path: Path) -> None:
    json_path = tmp_path / "agents.json"
    json_path.write_text(
        json.dumps(
            {
                "agents": {
                    "tester": {
                        "description": "test",
                        "prompt": "run tests",
                        "tools": ["Read"],
                        "model": "haiku",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    yaml_path = tmp_path / "agents.yaml"
    yaml_path.write_text(
        """
agents:
  developer:
    description: Build features
    prompt: Implement clean code
    tools:
      - Read
      - Write
    model: sonnet
""".strip(),
        encoding="utf-8",
    )

    json_agents = _load_agents_config(str(json_path), workspace=tmp_path)
    yaml_agents = _load_agents_config(str(yaml_path), workspace=tmp_path)

    assert "tester" in json_agents
    assert json_agents["tester"]["model"] == "haiku"
    assert "developer" in yaml_agents
    assert yaml_agents["developer"]["tools"] == ["Read", "Write"]

