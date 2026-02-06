#!/usr/bin/env python3
"""Simple integration verification script for ClaudeAgentProvider.

This script validates:
1) config schema + agents YAML loading
2) provider creation for `claude-agent`
3) tool-call mapping behavior using a mocked SDK stream

Run:
    python tests/verify_claude_agent_integration.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanobot.cli.commands import _create_provider, _load_agents_config
from nanobot.config.loader import load_config
from nanobot.providers.claude_agent_provider import ClaudeAgentProvider
from nanobot.providers import claude_agent_provider as cap


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


def _fake_query(*, prompt: str, options: Any):
    async def _iter():
        _ = prompt
        _ = options
        yield _AssistantMessage(
            content=[
                _TextBlock(text="I will read the file."),
                _ToolUseBlock(
                    id="call_1",
                    name="mcp__nanobot__read_file",
                    input={"path": "README.md"},
                ),
            ]
        )
        yield _ResultMessage(
            is_error=False,
            usage={"input_tokens": 12, "output_tokens": 8},
            result="done",
        )

    return _iter()


async def _verify_provider_mapping() -> None:
    provider = ClaudeAgentProvider(
        default_model="anthropic/claude-sonnet-4-5",
        query_func=_fake_query,
        agents={
            "developer": {
                "description": "Dev agent",
                "prompt": "Implement code.",
                "tools": ["Read", "Write"],
                "model": "sonnet",
            }
        },
    )

    response = await provider.chat(
        messages=[{"role": "user", "content": "Open README"}],
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

    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}
    assert response.usage == {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}


def _verify_config_and_loading() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        workspace = root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        (workspace / "agents.yaml").write_text(
            """
agents:
  requirement-analyst:
    description: Analyze requirements
    prompt: Ask clarifying questions
    tools:
      - Read
      - Write
    model: sonnet
""".strip(),
            encoding="utf-8",
        )

        config_path = root / "config.json"
        config_data = {
            "agents": {
                "defaults": {
                    "provider": "claude-agent",
                    "workspace": str(workspace),
                    "model": "anthropic/claude-sonnet-4-5",
                }
            },
            "providers": {
                "claudeAgent": {
                    "apiKey": "test-key",
                    "agentsConfigPath": "agents.yaml",
                    "permissionMode": "default",
                }
            },
        }
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        config = load_config(config_path)
        assert config.agents.defaults.provider == "claude-agent"

        agents = _load_agents_config("agents.yaml", workspace=config.workspace_path)
        assert "requirement-analyst" in agents

        # Verify local provider construction (works without external SDK install)
        provider = ClaudeAgentProvider(
            api_key=config.get_api_key("claude-agent"),
            api_base=config.get_api_base("claude-agent"),
            default_model=config.agents.defaults.model,
            agents=agents,
            query_func=_fake_query,
        )
        assert isinstance(provider, ClaudeAgentProvider)
        assert provider.get_default_model() == "anthropic/claude-sonnet-4-5"

        # Best-effort check for CLI wiring when SDK is available in runtime
        if getattr(cap, "_CLAUDE_SDK_IMPORT_ERROR", None) is None:
            provider_from_cli = _create_provider(config)
            assert isinstance(provider_from_cli, ClaudeAgentProvider)


def main() -> None:
    _verify_config_and_loading()
    asyncio.run(_verify_provider_mapping())
    print("Claude Agent integration verification passed.")


if __name__ == "__main__":
    main()
