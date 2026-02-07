"""OpenCode CLI provider with explicit plan/build workflow.

This provider is optimized for a two-phase workflow:

1. Phase 1 (`plan`):
   - Runs: ``opencode run --agent plan --format json <task>``
   - Parses JSON event stream and extracts a human-readable plan.
   - Returns the plan to user for approval.
2. Phase 2 (`build`):
   - Triggered only after user approval.
   - Runs: ``opencode run --agent build --format json <execution_prompt>``
   - Parses JSON event stream and returns execution result.

The provider keeps pending plans in memory per chat session (when session
metadata exists in system prompt).
"""

import asyncio
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse


@dataclass
class PendingPlan:
    """State for a generated plan waiting for user approval."""

    task: str
    plan_text: str
    plan_events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    plan_id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OpenCodeRunResult:
    """Parsed output for a single OpenCode agent command invocation."""

    agent: str
    returncode: int
    events: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    stderr: str = ""
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether the command completed successfully."""
        return self.returncode == 0 and self.error is None


class OpenCodeProvider(LLMProvider):
    """LLM provider that wraps OpenCode plan/build workflow.

    Approval flow:
    - First user task -> generate plan and request approval.
    - User replies with approval phrase -> execute build phase.
    - User replies with rejection phrase -> cancel pending plan.

    Notes:
    - ``tools`` are intentionally ignored because OpenCode handles execution.
    - ``model``, ``max_tokens``, and ``temperature`` are accepted for API
      compatibility but not passed to OpenCode CLI.
    """

    SESSION_PATTERN = re.compile(
        r"##\s*Current Session\s*"
        r"Channel:\s*(?P<channel>[^\n]+)\s*"
        r"Chat ID:\s*(?P<chat_id>[^\n]+)",
        re.IGNORECASE,
    )

    APPROVAL_EXACT = {
        "approve",
        "approved",
        "go",
        "go ahead",
        "ok",
        "okay",
        "yes",
        "y",
        "proceed",
        "run",
        "批准",
        "同意",
        "执行",
        "开始执行",
        "继续",
        "通过",
    }

    REJECTION_EXACT = {
        "reject",
        "rejected",
        "cancel",
        "deny",
        "stop",
        "no",
        "n",
        "拒绝",
        "取消",
        "不执行",
        "否",
        "终止",
    }

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "plan-build",
        command: str = "opencode",
        use_docker: bool = False,
        docker_image: str = "nanobot-opencode",
        timeout: int = 600,
        workspace: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.command = command
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.timeout = timeout
        self.workspace = self._normalize_workspace(workspace)
        self.pending_plans: dict[str, PendingPlan] = {}

    @staticmethod
    def _normalize_workspace(workspace: str | None) -> str | None:
        if not workspace:
            return None

        workspace_path = os.path.expandvars(os.fspath(workspace))
        workspace_path = os.path.expanduser(workspace_path)
        return os.path.abspath(workspace_path)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Handle one turn with plan/approval/build workflow."""
        chosen_model = model or self.default_model
        session_key = self._extract_session_key(messages)
        user_task = self._extract_latest_user_message(messages).strip()

        if not user_task:
            msg = "OpenCode provider requires a non-empty user task description."
            logger.error(msg)
            return LLMResponse(content=msg, finish_reason="error")

        if tools:
            logger.debug(
                "OpenCodeProvider ignores tools (count={}), workflow is CLI-native",
                len(tools),
            )

        if chosen_model != self.default_model:
            logger.debug("OpenCodeProvider ignores model='{}', CLI flow has no model flag", chosen_model)
        if max_tokens != 4096:
            logger.debug("OpenCodeProvider ignores max_tokens={}, CLI has no direct flag", max_tokens)
        if temperature != 0.7:
            logger.debug("OpenCodeProvider ignores temperature={}, CLI has no direct flag", temperature)

        pending = self.pending_plans.get(session_key)

        if not pending and self._is_approval(user_task):
            return LLMResponse(
                content="当前没有待审批的计划。如需执行代码修改任务，请先描述任务，我会先生成计划供你审批。",
                finish_reason="stop",
            )

        if not pending and self._is_rejection(user_task):
            return LLMResponse(
                content="当前没有待审批的计划可取消。如需开始新任务，请直接发送任务描述。",
                finish_reason="stop",
            )

        if pending and self._is_rejection(user_task):
            logger.info("OpenCode plan rejected by user: session={}, plan_id={}", session_key, pending.plan_id)
            self.pending_plans.pop(session_key, None)
            return LLMResponse(
                content="已取消当前计划。请发送新的任务描述，我会重新生成计划。",
                finish_reason="stop",
            )

        if pending and self._is_approval(user_task):
            logger.info("OpenCode plan approved: session={}, plan_id={}", session_key, pending.plan_id)
            self.pending_plans.pop(session_key, None)

            execution_prompt = self._build_execution_prompt(task=pending.task, plan_text=pending.plan_text)
            build_result = await self._run_agent(agent="build", prompt=execution_prompt)

            if not build_result.success:
                message = build_result.error or "Unknown OpenCode build phase error"
                logger.error("OpenCode build failed: session={}, plan_id={}, error={}", session_key, pending.plan_id, message)
                return LLMResponse(
                    content=f"OpenCode build 阶段执行失败：{message}",
                    finish_reason="error",
                    usage=self._merge_usage(pending.usage, build_result.usage),
                )

            return LLMResponse(
                content=self._format_build_result(pending, build_result),
                finish_reason="stop",
                usage=self._merge_usage(pending.usage, build_result.usage),
            )

        if pending:
            logger.info(
                "Pending plan replaced by a new user task: session={}, old_plan_id={}",
                session_key,
                pending.plan_id,
            )
            self.pending_plans.pop(session_key, None)

        plan_prompt = self._build_plan_triage_prompt(user_task)
        plan_result = await self._run_agent(agent="plan", prompt=plan_prompt)
        if not plan_result.success:
            message = plan_result.error or "Unknown OpenCode plan phase error"
            logger.error("OpenCode plan failed: session={}, error={}", session_key, message)
            return LLMResponse(
                content=f"OpenCode plan 阶段执行失败：{message}",
                finish_reason="error",
                usage=plan_result.usage,
            )

        plan_text = plan_result.text.strip()
        if not plan_text:
            logger.warning(
                "OpenCode plan returned empty text: session={}, events_count={}",
                session_key,
                len(plan_result.events),
            )
            plan_text = "（未从 OpenCode JSON 事件流中提取到计划文本，请检查 CLI 输出。）"

        triage_payload = self._extract_json_object(plan_text)
        if triage_payload is not None:
            requires_approval = bool(triage_payload.get("requires_approval"))
            content = triage_payload.get("content")
            if not isinstance(content, str):
                content = ""
            content = content.strip()

            if not requires_approval:
                return LLMResponse(
                    content=content or plan_text,
                    finish_reason="stop",
                    usage=plan_result.usage,
                )

            plan_text = content or plan_text
        else:
            requires_approval = self._infer_requires_approval(plan_text=plan_text, user_task=user_task)
            if not requires_approval:
                return LLMResponse(
                    content=plan_text,
                    finish_reason="stop",
                    usage=plan_result.usage,
                )

        pending = PendingPlan(
            task=user_task,
            plan_text=plan_text,
            plan_events=plan_result.events,
            usage=plan_result.usage,
        )
        self.pending_plans[session_key] = pending

        return LLMResponse(
            content=self._format_plan_for_approval(pending),
            finish_reason="stop",
            usage=plan_result.usage,
        )

    def get_default_model(self) -> str:
        """Get default model identifier (semantic only for compatibility)."""
        return self.default_model

    async def _run_agent(self, agent: str, prompt: str) -> OpenCodeRunResult:
        """Run ``opencode run`` for a specific agent and parse JSON event stream."""
        opencode_command = [
            self.command,
            "run",
            "--agent",
            agent,
            "--format",
            "json",
            prompt,
        ]

        if self.use_docker:
            if not self.workspace:
                message = "OpenCode docker mode requires workspace path."
                logger.error(message)
                return OpenCodeRunResult(agent=agent, returncode=1, error=message)

            if not os.path.isdir(self.workspace):
                message = f"OpenCode workspace directory not found: '{self.workspace}'"
                logger.error(message)
                return OpenCodeRunResult(agent=agent, returncode=1, error=message)

            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{self.workspace}:/workspace",
                "-w",
                "/workspace",
                self.docker_image,
                *opencode_command,
            ]
            cwd = None
            logger.debug(
                "Calling OpenCode via Docker: image='{}' agent='{}'",
                self.docker_image,
                agent,
            )
        else:
            command = opencode_command
            cwd = self.workspace
            logger.debug("Calling OpenCode CLI: command='{}'", " ".join(command[:-1]))

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            if self.use_docker:
                message = "Docker executable not found. Ensure docker is installed and available in PATH."
            else:
                message = f"OpenCode CLI not found: '{self.command}'"
            logger.error(message)
            return OpenCodeRunResult(agent=agent, returncode=127, error=message)
        except subprocess.TimeoutExpired:
            message = f"OpenCode CLI timed out after {self.timeout}s"
            logger.error(message)
            return OpenCodeRunResult(agent=agent, returncode=124, error=message)
        except Exception as e:
            message = f"Unexpected error calling OpenCode CLI: {e}"
            logger.error(message)
            return OpenCodeRunResult(agent=agent, returncode=1, error=message)

        events = self._parse_jsonl_events(completed.stdout)
        text = self._extract_text_from_events(events)
        usage = self._extract_usage(events)

        if completed.returncode != 0:
            error_text = self._extract_error_text(events, completed.stderr)
            message = error_text or f"OpenCode exited with status {completed.returncode}"
            logger.error("OpenCode agent '{}' failed (exit={}): {}", agent, completed.returncode, message)
            return OpenCodeRunResult(
                agent=agent,
                returncode=completed.returncode,
                events=events,
                text=text,
                usage=usage,
                stderr=completed.stderr,
                error=message,
            )

        return OpenCodeRunResult(
            agent=agent,
            returncode=completed.returncode,
            events=events,
            text=text,
            usage=usage,
            stderr=completed.stderr,
        )

    def _parse_jsonl_events(self, raw_stdout: str) -> list[dict[str, Any]]:
        """Parse OpenCode JSONL events from CLI stdout."""
        events: list[dict[str, Any]] = []
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            parsed = self._safe_json_loads(line)
            if isinstance(parsed, dict):
                events.append(parsed)
            else:
                logger.debug("OpenCode emitted non-JSON event line: {}", line[:200])

        return events

    def _extract_text_from_events(self, events: list[dict[str, Any]]) -> str:
        """Extract user-facing text from OpenCode events.

        Priority:
        1. Concatenate ``type == 'text'`` event chunks from ``part.text``.
        2. Fallback to common text fields in reverse event order.
        """
        chunks: list[str] = []

        for event in events:
            if event.get("type") != "text":
                continue
            part = event.get("part")
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)

        if chunks:
            return "".join(chunks).strip()

        candidate_keys = ["final_output", "output", "content", "text", "message"]
        for event in reversed(events):
            for key in candidate_keys:
                value = event.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            part = event.get("part")
            if isinstance(part, dict):
                for key in ("text", "content", "message"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        return ""

    def _extract_usage(self, events: list[dict[str, Any]]) -> dict[str, int]:
        """Extract token usage from OpenCode events.

        The PoC indicates ``type='step_finish'`` may carry token stats in
        ``part.tokens``; this implementation sums all such records.
        """
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        has_tokens = False

        for event in events:
            part = event.get("part")
            if not isinstance(part, dict):
                continue

            tokens = part.get("tokens")
            if not isinstance(tokens, dict):
                continue

            has_tokens = True
            prompt = self._safe_int(
                tokens.get("input_tokens")
                or tokens.get("prompt_tokens")
                or tokens.get("input")
            )
            completion = self._safe_int(
                tokens.get("output_tokens")
                or tokens.get("completion_tokens")
                or tokens.get("output")
            )
            total = self._safe_int(tokens.get("total_tokens") or tokens.get("total"))

            if total == 0:
                total = prompt + completion

            prompt_tokens += prompt
            completion_tokens += completion
            total_tokens += total

        if has_tokens:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        for event in reversed(events):
            usage = event.get("usage")
            if isinstance(usage, dict):
                prompt = self._safe_int(
                    usage.get("input_tokens")
                    or usage.get("prompt_tokens")
                    or usage.get("input")
                )
                completion = self._safe_int(
                    usage.get("output_tokens")
                    or usage.get("completion_tokens")
                    or usage.get("output")
                )
                total = self._safe_int(usage.get("total_tokens") or usage.get("total"))
                if total == 0:
                    total = prompt + completion

                return {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                }

        return {}

    def _extract_error_text(self, events: list[dict[str, Any]], stderr: str) -> str:
        """Extract a readable error message from OpenCode events and stderr."""
        for event in reversed(events):
            event_type = str(event.get("type") or "")

            if event_type in {"error", "fatal", "step_error"}:
                message = self._extract_error_message(event)
                if message:
                    return message

            if event.get("is_error") is True:
                message = self._extract_error_message(event)
                if message:
                    return message

            message = self._extract_error_message(event)
            if message and event_type in {"step_finish", "text"}:
                continue
            if message:
                return message

        stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        if stderr_lines:
            return stderr_lines[-1]

        return ""

    def _extract_error_message(self, event: dict[str, Any]) -> str:
        """Get possible error text from an event object."""
        error_candidate = event.get("error")
        if isinstance(error_candidate, str) and error_candidate.strip():
            return error_candidate.strip()
        if isinstance(error_candidate, dict):
            message = error_candidate.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

        part = event.get("part")
        if isinstance(part, dict):
            for key in ("error", "message"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    message = value.get("message")
                    if isinstance(message, str) and message.strip():
                        return message.strip()

        message = event.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

        return ""

    def _extract_latest_user_message(self, messages: list[dict[str, Any]]) -> str:
        """Extract latest user message content as current task description."""
        for message in reversed(messages):
            if message.get("role") == "user":
                return self._normalize_content(message.get("content"))
        return ""

    def _extract_session_key(self, messages: list[dict[str, Any]]) -> str:
        """Extract stable session key from system prompt metadata."""
        for message in messages:
            if message.get("role") != "system":
                continue

            content = self._normalize_content(message.get("content"))
            matched = self.SESSION_PATTERN.search(content)
            if matched:
                channel = matched.group("channel").strip()
                chat_id = matched.group("chat_id").strip()
                return f"{channel}:{chat_id}"

        return "default"

    def _normalize_content(self, content: Any) -> str:
        """Normalize message content to plain text."""
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

    def _normalize_approval_text(self, text: str) -> str:
        """Normalize user text for approval/rejection matching."""
        text = text.strip().lower()
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip("`'\"。，,!?！？；;：:")

    def _is_approval(self, text: str) -> bool:
        """Check whether user message approves executing pending plan."""
        normalized = self._normalize_approval_text(text)
        if normalized in self.APPROVAL_EXACT:
            return True

        approval_patterns = [
            r"\bplease proceed\b",
            r"\bgo ahead\b",
            r"\blooks good\b",
            r"\bapprove(?:d)?\b",
            r"请?执行",
            r"同意执行",
            r"批准执行",
        ]
        return any(re.search(pattern, normalized) for pattern in approval_patterns)

    def _is_rejection(self, text: str) -> bool:
        """Check whether user message rejects pending plan."""
        normalized = self._normalize_approval_text(text)
        if normalized in self.REJECTION_EXACT:
            return True

        rejection_patterns = [
            r"\bdo not proceed\b",
            r"\bdon't proceed\b",
            r"\bnot approved\b",
            r"\breject(?:ed)?\b",
            r"不(要|需)?执行",
            r"拒绝执行",
            r"取消执行",
        ]
        return any(re.search(pattern, normalized) for pattern in rejection_patterns)

    def _build_execution_prompt(self, task: str, plan_text: str) -> str:
        """Build build-phase prompt from plan and original task."""
        return f"Execute this plan:\n{plan_text}\n\nOriginal task: {task}"

    def _format_plan_for_approval(self, pending: PendingPlan) -> str:
        """Render user-facing plan output with clear approval instruction."""
        return (
            f"Phase 1 (plan) 已完成，计划 ID: {pending.plan_id}\n\n"
            f"任务描述:\n{pending.task}\n\n"
            "生成的计划:\n"
            f"{pending.plan_text}\n\n"
            "请审批该计划。\n"
            "- 批准并执行：回复 `approve` / `批准` / `执行`\n"
            "- 拒绝计划：回复 `reject` / `拒绝` / `取消`"
        )

    def _format_build_result(self, pending: PendingPlan, result: OpenCodeRunResult) -> str:
        """Render user-facing execution output."""
        body = result.text.strip() or "OpenCode build 阶段完成，但未返回可读文本输出。"
        return (
            f"Phase 2 (build) 已完成，计划 ID: {pending.plan_id}\n\n"
            f"原始任务:\n{pending.task}\n\n"
            "执行结果:\n"
            f"{body}"
        )

    def _build_plan_triage_prompt(self, user_task: str) -> str:
        """Build the prompt for the plan agent to decide approval requirement.

        The plan agent must output a strict JSON object:

        {
          "requires_approval": true/false,
          "content": "..."
        }

        - If requires_approval is false: content is the final user-facing reply.
        - If true: content is the plan text that should be approved before running build.
        """
        return (
            "你是一个\"任务分流 + 规划\"代理。你需要判断用户请求是否需要进入\"执行/修改代码\"的审批流程。\n\n"
            "规则：\n"
            "- 如果只是普通聊天、问答、解释概念、给建议、或者你需要向用户追问澄清信息：requires_approval=false，并在 content 里直接给出最终回复/问题。\n"
            "- 如果需要对仓库进行修改（创建/编辑/删除文件、生成补丁、运行命令、安装依赖、运行测试、构建部署等）：requires_approval=true，并在 content 里给出清晰的分步计划（用户将审批后再执行）。\n\n"
            "输出要求：\n"
            "- 只输出一段严格 JSON，不要使用 Markdown，不要添加代码块，不要添加额外解释文字。\n"
            "- JSON 格式：{\"requires_approval\": true/false, \"content\": \"...\"}\n\n"
            "用户输入：\n"
            f"{user_task.strip()}"
        )

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        """Best-effort extraction of a JSON object from free-form text.

        This is intentionally tolerant of minor formatting mistakes like:
        - Markdown fences (```json ... ```)
        - Leading commentary ("Here is the JSON: {...}")
        """
        if not text:
            return None

        candidate = text.strip()

        fence_match = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", candidate, re.IGNORECASE | re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()

        parsed = self._safe_json_loads(candidate)
        if isinstance(parsed, dict):
            return parsed

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", candidate):
            start = match.start()
            try:
                obj, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj

        return None

    def _infer_requires_approval(self, plan_text: str, user_task: str) -> bool:
        """Infer whether approval is required when JSON triage parsing fails.

        Conservative rule:
        - Return True if it looks like the plan involves file edits / code patches / shell commands.
        - Otherwise return False (treat as normal chat / Q&A).
        """
        combined = f"{user_task}\n{plan_text}".lower()
        combined = re.sub(r"\s+", " ", combined)

        greetings = {
            "hello",
            "hi",
            "hey",
            "你好",
            "在吗",
            "嗨",
            "早上好",
            "下午好",
            "晚上好",
        }
        normalized_task = self._normalize_approval_text(user_task)
        if normalized_task in greetings:
            return False

        patch_markers = [
            "*** begin patch",
            "diff --git",
            "@@ ",
            "+++ b/",
            "--- a/",
            "```diff",
        ]
        if any(marker in combined for marker in patch_markers):
            return True

        file_ops = [
            "modify",
            "edit",
            "update file",
            "create file",
            "delete file",
            "rename",
            "refactor",
            "implement",
            "fix ",
            "patch",
            "apply_patch",
            "修改",
            "编辑",
            "更新",
            "新增",
            "创建",
            "删除",
            "重命名",
            "重构",
            "补丁",
            "实现",
            "修复",
        ]
        if any(keyword in combined for keyword in file_ops):
            return True

        command_markers = [
            "run ",
            "execute ",
            "shell",
            "terminal",
            "command",
            "pip ",
            "npm ",
            "yarn ",
            "pnpm ",
            "pytest",
            "uv ",
            "make ",
            "docker ",
            "git ",
            "安装",
            "依赖",
            "运行",
            "执行",
            "命令",
            "测试",
            "构建",
            "部署",
        ]
        if any(keyword in combined for keyword in command_markers):
            return True

        file_path_pattern = r"\b[\w./-]+\.(py|js|ts|tsx|jsx|go|rs|java|kt|c|cc|cpp|h|hpp|md|toml|yaml|yml|json)\b"
        if re.search(file_path_pattern, combined):
            return True

        return False

    def _merge_usage(self, first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
        """Merge token usage dicts by summation."""
        if not first and not second:
            return {}

        prompt_tokens = self._safe_int(first.get("prompt_tokens")) + self._safe_int(second.get("prompt_tokens"))
        completion_tokens = self._safe_int(first.get("completion_tokens")) + self._safe_int(second.get("completion_tokens"))
        total_tokens = self._safe_int(first.get("total_tokens")) + self._safe_int(second.get("total_tokens"))

        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _safe_json_loads(self, text: str) -> Any | None:
        """Parse JSON safely. Returns ``None`` on decode failure."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _safe_int(self, value: Any) -> int:
        """Convert value to int safely, falling back to 0."""
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
