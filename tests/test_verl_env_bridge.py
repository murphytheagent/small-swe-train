from __future__ import annotations

import json
from dataclasses import dataclass

from config import resolve_feedback_deterministic_truncation_settings
from env.runtime_protocol import ToolRequest, ToolResponse
from verl_integration.env_bridge import run_env_bridge_step


@dataclass
class FakeExecutor:
    requests: list[ToolRequest]

    def run(self, request: ToolRequest) -> ToolResponse:
        self.requests.append(request)
        return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)


def test_run_env_bridge_step_executes_tool_calls_in_order() -> None:
    executor = FakeExecutor(requests=[])
    assistant_text = """
<think>inspect then patch</think>
<tool_call>{"tool":"search","args":{"query":"a"}}</tool_call>
<tool_call>{"tool":"apply_patch","args":{"path":"x.py","patch":"+x"}}</tool_call>
"""

    result = run_env_bridge_step(assistant_text, executor=executor, max_tool_calls=3)

    assert result.is_terminal is False
    assert [request.tool for request in executor.requests] == ["search", "apply_patch"]
    assert len(result.steps) == 2
    assert all("<tool_response>" in block for block in result.tool_response_blocks)


def test_run_env_bridge_step_submit_is_terminal_without_execution() -> None:
    executor = FakeExecutor(requests=[])
    assistant_text = "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"

    result = run_env_bridge_step(assistant_text, executor=executor)

    assert result.is_terminal is True
    assert result.steps == ()
    assert executor.requests == []


def test_run_env_bridge_step_invalid_submit_surfaces_validation_errors() -> None:
    executor = FakeExecutor(requests=[])
    assistant_text = "<tool_call>{\"tool\":\"submit\",\"args\":{}}</tool_call>"

    result = run_env_bridge_step(assistant_text, executor=executor)

    assert result.is_terminal is True
    assert len(result.steps) == 1
    assert result.steps[0].response.exit_code == 2
    assert "final_response" in result.steps[0].response.stderr
    assert len(result.tool_response_blocks) == 1
    assert executor.requests == []


def test_run_env_bridge_step_truncates_long_tool_response_payloads() -> None:
    truncation_settings = resolve_feedback_deterministic_truncation_settings()
    long_stdout = " ".join(
        f"tok{i}"
        for i in range(truncation_settings.head_tokens + truncation_settings.tail_tokens + 32)
    )

    @dataclass
    class LongOutputExecutor:
        requests: list[ToolRequest]

        def run(self, request: ToolRequest) -> ToolResponse:
            self.requests.append(request)
            return ToolResponse(stdout=long_stdout, stderr="", exit_code=0)

    executor = LongOutputExecutor(requests=[])
    assistant_text = '<tool_call>{"tool":"search","args":{"query":"long output"}}</tool_call>'

    result = run_env_bridge_step(assistant_text, executor=executor, max_tool_calls=3)

    assert len(result.tool_response_blocks) == 1
    payload = result.tool_response_blocks[0]
    assert payload.startswith("<tool_response>")
    assert payload.endswith("</tool_response>")
    parsed = json.loads(payload[len("<tool_response>") : -len("</tool_response>")])
    assert "<...truncated...>" in parsed["stdout"]
