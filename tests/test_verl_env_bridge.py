from __future__ import annotations

from dataclasses import dataclass

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
<tool_call>{"tool":"edit","args":{"path":"x.py","patch":"+x"}}</tool_call>
"""

    result = run_env_bridge_step(assistant_text, executor=executor, max_tool_calls=3)

    assert result.is_terminal is False
    assert [request.tool for request in executor.requests] == ["search", "edit"]
    assert len(result.steps) == 2
    assert all("<tool_response>" in block for block in result.tool_response_blocks)


def test_run_env_bridge_step_submit_is_terminal_without_execution() -> None:
    executor = FakeExecutor(requests=[])
    assistant_text = "<tool_call>{\"tool\":\"submit\",\"args\":{\"final_response\":\"done\"}}</tool_call>"

    result = run_env_bridge_step(assistant_text, executor=executor)

    assert result.is_terminal is True
    assert result.steps == ()
    assert executor.requests == []
