"""Environment bridge for assistant tool-call turns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from config import MAX_TOOL_CALLS_PER_TURN, resolve_feedback_deterministic_truncation_settings
from data.feedback_canonicalizer import truncate_tool_output_payload
from env.runtime_protocol import EnvironmentStep, ToolRequest, ToolResponse
from prompts.model_delimiters import default_delimiters
from rollout.action_format import parse_assistant_text as parse_assistant_action_text
from schemas import ActionEnvelope, ToolCall, validate_tool_call


class ToolExecutor(Protocol):
    """Executor protocol for runtime tool dispatch."""

    def run(self, request: ToolRequest) -> ToolResponse:  # pragma: no cover - protocol signature only
        ...


@dataclass(frozen=True)
class BridgeResult:
    envelope: ActionEnvelope
    steps: tuple[EnvironmentStep, ...]
    tool_response_blocks: tuple[str, ...]
    is_terminal: bool


def _parse_assistant_text(assistant_text: str, *, max_tool_calls: int) -> ActionEnvelope:
    return parse_assistant_action_text(assistant_text, max_tool_calls=max_tool_calls)


def build_tool_response_payload(response: ToolResponse) -> dict[str, Any]:
    """Build deterministic serialized payload for environment tool responses."""
    payload: dict[str, Any] = {
        "stdout": response.stdout,
        "stderr": response.stderr,
        "exit_code": response.exit_code,
    }
    if response.metadata:
        payload["metadata"] = response.metadata

    truncation_settings = resolve_feedback_deterministic_truncation_settings()
    truncated_payload, _ = truncate_tool_output_payload(
        payload,
        head_tokens=truncation_settings.head_tokens,
        tail_tokens=truncation_settings.tail_tokens,
    )
    return truncated_payload


def _format_tool_response_block(response: ToolResponse) -> str:
    delimiters = default_delimiters()
    payload = build_tool_response_payload(response)
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return f"{delimiters.tool_response_start}{serialized}{delimiters.tool_response_end}"


def _validate_or_build_error_response(call: ToolCall) -> ToolResponse | None:
    errors = validate_tool_call(call)
    if not errors:
        return None
    return ToolResponse(
        stdout="",
        stderr="; ".join(errors),
        exit_code=2,
        metadata={"validation_errors": errors},
    )


def _build_parse_error_response(error: Exception) -> ToolResponse:
    message = str(error).strip() or "assistant tool-call payload parsing failed"
    return ToolResponse(
        stdout="",
        stderr=message,
        exit_code=2,
        metadata={
            "parse_error": True,
            "validation_errors": [message],
        },
    )


def run_env_bridge_step(
    assistant_text: str,
    *,
    executor: ToolExecutor,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    step_index_start: int = 0,
) -> BridgeResult:
    """Parse one assistant turn and execute its non-terminal tool calls in order."""
    try:
        envelope = _parse_assistant_text(assistant_text, max_tool_calls=max_tool_calls)
    except Exception as exc:
        parse_response = _build_parse_error_response(exc)
        parse_step = EnvironmentStep(
            step_index=step_index_start,
            request=ToolRequest(tool="bash", args={"command": "", "timeout_sec": 0}),
            response=parse_response,
            thinking=None,
        )
        return BridgeResult(
            envelope=ActionEnvelope(tool_calls=(ToolCall(tool="bash", args={"command": ""}),), thinking=None),
            steps=(parse_step,),
            tool_response_blocks=(_format_tool_response_block(parse_response),),
            is_terminal=False,
        )

    if envelope.tool_calls[0].tool == "submit":
        submit_call = envelope.tool_calls[0]
        error_response = _validate_or_build_error_response(submit_call)
        if error_response is not None:
            request = ToolRequest(tool=submit_call.tool, args=dict(submit_call.args))
            step = EnvironmentStep(
                step_index=step_index_start,
                request=request,
                response=error_response,
                thinking=envelope.thinking,
            )
            return BridgeResult(
                envelope=envelope,
                steps=(step,),
                tool_response_blocks=(_format_tool_response_block(error_response),),
                is_terminal=False,
            )
        return BridgeResult(
            envelope=envelope,
            steps=(),
            tool_response_blocks=(),
            is_terminal=True,
        )

    steps: list[EnvironmentStep] = []
    blocks: list[str] = []

    for offset, call in enumerate(envelope.tool_calls):
        error_response = _validate_or_build_error_response(call)
        request = ToolRequest(tool=call.tool, args=dict(call.args))
        response = error_response if error_response is not None else executor.run(request)
        step = EnvironmentStep(
            step_index=step_index_start + offset,
            request=request,
            response=response,
            thinking=envelope.thinking,
        )
        steps.append(step)
        blocks.append(_format_tool_response_block(response))

    return BridgeResult(
        envelope=envelope,
        steps=tuple(steps),
        tool_response_blocks=tuple(blocks),
        is_terminal=False,
    )
