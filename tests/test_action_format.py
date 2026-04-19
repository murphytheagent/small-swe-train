from __future__ import annotations

from rollout.action_format import (
    is_chatml_assistant_turn,
    parse_assistant_text,
    render_assistant_action_text,
    render_tool_call_block,
    serialize_tool_call_payload,
)
from schemas import ActionEnvelope, ToolCall


def test_parse_assistant_text_accepts_bare_payload() -> None:
    envelope = parse_assistant_text(
        '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
    )

    assert envelope.tool_calls[0].tool == "submit"


def test_parse_assistant_text_accepts_chatml_assistant_turn() -> None:
    envelope = parse_assistant_text(
        "<|im_start|>assistant\n"
        '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>\n'
        "<|im_end|>"
    )

    assert envelope.tool_calls[0].tool == "submit"


def test_render_tool_call_block_preserves_legacy_json_contract() -> None:
    rendered = render_tool_call_block(
        ToolCall(tool="bash", args={"command": "pytest -q", "cwd": "."}),
    )

    assert rendered == '<tool_call>{"args": {"command": "pytest -q", "cwd": "."}, "tool": "bash"}</tool_call>'


def test_render_tool_call_block_supports_compact_mode() -> None:
    rendered = render_tool_call_block(
        {"tool": "submit", "args": {"final_response": "done"}},
        compact=True,
    )

    assert rendered == '<tool_call>{"args":{"final_response":"done"},"tool":"submit"}</tool_call>'


def test_render_assistant_action_text_includes_thinking_and_calls() -> None:
    envelope = ActionEnvelope(
        thinking="check file",
        tool_calls=(ToolCall(tool="read", args={"path": "src/app.py"}),),
    )

    assert render_assistant_action_text(envelope) == (
        '<think>check file</think>'
        '<tool_call>{"args": {"path": "src/app.py"}, "tool": "read"}</tool_call>'
    )


def test_is_chatml_assistant_turn_detects_assistant_prefix() -> None:
    assert is_chatml_assistant_turn("<|im_start|>assistant\nhello\n<|im_end|>")
    assert not is_chatml_assistant_turn('<tool_call>{"tool":"submit","args":{}}</tool_call>')


def test_serialize_tool_call_payload_supports_compact_mode() -> None:
    assert (
        serialize_tool_call_payload(
            ToolCall(tool="submit", args={"final_response": "done"}),
            compact=True,
        )
        == '{"args":{"final_response":"done"},"tool":"submit"}'
    )
