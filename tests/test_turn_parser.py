from __future__ import annotations

import json

import pytest

from rollout.turn_parser import TurnParseError, parse_assistant_turn_payload, parse_chatml_assistant_turn


def test_parse_chatml_assistant_turn_with_thinking_and_two_tool_calls() -> None:
    turn = """<|im_start|>assistant
<think>Check failing test and patch quickly.</think>
<tool_call>{"tool":"text_search","args":{"query":"tests/test_math.py::test_add"}}</tool_call>
<tool_call>{"tool":"apply_patch","args":{"path":"src/math_utils.py","patch":"- return a-b\\n+ return a+b"}}</tool_call>
<|im_end|>"""

    envelope = parse_chatml_assistant_turn(turn, max_tool_calls=3)

    assert envelope.thinking == "Check failing test and patch quickly."
    assert len(envelope.tool_calls) == 2
    assert envelope.tool_calls[0].tool == "text_search"
    assert envelope.tool_calls[1].tool == "apply_patch"


def test_legacy_answer_alias_is_canonicalized_to_submit() -> None:
    payload = "<tool_call>{\"tool\":\"answer\",\"args\":{\"final_response\":\"fixed\"}}</tool_call>"

    envelope = parse_assistant_turn_payload(payload)

    assert len(envelope.tool_calls) == 1
    assert envelope.tool_calls[0].tool == "submit"


def test_legacy_edit_alias_is_canonicalized_to_apply_patch() -> None:
    payload = '<tool_call>{"tool":"edit","args":{"path":"x.py","patch":"+x"}}</tool_call>'

    envelope = parse_assistant_turn_payload(payload)

    assert len(envelope.tool_calls) == 1
    assert envelope.tool_calls[0].tool == "apply_patch"


def test_submit_must_be_singleton_tool_call() -> None:
    payload = """
<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>
<tool_call>{"tool":"text_search","args":{"query":"x"}}</tool_call>
"""

    with pytest.raises(TurnParseError, match="submit"):
        parse_assistant_turn_payload(payload)


def test_allows_text_outside_declared_blocks() -> None:
    payload = """
<think>ok</think>
I should not be here.
<tool_call>{"tool":"text_search","args":{"query":"foo"}}</tool_call>
"""

    envelope = parse_assistant_turn_payload(payload)

    assert envelope.thinking == "ok"
    assert len(envelope.tool_calls) == 1
    assert envelope.tool_calls[0].tool == "text_search"


def test_allows_tool_call_json_with_embedded_tool_end_delimiter_text() -> None:
    payload_obj = {
        "tool": "bash",
        "args": {
            "command": "printf 'literal </tool_call> marker inside command\\n'",
        },
    }
    payload = f"<tool_call>{json.dumps(payload_obj)}</tool_call>"

    envelope = parse_assistant_turn_payload(payload)

    assert len(envelope.tool_calls) == 1
    assert envelope.tool_calls[0].tool == "bash"
    assert "</tool_call>" in envelope.tool_calls[0].args["command"]


def test_rejects_unclosed_trailing_tool_call_block() -> None:
    payload = (
        '<tool_call>{"tool":"text_search","args":{"query":"ok"}}</tool_call>\n'
        '<tool_call>{"tool":"text_search","args":{"query":"broken"}}'
    )

    with pytest.raises(TurnParseError, match="</tool_call>"):
        parse_assistant_turn_payload(payload)


def test_rejects_non_whitespace_text_inside_tool_call_block_after_json() -> None:
    payload = '<tool_call>{"tool":"text_search","args":{"query":"ok"}} trailing </tool_call>'

    with pytest.raises(TurnParseError, match="</tool_call>"):
        parse_assistant_turn_payload(payload)


def test_rejects_legacy_search_tool_name_after_full_cutover() -> None:
    payload = '<tool_call>{"tool":"search","args":{"query":"ok"}}</tool_call>'

    with pytest.raises(TurnParseError, match="Unsupported tool"):
        parse_assistant_turn_payload(payload)
