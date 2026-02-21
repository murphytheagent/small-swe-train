from __future__ import annotations

import pytest

from rollout.turn_parser import TurnParseError, parse_assistant_turn_payload, parse_chatml_assistant_turn


def test_parse_chatml_assistant_turn_with_thinking_and_two_tool_calls() -> None:
    turn = """<|im_start|>assistant
<think>Check failing test and patch quickly.</think>
<tool_call>{"tool":"search","args":{"query":"tests/test_math.py::test_add"}}</tool_call>
<tool_call>{"tool":"edit","args":{"path":"src/math_utils.py","patch":"- return a-b\\n+ return a+b"}}</tool_call>
<|im_end|>"""

    envelope = parse_chatml_assistant_turn(turn, max_tool_calls=3)

    assert envelope.thinking == "Check failing test and patch quickly."
    assert len(envelope.tool_calls) == 2
    assert envelope.tool_calls[0].tool == "search"
    assert envelope.tool_calls[1].tool == "edit"


def test_legacy_answer_alias_is_canonicalized_to_submit() -> None:
    payload = "<tool_call>{\"tool\":\"answer\",\"args\":{\"final_response\":\"fixed\"}}</tool_call>"

    envelope = parse_assistant_turn_payload(payload)

    assert len(envelope.tool_calls) == 1
    assert envelope.tool_calls[0].tool == "submit"


def test_submit_must_be_singleton_tool_call() -> None:
    payload = """
<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>
<tool_call>{"tool":"search","args":{"query":"x"}}</tool_call>
"""

    with pytest.raises(TurnParseError, match="submit"):
        parse_assistant_turn_payload(payload)


def test_rejects_text_outside_declared_blocks() -> None:
    payload = """
<think>ok</think>
I should not be here.
<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>
"""

    with pytest.raises(TurnParseError, match="outside"):
        parse_assistant_turn_payload(payload)
