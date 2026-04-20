from __future__ import annotations

import pytest

from rollout.action_format import (
    is_chatml_assistant_turn,
    parse_assistant_text,
    parse_assistant_text_result,
    render_assistant_action_text,
    render_tool_call_block,
    serialize_tool_call_payload,
)
from rollout.turn_parser import TurnParseError
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


def test_parse_assistant_text_accepts_xml_payload_in_xml_only_mode() -> None:
    envelope = parse_assistant_text(
        '<tool_call name="submit">'
        "<final_response><![CDATA[done]]></final_response>"
        "<changed_paths><path><![CDATA[src/app.py]]></path><path><![CDATA[tests/test_app.py]]></path></changed_paths>"
        "</tool_call>",
        parse_mode="xml_only",
    )

    assert envelope.tool_calls[0].tool == "submit"
    assert envelope.tool_calls[0].args["final_response"] == "done"
    assert envelope.tool_calls[0].args["changed_paths"] == ["src/app.py", "tests/test_app.py"]


def test_parse_assistant_text_result_reports_xml_payload_format() -> None:
    parsed = parse_assistant_text_result(
        '<tool_call name="bash"><command><![CDATA[pytest -q]]></command><cwd><![CDATA[.]]></cwd></tool_call>',
        parse_mode="dual",
    )

    assert parsed.payload_format == "xml"
    assert parsed.envelope.tool_calls[0].tool == "bash"


def test_render_tool_call_block_preserves_legacy_json_contract() -> None:
    rendered = render_tool_call_block(
        ToolCall(tool="bash", args={"command": "pytest -q", "cwd": "."}),
        payload_format="json",
    )

    assert rendered == '<tool_call>{"args": {"command": "pytest -q", "cwd": "."}, "tool": "bash"}</tool_call>'


def test_render_tool_call_block_supports_compact_mode() -> None:
    rendered = render_tool_call_block(
        {"tool": "submit", "args": {"final_response": "done"}},
        payload_format="json",
        compact=True,
    )

    assert rendered == '<tool_call>{"args":{"final_response":"done"},"tool":"submit"}</tool_call>'


def test_render_tool_call_block_supports_xml_payload_format() -> None:
    rendered = render_tool_call_block(
        ToolCall(tool="bash", args={"command": "pytest -q", "cwd": "."}),
        payload_format="xml",
    )

    assert rendered == (
        '<tool_call name="bash">'
        "<command><![CDATA[pytest -q]]></command>"
        "<cwd><![CDATA[.]]></cwd>"
        "</tool_call>"
    )


def test_render_tool_call_block_xml_splits_cdata_end_marker() -> None:
    rendered = render_tool_call_block(
        ToolCall(tool="submit", args={"final_response": "done ]]> now"}),
        payload_format="xml",
    )

    assert "]]]]><![CDATA[>" in rendered
    parsed = parse_assistant_text(rendered, parse_mode="xml_only")
    assert parsed.tool_calls[0].args["final_response"] == "done ]]> now"


def test_render_assistant_action_text_includes_thinking_and_calls() -> None:
    envelope = ActionEnvelope(
        thinking="check file",
        tool_calls=(ToolCall(tool="read", args={"path": "src/app.py"}),),
    )

    assert render_assistant_action_text(envelope, payload_format="json") == (
        '<think>check file</think>'
        '<tool_call>{"args": {"path": "src/app.py"}, "tool": "read"}</tool_call>'
    )


def test_render_assistant_action_text_supports_xml_payload_format() -> None:
    envelope = ActionEnvelope(
        thinking="check file",
        tool_calls=(ToolCall(tool="read", args={"path": "src/app.py", "start_line": 10}),),
    )

    assert render_assistant_action_text(envelope, payload_format="xml") == (
        "<think>check file</think>"
        '<tool_call name="read"><path><![CDATA[src/app.py]]></path><start_line>10</start_line></tool_call>'
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


def test_parse_assistant_text_dual_mode_rejects_mixed_json_and_xml_blocks() -> None:
    payload = (
        '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
        '<tool_call name="bash"><command><![CDATA[pytest -q]]></command></tool_call>'
    )

    with pytest.raises(TurnParseError, match="Mixed JSON/XML"):
        parse_assistant_text(payload, parse_mode="dual")


def test_parse_assistant_text_dual_mode_allows_xml_looking_text_inside_json_string() -> None:
    payload = (
        '<tool_call>{"tool":"submit","args":{"final_response":"see <tool_call name=\\"bash\\"> example"}}</tool_call>'
    )

    envelope = parse_assistant_text(payload, parse_mode="dual")

    assert envelope.tool_calls[0].tool == "submit"
    assert envelope.tool_calls[0].args["final_response"] == 'see <tool_call name="bash"> example'


def test_parse_assistant_text_dual_mode_allows_json_looking_text_inside_xml_cdata() -> None:
    payload = (
        '<tool_call name="submit">'
        '<final_response><![CDATA[old: <tool_call>{"tool":"submit","args":{}}</tool_call>]]></final_response>'
        "</tool_call>"
    )

    parsed = parse_assistant_text_result(payload, parse_mode="dual")

    assert parsed.payload_format == "xml"
    assert parsed.envelope.tool_calls[0].args["final_response"] == (
        'old: <tool_call>{"tool":"submit","args":{}}</tool_call>'
    )


def test_parse_assistant_text_dual_mode_falls_back_when_json_hint_came_from_xml_cdata() -> None:
    payload = (
        "<think><![CDATA[legacy <tool_call>{\"tool\":\"submit\",\"args\":{}}</tool_call>]]></think>"
        '<tool_call name="submit"><final_response><![CDATA[done]]></final_response></tool_call>'
    )

    parsed = parse_assistant_text_result(payload, parse_mode="dual")

    assert parsed.payload_format == "xml"
    assert parsed.envelope.thinking == 'legacy <tool_call>{"tool":"submit","args":{}}</tool_call>'
    assert parsed.envelope.tool_calls[0].tool == "submit"
    assert parsed.envelope.tool_calls[0].args["final_response"] == "done"


def test_parse_assistant_text_dual_mode_ignores_literal_xml_examples_before_json_block() -> None:
    payload = (
        'see <tool_call name="bash"><command><![CDATA[pytest -q]]></command></tool_call> example '
        '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
    )

    parsed = parse_assistant_text_result(payload, parse_mode="dual")

    assert parsed.payload_format == "json"
    assert parsed.envelope.tool_calls[0].tool == "submit"
    assert parsed.envelope.tool_calls[0].args["final_response"] == "done"


def test_parse_assistant_text_dual_mode_ignores_leading_literal_xml_examples_before_json_block() -> None:
    payload = (
        '<tool_call name="bash"><command><![CDATA[pytest -q]]></command></tool_call>'
        ' example '
        '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
    )

    parsed = parse_assistant_text_result(payload, parse_mode="dual")

    assert parsed.payload_format == "json"
    assert parsed.envelope.tool_calls[0].tool == "submit"
    assert parsed.envelope.tool_calls[0].args["final_response"] == "done"


def test_parse_assistant_text_dual_mode_rejects_xml_with_cdata_containing_think_after_json_call() -> None:
    payload = (
        '<tool_call>{"tool":"bash","args":{"command":"echo hi"}}</tool_call>'
        '<tool_call name="submit">'
        "<final_response><![CDATA[<think>not a real JSON-mode think block</think>]]></final_response>"
        "</tool_call>"
    )

    with pytest.raises(TurnParseError, match="Mixed JSON/XML"):
        parse_assistant_text(payload, parse_mode="dual")


def test_parse_assistant_text_dual_mode_rejects_xml_with_cdata_containing_json_tool_call_after_json_call() -> None:
    payload = (
        '<tool_call>{"tool":"bash","args":{"command":"echo hi"}}</tool_call>'
        '<tool_call name="submit">'
        '<final_response><![CDATA[old: <tool_call>{"tool":"submit","args":{"final_response":"inner"}}</tool_call>]]></final_response>'
        "</tool_call>"
    )

    with pytest.raises(TurnParseError, match="Mixed JSON/XML"):
        parse_assistant_text(payload, parse_mode="dual")


def test_parse_assistant_text_dual_mode_ignores_leading_xml_example_with_cdata_json_before_real_json_call() -> None:
    payload = (
        '<tool_call name="bash">'
        '<command><![CDATA[legacy <tool_call>{"tool":"bash","args":{"command":"fake"}}</tool_call>]]></command>'
        "</tool_call>"
        " example "
        '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
    )

    parsed = parse_assistant_text_result(payload, parse_mode="dual")

    assert parsed.payload_format == "json"
    assert tuple(call.tool for call in parsed.envelope.tool_calls) == ("submit",)
    assert parsed.envelope.thinking is None


def test_parse_assistant_text_dual_mode_ignores_leading_xml_example_with_cdata_think_before_real_json_call() -> None:
    payload = (
        '<tool_call name="bash">'
        "<command><![CDATA[<think>legacy plan</think>]]></command>"
        "</tool_call>"
        " example "
        '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
    )

    parsed = parse_assistant_text_result(payload, parse_mode="dual")

    assert parsed.payload_format == "json"
    assert tuple(call.tool for call in parsed.envelope.tool_calls) == ("submit",)
    assert parsed.envelope.thinking is None


def test_parse_assistant_text_dual_mode_rejects_xml_sequence_even_when_it_exceeds_max_tool_calls() -> None:
    payload = (
        '<tool_call>{"tool":"bash","args":{"command":"echo hi"}}</tool_call>'
        '<tool_call name="submit"><final_response><![CDATA[a]]></final_response></tool_call>'
        '<tool_call name="submit"><final_response><![CDATA[b]]></final_response></tool_call>'
    )

    with pytest.raises(TurnParseError, match="Mixed JSON/XML"):
        parse_assistant_text(payload, parse_mode="dual", max_tool_calls=1)


def test_parse_assistant_text_xml_rejects_comments() -> None:
    payload = (
        '<tool_call name="submit">'
        "<!-- note -->"
        "<final_response><![CDATA[done]]></final_response>"
        "</tool_call>"
    )

    with pytest.raises(TurnParseError, match="do not allow comments"):
        parse_assistant_text(payload, parse_mode="xml_only")


def test_parse_assistant_text_dual_mode_parses_xml_think_cdata_before_real_xml_tool_call() -> None:
    payload = (
        '<think><![CDATA[legacy </think> <tool_call>{"tool":"submit","args":{"final_response":"fake"}}</tool_call>]]></think>'
        '<tool_call name="submit"><final_response><![CDATA[real]]></final_response></tool_call>'
    )

    parsed = parse_assistant_text_result(payload, parse_mode="dual")

    assert parsed.payload_format == "xml"
    assert parsed.envelope.thinking == (
        'legacy </think> <tool_call>{"tool":"submit","args":{"final_response":"fake"}}</tool_call>'
    )
    assert tuple(call.tool for call in parsed.envelope.tool_calls) == ("submit",)
    assert parsed.envelope.tool_calls[0].args["final_response"] == "real"


def test_parse_assistant_text_xml_rejects_duplicate_scalar_fields() -> None:
    payload = (
        '<tool_call name="bash">'
        "<command><![CDATA[pytest -q]]></command>"
        "<command><![CDATA[pytest -q tests/test_app.py]]></command>"
        "</tool_call>"
    )

    with pytest.raises(TurnParseError, match="Duplicate XML arg field <command>"):
        parse_assistant_text(payload, parse_mode="xml_only")


def test_parse_assistant_text_xml_rejects_attributes_on_arg_elements() -> None:
    payload = (
        '<tool_call name="bash">'
        '<command foo="bar"><![CDATA[pytest -q]]></command>'
        "</tool_call>"
    )

    with pytest.raises(TurnParseError, match="Unsupported attributes on XML arg <command>"):
        parse_assistant_text(payload, parse_mode="xml_only")


def test_parse_assistant_text_xml_allows_literal_xml_text_inside_cdata() -> None:
    payload = (
        '<tool_call name="apply_patch">'
        "<path><![CDATA[src/app.py]]></path>"
        "<patch><![CDATA[<?xml version=\"1.0\"?><root xmlns=\"u\"/>]]></patch>"
        "</tool_call>"
    )

    envelope = parse_assistant_text(payload, parse_mode="xml_only")

    assert envelope.tool_calls[0].tool == "apply_patch"
    assert "<?xml version=\"1.0\"?>" in envelope.tool_calls[0].args["patch"]


def test_parse_assistant_text_xml_rejects_raw_text_outside_blocks() -> None:
    payload = 'oops<tool_call name="submit"><final_response><![CDATA[done]]></final_response></tool_call>'

    with pytest.raises(TurnParseError, match="Unexpected raw text outside XML assistant blocks"):
        parse_assistant_text(payload, parse_mode="xml_only")


def test_parse_assistant_text_xml_rejects_raw_text_after_think() -> None:
    payload = (
        "<think>plan</think>"
        'oops<tool_call name="submit"><final_response><![CDATA[done]]></final_response></tool_call>'
    )

    with pytest.raises(TurnParseError, match="Unexpected raw text outside XML assistant blocks"):
        parse_assistant_text(payload, parse_mode="xml_only")


def test_parse_assistant_text_xml_coerces_numeric_args() -> None:
    payload = (
        '<tool_call name="read">'
        "<path><![CDATA[src/app.py]]></path>"
        "<start_line>10</start_line>"
        "<end_line>40</end_line>"
        "</tool_call>"
    )

    envelope = parse_assistant_text(payload, parse_mode="xml_only")

    assert envelope.tool_calls[0].args["start_line"] == 10
    assert envelope.tool_calls[0].args["end_line"] == 40
