from __future__ import annotations

import pytest

from data.tool_schema_adapter import adapt_external_tool_call, map_external_tool


def test_map_str_replace_editor_view_to_read() -> None:
    tool = map_external_tool("str_replace_editor", subcommand="view")
    assert tool == "read"


def test_map_str_replace_editor_edit_commands_to_apply_patch() -> None:
    tool = map_external_tool("str_replace_editor", subcommand="insert")
    assert tool == "apply_patch"


def test_adapt_answer_alias_to_submit() -> None:
    call = adapt_external_tool_call("answer", {"answer": "final"})
    assert call.tool == "submit"
    assert call.args["final_response"] == "final"


def test_adapt_str_replace_editor_view_range_to_read_args() -> None:
    call = adapt_external_tool_call(
        "str_replace_editor",
        {
            "command": "view",
            "path": "requests/models.py",
            "view_range": [363, 420],
        },
    )

    assert call.tool == "read"
    assert call.args == {
        "path": "requests/models.py",
        "start_line": 363,
        "end_line": 420,
    }


def test_adapt_str_replace_editor_open_ended_view_range_omits_end_line() -> None:
    call = adapt_external_tool_call(
        "str_replace_editor",
        {
            "command": "view",
            "path": "requests/models.py",
            "view_range": [363, -1],
        },
    )

    assert call.tool == "read"
    assert call.args == {
        "path": "requests/models.py",
        "start_line": 363,
    }


@pytest.mark.parametrize(
    "view_range",
    [
        [10],
        [10, 9],
        [0, 10],
        [10, 0],
        ["10", 12],
    ],
)
def test_adapt_str_replace_editor_rejects_malformed_view_range(view_range) -> None:
    with pytest.raises(ValueError, match="view_range|start_line|end_line"):
        adapt_external_tool_call(
            "str_replace_editor",
            {
                "command": "view",
                "path": "requests/models.py",
                "view_range": view_range,
            },
        )


def test_unsupported_str_replace_subcommand_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        map_external_tool("str_replace_editor", subcommand="delete")
