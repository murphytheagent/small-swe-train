from __future__ import annotations

import pytest

from data.tool_schema_adapter import adapt_external_tool_call, map_external_tool


def test_map_str_replace_editor_view_to_search() -> None:
    tool = map_external_tool("str_replace_editor", subcommand="view")
    assert tool == "search"


def test_map_str_replace_editor_edit_commands_to_apply_patch() -> None:
    tool = map_external_tool("str_replace_editor", subcommand="insert")
    assert tool == "apply_patch"


def test_adapt_answer_alias_to_submit() -> None:
    call = adapt_external_tool_call("answer", {"answer": "final"})
    assert call.tool == "submit"
    assert call.args["final_response"] == "final"


def test_unsupported_str_replace_subcommand_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        map_external_tool("str_replace_editor", subcommand="delete")
