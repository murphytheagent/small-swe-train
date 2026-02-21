"""Deterministic adapter from external SWE trajectory tools into canonical tool schema."""

from __future__ import annotations

from typing import Any, Mapping

from schemas import ToolCall, canonical_tool_name

_STR_REPLACE_VIEW_COMMANDS = {"view"}
_STR_REPLACE_EDIT_COMMANDS = {"create", "str_replace", "insert", "undo_edit"}


def map_external_tool(tool_name: str, *, subcommand: str | None = None) -> str:
    """Map an external tool name into canonical runtime tool names."""
    normalized_tool = tool_name.strip().lower()
    if normalized_tool in {"bash", "submit", "answer"}:
        return canonical_tool_name(normalized_tool)

    if normalized_tool == "str_replace_editor":
        if not subcommand:
            raise ValueError("subcommand is required for str_replace_editor mapping")
        normalized_subcommand = subcommand.strip().lower()
        if normalized_subcommand in _STR_REPLACE_VIEW_COMMANDS:
            return "search"
        if normalized_subcommand in _STR_REPLACE_EDIT_COMMANDS:
            return "edit"
        raise ValueError(f"Unsupported str_replace_editor subcommand: {subcommand!r}")

    raise ValueError(f"Unsupported external tool: {tool_name!r}")


def adapt_external_tool_call(tool_name: str, args: Mapping[str, Any]) -> ToolCall:
    """Convert an external tool call object to canonical ToolCall."""
    subcommand = args.get("command") if tool_name == "str_replace_editor" else None
    canonical_tool = map_external_tool(tool_name, subcommand=subcommand if isinstance(subcommand, str) else None)

    if canonical_tool == "bash":
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("bash adapter requires non-empty 'command' argument")
        return ToolCall(tool="bash", args={"command": command, "cwd": args.get("cwd", ".")})

    if canonical_tool == "search":
        query = args.get("path") or args.get("query") or args.get("target")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search adapter requires a query/path-like source field")
        return ToolCall(tool="search", args={"query": query, "path_hint": args.get("path", "")})

    if canonical_tool == "edit":
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("edit adapter requires non-empty 'path' field")
        patch = args.get("patch") or args.get("new_str") or args.get("content")
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("edit adapter requires patch/new_str/content payload")
        return ToolCall(tool="edit", args={"path": path, "patch": patch})

    final_response = args.get("final_response") or args.get("answer") or ""
    if not isinstance(final_response, str) or not final_response.strip():
        raise ValueError("submit adapter requires final response text")
    return ToolCall(tool="submit", args={"final_response": final_response})
