"""Typed contracts and tool-call validation.

``contracts.py`` is the single source of truth for tool names, argument shapes,
and validation constraints.  Use ``validate_tool_call()`` to check a parsed
``ToolCall`` against ``TOOL_SCHEMAS``.
"""

from .contracts import (
    ALLOWED_TOOLS,
    TOOL_SCHEMAS,
    ActionEnvelope,
    AllowedTool,
    CanonicalFeedback,
    FeedbackPacket,
    SelfContainmentChecks,
    ToolCall,
    canonical_tool_name,
    make_tool_call,
    validate_tool_call,
)

__all__ = [
    "ALLOWED_TOOLS",
    "TOOL_SCHEMAS",
    "ActionEnvelope",
    "AllowedTool",
    "CanonicalFeedback",
    "FeedbackPacket",
    "SelfContainmentChecks",
    "ToolCall",
    "canonical_tool_name",
    "make_tool_call",
    "validate_tool_call",
]
