"""Runtime protocol data types for tool execution environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResponse:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentStep:
    step_index: int
    request: ToolRequest
    response: ToolResponse
    thinking: str | None = None
