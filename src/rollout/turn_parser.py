"""ChatML assistant-turn parser for optional thinking + ordered tool calls.

The ``TurnParser`` class compiles regex patterns from a ``ModelDelimiters``
config so the same parsing logic works across model families.  Module-level
convenience functions (``parse_chatml_assistant_turn``, etc.) use the Qwen3
defaults for backward compatibility.
"""

from __future__ import annotations

import functools
import json
import re

from prompts.model_delimiters import ModelDelimiters, default_delimiters
from config import MAX_TOOL_CALLS_PER_TURN
from schemas import ActionEnvelope, ToolCall, make_tool_call


class TurnParseError(ValueError):
    """Raised when assistant-turn payload cannot be parsed safely."""


# ------------------------------------------------------------------
# Configurable parser
# ------------------------------------------------------------------

class TurnParser:
    """Delimiter-aware parser for assistant turns."""

    def __init__(self, delimiters: ModelDelimiters) -> None:
        self._delimiters = delimiters
        self._assistant_prefix = f"{delimiters.role_start}assistant"
        self._assistant_end = delimiters.role_end
        self._think_pattern = re.compile(
            re.escape(delimiters.think_start)
            + r"(.*?)"
            + re.escape(delimiters.think_end),
            re.DOTALL,
        )
        self._tool_call_pattern = re.compile(
            re.escape(delimiters.tool_call_start)
            + r"(.*?)"
            + re.escape(delimiters.tool_call_end),
            re.DOTALL,
        )

    @property
    def delimiters(self) -> ModelDelimiters:
        return self._delimiters

    # ---- public API ------------------------------------------------

    def extract_chatml_assistant_payload(self, turn_text: str) -> str:
        """Extract assistant payload between role delimiters."""
        stripped = turn_text.strip()
        if not stripped.startswith(self._assistant_prefix):
            raise TurnParseError(
                f"Turn does not start with '{self._assistant_prefix}'."
            )
        end_index = stripped.rfind(self._assistant_end)
        if end_index < 0:
            raise TurnParseError(
                f"Missing '{self._assistant_end}' terminator."
            )
        tail = stripped[end_index + len(self._assistant_end) :].strip()
        if tail:
            raise TurnParseError("Unexpected text after ChatML end delimiter.")
        payload = stripped[len(self._assistant_prefix) : end_index]
        return payload.lstrip("\n").strip()

    def parse_assistant_turn_payload(
        self, payload: str, max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN
    ) -> ActionEnvelope:
        """Parse assistant payload into canonical action envelope."""
        d = self._delimiters
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")

        if payload.count(d.think_start) != payload.count(d.think_end):
            raise TurnParseError(f"Unbalanced {d.think_start} delimiters.")

        think_matches = list(self._think_pattern.finditer(payload))
        if len(think_matches) > 1:
            raise TurnParseError(
                f"At most one {d.think_start} block is allowed per assistant turn."
            )

        thinking: str | None = None
        if think_matches:
            match = think_matches[0]
            thinking = match.group(1).strip() or None

        tool_matches = list(self._tool_call_pattern.finditer(payload))
        if not tool_matches:
            raise TurnParseError(
                f"At least one {d.tool_call_start} block is required."
            )
        if len(tool_matches) > max_tool_calls:
            raise TurnParseError(
                f"Too many tool calls: got {len(tool_matches)}, max is {max_tool_calls}."
            )

        tool_calls: list[ToolCall] = []
        for match in tool_matches:
            raw_json = match.group(1).strip()
            try:
                payload_obj = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise TurnParseError(f"Invalid tool_call JSON: {exc.msg}") from exc
            if not isinstance(payload_obj, dict):
                raise TurnParseError(
                    f"Each {d.tool_call_start} payload must decode to a JSON object."
                )
            tool_calls.append(make_tool_call(payload_obj))

        try:
            return ActionEnvelope(tool_calls=tuple(tool_calls), thinking=thinking)
        except ValueError as exc:
            raise TurnParseError(str(exc)) from exc

    def parse_chatml_assistant_turn(
        self, turn_text: str, max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN
    ) -> ActionEnvelope:
        """Parse a full ChatML assistant turn string into an ActionEnvelope."""
        payload = self.extract_chatml_assistant_payload(turn_text)
        return self.parse_assistant_turn_payload(payload, max_tool_calls=max_tool_calls)


# ------------------------------------------------------------------
# Module-level convenience functions (default = Qwen3)
# ------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_default_parser() -> TurnParser:
    return TurnParser(default_delimiters())


def extract_chatml_assistant_payload(turn_text: str) -> str:
    """Extract assistant payload using default model-family delimiters."""
    return _get_default_parser().extract_chatml_assistant_payload(turn_text)


def parse_assistant_turn_payload(
    payload: str, max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN
) -> ActionEnvelope:
    """Parse assistant payload using default model-family delimiters."""
    return _get_default_parser().parse_assistant_turn_payload(payload, max_tool_calls)


def parse_chatml_assistant_turn(
    turn_text: str, max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN
) -> ActionEnvelope:
    """Parse a full ChatML assistant turn using default model-family delimiters."""
    return _get_default_parser().parse_chatml_assistant_turn(turn_text, max_tool_calls)
