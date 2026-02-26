"""ChatML assistant-turn parser for optional thinking + ordered tool calls.

The ``TurnParser`` class uses delimiter-aware scanning plus JSON decoding so
the same parsing logic works across model families. Module-level convenience
functions (``parse_chatml_assistant_turn``, etc.) use the Qwen3 defaults for
backward compatibility.
"""

from __future__ import annotations

import functools
import json

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
        self._json_decoder = json.JSONDecoder()

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
        if not stripped.endswith(self._assistant_end):
            end_index = stripped.rfind(self._assistant_end)
            if end_index < 0:
                raise TurnParseError(
                    f"Missing '{self._assistant_end}' terminator."
                )
            raise TurnParseError("Unexpected text after ChatML end delimiter.")

        end_index = len(stripped) - len(self._assistant_end)
        if end_index < len(self._assistant_prefix):
            raise TurnParseError(
                f"Missing '{self._assistant_end}' terminator."
            )
        payload = stripped[len(self._assistant_prefix) : end_index]
        return payload.lstrip("\n").strip()

    def parse_assistant_turn_payload(
        self, payload: str, max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN
    ) -> ActionEnvelope:
        """Parse assistant payload into canonical action envelope."""
        d = self._delimiters
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")

        thinking: str | None = None
        think_seen = False
        tool_calls: list[ToolCall] = []
        cursor = 0

        while cursor < len(payload):
            think_start = payload.find(d.think_start, cursor)
            tool_start = payload.find(d.tool_call_start, cursor)
            if think_start == -1 and tool_start == -1:
                break

            if think_start != -1 and (tool_start == -1 or think_start < tool_start):
                if think_seen:
                    raise TurnParseError(
                        f"At most one {d.think_start} block is allowed per assistant turn."
                    )
                think_end = payload.find(d.think_end, think_start + len(d.think_start))
                if think_end < 0:
                    raise TurnParseError(f"Unbalanced {d.think_start} delimiters.")
                think_seen = True
                raw_thinking = payload[think_start + len(d.think_start) : think_end].strip()
                thinking = raw_thinking or None
                cursor = think_end + len(d.think_end)
                continue

            json_start = tool_start + len(d.tool_call_start)
            while json_start < len(payload) and payload[json_start].isspace():
                json_start += 1

            try:
                payload_obj, json_end = self._json_decoder.raw_decode(payload, json_start)
            except json.JSONDecodeError as exc:
                raise TurnParseError(f"Invalid tool_call JSON: {exc.msg}") from exc

            end_tag_start = json_end
            while end_tag_start < len(payload) and payload[end_tag_start].isspace():
                end_tag_start += 1
            if not payload.startswith(d.tool_call_end, end_tag_start):
                raise TurnParseError(
                    f"Missing {d.tool_call_end} after {d.tool_call_start} JSON payload."
                )

            if not isinstance(payload_obj, dict):
                raise TurnParseError(
                    f"Each {d.tool_call_start} payload must decode to a JSON object."
                )
            tool_calls.append(make_tool_call(payload_obj))
            if len(tool_calls) > max_tool_calls:
                raise TurnParseError(
                    f"Too many tool calls: got {len(tool_calls)}, max is {max_tool_calls}."
                )
            cursor = end_tag_start + len(d.tool_call_end)

        if not tool_calls:
            raise TurnParseError(
                f"At least one {d.tool_call_start} block is required."
            )

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
