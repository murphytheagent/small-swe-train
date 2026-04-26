"""Rollout package."""

from .turn_parser import (
    TurnParseError,
    TurnParser,
    extract_chatml_assistant_payload,
    parse_assistant_turn_payload,
    parse_chatml_assistant_turn,
)

__all__ = [
    "OnPolicyRolloutCollector",
    "TurnParseError",
    "TurnParser",
    "extract_chatml_assistant_payload",
    "parse_assistant_turn_payload",
    "parse_chatml_assistant_turn",
]


def __getattr__(name: str):
    if name == "OnPolicyRolloutCollector":
        from .onpolicy_collector import OnPolicyRolloutCollector

        return OnPolicyRolloutCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
