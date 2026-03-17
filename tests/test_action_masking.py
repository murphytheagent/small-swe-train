from __future__ import annotations

from losses.action_masking import build_action_token_mask


def test_format_rft_includes_assistant_action_tokens() -> None:
    labels = ["think", "tool_call", "other", "tool_call"]
    mask = build_action_token_mask(labels, stage="format_rft")
    assert mask == [True, True, False, True]


def test_turn_sdpo_includes_assistant_action_tokens() -> None:
    labels = ["think", "tool_call", "other"]
    mask = build_action_token_mask(labels, stage="turn_sdpo")
    assert mask == [True, True, False]


def test_legacy_stage_aliases_map_to_canonical_masks() -> None:
    labels = ["think", "tool_call", "other"]
    mask = build_action_token_mask(labels, stage="step_sdpo")
    assert mask == [True, True, False]
