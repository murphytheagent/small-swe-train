from __future__ import annotations

from losses.action_masking import build_action_token_mask


def test_rft_excludes_think_tokens() -> None:
    labels = ["think", "tool_call", "other", "tool_call"]
    mask = build_action_token_mask(labels, stage="rft")
    assert mask == [False, True, False, True]


def test_step_sdpo_includes_think_and_tool_call_tokens() -> None:
    labels = ["think", "tool_call", "other"]
    mask = build_action_token_mask(labels, stage="step_sdpo")
    assert mask == [True, True, False]
