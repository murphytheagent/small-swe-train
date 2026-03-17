from __future__ import annotations

import pytest

from verl_integration.mask_injector import build_response_mask, inject_response_mask


def test_build_response_mask_turn_sdpo() -> None:
    labels = ["think", "tool_call", "other"]
    assert build_response_mask(labels, stage="turn_sdpo") == [True, True, False]


def test_inject_response_mask_for_format_rft() -> None:
    batch = [{"token_labels": ["think", "tool_call", "other", "tool_call"]}]

    injected = inject_response_mask(batch, stage="format_rft")

    assert injected[0]["response_mask"] == [True, True, False, True]


def test_build_response_mask_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unsupported token label"):
        build_response_mask(["think", "bad_label"], stage="rft")
