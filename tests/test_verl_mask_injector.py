from __future__ import annotations

import pytest

from verl_integration.mask_injector import build_response_mask, inject_response_mask


def test_build_response_mask_step_sdpo() -> None:
    labels = ["think", "tool_call", "other"]
    assert build_response_mask(labels, stage="step_sdpo") == [True, True, False]


def test_inject_response_mask_for_rft() -> None:
    batch = [{"token_labels": ["think", "tool_call", "other", "tool_call"]}]

    injected = inject_response_mask(batch, stage="rft")

    assert injected[0]["response_mask"] == [False, True, False, True]


def test_build_response_mask_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unsupported token label"):
        build_response_mask(["think", "bad_label"], stage="rft")
