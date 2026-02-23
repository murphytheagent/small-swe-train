from __future__ import annotations

import pytest

import qwen_vl_utils


def test_qwen_vl_utils_shim_raises_for_vision_helpers() -> None:
    with pytest.raises(RuntimeError, match="vision helpers are unavailable"):
        qwen_vl_utils.fetch_image("unused")
    with pytest.raises(RuntimeError, match="vision helpers are unavailable"):
        qwen_vl_utils.fetch_video("unused")
