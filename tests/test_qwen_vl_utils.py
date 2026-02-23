from __future__ import annotations

from types import SimpleNamespace

import pytest

import qwen_vl_utils


def test_qwen_vl_utils_shim_raises_for_vision_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen_vl_utils, "_load_external_module", lambda: None)
    fetch_image, fetch_video = qwen_vl_utils._resolve_helpers()

    with pytest.raises(RuntimeError, match="vision helpers are unavailable"):
        fetch_image("unused")
    with pytest.raises(RuntimeError, match="vision helpers are unavailable"):
        fetch_video("unused")


def test_qwen_vl_utils_shim_delegates_when_external_module_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = SimpleNamespace(
        fetch_image=lambda value: f"image::{value}",
        fetch_video=lambda value: f"video::{value}",
    )
    monkeypatch.setattr(qwen_vl_utils, "_load_external_module", lambda: external)

    fetch_image, fetch_video = qwen_vl_utils._resolve_helpers()

    assert fetch_image("sample.png") == "image::sample.png"
    assert fetch_video("sample.mp4") == "video::sample.mp4"
