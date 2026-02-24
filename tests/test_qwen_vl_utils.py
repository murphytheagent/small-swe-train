from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
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


def test_load_external_module_temporarily_registers_external_module_for_exec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[bool] = []

    class Loader:
        def create_module(self, spec: ModuleSpec):  # noqa: D401
            return None

        def exec_module(self, module: object) -> None:
            observed.append(sys.modules.get("qwen_vl_utils") is module)
            setattr(module, "fetch_image", lambda value: f"image::{value}")
            setattr(module, "fetch_video", lambda value: f"video::{value}")

    origin = str(tmp_path / "__init__.py")
    spec = ModuleSpec(name="qwen_vl_utils", loader=Loader(), origin=origin)
    original_module = sys.modules.get("qwen_vl_utils")

    monkeypatch.setattr(qwen_vl_utils, "_iter_external_search_paths", lambda: ["/fake/site-packages"])
    monkeypatch.setattr(qwen_vl_utils.PathFinder, "find_spec", lambda name, paths=None: spec)

    external = qwen_vl_utils._load_external_module()

    assert observed == [True]
    assert external is not None
    assert external.fetch_image("sample.png") == "image::sample.png"
    assert external.fetch_video("sample.mp4") == "video::sample.mp4"
    assert sys.modules.get("qwen_vl_utils") is original_module
