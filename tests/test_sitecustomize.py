from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec

import sitecustomize


def test_sitecustomize_can_hide_external_flash_attn(monkeypatch) -> None:
    marker = ModuleSpec(name="marker_mod", loader=None)

    def _fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "flash_attn":
            return ModuleSpec(name="flash_attn", loader=None)
        return marker

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setenv("SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN", "1")

    sitecustomize.apply_small_swe_runtime_patches()

    assert importlib.util.find_spec("flash_attn") is None
    assert importlib.util.find_spec("another_mod") is marker
