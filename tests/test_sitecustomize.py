from __future__ import annotations

import builtins
import importlib.util
import sys
from importlib.machinery import ModuleSpec

import pytest

import sitecustomize


def test_sitecustomize_can_hide_external_flash_attn(monkeypatch) -> None:
    marker = ModuleSpec(name="marker_mod", loader=None)

    def _fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "flash_attn":
            return ModuleSpec(name="flash_attn", loader=None)
        return marker

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setenv("SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN", "1")
    original_import = builtins.__import__
    original_root = sys.modules.get("flash_attn")
    original_child = sys.modules.get("flash_attn.flash_attn_interface")
    sys.modules["flash_attn"] = object()
    sys.modules["flash_attn.flash_attn_interface"] = object()

    try:
        sitecustomize.apply_small_swe_runtime_patches()

        assert importlib.util.find_spec("flash_attn") is None
        assert importlib.util.find_spec("another_mod") is marker
        assert "flash_attn" not in sys.modules
        assert "flash_attn.flash_attn_interface" not in sys.modules
        with pytest.raises(ModuleNotFoundError):
            builtins.__import__("flash_attn")
    finally:
        builtins.__import__ = original_import
        if original_root is None:
            sys.modules.pop("flash_attn", None)
        else:
            sys.modules["flash_attn"] = original_root
        if original_child is None:
            sys.modules.pop("flash_attn.flash_attn_interface", None)
        else:
            sys.modules["flash_attn.flash_attn_interface"] = original_child
