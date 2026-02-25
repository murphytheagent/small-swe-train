from __future__ import annotations

import builtins
import importlib.util
import sys
import types
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


def test_sitecustomize_preserves_existing_import_wrapper(monkeypatch) -> None:
    calls = {"count": 0}
    original_import = builtins.__import__

    def _wrapped_import(name, globals=None, locals=None, fromlist=(), level=0):
        calls["count"] += 1
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setenv("SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN", "1")
    builtins.__import__ = _wrapped_import
    try:
        sitecustomize.apply_small_swe_runtime_patches()
        with pytest.raises(ModuleNotFoundError):
            builtins.__import__("flash_attn")
        builtins.__import__("math")
    finally:
        builtins.__import__ = original_import

    assert calls["count"] >= 1


def test_sitecustomize_can_install_sdpo_patch_import_guard(monkeypatch) -> None:
    calls = {"count": 0, "module": None}

    fake_patch_module = types.ModuleType("verl_integration.ppo_runtime_patch")

    def _fake_apply_small_swe_sdpo_runtime_patch(module=None):
        calls["count"] += 1
        calls["module"] = module
        return True

    fake_patch_module.apply_small_swe_sdpo_runtime_patch = _fake_apply_small_swe_sdpo_runtime_patch

    fake_verl_pkg = types.ModuleType("verl")
    fake_trainer_pkg = types.ModuleType("verl.trainer")
    fake_ppo_pkg = types.ModuleType("verl.trainer.ppo")
    fake_ray_trainer = types.ModuleType("verl.trainer.ppo.ray_trainer")

    monkeypatch.setitem(sys.modules, "verl_integration.ppo_runtime_patch", fake_patch_module)
    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer", fake_trainer_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo", fake_ppo_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.ray_trainer", fake_ray_trainer)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    original_import = builtins.__import__
    try:
        sitecustomize.apply_small_swe_runtime_patches()
        builtins.__import__("verl.trainer.ppo.ray_trainer")
    finally:
        builtins.__import__ = original_import

    assert calls["count"] >= 1
    assert calls["module"] is fake_ray_trainer
