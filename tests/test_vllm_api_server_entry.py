from __future__ import annotations

import importlib.util
import runpy
from importlib.machinery import ModuleSpec
from typing import Any

import trainer.vllm_api_server_entry as vllm_entry


def test_should_hide_external_flash_attn_when_env_forced(monkeypatch) -> None:
    monkeypatch.setenv("SMALL_SWE_DISABLE_FLASH_ATTN", "1")
    monkeypatch.setattr(vllm_entry, "_probe_external_flash_attn", lambda: (True, ""))

    should_hide, reason = vllm_entry._should_hide_external_flash_attn()

    assert should_hide is True
    assert reason == "SMALL_SWE_DISABLE_FLASH_ATTN=1"


def test_should_hide_external_flash_attn_when_probe_fails(monkeypatch) -> None:
    monkeypatch.delenv("SMALL_SWE_DISABLE_FLASH_ATTN", raising=False)
    monkeypatch.setattr(
        vllm_entry,
        "_probe_external_flash_attn",
        lambda: (False, "ImportError: undefined symbol"),
    )

    should_hide, reason = vllm_entry._should_hide_external_flash_attn()

    assert should_hide is True
    assert "undefined symbol" in reason


def test_main_installs_find_spec_guard_for_flash_attn(monkeypatch) -> None:
    marker = ModuleSpec(name="marker_mod", loader=None)

    def _fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "flash_attn":
            return ModuleSpec(name="flash_attn", loader=None)
        return marker

    calls: list[tuple[str, str | None]] = []

    def _fake_run_module(mod_name: str, run_name: str | None = None, alter_sys: bool = False):
        calls.append((mod_name, run_name))
        return {"ok": True}

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setattr(vllm_entry, "_should_hide_external_flash_attn", lambda: (True, "test"))
    monkeypatch.setattr(runpy, "run_module", _fake_run_module)

    vllm_entry.main()

    assert importlib.util.find_spec("flash_attn") is None
    assert importlib.util.find_spec("another_mod") is marker
    assert calls == [("vllm.entrypoints.openai.api_server", "__main__")]


def test_clear_cached_flash_attn_modules_removes_parent_and_children() -> None:
    original_root = vllm_entry.sys.modules.get("flash_attn")
    original_child = vllm_entry.sys.modules.get("flash_attn.flash_attn_interface")
    original_other = vllm_entry.sys.modules.get("not_flash_attn")
    sentinel_root = object()
    sentinel_child = object()
    sentinel_other = object()
    try:
        vllm_entry.sys.modules["flash_attn"] = sentinel_root
        vllm_entry.sys.modules["flash_attn.flash_attn_interface"] = sentinel_child
        vllm_entry.sys.modules["not_flash_attn"] = sentinel_other

        vllm_entry._clear_cached_flash_attn_modules()

        assert "flash_attn" not in vllm_entry.sys.modules
        assert "flash_attn.flash_attn_interface" not in vllm_entry.sys.modules
        assert vllm_entry.sys.modules["not_flash_attn"] is sentinel_other
    finally:
        if original_root is None:
            vllm_entry.sys.modules.pop("flash_attn", None)
        else:
            vllm_entry.sys.modules["flash_attn"] = original_root
        if original_child is None:
            vllm_entry.sys.modules.pop("flash_attn.flash_attn_interface", None)
        else:
            vllm_entry.sys.modules["flash_attn.flash_attn_interface"] = original_child
        if original_other is None:
            vllm_entry.sys.modules.pop("not_flash_attn", None)
        else:
            vllm_entry.sys.modules["not_flash_attn"] = original_other
