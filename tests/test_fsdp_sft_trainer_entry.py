from __future__ import annotations

import importlib

import pytest


def _load_entry_module():
    pytest.importorskip("transformers")
    return importlib.import_module("verl_integration.fsdp_sft_trainer_entry")


def test_patched_from_pretrained_uses_sdpa_fallback_when_flash_attn_disabled(
    monkeypatch,
) -> None:
    entry = _load_entry_module()

    captured: dict[str, object] = {}

    def _fake_from_pretrained(*args, **kwargs):
        del args
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(entry, "_ORIGINAL_FROM_PRETRAINED", _fake_from_pretrained)
    monkeypatch.setattr(entry, "_FLASH_ATTN_DISABLED", True)
    monkeypatch.delenv("SMALL_SWE_RFT_ATTN_IMPL", raising=False)
    monkeypatch.delenv("SMALL_SWE_FALLBACK_ATTN_IMPL", raising=False)

    payload = entry._patched_from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    assert payload["attn_implementation"] == "sdpa"
    assert payload["use_flash_attention_2"] is False
    assert captured["attn_implementation"] == "sdpa"
    assert captured["use_flash_attention_2"] is False


def test_patched_from_pretrained_honors_explicit_attn_impl_override(monkeypatch) -> None:
    entry = _load_entry_module()

    captured: dict[str, object] = {}

    def _fake_from_pretrained(*args, **kwargs):
        del args
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(entry, "_ORIGINAL_FROM_PRETRAINED", _fake_from_pretrained)
    monkeypatch.setattr(entry, "_FLASH_ATTN_DISABLED", True)
    monkeypatch.setenv("SMALL_SWE_RFT_ATTN_IMPL", "flash_attention_2")

    payload = entry._patched_from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    assert payload["attn_implementation"] == "flash_attention_2"
    assert "use_flash_attention_2" not in payload
    assert captured["attn_implementation"] == "flash_attention_2"
