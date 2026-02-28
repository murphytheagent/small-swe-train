from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import config
from prompts.model_delimiters import default_delimiters


def _write_model_config(path: Path, *, model_family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"model_family: {model_family}",
                "delimiters:",
                '  role_start: "<|im_start|>"',
                '  role_end: "<|im_end|>"',
                '  think_start: "<think>"',
                '  think_end: "</think>"',
                '  tool_call_start: "<tool_call>"',
                '  tool_call_end: "</tool_call>"',
                '  tool_response_start: "<tool_response>"',
                '  tool_response_end: "</tool_response>"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_resolve_model_config_path_prefers_repo_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_dir = tmp_path / "override"
    bundled_dir = tmp_path / "bundled"
    override_path = override_dir / "qwen3.yaml"
    bundled_path = bundled_dir / "qwen3.yaml"
    _write_model_config(override_path, model_family="override-qwen3")
    _write_model_config(bundled_path, model_family="bundled-qwen3")

    monkeypatch.setattr(config, "_MODEL_CONFIG_OVERRIDE_DIR", override_dir)
    monkeypatch.setattr(config, "_BUNDLED_MODEL_CONFIGS_DIR", bundled_dir)

    resolved = config.resolve_model_config_path("qwen3")
    assert resolved == override_path

    default_delimiters.cache_clear()
    delimiters = default_delimiters("qwen3")
    assert delimiters.model_family == "override-qwen3"
    default_delimiters.cache_clear()


def test_resolve_model_config_path_falls_back_to_bundled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_dir = tmp_path / "override"
    bundled_dir = tmp_path / "bundled"
    bundled_path = bundled_dir / "qwen3.yaml"
    _write_model_config(bundled_path, model_family="bundled-qwen3")

    monkeypatch.setattr(config, "_MODEL_CONFIG_OVERRIDE_DIR", override_dir)
    monkeypatch.setattr(config, "_BUNDLED_MODEL_CONFIGS_DIR", bundled_dir)

    resolved = config.resolve_model_config_path("qwen3")
    assert resolved == bundled_path


def test_resolve_model_config_path_raises_for_missing_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_dir = tmp_path / "override"
    bundled_dir = tmp_path / "bundled"
    monkeypatch.setattr(config, "_MODEL_CONFIG_OVERRIDE_DIR", override_dir)
    monkeypatch.setattr(config, "_BUNDLED_MODEL_CONFIGS_DIR", bundled_dir)

    with pytest.raises(FileNotFoundError, match="No model config found"):
        config.resolve_model_config_path("missing-family")


def test_repo_model_config_overrides_match_bundled_defaults() -> None:
    override_dir = config._MODEL_CONFIG_OVERRIDE_DIR
    bundled_dir = config._BUNDLED_MODEL_CONFIGS_DIR

    bundled_paths = sorted(bundled_dir.glob("*.yaml"))
    assert bundled_paths, "No bundled model delimiter configs found."

    for bundled_path in bundled_paths:
        override_path = override_dir / bundled_path.name
        assert override_path.is_file(), f"Missing repo override for {bundled_path.name}"
        override_payload = yaml.safe_load(override_path.read_text(encoding="utf-8"))
        bundled_payload = yaml.safe_load(bundled_path.read_text(encoding="utf-8"))
        assert override_payload == bundled_payload, (
            f"Delimiter override drift for {bundled_path.name}; "
            "sync configs/model with src/prompts/model_configs."
        )
