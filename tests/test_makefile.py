from __future__ import annotations

from pathlib import Path


def _makefile_text() -> str:
    return (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")


def test_makefile_pins_python_abi_for_uv_sync_and_venv() -> None:
    text = _makefile_text()
    assert "PYTHON_VERSION ?= 3.13" in text
    assert "$(UV) sync --python $(PYTHON_VERSION)" in text
    assert "$(UV) venv --python $(PYTHON_VERSION)" in text

