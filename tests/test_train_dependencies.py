from __future__ import annotations

from pathlib import Path


def test_train_extra_includes_orjson_for_verl_tracking() -> None:
    pyproject_text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"orjson>=3.10"' in pyproject_text

