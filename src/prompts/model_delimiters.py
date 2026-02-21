"""Model-family delimiter configuration, loadable from YAML.

The YAML files under ``configs/model/`` are the single source of truth for
delimiter strings.  Call ``default_delimiters()`` (cached) or
``load_delimiters(path)`` to obtain a ``ModelDelimiters`` instance.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "model"


@dataclass(frozen=True)
class ModelDelimiters:
    model_family: str
    role_start: str
    role_end: str
    think_start: str
    think_end: str
    tool_call_start: str
    tool_call_end: str
    tool_response_start: str
    tool_response_end: str


def load_delimiters(config_path: str | Path) -> ModelDelimiters:
    """Load a ModelDelimiters instance from a YAML file.

    Requires ``pyyaml`` (listed as a project dependency).
    """
    import yaml

    path = Path(config_path)
    with path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    delims = raw["delimiters"]
    return ModelDelimiters(
        model_family=raw["model_family"],
        role_start=delims["role_start"],
        role_end=delims["role_end"],
        think_start=delims["think_start"],
        think_end=delims["think_end"],
        tool_call_start=delims["tool_call_start"],
        tool_call_end=delims["tool_call_end"],
        tool_response_start=delims["tool_response_start"],
        tool_response_end=delims["tool_response_end"],
    )


@functools.lru_cache(maxsize=4)
def default_delimiters(model_family: str = "qwen3") -> ModelDelimiters:
    """Load delimiters for *model_family* from ``configs/model/<family>.yaml``.

    Results are cached; subsequent calls with the same family are free.
    """
    return load_delimiters(_CONFIGS_DIR / f"{model_family}.yaml")
