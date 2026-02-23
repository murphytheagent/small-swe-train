"""Fallback shim for optional qwen_vl_utils dependency.

The upstream verl multiturn dataset imports `qwen_vl_utils` unconditionally for
vision helpers, even in text-only training flows. This project's RFT/SFT path
is text-only, so provide a lightweight shim that keeps imports working and
fails fast only when vision helpers are actually invoked.
"""

from __future__ import annotations

import sys
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path
from types import ModuleType
from typing import Callable

_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent


def _unsupported(*_args, **_kwargs):
    raise RuntimeError(
        "qwen_vl_utils vision helpers are unavailable in this runtime. "
        "Install and validate full vision dependencies to enable image/video data."
    )


def _iter_external_search_paths() -> list[str]:
    search_paths: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if resolved == _THIS_DIR:
            continue
        search_paths.append(entry)
    return search_paths


def _load_external_module() -> ModuleType | None:
    search_paths = _iter_external_search_paths()
    if not search_paths:
        return None

    spec = PathFinder.find_spec("qwen_vl_utils", search_paths)
    if spec is None or spec.loader is None:
        return None

    origin = getattr(spec, "origin", None)
    if isinstance(origin, str):
        try:
            if Path(origin).resolve() == _THIS_FILE:
                return None
        except OSError:
            return None

    module = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def _resolve_helper(module: ModuleType, name: str) -> Callable[..., object]:
    candidate = getattr(module, name, None)
    if callable(candidate):
        return candidate
    return _unsupported


def _resolve_helpers() -> tuple[Callable[..., object], Callable[..., object]]:
    external = _load_external_module()
    if external is None:
        return _unsupported, _unsupported
    return _resolve_helper(external, "fetch_image"), _resolve_helper(external, "fetch_video")


fetch_image, fetch_video = _resolve_helpers()
