"""Shared runtime filesystem path defaults."""

from __future__ import annotations

from pathlib import Path


DEFAULT_SDPO_TASK_CACHE_RELATIVE_DIR = Path("data") / "sdpo_task_cache"


def resolve_sdpo_task_cache_dir(*, project_root: str | Path) -> Path:
    """Resolve the canonical SDPO task-cache directory from a project root."""
    return Path(project_root) / DEFAULT_SDPO_TASK_CACHE_RELATIVE_DIR
