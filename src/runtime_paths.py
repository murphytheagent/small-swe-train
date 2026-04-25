"""Shared runtime filesystem path defaults."""

from __future__ import annotations

from pathlib import Path


DEFAULT_SDPO_TASK_CACHE_RELATIVE_DIR = Path("data") / "sdpo_task_cache"
DEFAULT_ON_POLICY_BAD_TASK_CACHE_RELATIVE_DIR = Path("data") / "on_policy_bad_task_cache"
DEFAULT_ON_POLICY_DIFFICULTY_BAND_CACHE_RELATIVE_DIR = (
    Path("data") / "on_policy_difficulty_band_cache"
)


def resolve_sdpo_task_cache_dir(*, project_root: str | Path) -> Path:
    """Resolve the canonical SDPO task-cache directory from a project root."""
    return Path(project_root) / DEFAULT_SDPO_TASK_CACHE_RELATIVE_DIR


def resolve_on_policy_bad_task_cache_dir(*, project_root: str | Path) -> Path:
    """Resolve the canonical on-policy bad-task cache directory from a project root."""
    return Path(project_root) / DEFAULT_ON_POLICY_BAD_TASK_CACHE_RELATIVE_DIR


def resolve_on_policy_difficulty_band_cache_dir(*, project_root: str | Path) -> Path:
    """Resolve the canonical on-policy difficulty-band cache directory from a project root."""
    return Path(project_root) / DEFAULT_ON_POLICY_DIFFICULTY_BAND_CACHE_RELATIVE_DIR
