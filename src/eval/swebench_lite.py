"""SWE-bench Lite evaluation scaffold signatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EpisodeResult:
    instance_id: str
    resolved: bool
    summary: str


def evaluate_swebench_lite(
    *,
    episodes: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> list[EpisodeResult]:
    """Return per-episode evaluation records (scaffold only)."""
    raise NotImplementedError("Benchmark evaluator wiring is not implemented yet.")
