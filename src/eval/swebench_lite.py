"""SWE-bench Lite evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EpisodeResult:
    instance_id: str
    resolved: bool
    summary: str


@dataclass(frozen=True)
class EvaluationSummary:
    total_episodes: int
    resolved_episodes: int
    unresolved_episodes: int
    resolve_rate: float


def _prediction_lookup(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        instance_id = prediction.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            lookup[instance_id] = prediction
    return lookup


def evaluate_swebench_lite(
    *,
    episodes: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> list[EpisodeResult]:
    """Return per-episode resolution records from evaluator outputs.

    A prediction resolves an episode when ``resolved`` is true, or when
    ``score`` is a numeric value greater than or equal to 1.0.
    """
    predictions_by_instance = _prediction_lookup(predictions)

    results: list[EpisodeResult] = []
    for episode in episodes:
        instance_id = str(episode.get("instance_id", "")).strip()
        if not instance_id:
            raise ValueError("Every episode must provide a non-empty 'instance_id'.")

        prediction = predictions_by_instance.get(instance_id)
        if prediction is None:
            results.append(
                EpisodeResult(
                    instance_id=instance_id,
                    resolved=False,
                    summary="missing prediction",
                )
            )
            continue

        raw_score = prediction.get("score")
        numeric_score = float(raw_score) if isinstance(raw_score, (int, float)) else None

        resolved = bool(prediction.get("resolved", False))
        if numeric_score is not None:
            resolved = resolved or numeric_score >= 1.0

        summary = str(prediction.get("summary") or ("resolved" if resolved else "unresolved"))
        results.append(
            EpisodeResult(
                instance_id=instance_id,
                resolved=resolved,
                summary=summary,
            )
        )

    return results


def summarize_episode_results(results: Sequence[EpisodeResult]) -> EvaluationSummary:
    total_episodes = len(results)
    resolved_episodes = sum(1 for result in results if result.resolved)
    unresolved_episodes = total_episodes - resolved_episodes
    resolve_rate = (resolved_episodes / total_episodes) if total_episodes else 0.0
    return EvaluationSummary(
        total_episodes=total_episodes,
        resolved_episodes=resolved_episodes,
        unresolved_episodes=unresolved_episodes,
        resolve_rate=resolve_rate,
    )


def compare_resolve_rates(
    *,
    baseline_results: Sequence[EpisodeResult],
    candidate_results: Sequence[EpisodeResult],
) -> dict[str, float | int]:
    baseline = summarize_episode_results(baseline_results)
    candidate = summarize_episode_results(candidate_results)
    return {
        "baseline_resolve_rate": baseline.resolve_rate,
        "candidate_resolve_rate": candidate.resolve_rate,
        "resolve_rate_delta": candidate.resolve_rate - baseline.resolve_rate,
        "baseline_resolved": baseline.resolved_episodes,
        "candidate_resolved": candidate.resolved_episodes,
        "baseline_total": baseline.total_episodes,
        "candidate_total": candidate.total_episodes,
    }
