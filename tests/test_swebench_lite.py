from __future__ import annotations

import pytest

from eval.swebench_lite import compare_resolve_rates, evaluate_swebench_lite, summarize_episode_results


def test_evaluate_swebench_lite_maps_predictions() -> None:
    episodes = [{"instance_id": "swe-1"}, {"instance_id": "swe-2"}]
    predictions = [
        {"instance_id": "swe-1", "resolved": True, "summary": "patch applies"},
        {"instance_id": "swe-2", "score": 0.0},
    ]

    results = evaluate_swebench_lite(episodes=episodes, predictions=predictions)

    assert [result.instance_id for result in results] == ["swe-1", "swe-2"]
    assert [result.resolved for result in results] == [True, False]
    assert results[0].summary == "patch applies"


def test_evaluate_swebench_lite_requires_instance_id() -> None:
    with pytest.raises(ValueError, match="instance_id"):
        evaluate_swebench_lite(episodes=[{}], predictions=[])


def test_summarize_episode_results_computes_resolve_rate() -> None:
    results = evaluate_swebench_lite(
        episodes=[{"instance_id": "swe-1"}, {"instance_id": "swe-2"}, {"instance_id": "swe-3"}],
        predictions=[
            {"instance_id": "swe-1", "resolved": True},
            {"instance_id": "swe-2", "score": 1.0},
        ],
    )

    summary = summarize_episode_results(results)

    assert summary.total_episodes == 3
    assert summary.resolved_episodes == 2
    assert summary.unresolved_episodes == 1
    assert summary.resolve_rate == pytest.approx(2 / 3)


def test_compare_resolve_rates_returns_delta() -> None:
    baseline = evaluate_swebench_lite(
        episodes=[{"instance_id": "swe-1"}, {"instance_id": "swe-2"}],
        predictions=[{"instance_id": "swe-1", "resolved": True}],
    )
    candidate = evaluate_swebench_lite(
        episodes=[{"instance_id": "swe-1"}, {"instance_id": "swe-2"}],
        predictions=[
            {"instance_id": "swe-1", "resolved": True},
            {"instance_id": "swe-2", "resolved": True},
        ],
    )

    comparison = compare_resolve_rates(
        baseline_results=baseline,
        candidate_results=candidate,
    )

    assert comparison["baseline_resolve_rate"] == 0.5
    assert comparison["candidate_resolve_rate"] == 1.0
    assert comparison["resolve_rate_delta"] == 0.5


def test_evaluate_swebench_lite_treats_false_string_as_unresolved() -> None:
    results = evaluate_swebench_lite(
        episodes=[{"instance_id": "swe-1"}],
        predictions=[{"instance_id": "swe-1", "resolved": "false"}],
    )

    assert len(results) == 1
    assert results[0].resolved is False
    assert results[0].summary == "unresolved"
