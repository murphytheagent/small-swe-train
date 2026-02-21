from __future__ import annotations

import pytest

from eval.swebench_lite import evaluate_swebench_lite


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
