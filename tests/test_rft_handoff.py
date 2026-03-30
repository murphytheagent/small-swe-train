from __future__ import annotations

import json
from pathlib import Path

from trainer.rft_handoff import collect_rft_sft_batch_for_steps


class _EmptyCollector:
    def collect_step(self, step_index: int):
        assert step_index == 0
        return []


def test_collect_rft_sft_batch_for_steps_allows_empty_eval_collection(tmp_path: Path) -> None:
    result = collect_rft_sft_batch_for_steps(
        total_steps=1,
        collector=_EmptyCollector(),
        tokenizer=object(),
        output_dir=tmp_path,
    )

    assert result["rollout_rows"] == []
    assert result["selected_rows"] == []
    assert result["rejected_rows"] == []
    assert result["sft_batch"]["meta_info"]["selected_count"] == 0
    assert result["dataproto_payload"]["meta_info"]["selected_count"] == 0

    summary = json.loads((tmp_path / "rollout_artifact_summary.json").read_text(encoding="utf-8"))
    assert summary["rollout_row_count"] == 0
    assert summary["selected_count"] == 0
    assert summary["rejected_count"] == 0
