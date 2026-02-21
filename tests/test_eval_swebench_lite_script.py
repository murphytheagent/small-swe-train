from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_eval_swebench_lite_script_reports_summary_and_comparison(tmp_path: Path) -> None:
    episodes_path = tmp_path / "episodes.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "eval_summary.json"

    episodes_path.write_text(
        json.dumps([{"instance_id": "swe-1"}, {"instance_id": "swe-2"}], ensure_ascii=True),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps([{"instance_id": "swe-1", "resolved": True}], ensure_ascii=True),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            [
                {"instance_id": "swe-1", "resolved": True},
                {"instance_id": "swe-2", "score": 1.0},
            ],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "scripts/eval_swebench_lite.py",
        "--episodes",
        str(episodes_path),
        "--predictions",
        str(candidate_path),
        "--baseline-predictions",
        str(baseline_path),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(result.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload["candidate_summary"]["resolve_rate"] == 1.0
    assert stdout_payload["comparison"]["baseline_resolve_rate"] == 0.5
    assert stdout_payload["comparison"]["resolve_rate_delta"] == 0.5
    assert file_payload == stdout_payload
