from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_step_sdpo_scaffold_script_writes_expected_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "batch.jsonl"
    output_dir = tmp_path / "sdpo_outputs"

    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "Fix failing unit test.",
                        "response_text": '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
                        "resolved": True,
                    },
                    ensure_ascii=True,
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "scripts/run_step_sdpo_scaffold.py",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(
        cmd,
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(result.stdout)
    summary_path = Path(stdout_payload["summary_path"])
    rollout_rows_path = Path(stdout_payload["rollout_rows_path"])
    teacher_prompts_path = Path(stdout_payload["teacher_prompts_path"])

    assert summary_path.exists()
    assert rollout_rows_path.exists()
    assert teacher_prompts_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["row_count"] == 1
    assert summary["reward_summary"]["mean"] == 1.0
    assert summary["training_stats"]["loss"] == 0.0

    teacher_rows = [
        json.loads(line)
        for line in teacher_prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(teacher_rows) == 1
    assert teacher_rows[0]["self_distillation_mask"] is True
    assert teacher_rows[0]["reward"] == 1.0
