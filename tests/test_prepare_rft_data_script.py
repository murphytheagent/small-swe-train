from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_prepare_rft_data_script_processes_100_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "trajectories.jsonl"
    output_path = tmp_path / "rft_rows.jsonl"
    summary_path = tmp_path / "summary.json"

    with input_path.open("w", encoding="utf-8") as handle:
        for idx in range(100):
            payload = {
                "prompt": f"task-{idx}",
                "step_index": idx,
                "assistant_response": "",
                "external_tool_calls": [{"tool": "answer", "args": {"answer": f"fixed-{idx}"}}],
                "tool_output": {"stdout": "", "stderr": "", "exit_code": 0},
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    cmd = [
        sys.executable,
        "scripts/prepare_rft_data.py",
        "--mode",
        "preprocess",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--summary-output",
        str(summary_path),
        "--validate-min-rows",
        "100",
        "--min-format-valid-rate",
        "1.0",
    ]
    result = subprocess.run(
        cmd,
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
    )

    summary_stdout = json.loads(result.stdout.strip())
    summary_file = json.loads(summary_path.read_text(encoding="utf-8"))
    output_lines = output_path.read_text(encoding="utf-8").splitlines()

    assert summary_stdout["total_rows"] == 100
    assert summary_stdout["format_valid_rows"] == 100
    assert summary_file["format_valid_rate"] == 1.0
    assert len(output_lines) == 100


def test_prepare_rft_data_script_enforces_min_row_validation(tmp_path: Path) -> None:
    input_path = tmp_path / "small.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "prompt": "task-a",
                    "assistant_response": "",
                    "external_tool_calls": [{"tool": "answer", "args": {"answer": "fixed"}}],
                }
            ],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "scripts/prepare_rft_data.py",
        "--mode",
        "preprocess",
        "--input",
        str(input_path),
        "--output",
        str(tmp_path / "out.jsonl"),
        "--validate-min-rows",
        "100",
    ]
    result = subprocess.run(
        cmd,
        cwd=_repo_root(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Row count check failed" in result.stderr
