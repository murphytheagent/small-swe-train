from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_sdpo_turn_integrity.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_integrity_script_passes_valid_fixture(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1],
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [2, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
        }
    ]
    input_path = tmp_path / "valid.jsonl"
    _write_jsonl(input_path, rows)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--turn-supervision-mode",
            "next_turn",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Integrity check passed." in result.stdout


def test_integrity_script_fails_on_mask_subset_violation(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 0],
            "turn_response_masks": [[1, 1]],
        }
    ]
    input_path = tmp_path / "subset_violation.jsonl"
    _write_jsonl(input_path, rows)

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--input", str(input_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not a subset of _response_mask" in result.stdout


def test_integrity_script_fails_on_overlong_turn_response_mask(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 0],
            "turn_response_masks": [[1, 0, 1]],
        }
    ]
    input_path = tmp_path / "overlong_mask.jsonl"
    _write_jsonl(input_path, rows)

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--input", str(input_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "turn_response_mask length" in result.stdout
    assert "exceeds _response_mask length" in result.stdout


def test_integrity_script_fails_on_missing_response_mask(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "Fix issue",
            "trajectory_assistant_turns": ["turn-0", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [1, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
        }
    ]
    input_path = tmp_path / "missing_response_mask.jsonl"
    _write_jsonl(input_path, rows)

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--input", str(input_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing or empty _response_mask" in result.stdout


def test_integrity_script_fails_on_target_turn_leakage(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1],
            "trajectory_assistant_turns": ["LEAKY_TURN_TEXT", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [2, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
        }
    ]
    input_path = tmp_path / "leakage_violation.jsonl"
    _write_jsonl(input_path, rows)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--turn-supervision-mode",
            "current_turn",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "target-turn leakage detected" in result.stdout


def test_integrity_script_allows_explicit_no_student_attempt(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1],
            "trajectory_assistant_turns": ["LEAKY_TURN_TEXT", "turn-1"],
            "trajectory_assistant_turn_token_lengths": [2, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
        }
    ]
    input_path = tmp_path / "no_student_attempt.jsonl"
    _write_jsonl(input_path, rows)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--turn-supervision-mode",
            "current_turn",
            "--no-include-student-attempt-for-teacher",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Integrity check passed." in result.stdout


def test_integrity_script_uses_turn_level_truncation_denominator(tmp_path: Path) -> None:
    long_turn = " ".join(f"tok{i}" for i in range(12050))
    rows = [
        {
            "prompt": "Fix issue",
            "_response_mask": [0, 0],
            "trajectory_assistant_turns": ["short turn", long_turn],
            "trajectory_assistant_turn_token_lengths": [0, 0],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
        }
    ]
    input_path = tmp_path / "turn_truncation_denominator.jsonl"
    _write_jsonl(input_path, rows)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--turn-supervision-mode",
            "current_turn",
            "--max-truncation-rate",
            "0.6",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "truncation_rate=0.5000" in result.stdout
    assert "Integrity check passed." in result.stdout
