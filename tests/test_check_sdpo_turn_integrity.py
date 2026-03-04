from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_sdpo_turn_integrity.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("check_sdpo_turn_integrity_module", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_integrity_script_allows_repeated_turn_text_without_leakage_false_positive(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "Fix issue",
            "_response_mask": [1, 1, 0, 1],
            "trajectory_assistant_turns": ["repeated output", "repeated output"],
            "trajectory_assistant_turn_token_lengths": [2, 1],
            "trajectory_turn_tool_response_blocks": [["r0"], ["r1"]],
        }
    ]
    input_path = tmp_path / "repeated_turn_text.jsonl"
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

    assert result.returncode == 0
    assert "Integrity check passed." in result.stdout


def test_recent_raw_leakage_detector_flags_target_turn_marker() -> None:
    module = _load_script_module()
    prompt = (
        "[TRAJECTORY_BLOCK]\n"
        "[RECENT_RAW_BLOCK]\n"
        "[TURN_0]\n"
        "[ASSISTANT]\n"
        "safe\n\n"
        "[TURN_2]\n"
        "[ASSISTANT]\n"
        "leak\n"
        "[COMPRESSED_MEMORY_BLOCK]\n"
        "memo\n"
    )

    assert module._recent_raw_block_contains_target_turn(prompt, turn_index=2) is True
    assert module._recent_raw_block_contains_target_turn(prompt, turn_index=1) is False


def test_coerce_binary_mask_parses_string_values() -> None:
    module = _load_script_module()

    parsed = module._coerce_binary_mask(["1", "0", "true", "false", "", "2", "0.0", "0.5"])

    assert parsed == [1, 0, 1, 0, 0, 1, 0, 1]


def test_integrity_script_forwards_verifier_feedback_mode_to_adapter(tmp_path: Path) -> None:
    rows = [{"prompt": "Fix issue", "_response_mask": [1]}]
    input_path = tmp_path / "verifier_feedback_mode.jsonl"
    _write_jsonl(input_path, rows)

    module = _load_script_module()
    call_args: dict[str, object] = {}

    def _fake_build_self_distillation_batch(
        _rows: list[dict],
        *,
        include_student_attempt_for_teacher: bool,
        turn_supervision_mode: str,
        verifier_feedback_mode: str,
    ) -> dict[str, list]:
        call_args["rows"] = _rows
        call_args["include_student_attempt_for_teacher"] = include_student_attempt_for_teacher
        call_args["turn_supervision_mode"] = turn_supervision_mode
        call_args["verifier_feedback_mode"] = verifier_feedback_mode
        return {
            "prompt_truncated": [False],
            "turn_prompt_truncated": [[]],
            "turn_teacher_prompts": [[]],
            "turn_response_masks": [[]],
            "turn_distillation_mask": [[]],
        }

    module.build_self_distillation_batch = _fake_build_self_distillation_batch
    rc = module.main(
        [
            "--input",
            str(input_path),
            "--turn-supervision-mode",
            "current_turn",
            "--verifier-feedback-mode",
            "all_turns",
        ]
    )

    assert rc == 0
    assert call_args["rows"] == rows
    assert call_args["include_student_attempt_for_teacher"] is True
    assert call_args["turn_supervision_mode"] == "current_turn"
    assert call_args["verifier_feedback_mode"] == "all_turns"


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
    long_tool_feedback = " ".join(f"tok{i}" for i in range(12050))
    rows = [
        {
            "prompt": "Fix issue",
            "_response_mask": [0, 0],
            "trajectory_assistant_turns": ["short turn", "second turn"],
            "trajectory_assistant_turn_token_lengths": [0, 0],
            "trajectory_turn_tool_response_blocks": [["r0"], [long_tool_feedback]],
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
