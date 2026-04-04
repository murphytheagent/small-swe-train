from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from env.runtime_protocol import ToolResponse
from env.shell_helpers import (
    build_executable_resolver_shell,
    build_python_interpreter_resolver_shell,
)
from verl_integration.submission_verifier import (
    _VERIFY_SCRIPT,
    _build_verifier_shell_command,
    run_submission_verifier,
)


@dataclass
class _FakeExecutor:
    responses: list[ToolResponse]

    def __post_init__(self) -> None:
        self.requests: list[object] = []

    def run(self, request):  # noqa: ANN001 - protocol-compatible test shim
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Unexpected verifier invocation with no queued response.")
        return self.responses.pop(0)


@dataclass
class _LocalShellExecutor:
    cwd: Path

    def __post_init__(self) -> None:
        self.requests: list[object] = []

    def run(self, request):  # noqa: ANN001 - protocol-compatible test shim
        self.requests.append(request)
        completed = subprocess.run(
            ["bash", "-lc", str(request.args["command"])],
            cwd=self.cwd,
            input=str(request.args.get("stdin", "")),
            text=True,
            capture_output=True,
            check=False,
            timeout=int(request.args.get("timeout_sec", 30)),
            env={
                **os.environ,
                "TASK_REPO_ROOT": str(self.cwd),
                "SMALL_SWE_REPO_ROOT": str(self.cwd),
            },
        )
        return ToolResponse(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )


def test_run_submission_verifier_uses_test_results_for_binary_resolution() -> None:
    executor = _FakeExecutor(
        responses=[
            ToolResponse(
                stdout='{"results":{"tests/test_bug.py::test_fix":true},"executed":1,"all_passed":true,"failures":{}}',
                stderr="",
                exit_code=0,
            ),
            ToolResponse(
                stdout='{"results":{"tests/test_ok.py::test_regression":false},"executed":1,"all_passed":false,"failures":{"tests/test_ok.py::test_regression":{"returncode":1}}}',
                stderr="",
                exit_code=0,
            ),
        ]
    )

    result = run_submission_verifier(
        executor=executor,
        fail_to_pass=["tests/test_bug.py::test_fix"],
        pass_to_pass=["tests/test_ok.py::test_regression"],
        verifier_timeout_sec=120,
        final_response="patched",
    )

    assert result["resolved"] is False
    assert result["fail_to_pass_verified"] is True
    assert result["pass_to_pass_verified"] is False
    assert result["fail_to_pass_results"] == {"tests/test_bug.py::test_fix": True}
    assert result["pass_to_pass_results"] == {"tests/test_ok.py::test_regression": False}
    assert result["submission_final_response"] == "patched"
    assert len(executor.requests) == 2
    for request in executor.requests:
        assert request.tool == "bash"
        assert "SMALL_SWE_TESTS_JSON=" in request.args["command"]
        assert "stdin" in request.args


def test_run_submission_verifier_marks_command_failures_unresolved() -> None:
    executor = _FakeExecutor(
        responses=[
            ToolResponse(stdout="", stderr="pytest unavailable", exit_code=127),
            ToolResponse(stdout='{"results":{},"executed":0,"all_passed":true,"failures":{}}', stderr="", exit_code=0),
        ]
    )

    result = run_submission_verifier(
        executor=executor,
        fail_to_pass=["tests/test_bug.py::test_fix"],
        pass_to_pass=[],
        verifier_timeout_sec=30,
        final_response="patched",
    )

    assert result["resolved"] is False
    assert result["fail_to_pass_verified"] is False
    assert "verifier command failed" in result["verification_feedback"]


def test_run_submission_verifier_handles_missing_test_groups() -> None:
    executor = _FakeExecutor(responses=[])
    result = run_submission_verifier(
        executor=executor,
        fail_to_pass=[],
        pass_to_pass=[],
        verifier_timeout_sec=30,
        final_response="done",
    )

    assert result["verification_missing"] is True
    assert result["resolved"] is True
    assert executor.requests == []


def test_run_submission_verifier_uses_generous_group_timeout_budget() -> None:
    executor = _FakeExecutor(
        responses=[
            ToolResponse(
                stdout='{"results":{"tests/test_bug.py::test_a":true,"tests/test_bug.py::test_b":true},"executed":2,"all_passed":true,"failures":{}}',
                stderr="",
                exit_code=0,
            )
        ]
    )

    run_submission_verifier(
        executor=executor,
        fail_to_pass=["tests/test_bug.py::test_a", "tests/test_bug.py::test_b"],
        pass_to_pass=[],
        verifier_timeout_sec=30,
        final_response="patched",
    )

    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.tool == "bash"
    # 2 tests * 180s each + 120s buffer
    assert request.args["timeout_sec"] == 480
    assert "SMALL_SWE_PER_TEST_TIMEOUT_SEC=180" in request.args["command"]


def test_build_verifier_shell_command_uses_shared_python_resolution() -> None:
    command = _build_verifier_shell_command(
        tests_json='["tests/test_bug.py::test_fix"]',
        per_test_timeout_sec=180,
        verifier_kind="pytest",
    )

    assert build_python_interpreter_resolver_shell(var_name="pybin") in command
    assert 'SMALL_SWE_PYBIN="${pybin}"' in command


def test_build_verifier_shell_command_resolves_go_binary_from_common_paths() -> None:
    command = _build_verifier_shell_command(
        tests_json='["TestBug"]',
        per_test_timeout_sec=180,
        verifier_kind="go_test",
    )

    assert build_executable_resolver_shell(
        var_name="gobin",
        command_names=("go",),
        fallback_paths=("/usr/local/go/bin/go", "/usr/lib/go/bin/go", "/opt/go/bin/go"),
        not_found_message="Go executable missing in task container.",
    ) in command
    assert 'SMALL_SWE_GO_BIN="${gobin}"' in command


def test_run_submission_verifier_parses_large_json_payload_from_full_stdout() -> None:
    payload = {
        "results": {"tests/test_bug.py::test_fix": True},
        "executed": 1,
        "all_passed": True,
        "failures": {},
        "padding": "x" * 6000,
    }
    executor = _FakeExecutor(
        responses=[
            ToolResponse(
                stdout=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                stderr="",
                exit_code=0,
            )
        ]
    )

    result = run_submission_verifier(
        executor=executor,
        fail_to_pass=["tests/test_bug.py::test_fix"],
        pass_to_pass=[],
        verifier_timeout_sec=120,
        final_response="patched",
    )

    assert result["resolved"] is True
    assert result["verification_error"] == ""
    assert result["fail_to_pass_results"] == {"tests/test_bug.py::test_fix": True}


def test_run_submission_verifier_returns_group_diagnostics() -> None:
    executor = _FakeExecutor(
        responses=[
            ToolResponse(
                stdout=json.dumps(
                    {
                        "results": {
                            "tests/test_bug.py::test_a": False,
                            "tests/test_bug.py::test_b": True,
                        },
                        "executed": 2,
                        "all_passed": False,
                        "failures": {
                            "tests/test_bug.py::test_a": {
                                "returncode": 1,
                                "command": [
                                    "python3",
                                    "-m",
                                    "pytest",
                                    "-q",
                                    "tests/test_bug.py::test_a",
                                ],
                            }
                        },
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                stderr="container stderr",
                exit_code=0,
            ),
            ToolResponse(
                stdout=json.dumps(
                    {
                        "results": {"tests/test_ok.py::test_regression": True},
                        "executed": 1,
                        "all_passed": True,
                        "failures": {},
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                stderr="",
                exit_code=0,
            ),
        ]
    )

    result = run_submission_verifier(
        executor=executor,
        fail_to_pass=["tests/test_bug.py::test_a", "tests/test_bug.py::test_b"],
        pass_to_pass=["tests/test_ok.py::test_regression"],
        verifier_timeout_sec=120,
        final_response="patched",
    )

    assert result["fail_to_pass_results"] == {
        "tests/test_bug.py::test_a": False,
        "tests/test_bug.py::test_b": True,
    }
    assert result["fail_to_pass_failures"]["tests/test_bug.py::test_a"]["returncode"] == 1
    assert result["fail_to_pass_stderr_tail"] == "container stderr"
    assert result["fail_to_pass_failures"]["tests/test_bug.py::test_a"]["command"] == [
        "python3",
        "-m",
        "pytest",
        "-q",
        "tests/test_bug.py::test_a",
    ]


def test_run_submission_verifier_marks_unimplemented_runtime_verifier_kind_invalid() -> None:
    executor = _FakeExecutor(responses=[])

    result = run_submission_verifier(
        executor=executor,
        fail_to_pass=["tests/smoke.js::testBug"],
        pass_to_pass=["tests/smoke.js::testRegression"],
        verifier_timeout_sec=30,
        final_response="patched",
        verifier_kind="node_test",
    )

    assert result["resolved"] is False
    assert result["infra_invalid"] is True
    assert result["invalid_reason"] == "verifier_crash"
    assert "not implemented" in result["verification_feedback"]
    assert executor.requests == []


def test_run_submission_verifier_executes_go_test_targets_locally(tmp_path: Path) -> None:
    if shutil.which("go") is None:
        pytest.skip("go toolchain unavailable")

    (tmp_path / "go.mod").write_text("module example.com/sample\n\ngo 1.20\n", encoding="utf-8")
    (tmp_path / "calc.go").write_text(
        "\n".join(
            [
                "package sample",
                "",
                "func add(a, b int) int { return a + b }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "calc_test.go").write_text(
        "\n".join(
            [
                "package sample",
                "",
                'import "testing"',
                "",
                "func TestBug(t *testing.T) {",
                "    if got := add(2, 2); got != 4 {",
                '        t.Fatalf("want 4, got %d", got)',
                "    }",
                "}",
                "",
                "func TestRegression(t *testing.T) {",
                "    if got := add(5, 3); got != 8 {",
                '        t.Fatalf("want 8, got %d", got)',
                "    }",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    executor = _LocalShellExecutor(cwd=tmp_path)
    result = run_submission_verifier(
        executor=executor,
        fail_to_pass=["TestBug"],
        pass_to_pass=["TestRegression"],
        verifier_timeout_sec=60,
        final_response="patched",
        verifier_kind="go_test",
    )

    assert result["resolved"] is True
    assert result["verifier_kind"] == "go_test"
    assert result["infra_invalid"] is False
    assert result["fail_to_pass_results"] == {"TestBug": True}
    assert result["pass_to_pass_results"] == {"TestRegression": True}
    assert len(executor.requests) == 2


def test_run_submission_verifier_marks_missing_go_test_targets_failed_locally(tmp_path: Path) -> None:
    if shutil.which("go") is None:
        pytest.skip("go toolchain unavailable")

    (tmp_path / "go.mod").write_text("module example.com/sample\n\ngo 1.20\n", encoding="utf-8")
    (tmp_path / "calc.go").write_text(
        "\n".join(
            [
                "package sample",
                "",
                "func add(a, b int) int { return a + b }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "calc_test.go").write_text(
        "\n".join(
            [
                "package sample",
                "",
                'import "testing"',
                "",
                "func TestRegression(t *testing.T) {",
                "    if got := add(5, 3); got != 8 {",
                '        t.Fatalf("want 8, got %d", got)',
                "    }",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    executor = _LocalShellExecutor(cwd=tmp_path)
    result = run_submission_verifier(
        executor=executor,
        fail_to_pass=["TestMissing"],
        pass_to_pass=["TestRegression"],
        verifier_timeout_sec=60,
        final_response="patched",
        verifier_kind="go_test",
    )

    assert result["resolved"] is False
    assert result["infra_invalid"] is False
    assert result["fail_to_pass_results"] == {"TestMissing": False}
    assert result["fail_to_pass_failures"]["TestMissing"]["selector_missing"] is True
    assert result["pass_to_pass_results"] == {"TestRegression": True}


def test_verify_script_runs_all_tests_after_a_failure(tmp_path: Path) -> None:
    test_file = tmp_path / "test_ordering.py"
    test_file.write_text(
        "\n".join(
            [
                "def test_first_fail():",
                "    assert False",
                "",
                "def test_second_pass():",
                "    assert True",
                "",
                "def test_third_pass():",
                "    assert True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-c", _VERIFY_SCRIPT],
        cwd=tmp_path,
        env={
            **os.environ,
            "SMALL_SWE_TESTS_JSON": json.dumps(
                [
                    "test_ordering.py::test_first_fail",
                    "test_ordering.py::test_second_pass",
                    "test_ordering.py::test_third_pass",
                ]
            ),
            "TASK_REPO_ROOT": str(tmp_path),
            "SMALL_SWE_PYBIN": sys.executable,
            "SMALL_SWE_PER_TEST_TIMEOUT_SEC": "30",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["executed"] == 3
    assert payload["results"] == {
        "test_ordering.py::test_first_fail": False,
        "test_ordering.py::test_second_pass": True,
        "test_ordering.py::test_third_pass": True,
    }
    assert payload["all_passed"] is False
    assert payload["failures"]["test_ordering.py::test_first_fail"]["command"] == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "test_ordering.py::test_first_fail",
    ]


def test_verify_script_uses_explicit_go_binary_when_path_missing(tmp_path: Path) -> None:
    fake_go = tmp_path / "fake-go"
    log_path = tmp_path / "fake-go.log"
    fake_go.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                'printf "%s\\n" "$*" >> "$FAKE_GO_LOG"',
                'printf \'{"Action":"run","Test":"TestBug"}\\n\'',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_go.chmod(0o755)

    completed = subprocess.run(
        [sys.executable, "-c", _VERIFY_SCRIPT],
        cwd=tmp_path,
        env={
            **os.environ,
            "FAKE_GO_LOG": str(log_path),
            "PATH": "",
            "SMALL_SWE_TESTS_JSON": json.dumps(["TestBug"]),
            "SMALL_SWE_GO_BIN": str(fake_go),
            "SMALL_SWE_VERIFIER_KIND": "go_test",
            "TASK_REPO_ROOT": str(tmp_path),
            "SMALL_SWE_PER_TEST_TIMEOUT_SEC": "30",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["executed"] == 1
    assert payload["results"] == {"TestBug": True}
    logged_command = log_path.read_text(encoding="utf-8").strip().split()
    assert logged_command[:5] == ["test", "./...", "-count=1", "-json", "-run"]
