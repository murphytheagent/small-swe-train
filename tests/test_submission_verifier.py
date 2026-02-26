from __future__ import annotations

from dataclasses import dataclass

from env.runtime_protocol import ToolResponse
from verl_integration.submission_verifier import run_submission_verifier


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
