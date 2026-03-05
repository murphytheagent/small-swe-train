from __future__ import annotations

from dataclasses import replace
import threading
import time

from config import (
    OnPolicyDataConfig,
    OnPolicyDatasetColumns,
    OnPolicyRuntimeConfig,
    OnPolicySettings,
    resolve_feedback_deterministic_truncation_settings,
)
from env.container_pool import ContainerHandle
from env.runtime_protocol import ToolRequest, ToolResponse
from env.task_dataset import TaskSample
import rollout.onpolicy_collector as onpolicy_collector_module
from rollout.onpolicy_collector import OnPolicyRolloutCollector


class _FakePool:
    def __init__(self) -> None:
        self.release_called = False

    def acquire(self, tasks: list[TaskSample]) -> tuple[ContainerHandle, ...]:
        return tuple(
            ContainerHandle(
                task_id=task.task_id,
                image_name=task.image_name,
                container_id=f"cid-{index}",
                container_name=f"cname-{index}",
            )
            for index, task in enumerate(tasks)
        )

    def release_all(self) -> None:
        self.release_called = True


class _PerAcquirePool:
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.release_calls = 0
        self._active = False

    def acquire(self, tasks: list[TaskSample]) -> tuple[ContainerHandle, ...]:
        if self._active:
            raise RuntimeError("acquire called before release_all")
        self._active = True
        call_index = self.acquire_calls
        self.acquire_calls += 1
        task = tasks[0]
        return (
            ContainerHandle(
                task_id=task.task_id,
                image_name=task.image_name,
                container_id=f"cid-{call_index}",
                container_name=f"cname-{call_index}",
            ),
        )

    def release_all(self) -> None:
        self._active = False
        self.release_calls += 1


class _BatchTrackingPool:
    def __init__(self) -> None:
        self.acquire_inputs: list[list[str]] = []
        self.release_calls = 0
        self._lock = threading.Lock()

    def acquire(self, tasks: list[TaskSample]) -> tuple[ContainerHandle, ...]:
        with self._lock:
            self.acquire_inputs.append([task.task_id for task in tasks])
        return tuple(
            ContainerHandle(
                task_id=task.task_id,
                image_name=task.image_name,
                container_id=f"cid-{task.task_id}",
                container_name=f"cname-{task.task_id}",
            )
            for task in tasks
        )

    def release_all(self) -> None:
        with self._lock:
            self.release_calls += 1


class _FakeExecutor:
    def __init__(self) -> None:
        self.requests: list[ToolRequest] = []

    def run(self, request: ToolRequest) -> ToolResponse:
        self.requests.append(request)
        return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)


class _FailingInitExecutor:
    def __init__(self, *, message: str) -> None:
        self._message = message
        self.requests: list[ToolRequest] = []

    def run(self, request: ToolRequest) -> ToolResponse:
        self.requests.append(request)
        raise OSError(self._message)


class _FlakyInitExecutor:
    def __init__(self, *, message: str) -> None:
        self._message = message
        self.requests: list[ToolRequest] = []
        self._call_count = 0

    def run(self, request: ToolRequest) -> ToolResponse:
        self.requests.append(request)
        self._call_count += 1
        if self._call_count == 1:
            raise OSError(self._message)
        return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)


def _settings() -> OnPolicySettings:
    return OnPolicySettings(
        data=OnPolicyDataConfig(
            dataset_id="dummy/ds",
            dataset_split="train",
            columns=OnPolicyDatasetColumns(
                image_name="image_name",
                problem_statement="problem_statement",
                fail_to_pass="FAIL_TO_PASS",
                pass_to_pass="PASS_TO_PASS",
            ),
        ),
        runtime=OnPolicyRuntimeConfig(
            enabled=True,
            rollout_only=True,
            task_batch_size=1,
            attempts_per_task=1,
            max_turns_per_attempt=3,
            env_pool_size=1,
            tool_timeout_sec=10,
            container_start_timeout_sec=10,
            attempt_timeout_sec=60,
            max_tool_calls_per_turn=3,
        ),
    )


def _dataset_loader(_dataset_id: str, _split: str) -> list[dict[str, object]]:
    return [
        {
            "task_id": "task-1",
            "image_name": "img:1",
            "problem_statement": "Fix the bug",
            "FAIL_TO_PASS": ["a"],
            "PASS_TO_PASS": ["b"],
        }
    ]


def _four_task_dataset_loader(_dataset_id: str, _split: str) -> list[dict[str, object]]:
    return [
        {
            "task_id": "task-a",
            "image_name": "img:1",
            "problem_statement": "Fix A",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "task-b",
            "image_name": "img:2",
            "problem_statement": "Fix B",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "task-c",
            "image_name": "img:3",
            "problem_statement": "Fix C",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
        {
            "task_id": "task-d",
            "image_name": "img:4",
            "problem_statement": "Fix D",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
            "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
        },
    ]


def test_onpolicy_collector_collects_terminal_attempt_rows() -> None:
    pool = _FakePool()
    executor = _FakeExecutor()

    def turn_generator(**kwargs: object) -> str:
        turn_index = int(kwargs["turn_index"])
        if turn_index == 0:
            return '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=_settings(),
        turn_generator=turn_generator,
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    row = rows[0]
    assert row["resolved"] is True
    assert row["is_terminal"] is True
    assert row["task_id"] == "task-1"
    assert row["image_name"] == "img:1"
    assert row["fail_to_pass"] == ["a"]
    assert row["pass_to_pass"] == ["b"]
    assert row["trajectory_steps"]
    assert row["trajectory_history"]
    assert row["trajectory_assistant_turns"]
    assert row["trajectory_tool_validation_errors"] == []
    assert row["trajectory_format_valid"] is True
    assert row["final_turn_has_submit"] is True
    assert row["final_submit_format_valid"] is True
    assert row["container_init_succeeded"] is True
    assert row["attempt_index"] == 0
    assert pool.release_called is True


def test_onpolicy_collector_truncates_tool_output_payload_fields() -> None:
    pool = _FakePool()
    truncation_settings = resolve_feedback_deterministic_truncation_settings()
    long_stdout = " ".join(
        f"tok{i}"
        for i in range(truncation_settings.head_tokens + truncation_settings.tail_tokens + 32)
    )

    class _LongOutputExecutor:
        def run(self, request: ToolRequest) -> ToolResponse:
            del request
            return ToolResponse(stdout=long_stdout, stderr="", exit_code=0)

    def turn_generator(**kwargs: object) -> str:
        turn_index = int(kwargs["turn_index"])
        if turn_index == 0:
            return '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=_settings(),
        turn_generator=turn_generator,
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: _LongOutputExecutor(),
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    row = rows[0]
    assert "<...truncated...>" in str(row["tool_output"]["stdout"])
    first_step = row["trajectory_steps"][0]
    assert "<...truncated...>" in str(first_step["stdout"])


def test_onpolicy_collector_keeps_failed_rows() -> None:
    settings = _settings()
    settings = OnPolicySettings(
        data=settings.data,
        runtime=replace(settings.runtime, attempts_per_task=2),
    )

    pool = _FakePool()
    executor = _FakeExecutor()

    def turn_generator(**kwargs: object) -> str:
        attempt_index = int(kwargs["attempt_index"])
        if attempt_index == 0:
            raise RuntimeError("generation failed")
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 2
    assert rows[0]["resolved"] is False
    assert "collector_error" in rows[0]
    assert rows[1]["resolved"] is False


def test_onpolicy_collector_tracks_invalid_submit_metadata_without_forcing_terminal_stop() -> None:
    pool = _FakePool()
    executor = _FakeExecutor()

    collector = OnPolicyRolloutCollector(
        settings=_settings(),
        turn_generator=lambda **_kwargs: '<tool_call>{"tool":"submit","args":{}}</tool_call>',
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    row = rows[0]
    assert row["is_terminal"] is False
    assert row["final_turn_has_submit"] is True
    assert row["final_submit_format_valid"] is False
    assert row["trajectory_format_valid"] is False
    assert row["trajectory_tool_validation_errors"]
    assert "Missing required arg 'final_response'" in row["trajectory_tool_validation_errors"][0]


def test_onpolicy_collector_resolver_sees_full_attempt_history() -> None:
    settings = _settings()
    pool = _FakePool()
    executor = _FakeExecutor()
    observed_step_counts: list[int] = []

    def turn_generator(**kwargs: object) -> str:
        turn_index = int(kwargs["turn_index"])
        if turn_index == 0:
            return '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
        if turn_index == 1:
            return '<tool_call>{"tool":"bash","args":{"command":"echo ok"}}</tool_call>'
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    def resolver(task, attempt_index, is_terminal, steps):
        del task, attempt_index
        observed_step_counts.append(len(steps))
        return is_terminal and len(steps) == 2

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
        attempt_resolver=resolver,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    assert observed_step_counts == [2]
    assert rows[0]["resolved"] is True


def test_onpolicy_collector_uses_fresh_container_per_attempt() -> None:
    settings = _settings()
    settings = OnPolicySettings(
        data=settings.data,
        runtime=replace(settings.runtime, attempts_per_task=2),
    )

    pool = _PerAcquirePool()
    observed_container_ids: list[str] = []

    def turn_generator(**kwargs: object) -> str:
        del kwargs
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    def executor_factory(handle: ContainerHandle, _runtime: OnPolicyRuntimeConfig):
        observed_container_ids.append(handle.container_id)
        return _FakeExecutor()

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=executor_factory,
        attempt_resolver=lambda _task, _attempt, is_terminal, _steps: is_terminal,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 2
    assert observed_container_ids == ["cid-0", "cid-1"]
    assert [row["container_id"] for row in rows] == ["cid-0", "cid-1"]
    assert pool.acquire_calls == 2
    assert pool.release_calls == 2


def test_onpolicy_collector_dispatches_one_task_per_trajectory_attempt() -> None:
    settings = _settings()
    settings = OnPolicySettings(
        data=settings.data,
        runtime=replace(
            settings.runtime,
            task_batch_size=2,
            attempts_per_task=2,
            env_pool_size=2,
        ),
    )
    pool = _BatchTrackingPool()
    executor = _FakeExecutor()

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=lambda **_kwargs: (
            '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
        ),
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-a",
                "image_name": "img:1",
                "problem_statement": "Fix A",
                "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
                "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
            },
            {
                "task_id": "task-b",
                "image_name": "img:2",
                "problem_statement": "Fix B",
                "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
                "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
            },
        ],
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
        attempt_resolver=lambda _task, _attempt, is_terminal, _steps: is_terminal,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 4
    assert pool.acquire_inputs == [["task-a"], ["task-a"], ["task-b"], ["task-b"]]
    assert pool.release_calls == 4
    assert [(row["task_id"], row["attempt_index"]) for row in rows] == [
        ("task-a", 0),
        ("task-a", 1),
        ("task-b", 0),
        ("task-b", 1),
    ]
    assert all(row["batch_container_count"] == 2 for row in rows)


def test_onpolicy_collector_runs_task_attempts_concurrently_when_enabled(
    monkeypatch,
) -> None:
    settings = _settings()
    settings = OnPolicySettings(
        data=settings.data,
        runtime=replace(
            settings.runtime,
            task_batch_size=4,
            attempts_per_task=1,
            env_pool_size=4,
            max_in_flight_tasks=4,
        ),
    )
    pool = _BatchTrackingPool()
    executor = _FakeExecutor()
    active = 0
    max_active = 0
    lock = threading.Lock()

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=lambda **_kwargs: (
            '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
        ),
        dataset_loader=_four_task_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
    )

    def _fake_collect_attempt(**kwargs):
        nonlocal active, max_active
        task_position = int(kwargs["task_position"])
        task = kwargs["task"]
        with lock:
            active += 1
            if active > max_active:
                max_active = active
        time.sleep(0.05)
        with lock:
            active -= 1
        return {
            "task_id": task.task_id,
            "resolved": True,
            "task_position": task_position,
        }

    monkeypatch.setattr(collector, "_collect_attempt", _fake_collect_attempt)

    rows = collector.collect_step(0)

    assert len(rows) == 4
    assert max_active >= 2
    assert sorted(pool.acquire_inputs) == [["task-a"], ["task-b"], ["task-c"], ["task-d"]]
    assert [row["task_position"] for row in rows] == [0, 1, 2, 3]
    assert pool.release_calls == 4


def test_onpolicy_collector_parallel_dispatch_reduces_rollout_wall_time(
    monkeypatch,
) -> None:
    def _run_once(max_in_flight_tasks: int) -> float:
        settings = _settings()
        settings = OnPolicySettings(
            data=settings.data,
            runtime=replace(
                settings.runtime,
                task_batch_size=4,
                attempts_per_task=1,
                env_pool_size=4,
                max_in_flight_tasks=max_in_flight_tasks,
            ),
        )
        pool = _BatchTrackingPool()
        executor = _FakeExecutor()
        collector = OnPolicyRolloutCollector(
            settings=settings,
            turn_generator=lambda **_kwargs: (
                '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
            ),
            dataset_loader=_four_task_dataset_loader,
            pool_factory=lambda _runtime: pool,
            executor_factory=lambda _handle, _runtime: executor,
        )

        def _fake_collect_attempt(**kwargs):
            task = kwargs["task"]
            task_position = int(kwargs["task_position"])
            time.sleep(0.08)
            return {
                "task_id": task.task_id,
                "resolved": True,
                "task_position": task_position,
            }

        monkeypatch.setattr(collector, "_collect_attempt", _fake_collect_attempt)
        start = time.perf_counter()
        rows = collector.collect_step(0)
        elapsed = time.perf_counter() - start
        assert len(rows) == 4
        return elapsed

    serial_elapsed = _run_once(1)
    parallel_elapsed = _run_once(4)
    assert parallel_elapsed < serial_elapsed * 0.6


def test_onpolicy_collector_applies_task_patch_before_rollout_turns() -> None:
    settings = _settings()
    pool = _FakePool()
    executor = _FakeExecutor()
    task_patch = (
        "diff --git a/a.txt b/a.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )

    def turn_generator(**kwargs: object) -> str:
        turn_index = int(kwargs["turn_index"])
        if turn_index == 0:
            return '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-1",
                "image_name": "img:1",
                "problem_statement": "Fix patch flow",
                "patch": task_patch,
                "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
                "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
            }
        ],
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
        attempt_resolver=lambda _task, _attempt, is_terminal, _steps: is_terminal,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    assert len(executor.requests) >= 2
    assert executor.requests[0].tool == "bash"
    assert executor.requests[0].args.get("stdin") == task_patch
    assert task_patch not in str(executor.requests[0].args.get("command", ""))
    assert "git apply" in str(executor.requests[0].args.get("command", ""))
    assert "git apply --3way" in str(executor.requests[0].args.get("command", ""))
    assert "patch --batch --forward -p1" in str(executor.requests[0].args.get("command", ""))
    assert "git apply --reverse --check" in str(executor.requests[0].args.get("command", ""))
    assert executor.requests[1].tool == "search"
    assert rows[0]["resolved"] is True
    assert rows[0]["task_patch_applied"] is True
    assert rows[0]["container_init_succeeded"] is True
    assert rows[0]["batch_container_count"] == 1


def test_onpolicy_collector_retries_task_patch_init_once_on_transient_executor_error() -> None:
    settings = _settings()
    pool = _FakePool()
    flaky_executor = _FlakyInitExecutor(message="temporary docker exec failure")
    task_patch = "diff --git a/a.txt b/a.txt\n"

    def turn_generator(**kwargs: object) -> str:
        turn_index = int(kwargs["turn_index"])
        if turn_index == 0:
            return '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-1",
                "image_name": "img:1",
                "problem_statement": "Fix patch flow",
                "patch": task_patch,
                "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
                "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
            }
        ],
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: flaky_executor,
        attempt_resolver=lambda _task, _attempt, is_terminal, _steps: is_terminal,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    assert rows[0]["resolved"] is True
    assert rows[0]["task_patch_applied"] is True
    assert rows[0]["container_init_succeeded"] is True
    assert "executor_error" not in rows[0]
    assert [request.tool for request in flaky_executor.requests[:2]] == ["bash", "bash"]
    assert flaky_executor.requests[2].tool == "search"


def test_onpolicy_collector_keeps_batch_running_when_patch_init_executor_raises() -> None:
    settings = _settings()
    settings = OnPolicySettings(
        data=settings.data,
        runtime=replace(
            settings.runtime,
            task_batch_size=2,
            env_pool_size=2,
        ),
    )
    pool = _BatchTrackingPool()
    success_executor = _FakeExecutor()
    failing_executor = _FailingInitExecutor(message="argument list too long")

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=lambda **_kwargs: (
            '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
        ),
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-a",
                "image_name": "img:1",
                "problem_statement": "Fix A",
                "patch": "diff --git a/a.txt b/a.txt\n",
                "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
                "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
            },
            {
                "task_id": "task-b",
                "image_name": "img:2",
                "problem_statement": "Fix B",
                "patch": "diff --git a/b.txt b/b.txt\n",
                "FAIL_TO_PASS": ["tests/test_bug.py::test_bugfix"],
                "PASS_TO_PASS": ["tests/test_ok.py::test_regression"],
            },
        ],
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda handle, _runtime: (
            failing_executor if handle.task_id == "task-a" else success_executor
        ),
        attempt_resolver=lambda _task, _attempt, is_terminal, _steps: is_terminal,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 2
    assert rows[0]["task_id"] == "task-a"
    assert rows[0]["resolved"] is False
    assert "executor_error" in rows[0]
    assert rows[0]["container_init_succeeded"] is False
    assert "argument list too long" in str(rows[0]["executor_error"])
    assert "attempt 1/2" in str(rows[0]["executor_error"])
    assert "attempt 2/2" in str(rows[0]["executor_error"])
    assert rows[1]["task_id"] == "task-b"
    assert rows[1]["resolved"] is True
    assert rows[1]["task_patch_applied"] is True
    assert rows[1]["container_init_succeeded"] is True
    assert pool.release_calls == 2


def test_onpolicy_collector_keeps_tool_output_aligned_with_first_tool_call() -> None:
    pool = _FakePool()
    executor = _FakeExecutor()

    def turn_generator(**kwargs: object) -> str:
        turn_index = int(kwargs["turn_index"])
        if turn_index == 0:
            return (
                '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
                '\n<tool_call>{"tool":"bash","args":{"command":"echo second"}}</tool_call>'
            )
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=_settings(),
        turn_generator=turn_generator,
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    row = rows[0]
    assert row["is_terminal"] is True
    assert row["tool_name"] == "search"
    assert row["tool_output"]["stdout"] == "ran:search"
    assert row["turn_index"] == 0
    assert '"tool":"search"' in row["assistant_response"]
    assert row["trajectory_steps"][0]["tool"] == "search"


def test_onpolicy_collector_keeps_failed_bridge_turn_in_trajectory_history(
    monkeypatch,
) -> None:
    pool = _FakePool()
    executor = _FakeExecutor()
    assistant_turn = '<tool_call>{"tool":"bash","args":{"command":"echo broken"}}</tool_call>'

    def broken_bridge_step(*_args, **_kwargs):
        raise ValueError("bridge parse failed")

    monkeypatch.setattr(
        onpolicy_collector_module,
        "run_env_bridge_step",
        broken_bridge_step,
    )

    collector = OnPolicyRolloutCollector(
        settings=_settings(),
        turn_generator=lambda **_kwargs: assistant_turn,
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    row = rows[0]
    assert row["resolved"] is False
    assert row["is_terminal"] is False
    assert row["assistant_response"] == assistant_turn
    assert row["trajectory_history"] == [assistant_turn]
    assert row["trajectory_assistant_turns"] == [assistant_turn]
    assert row["final_turn_has_submit"] is False
    assert row["final_submit_format_valid"] is False
    assert row["trajectory_format_valid"] is False
    assert "bridge_error" in row
    assert "bridge parse failed" in str(row["bridge_error"])


def test_onpolicy_collector_default_turn_generator_produces_resolved_attempt() -> None:
    pool = _FakePool()
    executor = _FakeExecutor()

    collector = OnPolicyRolloutCollector(
        settings=_settings(),
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    assert rows[0]["is_terminal"] is True
    assert rows[0]["resolved"] is True
    assert rows[0]["turn_index"] == 0
    assert rows[0]["tool_name"] == "bash"
    assert rows[0]["trajectory_format_valid"] is True
    assert rows[0]["final_turn_has_submit"] is True
    assert rows[0]["final_submit_format_valid"] is True
    assert [request.tool for request in executor.requests] == ["bash"]


def test_onpolicy_collector_continues_after_nonzero_tool_exit_until_submit() -> None:
    pool = _FakePool()

    class _NonZeroSearchExecutor:
        def __init__(self) -> None:
            self.requests: list[ToolRequest] = []

        def run(self, request: ToolRequest) -> ToolResponse:
            self.requests.append(request)
            if request.tool == "search":
                return ToolResponse(
                    stdout="",
                    stderr="simulated lookup failure",
                    exit_code=2,
                )
            return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)

    executor = _NonZeroSearchExecutor()

    def turn_generator(**kwargs: object) -> str:
        turn_index = int(kwargs["turn_index"])
        if turn_index == 0:
            return '<tool_call>{"tool":"search","args":{"query":"foo"}}</tool_call>'
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=_settings(),
        turn_generator=turn_generator,
        dataset_loader=_dataset_loader,
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 1
    row = rows[0]
    assert row["is_terminal"] is True
    assert row["final_turn_has_submit"] is True
    assert row["final_submit_format_valid"] is True
    assert row["resolved"] is False
    assert "executor_error" in row
    assert "simulated lookup failure" in str(row["executor_error"])
    assert len(row["trajectory_steps"]) == 1
    assert row["trajectory_steps"][0]["tool"] == "search"
    assert [request.tool for request in executor.requests] == ["search"]
