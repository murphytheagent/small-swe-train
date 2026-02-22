from __future__ import annotations

from dataclasses import replace

from config import (
    OnPolicyDataConfig,
    OnPolicyDatasetColumns,
    OnPolicyRuntimeConfig,
    OnPolicySettings,
)
from env.container_pool import ContainerHandle
from env.runtime_protocol import ToolRequest, ToolResponse
from env.task_dataset import TaskSample
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

    def acquire(self, tasks: list[TaskSample]) -> tuple[ContainerHandle, ...]:
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
        self.release_calls += 1


class _FakeExecutor:
    def __init__(self) -> None:
        self.requests: list[ToolRequest] = []

    def run(self, request: ToolRequest) -> ToolResponse:
        self.requests.append(request)
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
    assert row["attempt_index"] == 0
    assert pool.release_called is True


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


def test_onpolicy_collector_acquires_full_task_batch_once_per_attempt() -> None:
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
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            },
            {
                "task_id": "task-b",
                "image_name": "img:2",
                "problem_statement": "Fix B",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            },
        ],
        pool_factory=lambda _runtime: pool,
        executor_factory=lambda _handle, _runtime: executor,
        attempt_resolver=lambda _task, _attempt, is_terminal, _steps: is_terminal,
    )

    rows = collector.collect_step(0)

    assert len(rows) == 4
    assert pool.acquire_inputs == [["task-a", "task-b"], ["task-a", "task-b"]]
    assert pool.release_calls == 2
    assert all(row["batch_container_count"] == 2 for row in rows)


def test_onpolicy_collector_applies_task_patch_before_rollout_turns() -> None:
    settings = _settings()
    pool = _FakePool()
    executor = _FakeExecutor()

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
                "patch": "diff --git a/a.txt b/a.txt\nindex 1111111..2222222 100644\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
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
    assert "git apply" in str(executor.requests[0].args.get("command", ""))
    assert executor.requests[1].tool == "search"
    assert rows[0]["resolved"] is True
    assert rows[0]["task_patch_applied"] is True
    assert rows[0]["batch_container_count"] == 1


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
    assert [request.tool for request in executor.requests] == ["bash"]
