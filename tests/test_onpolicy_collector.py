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
    assert rows[1]["resolved"] is True
