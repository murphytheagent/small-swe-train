"""On-policy rollout collector that runs before trajectory preprocessing."""

from __future__ import annotations

import time
from typing import Callable, Protocol, Sequence

from config import OnPolicyRuntimeConfig, OnPolicySettings
from env.container_pool import BatchContainerPool, ContainerHandle
from env.docker_executor import DockerToolExecutor
from env.runtime_protocol import EnvironmentStep, ToolRequest, ToolResponse
from env.task_dataset import DatasetLoader, TaskSample, load_task_batch
from schemas import RolloutRow
from verl_integration.env_bridge import run_env_bridge_step


class ToolExecutorLike(Protocol):
    def run(self, request: ToolRequest) -> ToolResponse:
        ...


class CollectorPool(Protocol):
    def acquire(self, tasks: Sequence[TaskSample]) -> tuple[ContainerHandle, ...]:
        ...

    def release_all(self) -> None:
        ...


class AssistantTurnGenerator(Protocol):
    def __call__(
        self,
        *,
        task: TaskSample,
        attempt_index: int,
        turn_index: int,
        step_index: int,
        history: Sequence[str],
    ) -> str:
        ...


AttemptResolver = Callable[[TaskSample, int, bool, Sequence[EnvironmentStep]], bool]
PoolFactory = Callable[[OnPolicyRuntimeConfig], CollectorPool]
ExecutorFactory = Callable[[ContainerHandle, OnPolicyRuntimeConfig], ToolExecutorLike]


def _default_turn_generator(
    *,
    task: TaskSample,
    attempt_index: int,
    turn_index: int,
    step_index: int,
    history: Sequence[str],
) -> str:
    del task, attempt_index, step_index, history
    if turn_index == 0:
        return (
            "<tool_call>"
            '{"tool":"bash","args":{"command":"true"}}'
            "</tool_call>"
        )
    return (
        "<tool_call>"
        '{"tool":"submit","args":{"final_response":"collector default terminal submit"}}'
        "</tool_call>"
    )


def _default_attempt_resolver(
    task: TaskSample,
    attempt_index: int,
    is_terminal: bool,
    steps: Sequence[EnvironmentStep],
) -> bool:
    del task, attempt_index
    if not is_terminal:
        return False
    if not steps:
        return False
    return all(step.response.exit_code == 0 for step in steps)


class OnPolicyRolloutCollector:
    """Collect task-attempt rollout rows with per-batch container pooling."""

    def __init__(
        self,
        *,
        settings: OnPolicySettings,
        turn_generator: AssistantTurnGenerator | None = None,
        dataset_loader: DatasetLoader | None = None,
        pool_factory: PoolFactory | None = None,
        executor_factory: ExecutorFactory | None = None,
        attempt_resolver: AttemptResolver | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._turn_generator = turn_generator or _default_turn_generator
        self._dataset_loader = dataset_loader
        self._pool_factory = pool_factory or _default_pool_factory
        self._executor_factory = executor_factory or _default_executor_factory
        self._attempt_resolver = attempt_resolver or _default_attempt_resolver
        self._monotonic_clock = monotonic_clock or time.monotonic

    @property
    def settings(self) -> OnPolicySettings:
        return self._settings

    def collect_step(self, step_index: int) -> list[RolloutRow]:
        if step_index < 0:
            raise ValueError("step_index must be >= 0")

        runtime = self._settings.runtime
        tasks = load_task_batch(
            step_index=step_index,
            batch_size=runtime.task_batch_size,
            config=self._settings.data,
            dataset_loader=self._dataset_loader,
        )

        rows: list[RolloutRow] = []
        for task_position, task in enumerate(tasks):
            for attempt_index in range(runtime.attempts_per_task):
                pool = self._pool_factory(runtime)
                try:
                    handles = pool.acquire([task])
                    if len(handles) != 1:
                        raise RuntimeError(
                            "Container pool must provide exactly one handle per task attempt."
                        )
                    handle = handles[0]
                    executor = self._executor_factory(handle, runtime)
                    row = self._collect_attempt(
                        step_index=step_index,
                        task_position=task_position,
                        task=task,
                        handle=handle,
                        attempt_index=attempt_index,
                        runtime=runtime,
                        executor=executor,
                    )
                    rows.append(row)
                finally:
                    pool.release_all()

        return rows

    def _collect_attempt(
        self,
        *,
        step_index: int,
        task_position: int,
        task: TaskSample,
        handle: ContainerHandle,
        attempt_index: int,
        runtime: OnPolicyRuntimeConfig,
        executor: ToolExecutorLike,
    ) -> RolloutRow:
        attempt_start = self._monotonic_clock()
        history: list[str] = []
        assistant_response = ""
        assistant_response_for_feedback = ""
        turn_index_for_feedback = -1
        tool_output: dict[str, object] = {}
        turn_index = -1
        resolved = False
        is_terminal = False
        collector_error = ""
        bridge_error = ""
        timeout_error = ""
        executor_error = ""
        tool_name = ""
        exit_code = 0
        attempt_steps: list[EnvironmentStep] = []

        for turn_index in range(runtime.max_turns_per_attempt):
            elapsed_sec = self._monotonic_clock() - attempt_start
            if elapsed_sec > runtime.attempt_timeout_sec:
                timeout_error = (
                    f"Attempt exceeded timeout of {runtime.attempt_timeout_sec}s before turn {turn_index}."
                )
                break

            try:
                assistant_response = self._turn_generator(
                    task=task,
                    attempt_index=attempt_index,
                    turn_index=turn_index,
                    step_index=step_index,
                    history=tuple(history),
                )
            except Exception as exc:
                collector_error = str(exc)
                break

            try:
                bridge_result = run_env_bridge_step(
                    assistant_response,
                    executor=executor,
                    max_tool_calls=runtime.max_tool_calls_per_turn,
                    step_index_start=turn_index * runtime.max_tool_calls_per_turn,
                )
            except Exception as exc:
                bridge_error = str(exc)
                break

            history.append(assistant_response)
            history.extend(bridge_result.tool_response_blocks)
            if bridge_result.steps:
                attempt_steps.extend(bridge_result.steps)
                assistant_response_for_feedback = assistant_response
                turn_index_for_feedback = turn_index

            if bridge_result.steps:
                first_step = bridge_result.steps[0]
                tool_name = first_step.request.tool
                exit_code = first_step.response.exit_code
                tool_output = {
                    "stdout": first_step.response.stdout,
                    "stderr": first_step.response.stderr,
                    "exit_code": first_step.response.exit_code,
                    "metadata": dict(first_step.response.metadata),
                }
                failing_step = next(
                    (step for step in bridge_result.steps if step.response.exit_code != 0),
                    None,
                )
                if failing_step is not None:
                    executor_error = failing_step.response.stderr or (
                        f"Tool {failing_step.request.tool!r} failed with exit code {failing_step.response.exit_code}."
                    )

            if bridge_result.is_terminal:
                is_terminal = True
                resolved = self._attempt_resolver(
                    task,
                    attempt_index,
                    bridge_result.is_terminal,
                    tuple(attempt_steps),
                )
                break
        else:
            timeout_error = (
                f"Attempt reached max_turns_per_attempt={runtime.max_turns_per_attempt} without terminal submit."
            )

        elapsed_ms = (self._monotonic_clock() - attempt_start) * 1000.0
        row_step_index = (
            step_index * runtime.task_batch_size * runtime.attempts_per_task
            + task_position * runtime.attempts_per_task
            + attempt_index
        )
        row_assistant_response = assistant_response_for_feedback or assistant_response
        if turn_index_for_feedback >= 0:
            row_turn_index = turn_index_for_feedback
        else:
            row_turn_index = max(turn_index, 0)

        row: RolloutRow = {
            "prompt": task.problem_statement,
            "assistant_response": row_assistant_response,
            "tool_output": tool_output,
            "resolved": bool(resolved),
            "step_index": row_step_index,
            "task_id": task.task_id,
            "attempt_index": attempt_index,
            "turn_index": row_turn_index,
            "container_id": handle.container_id,
            "is_terminal": is_terminal,
            "latency_ms": elapsed_ms,
        }
        if collector_error:
            row["collector_error"] = collector_error
        if bridge_error:
            row["bridge_error"] = bridge_error
        if timeout_error:
            row["timeout_error"] = timeout_error
        if executor_error:
            row["executor_error"] = executor_error
        if tool_name:
            row["tool_name"] = tool_name
            row["exit_code"] = exit_code

        return row


def _default_pool_factory(runtime: OnPolicyRuntimeConfig) -> CollectorPool:
    return BatchContainerPool(
        env_pool_size=runtime.env_pool_size,
        container_start_timeout_sec=runtime.container_start_timeout_sec,
    )


def _default_executor_factory(
    handle: ContainerHandle,
    runtime: OnPolicyRuntimeConfig,
) -> ToolExecutorLike:
    return DockerToolExecutor(
        container_id=handle.container_id,
        tool_timeout_sec=runtime.tool_timeout_sec,
    )
