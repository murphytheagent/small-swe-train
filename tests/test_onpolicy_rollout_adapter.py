from __future__ import annotations

from pathlib import Path

from config import (
    OnPolicyDataConfig,
    OnPolicyDatasetColumns,
    OnPolicyRuntimeConfig,
    OnPolicySettings,
)
from rollout.onpolicy_collector import OnPolicyRolloutCollector
from verl_integration.onpolicy_rollout_adapter import collect_rollouts_for_steps


def test_collect_rollouts_for_steps_writes_jsonl_artifacts(tmp_path: Path) -> None:
    settings = OnPolicySettings(
        data=OnPolicyDataConfig(
            dataset_id="dummy/local",
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
            max_turns_per_attempt=1,
            env_pool_size=1,
            tool_timeout_sec=1,
            container_start_timeout_sec=1,
            attempt_timeout_sec=10,
            max_tool_calls_per_turn=3,
        ),
    )

    class _FakePool:
        def acquire(self, tasks):
            from env.container_pool import ContainerHandle

            return (
                ContainerHandle(
                    task_id=tasks[0].task_id,
                    image_name=tasks[0].image_name,
                    container_id="cid-1",
                    container_name="cname-1",
                ),
            )

        def release_all(self) -> None:
            return None

    class _FakeExecutor:
        def run(self, request):
            from env.runtime_protocol import ToolResponse

            return ToolResponse(stdout=f"ran:{request.tool}", stderr="", exit_code=0)

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=lambda **_kwargs: '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
        dataset_loader=lambda _dataset_id, _split: [
            {
                "task_id": "task-1",
                "image_name": "img:1",
                "problem_statement": "Fix bug",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            }
        ],
        pool_factory=lambda _runtime: _FakePool(),
        executor_factory=lambda _handle, _runtime: _FakeExecutor(),
    )

    rows = collect_rollouts_for_steps(total_steps=2, collector=collector, output_dir=tmp_path)

    assert len(rows) == 2
    assert len(rows[0]) == 1
    assert (tmp_path / "step_00000.jsonl").exists()
    assert (tmp_path / "step_00001.jsonl").exists()
