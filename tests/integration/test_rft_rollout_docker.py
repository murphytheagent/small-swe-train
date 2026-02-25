from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

import pytest

from config import resolve_on_policy_settings
from env.docker_executor import DockerToolExecutor
from env.task_dataset import load_hf_dataset
from rollout.onpolicy_collector import OnPolicyRolloutCollector


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DOCKER_INTEGRATION") != "1" or not _docker_ready(),
    reason="Set RUN_DOCKER_INTEGRATION=1 with Docker daemon available to run this test.",
)


def _assert_and_pull_image_if_missing(image_name: str) -> None:
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspect_result.returncode == 0:
        return

    pull_result = subprocess.run(
        ["docker", "pull", image_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if pull_result.returncode != 0:
        raise RuntimeError(
            f"Failed to pull dataset task image {image_name!r}: {pull_result.stderr.strip()}"
        )


def _load_real_rows_for_single_image() -> list[dict[str, Any]]:
    settings = resolve_on_policy_settings()
    dataset = load_hf_dataset(settings.data.dataset_id, settings.data.dataset_split)
    columns = settings.data.columns

    for row in dataset:
        if not isinstance(row, dict):
            continue
        image_name_raw = row.get(columns.image_name)
        prompt_raw = row.get(columns.problem_statement)
        if not isinstance(image_name_raw, str) or not image_name_raw.strip():
            continue
        if not isinstance(prompt_raw, str) or not prompt_raw.strip():
            continue
        return [dict(row)]
    raise AssertionError(f"No valid task rows found in {settings.data.dataset_id}:{settings.data.dataset_split}.")


def test_full_episode_rollout_executes_all_tools_in_order() -> None:
    settings = resolve_on_policy_settings(
        runtime_overrides={
            "enabled": True,
            "rollout_only": True,
            "task_batch_size": 1,
            "attempts_per_task": 1,
            "max_turns_per_attempt": 5,
            "env_pool_size": 1,
            "tool_timeout_sec": 30,
            "container_start_timeout_sec": 300,
            "attempt_timeout_sec": 180,
            "max_tool_calls_per_turn": 3,
        }
    )

    rows = _load_real_rows_for_single_image()
    image_name = str(rows[0][settings.data.columns.image_name])
    _assert_and_pull_image_if_missing(image_name)

    executed_tools: list[str] = []

    class LoggingExecutor:
        def __init__(self, container_id: str) -> None:
            self._inner = DockerToolExecutor(
                container_id=container_id,
                tool_timeout_sec=settings.runtime.tool_timeout_sec,
            )

        def run(self, request):
            executed_tools.append(request.tool)
            return self._inner.run(request)

    def turn_generator(**kwargs: object) -> str:
        turn_index = int(kwargs["turn_index"])
        if turn_index == 0:
            return (
                '<tool_call>{"tool":"bash","args":{"command":"printf \'from_bash\\n\' > /tmp/episode.txt"}}</tool_call>'
            )
        if turn_index == 1:
            return (
                '<tool_call>{"tool":"search","args":{"query":"from_bash","path_hint":"/tmp","top_k":10}}</tool_call>'
            )
        if turn_index == 2:
            return (
                '<tool_call>{"tool":"apply_patch","args":{"path":"/tmp/episode.txt","patch":"from_edit"}}</tool_call>'
            )
        if turn_index == 3:
            return (
                '<tool_call>{"tool":"search","args":{"query":"from_edit","path_hint":"/tmp","top_k":10}}</tool_call>'
            )
        return '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'

    collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=turn_generator,
        dataset_loader=lambda _dataset_id, _split: rows,
        executor_factory=lambda handle, _runtime: LoggingExecutor(handle.container_id),
    )

    collected = collector.collect_step(0)

    assert len(collected) == 1
    row = collected[0]
    assert row["resolved"] is True
    assert row["is_terminal"] is True
    assert row.get("tool_name") == "search"
    assert "from_edit" in row["tool_output"].get("stdout", "")
    expected_chain = ["bash", "search", "apply_patch", "search"]
    trajectory_tools = [step.get("tool") for step in row.get("trajectory_steps", [])]
    assert trajectory_tools == expected_chain
    if row.get("task_patch_applied"):
        assert executed_tools[0] == "bash"
        assert executed_tools[1:] == expected_chain
    else:
        assert executed_tools == expected_chain
