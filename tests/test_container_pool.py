from __future__ import annotations

import pytest

from env.command_runner import CommandResult
from env.container_pool import BatchContainerPool
from env.task_dataset import TaskSample


def _task(task_id: str, image_name: str) -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_name=image_name,
        problem_statement=f"problem-{task_id}",
        fail_to_pass=[],
        pass_to_pass=[],
        raw={},
    )


def test_batch_container_pool_acquire_and_release() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], *, timeout_sec: int) -> CommandResult:
        del timeout_sec
        commands.append(list(command))
        if command[:2] == ["docker", "run"]:
            suffix = len([cmd for cmd in commands if cmd[:2] == ["docker", "run"]])
            return CommandResult(returncode=0, stdout=f"container-{suffix}\n")
        return CommandResult(returncode=0, stdout="")

    pool = BatchContainerPool(
        env_pool_size=2,
        container_start_timeout_sec=10,
        runner=runner,
    )
    handles = pool.acquire([_task("t1", "image:1"), _task("t2", "image:2")])

    assert [handle.container_id for handle in handles] == ["container-1", "container-2"]
    assert len(pool.active_handles) == 2

    pool.release_all()

    rm_commands = [cmd for cmd in commands if cmd[:3] == ["docker", "rm", "-f"]]
    assert len(rm_commands) == 2
    assert pool.active_handles == ()


def test_batch_container_pool_cleans_up_on_partial_start_failure() -> None:
    commands: list[list[str]] = []
    run_calls = 0

    def runner(command: list[str], *, timeout_sec: int) -> CommandResult:
        nonlocal run_calls
        del timeout_sec
        commands.append(list(command))
        if command[:2] == ["docker", "run"]:
            run_calls += 1
            if run_calls == 1:
                return CommandResult(returncode=0, stdout="container-ok\n")
            return CommandResult(returncode=1, stderr="pull access denied")
        return CommandResult(returncode=0)

    pool = BatchContainerPool(
        env_pool_size=2,
        container_start_timeout_sec=10,
        runner=runner,
    )

    with pytest.raises(RuntimeError, match="Failed to start container"):
        pool.acquire([_task("t1", "image:1"), _task("t2", "missing:image")])

    rm_commands = [cmd for cmd in commands if cmd[:3] == ["docker", "rm", "-f"]]
    assert len(rm_commands) == 1
    assert rm_commands[0][3] == "container-ok"
    assert pool.active_handles == ()


def test_batch_container_pool_adds_management_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], *, timeout_sec: int) -> CommandResult:
        del timeout_sec
        commands.append(list(command))
        if command[:2] == ["docker", "run"]:
            return CommandResult(returncode=0, stdout="container-1\n")
        return CommandResult(returncode=0)

    monkeypatch.setenv("SLURM_JOB_ID", "606")
    monkeypatch.setenv("SDPO_RUN_LABEL", "20260228T020803Z_job606")
    monkeypatch.setenv("EXPERIMENT", "small-swe-sdpo_20260228T020803Z_job606")

    pool = BatchContainerPool(
        env_pool_size=1,
        container_start_timeout_sec=10,
        name_prefix="sdpo-swe-bridge",
        runner=runner,
    )
    pool.acquire([_task("t1", "image:1")])
    pool.release_all()

    run_commands = [cmd for cmd in commands if cmd[:2] == ["docker", "run"]]
    assert len(run_commands) == 1
    run_cmd = run_commands[0]
    assert "--label" in run_cmd
    assert "small_swe.managed=1" in run_cmd
    assert "small_swe.pool_name=sdpo-swe-bridge" in run_cmd
    assert "small_swe.slurm_job_id=606" in run_cmd
    assert "small_swe.run_label=20260228T020803Z_job606" in run_cmd


def test_batch_container_pool_uses_slurm_jobid_alias_for_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], *, timeout_sec: int) -> CommandResult:
        del timeout_sec
        commands.append(list(command))
        if command[:2] == ["docker", "run"]:
            return CommandResult(returncode=0, stdout="container-1\n")
        return CommandResult(returncode=0)

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("SLURM_JOBID", "707")
    monkeypatch.setenv("SDPO_RUN_LABEL", "20260228T020803Z_job707")

    pool = BatchContainerPool(
        env_pool_size=1,
        container_start_timeout_sec=10,
        name_prefix="sdpo-swe-bridge",
        runner=runner,
    )
    pool.acquire([_task("t1", "image:1")])
    pool.release_all()

    run_commands = [cmd for cmd in commands if cmd[:2] == ["docker", "run"]]
    assert len(run_commands) == 1
    run_cmd = run_commands[0]
    assert "small_swe.slurm_job_id=707" in run_cmd
