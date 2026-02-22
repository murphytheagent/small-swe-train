from __future__ import annotations

import subprocess

from env.command_runner import CommandResult
from env.docker_executor import DockerToolExecutor
from env.runtime_protocol import ToolRequest


def test_docker_executor_runs_bash_in_container() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], *, timeout_sec: int) -> CommandResult:
        del timeout_sec
        commands.append(list(command))
        return CommandResult(returncode=0, stdout="ok\n", stderr="")

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=runner,
    )

    response = executor.run(
        ToolRequest(tool="bash", args={"command": "echo hi", "cwd": "/repo"})
    )

    assert response.exit_code == 0
    assert response.stdout == "ok\n"
    assert commands[0][:4] == ["docker", "exec", "-w", "/repo"]


def test_docker_executor_submit_is_local_terminal_tool() -> None:
    executor = DockerToolExecutor(container_id="container-1", tool_timeout_sec=30)
    response = executor.run(
        ToolRequest(tool="submit", args={"final_response": "done"})
    )

    assert response.exit_code == 0
    assert response.metadata["submitted"] is True


def test_docker_executor_maps_timeout_to_nonzero_response() -> None:
    def runner(command: list[str], *, timeout_sec: int) -> CommandResult:
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_sec)

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=1,
        runner=runner,
    )
    response = executor.run(ToolRequest(tool="bash", args={"command": "sleep 5"}))

    assert response.exit_code == 124
    assert "timed out" in response.stderr.lower()
