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


def test_docker_executor_submit_requires_non_empty_final_response() -> None:
    executor = DockerToolExecutor(container_id="container-1", tool_timeout_sec=30)
    response = executor.run(ToolRequest(tool="submit", args={"final_response": "  "}))

    assert response.exit_code == 2
    assert "final_response" in response.stderr
    assert response.metadata.get("validation_errors")


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


def test_docker_executor_apply_patch_uses_stdin_payload_and_non_silent_script() -> None:
    commands: list[list[str]] = []
    stdin_payloads: list[str | None] = []

    def runner(
        command: list[str],
        *,
        timeout_sec: int,
        stdin_text: str | None = None,
    ) -> CommandResult:
        del timeout_sec
        commands.append(list(command))
        stdin_payloads.append(stdin_text)
        return CommandResult(returncode=0, stdout="patched\n", stderr="")

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=runner,
    )
    response = executor.run(
        ToolRequest(tool="apply_patch", args={"path": "/repo/file.txt", "patch": "new content"})
    )

    assert response.exit_code == 0
    assert len(commands) == 1
    assert commands[0][:3] == ["docker", "exec", "-i"]
    assert "PATCH_PAYLOAD=" not in " ".join(commands[0])
    assert stdin_payloads == ["new content"]
    script = commands[0][-1]
    assert 'cat >"$PATCH_FILE"' in script
    assert "patch --batch --forward" in script
    assert "Failed to apply patch" in script
    assert "PATCH_PAYLOAD" not in script


def test_docker_executor_apply_patch_legacy_edit_alias_is_supported() -> None:
    commands: list[list[str]] = []
    stdin_payloads: list[str | None] = []

    def runner(
        command: list[str],
        *,
        timeout_sec: int,
        stdin_text: str | None = None,
    ) -> CommandResult:
        del timeout_sec
        commands.append(list(command))
        stdin_payloads.append(stdin_text)
        return CommandResult(returncode=0, stdout="ok\n", stderr="")

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=runner,
    )
    response = executor.run(
        ToolRequest(tool="edit", args={"path": "/repo/file.txt", "patch": "legacy mode"})
    )

    assert response.exit_code == 0
    assert commands[0][:3] == ["docker", "exec", "-i"]
    assert stdin_payloads == ["legacy mode"]


def test_docker_executor_apply_patch_codex_format_prefers_apply_patch_command() -> None:
    commands: list[list[str]] = []
    stdin_payloads: list[str | None] = []

    def runner(
        command: list[str],
        *,
        timeout_sec: int,
        stdin_text: str | None = None,
    ) -> CommandResult:
        del timeout_sec
        commands.append(list(command))
        stdin_payloads.append(stdin_text)
        return CommandResult(returncode=0, stdout="No files were modified.\n", stderr="")

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=runner,
    )
    codex_patch = "*** Begin Patch\n*** End Patch\n"
    response = executor.run(
        ToolRequest(tool="apply_patch", args={"patch": codex_patch})
    )

    assert response.exit_code == 0
    script = commands[0][-1]
    assert "command -v apply_patch" in script
    assert "apply_patch;" in script
    assert stdin_payloads == [codex_patch]


def test_docker_executor_apply_patch_requires_path_for_legacy_payload() -> None:
    executor = DockerToolExecutor(container_id="container-1", tool_timeout_sec=30)
    response = executor.run(ToolRequest(tool="apply_patch", args={"patch": "raw content"}))

    assert response.exit_code == 2
    assert "Missing required arg 'path'" in response.stderr


def test_docker_executor_search_command_does_not_suppress_errors() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], *, timeout_sec: int) -> CommandResult:
        del timeout_sec
        commands.append(list(command))
        return CommandResult(returncode=0, stdout="", stderr="")

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=runner,
    )
    response = executor.run(ToolRequest(tool="search", args={"query": "needle", "top_k": 5}))

    assert response.exit_code == 0
    script = commands[0][-1]
    assert "2>/dev/null" not in script
    assert "|| true" not in script
    assert 'if [ ! -e "$SEARCH_PATH" ]; then ' in script
    assert 'search path_hint not found: %s; falling back to .\\n' in script
    assert 'SEARCH_PATH="."; ' in script
    assert 'status=0; grep -R -n -F -m "$TOP_K" -- "$QUERY" "$SEARCH_PATH"' in script
    assert 'if [ "$status" -eq 0 ] || [ "$status" -eq 1 ]; then exit 0; fi;' in script


def test_docker_executor_rejects_out_of_range_bash_timeout() -> None:
    executor = DockerToolExecutor(container_id="container-1", tool_timeout_sec=30)
    response = executor.run(
        ToolRequest(tool="bash", args={"command": "echo hi", "timeout_sec": 601})
    )

    assert response.exit_code == 2
    assert "timeout_sec" in response.stderr


def test_docker_executor_rejects_out_of_range_search_top_k() -> None:
    executor = DockerToolExecutor(container_id="container-1", tool_timeout_sec=30)
    response = executor.run(ToolRequest(tool="search", args={"query": "x", "top_k": 51}))

    assert response.exit_code == 2
    assert "top_k" in response.stderr


def test_docker_executor_runs_bash_with_stdin_stream() -> None:
    commands: list[list[str]] = []
    stdin_payloads: list[str | None] = []

    def runner(
        command: list[str],
        *,
        timeout_sec: int,
        stdin_text: str | None = None,
    ) -> CommandResult:
        del timeout_sec
        commands.append(list(command))
        stdin_payloads.append(stdin_text)
        return CommandResult(returncode=0, stdout="ok\n", stderr="")

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=runner,
    )

    response = executor.run(
        ToolRequest(
            tool="bash",
            args={"command": "cat >/tmp/payload", "stdin": "payload-by-stdin"},
        )
    )

    assert response.exit_code == 0
    assert commands[0][:3] == ["docker", "exec", "-i"]
    assert stdin_payloads == ["payload-by-stdin"]
