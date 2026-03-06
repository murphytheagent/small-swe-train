from __future__ import annotations

import os
from pathlib import Path
import subprocess

from env.command_runner import CommandResult
from env.docker_executor import DockerToolExecutor
from env.runtime_protocol import ToolRequest


def _make_local_docker_exec_script_runner(*, cwd: Path | None = None):
    def runner(
        command: list[str],
        *,
        timeout_sec: int,
        stdin_text: str | None = None,
    ) -> CommandResult:
        del stdin_text
        assert command[:2] == ["docker", "exec"]

        env = os.environ.copy()
        index = 2
        while index + 1 < len(command) and command[index] == "-e":
            key, _, value = command[index + 1].partition("=")
            env[key] = value
            index += 2

        assert index + 2 < len(command)
        index += 1  # container id
        assert command[index:index + 2] == ["sh", "-lc"]
        script = command[index + 2]
        completed = subprocess.run(
            ["sh", "-lc", script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_sec,
            cwd=cwd,
        )
        return CommandResult(
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return runner


def _run_local_docker_exec_script(
    command: list[str],
    *,
    timeout_sec: int,
    stdin_text: str | None = None,
) -> CommandResult:
    return _make_local_docker_exec_script_runner()(command, timeout_sec=timeout_sec, stdin_text=stdin_text)


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


def test_docker_executor_prefers_bash_login_shell_for_bash_tool() -> None:
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

    response = executor.run(ToolRequest(tool="bash", args={"command": "python -m pytest -q"}))

    assert response.exit_code == 0
    assert commands[0][2] == "container-1"
    assert commands[0][3] == "sh"
    assert commands[0][4] == "-lc"
    assert 'exec bash -lc "$1"; ' in commands[0][5]
    assert 'exec sh -lc "$1"; ' in commands[0][5]
    assert commands[0][6] == "small-swe-shell"
    assert commands[0][7] == "python -m pytest -q"


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
    assert 'for candidate in python3 python; do ' in script
    assert 'search path_hint not found: ' in script
    assert "matches >= top_k" in script


def test_docker_executor_rejects_out_of_range_bash_timeout() -> None:
    executor = DockerToolExecutor(container_id="container-1", tool_timeout_sec=30)
    response = executor.run(
        ToolRequest(tool="bash", args={"command": "echo hi", "timeout_sec": 7201})
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


def test_docker_executor_read_resolves_repo_relative_paths_and_numbers_lines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    target_path = repo_root / "src" / "sample.txt"
    target_path.parent.mkdir(parents=True)
    target_path.write_text("line 1\nline 2\nline 3\nline 4\n", encoding="utf-8")
    monkeypatch.setenv("TASK_REPO_ROOT", str(repo_root))

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=_run_local_docker_exec_script,
    )
    response = executor.run(
        ToolRequest(tool="read", args={"path": "src/sample.txt", "start_line": 2, "end_line": 3})
    )

    assert response.exit_code == 0
    assert response.stderr == ""
    assert response.stdout.splitlines() == [
        "       2\tline 2",
        "       3\tline 3",
    ]


def test_docker_executor_read_supports_path_only_start_only_end_only_and_both_bounds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    target_path = repo_root / "src" / "sample.txt"
    target_path.parent.mkdir(parents=True)
    target_path.write_text("line 1\nline 2\nline 3\nline 4\n", encoding="utf-8")
    monkeypatch.setenv("TASK_REPO_ROOT", str(repo_root))

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=_run_local_docker_exec_script,
    )

    path_only = executor.run(ToolRequest(tool="read", args={"path": "src/sample.txt"}))
    start_only = executor.run(ToolRequest(tool="read", args={"path": "src/sample.txt", "start_line": 3}))
    end_only = executor.run(ToolRequest(tool="read", args={"path": "src/sample.txt", "end_line": 2}))
    bounded = executor.run(
        ToolRequest(tool="read", args={"path": "src/sample.txt", "start_line": 2, "end_line": 4})
    )

    assert path_only.exit_code == 0
    assert start_only.exit_code == 0
    assert end_only.exit_code == 0
    assert bounded.exit_code == 0
    assert path_only.stdout.splitlines() == [
        "       1\tline 1",
        "       2\tline 2",
        "       3\tline 3",
        "       4\tline 4",
    ]
    assert start_only.stdout.splitlines() == [
        "       3\tline 3",
        "       4\tline 4",
    ]
    assert end_only.stdout.splitlines() == [
        "       1\tline 1",
        "       2\tline 2",
    ]
    assert bounded.stdout.splitlines() == [
        "       2\tline 2",
        "       3\tline 3",
        "       4\tline 4",
    ]


def test_docker_executor_read_rejects_missing_file_and_directory_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    target_dir = repo_root / "src"
    target_dir.mkdir(parents=True)
    monkeypatch.setenv("TASK_REPO_ROOT", str(repo_root))

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=_run_local_docker_exec_script,
    )

    missing = executor.run(ToolRequest(tool="read", args={"path": "src/missing.txt"}))
    directory = executor.run(ToolRequest(tool="read", args={"path": "src"}))

    assert missing.exit_code != 0
    assert "read path not found" in missing.stderr
    assert directory.exit_code != 0
    assert "directory" in directory.stderr.lower()


def test_docker_executor_read_guard_caps_output_and_signals_truncation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    target_path = repo_root / "src" / "large.txt"
    target_path.parent.mkdir(parents=True)
    lines = [f"line {index} {'x' * 120}" for index in range(1, 400)]
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("TASK_REPO_ROOT", str(repo_root))

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=_run_local_docker_exec_script,
    )
    response = executor.run(ToolRequest(tool="read", args={"path": "src/large.txt"}))

    assert response.exit_code == 0
    assert len(response.stdout.splitlines()) <= 200
    assert len(response.stdout) <= 8192
    assert "narrow with start_line/end_line" in response.stderr


def test_docker_executor_read_prefers_repo_relative_path_over_process_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    cwd_root = tmp_path / "other"
    repo_target = repo_root / "src" / "sample.txt"
    cwd_target = cwd_root / "src" / "sample.txt"
    repo_target.parent.mkdir(parents=True)
    cwd_target.parent.mkdir(parents=True)
    repo_target.write_text("repo copy\n", encoding="utf-8")
    cwd_target.write_text("cwd copy\n", encoding="utf-8")
    monkeypatch.setenv("TASK_REPO_ROOT", str(repo_root))

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=_make_local_docker_exec_script_runner(cwd=cwd_root),
    )
    response = executor.run(ToolRequest(tool="read", args={"path": "src/sample.txt"}))

    assert response.exit_code == 0
    assert response.stdout.splitlines() == ["       1\trepo copy"]


def test_docker_executor_read_handles_unicode_without_decode_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    target_path = repo_root / "src" / "unicode.txt"
    target_path.parent.mkdir(parents=True)
    target_path.write_text(("é" * 9000) + "\n", encoding="utf-8")
    monkeypatch.setenv("TASK_REPO_ROOT", str(repo_root))

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=_run_local_docker_exec_script,
    )
    response = executor.run(ToolRequest(tool="read", args={"path": "src/unicode.txt"}))

    assert response.exit_code == 0
    assert len(response.stdout) <= 8192
    assert "read output truncated" in response.stderr
    assert "é" in response.stdout


def test_docker_executor_search_applies_top_k_globally(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "a.txt").write_text("needle\nneedle\n", encoding="utf-8")
    (repo_root / "b.txt").write_text("needle\nneedle\n", encoding="utf-8")
    monkeypatch.setenv("TASK_REPO_ROOT", str(repo_root))

    executor = DockerToolExecutor(
        container_id="container-1",
        tool_timeout_sec=30,
        runner=_run_local_docker_exec_script,
    )
    response = executor.run(ToolRequest(tool="search", args={"query": "needle", "top_k": 1}))

    assert response.exit_code == 0
    assert response.stdout.splitlines() == [f"{repo_root / 'a.txt'}:1:needle"]


def test_docker_executor_read_rejects_descending_ranges() -> None:
    executor = DockerToolExecutor(container_id="container-1", tool_timeout_sec=30)
    response = executor.run(
        ToolRequest(tool="read", args={"path": "src/sample.txt", "start_line": 10, "end_line": 9})
    )

    assert response.exit_code == 2
    assert "end_line" in response.stderr
