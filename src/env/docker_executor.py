"""Docker-backed implementation of the tool execution protocol."""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

from .command_runner import CommandRunner, default_command_runner
from .runtime_protocol import ToolRequest, ToolResponse


class DockerToolExecutor:
    """Execute canonical tools inside one running Docker container."""

    def __init__(
        self,
        *,
        container_id: str,
        tool_timeout_sec: int,
        runner: CommandRunner | None = None,
    ) -> None:
        if not container_id.strip():
            raise ValueError("container_id must be a non-empty string.")
        if tool_timeout_sec < 1:
            raise ValueError("tool_timeout_sec must be >= 1.")
        self._container_id = container_id
        self._tool_timeout_sec = tool_timeout_sec
        self._runner = runner or default_command_runner

    def run(self, request: ToolRequest) -> ToolResponse:
        if request.tool == "submit":
            final_response = str(request.args.get("final_response", ""))
            return ToolResponse(
                stdout=final_response,
                stderr="",
                exit_code=0,
                metadata={"submitted": True, "container_id": self._container_id},
            )

        if request.tool == "bash":
            return self._run_bash(request.args)
        if request.tool == "search":
            return self._run_search(request.args)
        if request.tool == "edit":
            return self._run_edit(request.args)

        return ToolResponse(
            stdout="",
            stderr=f"Unsupported tool for Docker executor: {request.tool!r}",
            exit_code=2,
            metadata={"container_id": self._container_id},
        )

    def _run_bash(self, args: dict[str, Any]) -> ToolResponse:
        command = str(args.get("command", ""))
        if not command.strip():
            return ToolResponse(
                stdout="",
                stderr="Missing required 'command' for bash tool.",
                exit_code=2,
                metadata={"container_id": self._container_id},
            )
        cwd = args.get("cwd")
        timeout_sec = _coerce_timeout(args.get("timeout_sec"), fallback=self._tool_timeout_sec)

        docker_cmd = ["docker", "exec"]
        if isinstance(cwd, str) and cwd.strip():
            docker_cmd.extend(["-w", cwd.strip()])
        docker_cmd.extend([self._container_id, "sh", "-lc", command])
        return self._run_command(docker_cmd, timeout_sec=timeout_sec)

    def _run_search(self, args: dict[str, Any]) -> ToolResponse:
        query = str(args.get("query", ""))
        if not query.strip():
            return ToolResponse(
                stdout="",
                stderr="Missing required 'query' for search tool.",
                exit_code=2,
                metadata={"container_id": self._container_id},
            )
        path_hint = str(args.get("path_hint") or ".")
        top_k = _coerce_positive_int(args.get("top_k"), fallback=10)
        timeout_sec = self._tool_timeout_sec

        quoted_query = shlex.quote(query)
        quoted_path = shlex.quote(path_hint)
        search_cmd = (
            "grep -R -n -F -- "
            f"{quoted_query} {quoted_path} 2>/dev/null | head -n {top_k} || true"
        )
        docker_cmd = ["docker", "exec", self._container_id, "sh", "-lc", search_cmd]
        return self._run_command(docker_cmd, timeout_sec=timeout_sec)

    def _run_edit(self, args: dict[str, Any]) -> ToolResponse:
        path = str(args.get("path", ""))
        patch = str(args.get("patch", ""))
        if not path.strip() or not patch:
            return ToolResponse(
                stdout="",
                stderr="Missing required 'path' or 'patch' for edit tool.",
                exit_code=2,
                metadata={"container_id": self._container_id},
            )
        timeout_sec = self._tool_timeout_sec
        script = (
            'mkdir -p "$(dirname "$TARGET_PATH")" && '
            'PATCH_FILE="$(mktemp)" && '
            'printf "%s\\n" "$PATCH_PAYLOAD" > "$PATCH_FILE" && '
            'if [ -f "$TARGET_PATH" ] && grep -Eq "^(--- |\\+\\+\\+ |@@ |\\*\\*\\* Begin Patch)" "$PATCH_FILE"; then '
            'if patch --batch --forward --silent "$TARGET_PATH" "$PATCH_FILE" >/dev/null 2>&1; then '
            ":; "
            "else "
            'printf "%s\\n" "$PATCH_PAYLOAD" > "$TARGET_PATH"; '
            "fi; "
            "else "
            'printf "%s\\n" "$PATCH_PAYLOAD" > "$TARGET_PATH"; '
            "fi && "
            'rm -f "$PATCH_FILE"'
        )
        docker_cmd = [
            "docker",
            "exec",
            "-e",
            f"TARGET_PATH={path}",
            "-e",
            f"PATCH_PAYLOAD={patch}",
            self._container_id,
            "sh",
            "-lc",
            script,
        ]
        return self._run_command(docker_cmd, timeout_sec=timeout_sec)

    def _run_command(self, command: list[str], *, timeout_sec: int) -> ToolResponse:
        try:
            result = self._runner(command, timeout_sec=timeout_sec)
        except subprocess.TimeoutExpired:
            return ToolResponse(
                stdout="",
                stderr=f"Tool execution timed out after {timeout_sec}s.",
                exit_code=124,
                metadata={"container_id": self._container_id, "timed_out": True},
            )
        except FileNotFoundError as exc:
            return ToolResponse(
                stdout="",
                stderr=f"Docker command not found: {exc}",
                exit_code=127,
                metadata={"container_id": self._container_id},
            )
        return ToolResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            metadata={"container_id": self._container_id},
        )


def _coerce_timeout(value: Any, *, fallback: int) -> int:
    if value is None:
        return fallback
    return _coerce_positive_int(value, fallback=fallback)


def _coerce_positive_int(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int) and value >= 1:
        return value
    return fallback
