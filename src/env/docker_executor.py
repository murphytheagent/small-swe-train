"""Docker-backed implementation of the tool execution protocol."""

from __future__ import annotations

import subprocess
from typing import Any

from .command_runner import CommandRunner, default_command_runner
from .runtime_protocol import ToolRequest, ToolResponse

_BASH_TIMEOUT_MIN = 1
_BASH_TIMEOUT_MAX = 600
_SEARCH_TOP_K_DEFAULT = 10
_SEARCH_TOP_K_MIN = 1
_SEARCH_TOP_K_MAX = 50
_APPLY_PATCH_BEGIN_MARKER = "*** Begin Patch"


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
            errors: list[str] = []
            self._reject_unknown_args(
                request.args,
                allowed={"final_response", "changed_paths"},
                tool_name="submit",
                errors=errors,
            )
            final_response = self._require_non_empty_str(
                request.args,
                key="final_response",
                tool_name="submit",
                errors=errors,
            )
            self._validate_string_list(
                request.args,
                key="changed_paths",
                tool_name="submit",
                errors=errors,
            )
            if errors:
                return self._validation_error(errors)
            return ToolResponse(
                stdout=final_response or "",
                stderr="",
                exit_code=0,
                metadata={"submitted": True, "container_id": self._container_id},
            )

        if request.tool == "bash":
            return self._run_bash(request.args)
        if request.tool == "search":
            return self._run_search(request.args)
        if request.tool in {"apply_patch", "edit"}:
            return self._run_apply_patch(request.args)

        return ToolResponse(
            stdout="",
            stderr=f"Unsupported tool for Docker executor: {request.tool!r}",
            exit_code=2,
            metadata={"container_id": self._container_id},
        )

    def _run_bash(self, args: dict[str, Any]) -> ToolResponse:
        errors: list[str] = []
        self._reject_unknown_args(
            args,
            allowed={"command", "cwd", "timeout_sec", "stdin"},
            tool_name="bash",
            errors=errors,
        )
        command = self._require_non_empty_str(args, key="command", tool_name="bash", errors=errors)
        cwd = self._optional_non_empty_str(args, key="cwd", tool_name="bash", errors=errors)
        stdin_payload = self._optional_str(args, key="stdin", tool_name="bash", errors=errors)
        timeout_sec = self._optional_int_in_range(
            args,
            key="timeout_sec",
            tool_name="bash",
            minimum=_BASH_TIMEOUT_MIN,
            maximum=_BASH_TIMEOUT_MAX,
            default=self._tool_timeout_sec,
            errors=errors,
        )
        if errors:
            return self._validation_error(errors)

        docker_cmd = ["docker", "exec"]
        if stdin_payload is not None:
            docker_cmd.append("-i")
        if cwd is not None:
            docker_cmd.extend(["-w", cwd])
        docker_cmd.extend([self._container_id, "sh", "-lc", command or ""])
        return self._run_command(
            docker_cmd,
            timeout_sec=timeout_sec or self._tool_timeout_sec,
            stdin_text=stdin_payload,
        )

    def _run_search(self, args: dict[str, Any]) -> ToolResponse:
        errors: list[str] = []
        self._reject_unknown_args(
            args,
            allowed={"query", "path_hint", "top_k"},
            tool_name="search",
            errors=errors,
        )
        query = self._require_non_empty_str(args, key="query", tool_name="search", errors=errors)
        path_hint = self._optional_str(args, key="path_hint", tool_name="search", errors=errors)
        top_k = self._optional_int_in_range(
            args,
            key="top_k",
            tool_name="search",
            minimum=_SEARCH_TOP_K_MIN,
            maximum=_SEARCH_TOP_K_MAX,
            default=_SEARCH_TOP_K_DEFAULT,
            errors=errors,
        )
        if errors:
            return self._validation_error(errors)

        resolved_path = path_hint if path_hint else "."
        search_cmd = (
            'status=0; grep -R -n -F -m "$TOP_K" -- "$QUERY" "$PATH_HINT" || status=$?; '
            'if [ "$status" -eq 0 ] || [ "$status" -eq 1 ]; then exit 0; fi; exit "$status"'
        )
        docker_cmd = [
            "docker",
            "exec",
            "-e",
            f"QUERY={query or ''}",
            "-e",
            f"PATH_HINT={resolved_path}",
            "-e",
            f"TOP_K={top_k or _SEARCH_TOP_K_DEFAULT}",
            self._container_id,
            "sh",
            "-lc",
            search_cmd,
        ]
        return self._run_command(docker_cmd, timeout_sec=self._tool_timeout_sec)

    def _run_apply_patch(self, args: dict[str, Any]) -> ToolResponse:
        errors: list[str] = []
        self._reject_unknown_args(
            args,
            allowed={"path", "patch", "description"},
            tool_name="apply_patch",
            errors=errors,
        )
        patch = self._require_non_empty_str(
            args,
            key="patch",
            tool_name="apply_patch",
            errors=errors,
        )
        path = self._optional_non_empty_str(
            args,
            key="path",
            tool_name="apply_patch",
            errors=errors,
        )
        self._optional_str(
            args,
            key="description",
            tool_name="apply_patch",
            errors=errors,
        )
        if errors:
            return self._validation_error(errors)

        patch_text = patch or ""
        timeout_sec = self._tool_timeout_sec
        if _is_codex_apply_patch_payload(patch_text):
            codex_script = (
                "set -eu; "
                "if command -v apply_patch >/dev/null 2>&1; then "
                "apply_patch; "
                "else "
                'printf "Codex apply_patch format requires an apply_patch command in the container.\\n" >&2; '
                "exit 127; "
                "fi"
            )
            codex_cmd = ["docker", "exec", "-i", self._container_id, "sh", "-lc", codex_script]
            return self._run_command(codex_cmd, timeout_sec=timeout_sec, stdin_text=patch_text)

        if path is None:
            return self._validation_error(
                [
                    "Missing required arg 'path' for tool 'apply_patch' when patch is not Codex apply_patch format."
                ]
            )

        script = (
            "set -eu; "
            'PATCH_FILE="$(mktemp)"; '
            'WRITE_FILE=""; '
            "cleanup() { "
            'rm -f "$PATCH_FILE"; '
            'if [ -n "$WRITE_FILE" ]; then rm -f "$WRITE_FILE"; fi; '
            "}; "
            "trap cleanup EXIT; "
            'cat >"$PATCH_FILE"; '
            'mkdir -p "$(dirname "$TARGET_PATH")"; '
            'if grep -Eq "^(--- |\\+\\+\\+ |@@ )" "$PATCH_FILE"; then '
            'if patch --batch --forward "$TARGET_PATH" "$PATCH_FILE"; then '
            'printf "Applied patch to %s\\n" "$TARGET_PATH"; '
            "else "
            'printf "Failed to apply patch to %s\\n" "$TARGET_PATH" >&2; '
            "exit 1; "
            "fi; "
            "else "
            'WRITE_FILE="$(mktemp)"; '
            'cat "$PATCH_FILE" >"$WRITE_FILE"; '
            'mv "$WRITE_FILE" "$TARGET_PATH"; '
            'WRITE_FILE=""; '
            'printf "Wrote file %s\\n" "$TARGET_PATH"; '
            "fi"
        )
        docker_cmd = [
            "docker",
            "exec",
            "-i",
            "-e",
            f"TARGET_PATH={path or ''}",
            self._container_id,
            "sh",
            "-lc",
            script,
        ]
        return self._run_command(docker_cmd, timeout_sec=timeout_sec, stdin_text=patch_text)

    def _run_command(
        self,
        command: list[str],
        *,
        timeout_sec: int,
        stdin_text: str | None = None,
    ) -> ToolResponse:
        try:
            if stdin_text is None:
                result = self._runner(command, timeout_sec=timeout_sec)
            else:
                result = self._runner(
                    command,
                    timeout_sec=timeout_sec,
                    stdin_text=stdin_text,
                )
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

    def _validation_error(self, errors: list[str]) -> ToolResponse:
        return ToolResponse(
            stdout="",
            stderr="; ".join(errors),
            exit_code=2,
            metadata={"container_id": self._container_id, "validation_errors": list(errors)},
        )

    def _reject_unknown_args(
        self,
        args: dict[str, Any],
        *,
        allowed: set[str],
        tool_name: str,
        errors: list[str],
    ) -> None:
        for key in args:
            if key not in allowed:
                errors.append(f"Unknown arg '{key}' for tool '{tool_name}'")

    def _require_non_empty_str(
        self,
        args: dict[str, Any],
        *,
        key: str,
        tool_name: str,
        errors: list[str],
    ) -> str | None:
        raw = args.get(key)
        if raw is None:
            errors.append(f"Missing required arg '{key}' for tool '{tool_name}'")
            return None
        if not isinstance(raw, str):
            errors.append(f"Arg '{key}': expected str, got {type(raw).__name__}")
            return None
        if not raw.strip():
            errors.append(f"Arg '{key}': length must be >= 1")
            return None
        return raw

    def _optional_str(
        self,
        args: dict[str, Any],
        *,
        key: str,
        tool_name: str,
        errors: list[str],
    ) -> str | None:
        if key not in args:
            return None
        raw = args[key]
        if not isinstance(raw, str):
            errors.append(f"Arg '{key}': expected str, got {type(raw).__name__}")
            return None
        return raw

    def _optional_non_empty_str(
        self,
        args: dict[str, Any],
        *,
        key: str,
        tool_name: str,
        errors: list[str],
    ) -> str | None:
        value = self._optional_str(args, key=key, tool_name=tool_name, errors=errors)
        if value is None:
            return None
        if not value.strip():
            errors.append(f"Arg '{key}': length must be >= 1")
            return None
        return value

    def _optional_int_in_range(
        self,
        args: dict[str, Any],
        *,
        key: str,
        tool_name: str,
        minimum: int,
        maximum: int,
        default: int,
        errors: list[str],
    ) -> int | None:
        if key not in args:
            return default
        raw = args[key]
        if isinstance(raw, bool) or not isinstance(raw, int):
            errors.append(f"Arg '{key}': expected int, got {type(raw).__name__}")
            return None
        if raw < minimum:
            errors.append(f"Arg '{key}': must be >= {minimum}")
            return None
        if raw > maximum:
            errors.append(f"Arg '{key}': must be <= {maximum}")
            return None
        return raw

    def _validate_float_range(
        self,
        args: dict[str, Any],
        *,
        key: str,
        tool_name: str,
        minimum: float,
        maximum: float,
        errors: list[str],
    ) -> float | None:
        if key not in args:
            return None
        raw = args[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            errors.append(f"Arg '{key}': expected float, got {type(raw).__name__}")
            return None
        value = float(raw)
        if value < minimum:
            errors.append(f"Arg '{key}': must be >= {minimum}")
            return None
        if value > maximum:
            errors.append(f"Arg '{key}': must be <= {maximum}")
            return None
        return value

    def _validate_string_list(
        self,
        args: dict[str, Any],
        *,
        key: str,
        tool_name: str,
        errors: list[str],
    ) -> None:
        if key not in args:
            return
        raw = args[key]
        if not isinstance(raw, list):
            errors.append(f"Arg '{key}': expected list[str], got {type(raw).__name__}")
            return
        for item in raw:
            if not isinstance(item, str):
                errors.append(f"Arg '{key}': expected list[str], got list[{type(item).__name__}]")
                return


def _is_codex_apply_patch_payload(patch: str) -> bool:
    stripped = patch.lstrip()
    return stripped.startswith(_APPLY_PATCH_BEGIN_MARKER)
