"""Docker-backed implementation of the tool execution protocol."""

from __future__ import annotations

import subprocess
import textwrap
from typing import Any

from .command_runner import CommandResult, CommandRunner, default_command_runner
from .runtime_protocol import ToolRequest, ToolResponse

_BASH_TIMEOUT_MIN = 1
_BASH_TIMEOUT_MAX = 7200
_READ_LINE_NUMBER_MIN = 1
_READ_LINE_NUMBER_MAX = 1_000_000_000
_READ_MAX_LINES = 200
_READ_MAX_STDOUT_CHARS = 8192
_FILE_SEARCH_TOP_K_DEFAULT = 10
_FILE_SEARCH_TOP_K_MIN = 1
_FILE_SEARCH_TOP_K_MAX = 50
_TEXT_SEARCH_TOP_K_DEFAULT = 10
_TEXT_SEARCH_TOP_K_MIN = 1
_TEXT_SEARCH_TOP_K_MAX = 50
_APPLY_PATCH_BEGIN_MARKER = "*** Begin Patch"
_FILE_SEARCH_ENGINE = "fuzzy_path"
_TEXT_SEARCH_ENGINE = "grep"
_PREFER_BASH_LOGIN_SHELL_WRAPPER = (
    'if command -v bash >/dev/null 2>&1; then '
    'exec bash -lc "$1"; '
    "else "
    'exec sh -lc "$1"; '
    "fi"
)
_PREFER_BASH_LOGIN_SHELL_ARG0 = "small-swe-shell"
_PYTHON_INTERPRETER_DISCOVERY_SNIPPET = (
    'pybin=""; '
    'for candidate in python3 python; do '
    'if command -v "$candidate" >/dev/null 2>&1; then pybin="$candidate"; break; fi; '
    "done; "
    'if [ -z "$pybin" ]; then '
    'printf "Python interpreter missing in task container.\\n" >&2; '
    "exit 127; "
    "fi; "
)
_FILE_SEARCH_PYTHON_SCRIPT = textwrap.dedent(
    """\
    import os
    import sys

    CANDIDATES = ("/testbed", "/workspace", "/repo", "/app")
    IGNORED_DIRS = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".venv",
        "venv",
        "node_modules",
        "build",
        "dist",
        ".tox",
    }


    def discover_repo_root() -> str:
        for value in (os.environ.get("TASK_REPO_ROOT"), os.environ.get("SMALL_SWE_REPO_ROOT")):
            if value and os.path.exists(value):
                return os.path.abspath(value)
        for candidate in CANDIDATES:
            if os.path.isdir(os.path.join(candidate, ".git")):
                return candidate
        for candidate in CANDIDATES:
            if os.path.isdir(candidate):
                return candidate
        return ""


    def is_within_repo(candidate: str, repo_root: str) -> bool:
        try:
            return os.path.commonpath([os.path.abspath(candidate), repo_root]) == repo_root
        except ValueError:
            return False


    def resolve_search_root(root: str, repo_root: str) -> str:
        if not repo_root:
            raise ValueError("file_search could not determine repo root")
        candidate = repo_root
        if root:
            if os.path.isabs(root):
                candidate = os.path.abspath(root)
            else:
                candidate = os.path.abspath(os.path.join(repo_root, root))
        if not os.path.exists(candidate):
            raise ValueError(f"file_search root not found: {root or repo_root}")
        if not is_within_repo(candidate, repo_root):
            raise ValueError(f"file_search root must resolve inside repo root: {root}")
        return candidate


    def iter_files(search_root: str):
        if os.path.isfile(search_root):
            if os.path.isfile(search_root):
                yield search_root
            return
        for root, dirnames, filenames in os.walk(search_root):
            dirnames[:] = sorted(
                dirname
                for dirname in dirnames
                if dirname not in IGNORED_DIRS
            )
            filenames.sort()
            for filename in filenames:
                yield os.path.join(root, filename)


    def is_subsequence(needle: str, haystack: str) -> bool:
        if not needle:
            return False
        index = 0
        for char in haystack:
            if char == needle[index]:
                index += 1
                if index == len(needle):
                    return True
        return False


    def score_path(relative_path: str, query: str) -> int | None:
        lowered_query = query.lower().strip()
        relative_lower = relative_path.lower()
        basename_lower = os.path.basename(relative_path).lower()
        tokens = [token for token in lowered_query.split() if token]
        compact_query = "".join(tokens) if tokens else lowered_query.replace(" ", "")

        matched = False
        score = 0

        if basename_lower == lowered_query:
            score += 4000
            matched = True
        elif basename_lower.startswith(lowered_query):
            score += 2600
            matched = True
        elif lowered_query in basename_lower:
            score += 1800
            matched = True

        if relative_lower == lowered_query:
            score += 3200
            matched = True
        elif relative_lower.startswith(lowered_query):
            score += 1400
            matched = True
        elif lowered_query in relative_lower:
            score += 900
            matched = True

        token_hits = 0
        for token in tokens:
            if token in basename_lower:
                token_hits += 1
                matched = True
                if basename_lower == token:
                    score += 900
                elif basename_lower.startswith(token):
                    score += 600
                else:
                    score += 350
            elif token in relative_lower:
                token_hits += 1
                matched = True
                score += 180

        if tokens and token_hits == len(tokens):
            score += 500

        if compact_query and compact_query != lowered_query and compact_query in basename_lower:
            matched = True
            score += 220
        elif compact_query and is_subsequence(compact_query, basename_lower):
            matched = True
            score += 140
        elif compact_query and is_subsequence(compact_query, relative_lower):
            matched = True
            score += 80

        if not matched:
            return None

        depth = relative_path.count("/")
        score -= depth * 25
        score -= len(relative_path)
        return score


    query = os.environ["QUERY"].strip()
    top_k_plus_one = int(os.environ["TOP_K_PLUS_ONE"])
    repo_root = discover_repo_root()
    try:
        search_root = resolve_search_root(os.environ.get("SEARCH_ROOT", ""), repo_root)
    except ValueError as exc:
        sys.stderr.write(str(exc) + "\\n")
        raise SystemExit(1)

    ranked_paths: list[tuple[int, str]] = []
    for file_path in iter_files(search_root):
        relative_path = os.path.relpath(file_path, repo_root).replace(os.sep, "/")
        score = score_path(relative_path, query)
        if score is None:
            continue
        ranked_paths.append((score, relative_path))

    ranked_paths.sort(key=lambda item: (-item[0], item[1]))
    for _score, relative_path in ranked_paths[:top_k_plus_one]:
        sys.stdout.write(relative_path + "\\n")
    raise SystemExit(0)
    """
)
_TEXT_SEARCH_PYTHON_SCRIPT = textwrap.dedent(
    """\
    import os
    import subprocess
    import sys

    CANDIDATES = ("/testbed", "/workspace", "/repo", "/app")


    def discover_repo_root() -> str:
        for value in (os.environ.get("TASK_REPO_ROOT"), os.environ.get("SMALL_SWE_REPO_ROOT")):
            if value and os.path.exists(value):
                return os.path.abspath(value)
        for candidate in CANDIDATES:
            if os.path.isdir(os.path.join(candidate, ".git")):
                return candidate
        for candidate in CANDIDATES:
            if os.path.isdir(candidate):
                return candidate
        return ""


    def resolve_path_hint(path_hint: str, repo_root: str) -> str:
        if not path_hint:
            if not repo_root:
                raise ValueError("text_search could not determine repo root")
            return repo_root
        if os.path.isabs(path_hint):
            return os.path.abspath(path_hint)
        if repo_root:
            return os.path.abspath(os.path.join(repo_root, path_hint))
        return os.path.abspath(path_hint)


    def normalize_match_line(raw_line: str, repo_root: str) -> str | None:
        stripped = raw_line.rstrip("\\n").rstrip("\\r")
        parts = stripped.split(":", 2)
        if len(parts) != 3:
            return None
        raw_path, line_number, snippet = parts
        if not line_number.isdigit():
            return None

        absolute_path = os.path.abspath(raw_path)
        normalized_path = absolute_path
        if repo_root:
            try:
                in_repo = os.path.commonpath([absolute_path, repo_root]) == repo_root
            except ValueError:
                in_repo = False
            if in_repo:
                normalized_path = os.path.relpath(absolute_path, repo_root).replace(os.sep, "/")
        return f"{normalized_path}:{line_number}:{snippet}"


    query = os.environ["QUERY"]
    top_k_plus_one = int(os.environ["TOP_K_PLUS_ONE"])
    repo_root = discover_repo_root()
    path_hint = os.environ.get("PATH_HINT", "")
    try:
        search_target = resolve_path_hint(path_hint, repo_root)
    except ValueError as exc:
        sys.stderr.write(str(exc) + "\\n")
        raise SystemExit(1)

    if not os.path.exists(search_target):
        sys.stderr.write(f"text_search path_hint not found: {path_hint or search_target}\\n")
        raise SystemExit(1)

    if os.path.isdir(search_target):
        command = ["grep", "-RInH", "-I", "-F", "--", query, search_target]
    else:
        command = ["grep", "-Hn", "-I", "-F", "--", query, search_target]

    normalized_lines: list[str] = []
    stderr_text = ""
    stopped_early = False
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            normalized = normalize_match_line(raw_line, repo_root)
            if normalized is None:
                continue
            normalized_lines.append(normalized)
            if len(normalized_lines) >= top_k_plus_one:
                stopped_early = True
                process.terminate()
                break
        try:
            _unused_stdout, stderr_text = process.communicate(timeout=1 if stopped_early else None)
        except subprocess.TimeoutExpired:
            process.kill()
            _unused_stdout, stderr_text = process.communicate()
    except Exception:
        process.kill()
        process.communicate()
        raise

    process_returncode = process.returncode
    if stopped_early and process_returncode == -15:
        process_returncode = 0
    if process_returncode not in (0, 1):
        if normalized_lines:
            sys.stdout.write("\\n".join(normalized_lines) + "\\n")
        if stderr_text:
            sys.stderr.write(stderr_text)
        raise SystemExit(process_returncode)

    if normalized_lines:
        sys.stdout.write("\\n".join(normalized_lines) + "\\n")
    if stderr_text:
        sys.stderr.write(stderr_text)
    raise SystemExit(0)
    """
)
_READ_PYTHON_SCRIPT = textwrap.dedent(
    """\
    import os
    import sys

    CANDIDATES = ("/testbed", "/workspace", "/repo", "/app")


    def discover_repo_root() -> str:
        for value in (os.environ.get("TASK_REPO_ROOT"), os.environ.get("SMALL_SWE_REPO_ROOT")):
            if value and os.path.exists(value):
                return os.path.abspath(value)
        for candidate in CANDIDATES:
            if os.path.isdir(os.path.join(candidate, ".git")):
                return candidate
        for candidate in CANDIDATES:
            if os.path.isdir(candidate):
                return candidate
        return ""


    def resolve_read_path(path: str, repo_root: str) -> str:
        if os.path.isabs(path):
            return path
        if repo_root:
            return os.path.join(repo_root, path)
        return path


    target_path = os.environ["TARGET_PATH"]
    read_path = resolve_read_path(target_path, discover_repo_root())
    if not os.path.exists(read_path):
        sys.stderr.write(f"read path not found: {target_path}\\n")
        raise SystemExit(1)
    if os.path.isdir(read_path):
        sys.stderr.write(f"read target is a directory, not a file: {target_path}\\n")
        raise SystemExit(1)
    if not os.path.isfile(read_path):
        sys.stderr.write(f"read target is not a regular file: {target_path}\\n")
        raise SystemExit(1)

    start_line = int(os.environ["START_LINE"]) if os.environ.get("START_LINE") else 1
    end_line = int(os.environ["END_LINE"]) if os.environ.get("END_LINE") else None
    max_lines = int(os.environ["MAX_LINES"])
    max_chars = int(os.environ["MAX_CHARS"])

    selected_line_count = 0
    selected_char_count = 0
    rendered_lines: list[str] = []
    with open(read_path, "r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number < start_line:
                continue
            if end_line is not None and line_number > end_line:
                break
            rendered = f"{line_number:8d}\\t{raw_line.rstrip(chr(10)).rstrip(chr(13))}\\n"
            selected_line_count += 1
            selected_char_count += len(rendered)
            if len(rendered_lines) < max_lines:
                rendered_lines.append(rendered)

    output = "".join(rendered_lines)
    if len(output) > max_chars:
        output = output[:max_chars]
    sys.stdout.write(output)
    if selected_line_count > max_lines or selected_char_count > max_chars:
        sys.stderr.write(
            f"read output truncated to {max_lines} lines or {max_chars} chars; narrow with start_line/end_line.\\n"
        )
    """
)


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
        if request.tool == "read":
            return self._run_read(request.args)
        if request.tool == "file_search":
            return self._run_file_search(request.args)
        if request.tool == "text_search":
            return self._run_text_search(request.args)
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
        docker_cmd.extend(
            [
                self._container_id,
                "sh",
                "-lc",
                _PREFER_BASH_LOGIN_SHELL_WRAPPER,
                _PREFER_BASH_LOGIN_SHELL_ARG0,
                command or "",
            ]
        )
        return self._run_command(
            docker_cmd,
            timeout_sec=timeout_sec or self._tool_timeout_sec,
            stdin_text=stdin_payload,
        )

    def _run_file_search(self, args: dict[str, Any]) -> ToolResponse:
        errors: list[str] = []
        self._reject_unknown_args(
            args,
            allowed={"query", "root", "top_k"},
            tool_name="file_search",
            errors=errors,
        )
        query = self._require_non_empty_str(args, key="query", tool_name="file_search", errors=errors)
        root = self._optional_non_empty_str(args, key="root", tool_name="file_search", errors=errors)
        top_k = self._optional_int_in_range(
            args,
            key="top_k",
            tool_name="file_search",
            minimum=_FILE_SEARCH_TOP_K_MIN,
            maximum=_FILE_SEARCH_TOP_K_MAX,
            default=_FILE_SEARCH_TOP_K_DEFAULT,
            errors=errors,
        )
        if errors:
            return self._validation_error(errors)

        search_cmd = _build_container_python_shell(_FILE_SEARCH_PYTHON_SCRIPT)
        docker_cmd = [
            "docker",
            "exec",
            "-e",
            f"QUERY={query or ''}",
            "-e",
            f"SEARCH_ROOT={root or ''}",
            "-e",
            f"TOP_K_PLUS_ONE={(top_k or _FILE_SEARCH_TOP_K_DEFAULT) + 1}",
            self._container_id,
            "sh",
            "-lc",
            search_cmd,
        ]
        return self._run_compact_search_command(
            docker_cmd,
            timeout_sec=self._tool_timeout_sec,
            top_k=top_k or _FILE_SEARCH_TOP_K_DEFAULT,
            engine=_FILE_SEARCH_ENGINE,
        )

    def _run_text_search(self, args: dict[str, Any]) -> ToolResponse:
        errors: list[str] = []
        self._reject_unknown_args(
            args,
            allowed={"query", "path_hint", "top_k"},
            tool_name="text_search",
            errors=errors,
        )
        query = self._require_non_empty_str(args, key="query", tool_name="text_search", errors=errors)
        path_hint = self._optional_non_empty_str(
            args,
            key="path_hint",
            tool_name="text_search",
            errors=errors,
        )
        top_k = self._optional_int_in_range(
            args,
            key="top_k",
            tool_name="text_search",
            minimum=_TEXT_SEARCH_TOP_K_MIN,
            maximum=_TEXT_SEARCH_TOP_K_MAX,
            default=_TEXT_SEARCH_TOP_K_DEFAULT,
            errors=errors,
        )
        if errors:
            return self._validation_error(errors)

        search_cmd = (
            "set -eu; "
            "if ! command -v grep >/dev/null 2>&1; then "
            'printf "grep is required for text_search but was not found.\\n" >&2; '
            "exit 127; "
            "fi; "
            + _build_container_python_shell(_TEXT_SEARCH_PYTHON_SCRIPT)
        )
        docker_cmd = [
            "docker",
            "exec",
            "-e",
            f"QUERY={query or ''}",
            "-e",
            f"PATH_HINT={path_hint or ''}",
            "-e",
            f"TOP_K_PLUS_ONE={(top_k or _TEXT_SEARCH_TOP_K_DEFAULT) + 1}",
            self._container_id,
            "sh",
            "-lc",
            search_cmd,
        ]
        return self._run_compact_search_command(
            docker_cmd,
            timeout_sec=self._tool_timeout_sec,
            top_k=top_k or _TEXT_SEARCH_TOP_K_DEFAULT,
            engine=_TEXT_SEARCH_ENGINE,
        )

    def _run_read(self, args: dict[str, Any]) -> ToolResponse:
        errors: list[str] = []
        self._reject_unknown_args(
            args,
            allowed={"path", "start_line", "end_line"},
            tool_name="read",
            errors=errors,
        )
        path = self._require_non_empty_str(args, key="path", tool_name="read", errors=errors)
        start_line = self._optional_int_in_range(
            args,
            key="start_line",
            tool_name="read",
            minimum=_READ_LINE_NUMBER_MIN,
            maximum=_READ_LINE_NUMBER_MAX,
            default=0,
            errors=errors,
        )
        end_line = self._optional_int_in_range(
            args,
            key="end_line",
            tool_name="read",
            minimum=_READ_LINE_NUMBER_MIN,
            maximum=_READ_LINE_NUMBER_MAX,
            default=0,
            errors=errors,
        )
        if (
            start_line is not None
            and end_line is not None
            and start_line > 0
            and end_line > 0
            and end_line < start_line
        ):
            errors.append("Arg 'end_line': must be >= start_line")
        if errors:
            return self._validation_error(errors)

        read_cmd = _build_container_python_shell(_READ_PYTHON_SCRIPT)
        docker_cmd = [
            "docker",
            "exec",
            "-e",
            f"TARGET_PATH={path or ''}",
            "-e",
            f"START_LINE={start_line if start_line and start_line > 0 else ''}",
            "-e",
            f"END_LINE={end_line if end_line and end_line > 0 else ''}",
            "-e",
            f"MAX_LINES={_READ_MAX_LINES}",
            "-e",
            f"MAX_CHARS={_READ_MAX_STDOUT_CHARS}",
            self._container_id,
            "sh",
            "-lc",
            read_cmd,
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

    def _invoke_runner(
        self,
        command: list[str],
        *,
        timeout_sec: int,
        stdin_text: str | None = None,
    ) -> CommandResult | ToolResponse:
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
        return result

    def _run_command(
        self,
        command: list[str],
        *,
        timeout_sec: int,
        stdin_text: str | None = None,
    ) -> ToolResponse:
        result = self._invoke_runner(command, timeout_sec=timeout_sec, stdin_text=stdin_text)
        if isinstance(result, ToolResponse):
            return result
        return ToolResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            metadata={"container_id": self._container_id},
        )

    def _run_compact_search_command(
        self,
        command: list[str],
        *,
        timeout_sec: int,
        top_k: int,
        engine: str,
    ) -> ToolResponse:
        result = self._invoke_runner(command, timeout_sec=timeout_sec)
        if isinstance(result, ToolResponse):
            return result

        output_lines = result.stdout.splitlines()
        truncated = len(output_lines) > top_k
        returned_lines = output_lines[:top_k]
        stdout = "\n".join(returned_lines)
        if stdout:
            stdout += "\n"
        return ToolResponse(
            stdout=stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            metadata={
                "engine": engine,
                "returned_count": len(returned_lines),
                "truncated": truncated,
            },
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


def _build_container_python_shell(script: str) -> str:
    normalized_script = script if script.endswith("\n") else f"{script}\n"
    return (
        "set -eu; "
        + _PYTHON_INTERPRETER_DISCOVERY_SNIPPET
        + "exec \"$pybin\" - <<'PY'\n"
        + normalized_script
        + "PY"
    )
