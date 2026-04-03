"""Shared shell-script snippets for Docker-backed task runtime commands."""

from __future__ import annotations

import shlex


def build_executable_resolver_shell(
    *,
    var_name: str,
    command_names: tuple[str, ...] | list[str],
    fallback_paths: tuple[str, ...] | list[str] = (),
    not_found_message: str,
) -> str:
    """Return shell statements that resolve an executable into one variable."""
    normalized_var_name = str(var_name).strip()
    if not normalized_var_name:
        raise ValueError("var_name must be a non-empty string.")

    normalized_command_names = tuple(str(name).strip() for name in command_names if str(name).strip())
    normalized_fallback_paths = tuple(
        str(path).strip() for path in fallback_paths if str(path).strip()
    )
    if not normalized_command_names and not normalized_fallback_paths:
        raise ValueError("At least one command name or fallback path must be provided.")

    command_candidates = " ".join(shlex.quote(name) for name in normalized_command_names)
    path_candidates = " ".join(shlex.quote(path) for path in normalized_fallback_paths)
    message = str(not_found_message)

    command_probe = ""
    if command_candidates:
        command_probe = (
            f"for candidate in {command_candidates}; do "
            f'if command -v "${{candidate}}" >/dev/null 2>&1; then {normalized_var_name}="$(command -v "${{candidate}}")"; break; fi; '
            "done; "
        )
    path_probe = ""
    if path_candidates:
        path_probe = (
            f'if [ -z "${{{normalized_var_name}}}" ]; then '
            f"for candidate in {path_candidates}; do "
            f'if [ -x "${{candidate}}" ]; then {normalized_var_name}="${{candidate}}"; break; fi; '
            "done; "
            "fi; "
        )
    return (
        f'{normalized_var_name}=""; '
        + command_probe
        + path_probe
        + f'if [ -z "${{{normalized_var_name}}}" ]; then '
        + f"printf '%s\\n' {shlex.quote(message)} >&2; "
        + "exit 127; "
        + "fi; "
    )


def build_python_interpreter_resolver_shell(
    *,
    var_name: str = "pybin",
    not_found_message: str = "Python interpreter missing in task container.",
) -> str:
    """Return shell statements that resolve `python3`/`python` into one variable."""
    return build_executable_resolver_shell(
        var_name=var_name,
        command_names=("python3", "python"),
        fallback_paths=(),
        not_found_message=not_found_message,
    )
