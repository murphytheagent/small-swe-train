"""Shared shell-script snippets for Docker-backed task runtime commands."""

from __future__ import annotations

import shlex


def build_python_interpreter_resolver_shell(
    *,
    var_name: str = "pybin",
    not_found_message: str = "Python interpreter missing in task container.",
) -> str:
    """Return shell statements that resolve `python3`/`python` into one variable."""
    normalized_var_name = str(var_name).strip()
    if not normalized_var_name:
        raise ValueError("var_name must be a non-empty string.")

    message = str(not_found_message)
    return (
        f'{normalized_var_name}=""; '
        'for candidate in python3 python; do '
        f'if command -v "${{candidate}}" >/dev/null 2>&1; then {normalized_var_name}="${{candidate}}"; break; fi; '
        "done; "
        f'if [ -z "${{{normalized_var_name}}}" ]; then '
        f"printf '%s\\n' {shlex.quote(message)} >&2; "
        "exit 127; "
        "fi; "
    )
