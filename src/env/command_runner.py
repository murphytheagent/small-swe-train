"""Command runner primitives for Docker-backed environment modules."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_sec: int,
        stdin_text: str | None = None,
    ) -> CommandResult:
        ...


def default_command_runner(
    command: Sequence[str],
    *,
    timeout_sec: int,
    stdin_text: str | None = None,
) -> CommandResult:
    """Execute a command and return captured text output."""
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        input=stdin_text,
        timeout=timeout_sec,
    )
    return CommandResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
