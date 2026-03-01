"""Per-step Docker container pool for on-policy rollout collection."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Sequence

from .command_runner import CommandRunner, default_command_runner
from .task_dataset import TaskSample

_CONTAINER_START_MAX_ATTEMPTS = 3
_CONTAINER_START_RETRY_BASE_DELAY_SEC = 1.0
_CONTAINER_START_RETRY_MAX_DELAY_SEC = 5.0


@dataclass(frozen=True)
class ContainerHandle:
    task_id: str
    image_name: str
    container_id: str
    container_name: str


class BatchContainerPool:
    """Create and teardown task containers for one global training step."""

    def __init__(
        self,
        *,
        env_pool_size: int,
        container_start_timeout_sec: int,
        cleanup_timeout_sec: int = 30,
        name_prefix: str = "onpolicy-task",
        runner: CommandRunner | None = None,
    ) -> None:
        if env_pool_size < 1:
            raise ValueError("env_pool_size must be >= 1")
        if container_start_timeout_sec < 1:
            raise ValueError("container_start_timeout_sec must be >= 1")
        if cleanup_timeout_sec < 1:
            raise ValueError("cleanup_timeout_sec must be >= 1")

        self._env_pool_size = env_pool_size
        self._container_start_timeout_sec = container_start_timeout_sec
        self._cleanup_timeout_sec = cleanup_timeout_sec
        self._name_prefix = name_prefix
        self._runner = runner or default_command_runner
        self._active_handles: list[ContainerHandle] = []

    @property
    def active_handles(self) -> tuple[ContainerHandle, ...]:
        return tuple(self._active_handles)

    def acquire(self, tasks: Sequence[TaskSample]) -> tuple[ContainerHandle, ...]:
        if len(tasks) > self._env_pool_size:
            raise ValueError(
                "Task batch exceeds env pool size. "
                f"tasks={len(tasks)} pool={self._env_pool_size}"
            )
        if self._active_handles:
            raise RuntimeError("Pool already has active containers. Call release_all() first.")

        created: list[ContainerHandle] = []
        try:
            for task in tasks:
                handle = self._start_container(task)
                created.append(handle)
            self._active_handles = created
            return tuple(created)
        except Exception:
            self._active_handles = created
            self.release_all()
            raise

    def _start_container(self, task: TaskSample) -> ContainerHandle:
        suffix = uuid.uuid4().hex[:8]
        container_name = f"{self._name_prefix}-{suffix}"
        label_args = self._build_container_label_args()
        last_timeout = False
        last_error = "<unknown error>"
        for attempt_index in range(_CONTAINER_START_MAX_ATTEMPTS):
            command = [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,
                *label_args,
                task.image_name,
                "sh",
                "-lc",
                "sleep infinity",
            ]
            try:
                result = self._runner(command, timeout_sec=self._container_start_timeout_sec)
            except subprocess.TimeoutExpired:
                last_timeout = True
                last_error = f"docker run timed out after {self._container_start_timeout_sec}s"
                self._best_effort_remove_container(container_name)
            else:
                if result.returncode == 0:
                    container_id = (
                        result.stdout.strip().splitlines()[0]
                        if result.stdout.strip()
                        else container_name
                    )
                    return ContainerHandle(
                        task_id=task.task_id,
                        image_name=task.image_name,
                        container_id=container_id,
                        container_name=container_name,
                    )

                last_timeout = False
                last_error = result.stderr.strip() or "<empty stderr>"
                self._best_effort_remove_container(container_name)

            if attempt_index + 1 < _CONTAINER_START_MAX_ATTEMPTS:
                backoff_sec = min(
                    _CONTAINER_START_RETRY_BASE_DELAY_SEC * (2**attempt_index),
                    _CONTAINER_START_RETRY_MAX_DELAY_SEC,
                )
                time.sleep(backoff_sec)

        if last_timeout:
            raise RuntimeError(
                f"Container start timed out for task {task.task_id!r} with image {task.image_name!r} "
                f"after {_CONTAINER_START_MAX_ATTEMPTS} attempts."
            )
        raise RuntimeError(
            f"Failed to start container for task {task.task_id!r} after {_CONTAINER_START_MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _build_container_label_args(self) -> list[str]:
        labels = {
            "small_swe.managed": "1",
            "small_swe.pool_name": self._name_prefix,
        }
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip() or os.environ.get(
            "SLURM_JOBID", ""
        ).strip()
        if slurm_job_id:
            labels["small_swe.slurm_job_id"] = slurm_job_id

        env_to_label = {
            "SDPO_RUN_LABEL": "small_swe.run_label",
            "EXPERIMENT": "small_swe.experiment",
        }
        for env_key, label_key in env_to_label.items():
            value = os.environ.get(env_key, "").strip()
            if value:
                labels[label_key] = value

        args: list[str] = []
        for key, value in labels.items():
            args.extend(["--label", f"{key}={value}"])
        return args

    def release_all(self) -> None:
        handles = list(self._active_handles)
        self._active_handles = []
        for handle in handles:
            command = ["docker", "rm", "-f", handle.container_id]
            try:
                self._runner(command, timeout_sec=self._cleanup_timeout_sec)
            except subprocess.TimeoutExpired:
                # Best-effort cleanup in all error cases.
                continue

    def _best_effort_remove_container(self, container_ref: str) -> None:
        try:
            self._runner(["docker", "rm", "-f", container_ref], timeout_sec=self._cleanup_timeout_sec)
        except subprocess.TimeoutExpired:
            # Best-effort cleanup in all error cases.
            return
