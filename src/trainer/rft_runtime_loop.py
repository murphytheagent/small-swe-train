"""End-to-end RFT loop orchestration for live rollout -> train -> checkpoint refresh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from config import resolve_rft_collector_max_in_flight_default
from trainer.rft_multiturn_dataset import (
    build_multiturn_messages,
    write_selected_rows_to_multiturn_parquet,
)
from trainer.rft_runtime import OnPolicyRFTRuntimeRequest, collect_onpolicy_rft_runtime_batch

_GLOBAL_STEP_PATTERN = re.compile(r"^global_step_(\d+)$")
_VERL_SFT_TRAINER_DOC = (
    "https://github.com/lasgroup/SDPO/blob/main/verl/trainer/fsdp_sft_trainer.py"
)
_VLLM_OPENAI_SERVER_DOC = (
    "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html"
)
_VLLM_OPENAI_SERVER_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py"
)
_MICRO_BATCH_SIZE_KEY = "data.micro_batch_size_per_gpu"
_DATA_MAX_LENGTH_KEY = "data.max_length"


@dataclass(frozen=True)
class RFTLoopConfig:
    project_root: Path
    config_dir: Path
    config_name: str
    trainer_module: str
    python_bin: str
    nnodes: int
    nproc_per_node: int
    rft_steps: int
    samples_per_task: int
    task_batch_size: int
    sft_num_epoch_per_batch: int
    checkpoint_keep_last: int
    train_batch_size: int
    output_dir: Path
    data_config_name: str
    turn_generator_mode: str
    initial_model: str
    vllm_base_url: str
    vllm_served_model: str
    manage_vllm: bool
    vllm_launch_module: str
    vllm_ready_timeout_sec: int
    vllm_stop_timeout_sec: int
    vllm_extra_args: tuple[str, ...]
    trainer_overrides: tuple[str, ...]
    dry_run: bool
    collector_max_in_flight_tasks: int | None = None
    collector_max_turns_per_attempt: int | None = None
    eval_split_fraction: float = 0.1
    eval_min_rows: int = 1


class VLLMServerController:
    """Manage an OpenAI-compatible vLLM server process for the RFT loop.

    Grounding: vLLM serves OpenAI-compatible chat/completions via
    `python -m vllm.entrypoints.openai.api_server` as documented in:
    https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

    def __init__(self, *, config: RFTLoopConfig, log_path: Path) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._log_path = log_path
        self._models_url = _build_models_url(config.vllm_base_url)
        self._api_key = _resolve_vllm_api_key()

    def start(self, *, model_path: str) -> None:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("vLLM server is already running; stop it before starting a new model.")

        command = build_vllm_server_command(
            python_bin=self._config.python_bin,
            launch_module=self._config.vllm_launch_module,
            base_url=self._config.vllm_base_url,
            model_path=model_path,
            served_model_name=self._config.vllm_served_model,
            extra_args=self._config.vllm_extra_args,
        )

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self._log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            command,
            cwd=self._config.project_root,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
        )
        self._wait_until_ready()

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=self._config.vllm_stop_timeout_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self._config.vllm_stop_timeout_sec)

    def _wait_until_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self._config.vllm_ready_timeout_sec
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early with code {self._process.returncode}. "
                    f"Inspect logs at {self._log_path}."
                )
            if _is_http_endpoint_ready(self._models_url, api_key=self._api_key):
                return
            time.sleep(1.0)
        raise RuntimeError(
            "Timed out waiting for vLLM readiness at "
            f"{self._models_url}. Inspect logs at {self._log_path}."
        )


def run_rft_runtime_loop(config: RFTLoopConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    vllm_logs = config.output_dir / "vllm_server.log"
    collector_max_in_flight_tasks = config.collector_max_in_flight_tasks
    if collector_max_in_flight_tasks is None:
        collector_max_in_flight_tasks = resolve_rft_collector_max_in_flight_default(
            task_batch_size=config.task_batch_size,
        )
    collector_max_in_flight_tasks = max(1, min(collector_max_in_flight_tasks, config.task_batch_size))
    collector_max_turns_per_attempt = config.collector_max_turns_per_attempt
    micro_batch_size_per_gpu = resolve_micro_batch_size_per_gpu(
        config_dir=config.config_dir,
        config_name=config.config_name,
        trainer_overrides=config.trainer_overrides,
    )
    trainer_data_max_length = resolve_data_max_length(
        config_dir=config.config_dir,
        config_name=config.config_name,
        trainer_overrides=config.trainer_overrides,
    )
    runtime_manifest: dict[str, Any] = {
        "generated_utc": _utc_now(),
        "config": {
            "rft_steps": config.rft_steps,
            "samples_per_task": config.samples_per_task,
            "task_batch_size": config.task_batch_size,
            "eval_split_fraction": config.eval_split_fraction,
            "eval_min_rows": config.eval_min_rows,
            "collector_max_in_flight_tasks": collector_max_in_flight_tasks,
            "collector_max_turns_per_attempt": collector_max_turns_per_attempt,
            "sft_num_epoch_per_batch": config.sft_num_epoch_per_batch,
            "checkpoint_keep_last": config.checkpoint_keep_last,
            "train_batch_size": config.train_batch_size,
            "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
            "trainer_data_max_length": trainer_data_max_length,
            "data_config_name": config.data_config_name,
            "turn_generator_mode": config.turn_generator_mode,
            "initial_model": config.initial_model,
            "vllm_base_url": config.vllm_base_url,
            "vllm_served_model": config.vllm_served_model,
            "manage_vllm": config.manage_vllm,
        },
        "steps": [],
    }

    if config.dry_run:
        _print_dry_run_plan(
            config=config,
            collector_max_in_flight_tasks=collector_max_in_flight_tasks,
        )
        return

    tokenizer = _load_tokenizer(config.initial_model)
    current_model_path = config.initial_model
    vllm_controller = VLLMServerController(config=config, log_path=vllm_logs)
    run_step_dirs: list[Path] = []
    checkpoint_step_dirs: list[Path] = []

    try:
        if config.manage_vllm:
            vllm_controller.start(model_path=current_model_path)

        for step_index in range(config.rft_steps):
            step_start = time.monotonic()
            step_dir = config.output_dir / f"rft_step_{step_index:05d}"
            collector_dir = step_dir / "collector_artifacts"
            train_parquet_path = step_dir / "accepted_trajectories.parquet"
            eval_parquet_path = step_dir / "accepted_trajectories_eval.parquet"
            trainer_checkpoint_root = step_dir / "trainer_checkpoints"
            reset_step_artifacts(step_dir)
            step_dir.mkdir(parents=True, exist_ok=True)
            run_step_dirs.append(step_dir)

            runtime_overrides: dict[str, int] = {
                "task_batch_size": config.task_batch_size,
                "attempts_per_task": config.samples_per_task,
                "max_in_flight_tasks": collector_max_in_flight_tasks,
            }
            if collector_max_turns_per_attempt is not None:
                runtime_overrides["max_turns_per_attempt"] = collector_max_turns_per_attempt
            request = OnPolicyRFTRuntimeRequest(
                data_config_name=config.data_config_name,
                turn_generator_mode=config.turn_generator_mode,
                total_steps=1,
                start_step_index=step_index,
                runtime_overrides=runtime_overrides,
                output_dir=str(collector_dir),
            )
            collect_start = time.monotonic()
            handoff = collect_onpolicy_rft_runtime_batch(
                request=request,
                tokenizer=tokenizer,
            )
            collect_duration_sec = time.monotonic() - collect_start
            selected_rows = _coerce_rows(handoff.get("selected_rows"))
            rejected_rows = _coerce_rows(handoff.get("rejected_rows"))
            selected_count_raw = len(selected_rows)
            (
                selected_rows,
                selected_rows_over_max_length_dropped,
            ) = filter_selected_rows_by_token_length(
                selected_rows=selected_rows,
                tokenizer=tokenizer,
                max_sequence_length=trainer_data_max_length,
            )
            selected_count_after_max_length_filter = len(selected_rows)
            selected_count_for_train_raw = 0
            selected_count_for_train = 0
            selected_count_for_eval = 0
            selected_rows_upsampled = 0
            eval_split_fallback_to_train = False
            resolved_val_parquet_path = train_parquet_path
            effective_train_batch_size: int | None = None
            trainer_command: list[str] | None = None
            latest_hf_checkpoint: Path | None = None
            pruned_global_step_checkpoints: list[Path] = []
            trainer_duration_sec: float | None = None
            trainer_skipped = False
            skip_reason: str | None = None

            if selected_count_after_max_length_filter < 1:
                trainer_skipped = True
                skip_reason = "no_selected_rows_after_length_filter"
            else:
                selected_rows_for_train, selected_rows_for_eval = split_selected_rows_for_eval(
                    selected_rows,
                    eval_split_fraction=config.eval_split_fraction,
                    min_eval_rows=config.eval_min_rows,
                )
                selected_count_for_train_raw = len(selected_rows_for_train)
                selected_count_for_eval = len(selected_rows_for_eval)
                if selected_count_for_train_raw < 1:
                    trainer_skipped = True
                    skip_reason = "empty_train_split"

                world_size = config.nnodes * config.nproc_per_node
                if not trainer_skipped:
                    selected_count_for_train = write_selected_rows_to_multiturn_parquet(
                        selected_rows_for_train,
                        train_parquet_path,
                    )
                    if selected_rows_for_eval:
                        selected_count_for_eval = write_selected_rows_to_multiturn_parquet(
                            selected_rows_for_eval,
                            eval_parquet_path,
                        )
                        resolved_val_parquet_path = eval_parquet_path
                    else:
                        eval_split_fallback_to_train = True
                        resolved_val_parquet_path = train_parquet_path

                    effective_train_batch_size = resolve_effective_train_batch_size(
                        requested=config.train_batch_size,
                        selected_count=selected_count_for_train,
                        world_size=world_size,
                        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
                    )
                    if effective_train_batch_size is None:
                        trainer_skipped = True
                        skip_reason = "insufficient_selected_rows_for_batch_constraints"
                    else:
                        selected_rows_for_train, selected_rows_upsampled = (
                            upsample_selected_rows_to_batch_multiple(
                                selected_rows_for_train,
                                global_batch_size=effective_train_batch_size,
                            )
                        )
                        if selected_rows_upsampled > 0:
                            selected_count_for_train = write_selected_rows_to_multiturn_parquet(
                                selected_rows_for_train,
                                train_parquet_path,
                            )
                        trainer_command = build_trainer_step_command(
                            python_bin=config.python_bin,
                            nnodes=config.nnodes,
                            nproc_per_node=config.nproc_per_node,
                            trainer_module=config.trainer_module,
                            config_name=config.config_name,
                            config_dir=config.config_dir,
                            model_path=current_model_path,
                            train_parquet_path=train_parquet_path,
                            val_parquet_path=resolved_val_parquet_path,
                            trainer_output_dir=trainer_checkpoint_root,
                            train_batch_size=effective_train_batch_size,
                            sft_num_epoch_per_batch=config.sft_num_epoch_per_batch,
                            trainer_overrides=config.trainer_overrides,
                        )

                        if config.manage_vllm:
                            vllm_controller.stop()
                        trainer_start = time.monotonic()
                        _run_command(trainer_command, cwd=config.project_root)
                        trainer_duration_sec = time.monotonic() - trainer_start

                        latest_hf_checkpoint = resolve_latest_hf_checkpoint(trainer_checkpoint_root)
                        pruned_global_step_checkpoints = prune_old_global_step_checkpoints(
                            checkpoint_root=trainer_checkpoint_root,
                            keep_last=config.checkpoint_keep_last,
                        )
                        current_model_path = str(latest_hf_checkpoint)
                        checkpoint_step_dirs.append(step_dir)

                        # Restart vLLM only when another collection step remains.
                        # Restarting after the final step adds unnecessary startup cost
                        # and can surface avoidable restart-path failures.
                        if config.manage_vllm and step_index + 1 < config.rft_steps:
                            vllm_controller.start(model_path=current_model_path)

            pruned_checkpoint_roots: list[Path] = []
            if latest_hf_checkpoint is not None:
                pruned_checkpoint_roots = prune_old_step_checkpoints(
                    step_dirs=checkpoint_step_dirs,
                    keep_last=config.checkpoint_keep_last,
                )
            pruned_step_payloads = prune_old_step_payloads(
                step_dirs=run_step_dirs,
                keep_last=config.checkpoint_keep_last,
            )
            step_summary = {
                "step_index": step_index,
                "selected_count": selected_count_raw,
                "selected_count_raw": selected_count_raw,
                "selected_count_after_length_filter": selected_count_after_max_length_filter,
                "selected_rows_over_max_length_dropped": selected_rows_over_max_length_dropped,
                "selected_count_for_train_raw": selected_count_for_train_raw,
                "selected_count_for_train": selected_count_for_train,
                "selected_count_for_eval": selected_count_for_eval,
                "selected_rows_upsampled": selected_rows_upsampled,
                "eval_split_fallback_to_train": eval_split_fallback_to_train,
                "rejected_count": len(rejected_rows),
                "trainer_skipped": trainer_skipped,
                "skip_reason": skip_reason,
                "effective_train_batch_size": effective_train_batch_size,
                "collector_duration_sec": collect_duration_sec,
                "trainer_duration_sec": trainer_duration_sec,
                "step_duration_sec": time.monotonic() - step_start,
                "train_parquet": str(train_parquet_path),
                "eval_parquet": str(resolved_val_parquet_path),
                "trainer_checkpoint_root": str(trainer_checkpoint_root),
                "latest_hf_checkpoint": str(latest_hf_checkpoint) if latest_hf_checkpoint else None,
                "trainer_command": trainer_command,
                "pruned_global_step_checkpoints": [
                    str(path) for path in pruned_global_step_checkpoints
                ],
                "pruned_checkpoint_roots": [str(path) for path in pruned_checkpoint_roots],
                "pruned_step_payloads": [str(path) for path in pruned_step_payloads],
            }
            runtime_manifest["steps"].append(step_summary)
            _write_json(step_dir / "rft_step_summary.json", step_summary)
    finally:
        if config.manage_vllm:
            vllm_controller.stop()

    runtime_manifest["final_model_path"] = current_model_path
    runtime_manifest["completed_utc"] = _utc_now()
    _write_json(config.output_dir / "rft_runtime_loop_manifest.json", runtime_manifest)


def build_trainer_step_command(
    *,
    python_bin: str,
    nnodes: int,
    nproc_per_node: int,
    trainer_module: str,
    config_name: str,
    config_dir: Path,
    model_path: str,
    train_parquet_path: Path,
    val_parquet_path: Path,
    trainer_output_dir: Path,
    train_batch_size: int,
    sft_num_epoch_per_batch: int,
    trainer_overrides: Sequence[str],
) -> list[str]:
    """Build the documented verl SFT trainer launch command.

    Grounding: SDPO/verl SFT entrypoint (`torchrun -m verl.trainer.fsdp_sft_trainer`)
    in project source/docs:
    https://github.com/lasgroup/SDPO/blob/main/verl/trainer/fsdp_sft_trainer.py
    """
    required_overrides = [
        f"trainer.total_epochs={sft_num_epoch_per_batch}",
        f"trainer.n_gpus_per_node={nproc_per_node}",
        "trainer.resume_mode=disable",
        f"trainer.default_local_dir={trainer_output_dir}",
        # Runtime loop only consumes HuggingFace exports for vLLM restarts; keeping
        # checkpoint payloads hf_model-only avoids redundant dense/FSDP artifacts.
        "trainer.checkpoint.save_contents=[hf_model]",
        "trainer.checkpoint.load_contents=[hf_model]",
        f"data.train_batch_size={train_batch_size}",
        "data.on_policy.enabled=false",
        "data.multiturn.enable=true",
        "data.multiturn.messages_key=messages",
        "data.custom_cls.path=null",
        "data.custom_cls.name=null",
        f"data.train_files={train_parquet_path}",
        f"data.val_files={val_parquet_path}",
        f"model.partial_pretrain={model_path}",
    ]

    return [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        str(nnodes),
        "--nproc_per_node",
        str(nproc_per_node),
        "-m",
        trainer_module,
        "--config-name",
        config_name,
        "--config-dir",
        str(config_dir),
        *trainer_overrides,
        *required_overrides,
    ]


def build_vllm_server_command(
    *,
    python_bin: str,
    launch_module: str,
    base_url: str,
    model_path: str,
    served_model_name: str,
    extra_args: Sequence[str],
) -> list[str]:
    """Build the documented vLLM OpenAI-compatible API server launch command.

    Grounding:
    https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"vLLM base URL must include http/https scheme, got {base_url!r}.")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"vLLM base URL must include host and port, got {base_url!r}.")

    command = [
        python_bin,
        "-m",
        launch_module,
        "--host",
        parsed.hostname,
        "--port",
        str(parsed.port),
        "--model",
        model_path,
        "--served-model-name",
        served_model_name,
    ]
    command.extend(extra_args)
    return command


def resolve_latest_hf_checkpoint(checkpoint_root: str | Path) -> Path:
    root = Path(checkpoint_root)
    if not root.exists():
        raise FileNotFoundError(f"Trainer checkpoint root does not exist: {root}")

    candidates = list(_iter_global_step_dirs(root))
    if not candidates:
        raise FileNotFoundError(f"No global_step_* checkpoint directories found in {root}")

    candidates.sort(key=lambda item: (item[1], item[0]))
    _, _, latest_step_dir = candidates[-1]
    huggingface_dir = latest_step_dir / "huggingface"
    if not huggingface_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint {latest_step_dir} is missing huggingface export directory."
        )
    return huggingface_dir


def prune_old_step_checkpoints(*, step_dirs: Sequence[str | Path], keep_last: int) -> list[Path]:
    """Delete old per-step trainer checkpoint trees beyond the keep-last window."""
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1 to preserve the current model checkpoint.")

    ordered_step_dirs = _coerce_step_dirs_in_order(step_dirs)
    if len(ordered_step_dirs) <= keep_last:
        return []

    to_prune = ordered_step_dirs[: len(ordered_step_dirs) - keep_last]
    pruned: list[Path] = []
    for step_dir in to_prune:
        checkpoint_root = step_dir / "trainer_checkpoints"
        if checkpoint_root.is_dir():
            shutil.rmtree(checkpoint_root)
            pruned.append(checkpoint_root)
    return pruned


def prune_old_global_step_checkpoints(*, checkpoint_root: str | Path, keep_last: int) -> list[Path]:
    """Delete old global_step_* directories in one trainer checkpoint root."""
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1 to preserve the latest global step checkpoint.")

    resolved_root = Path(checkpoint_root)
    if not resolved_root.exists():
        return []

    global_step_dirs = list(_iter_global_step_dirs(resolved_root))
    if len(global_step_dirs) <= keep_last:
        return []

    global_step_dirs.sort(key=lambda item: (item[1], item[0]))
    to_prune = global_step_dirs[: len(global_step_dirs) - keep_last]
    pruned: list[Path] = []
    for _, _, path in to_prune:
        if path.exists():
            shutil.rmtree(path)
            pruned.append(path)
    return pruned


def prune_old_step_payloads(*, step_dirs: Sequence[str | Path], keep_last: int) -> list[Path]:
    """Delete old per-step rollout payloads beyond the keep-last window.

    Retained summaries (`rft_step_summary.json`) remain in each step directory for
    lightweight auditability while bulky artifacts are pruned.
    """
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1 to preserve current step payload artifacts.")

    ordered_step_dirs = _coerce_step_dirs_in_order(step_dirs)
    if len(ordered_step_dirs) <= keep_last:
        return []

    to_prune = ordered_step_dirs[: len(ordered_step_dirs) - keep_last]
    pruned: list[Path] = []
    for step_dir in to_prune:
        for relative in (
            "collector_artifacts",
            "accepted_trajectories.parquet",
            "accepted_trajectories_eval.parquet",
        ):
            target = step_dir / relative
            if target.is_dir():
                shutil.rmtree(target)
                pruned.append(target)
            elif target.is_file():
                target.unlink()
                pruned.append(target)
    return pruned


def reset_step_artifacts(step_dir: str | Path) -> None:
    """Clear mutable per-step outputs so reruns do not mix stale artifacts."""
    resolved_step = Path(step_dir)
    for relative in (
        "collector_artifacts",
        "trainer_checkpoints",
        "accepted_trajectories.parquet",
        "accepted_trajectories_eval.parquet",
        "rft_step_summary.json",
    ):
        target = resolved_step / relative
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _coerce_step_dirs_in_order(step_dirs: Sequence[str | Path]) -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for item in step_dirs:
        path = Path(item)
        if path in seen:
            continue
        ordered.append(path)
        seen.add(path)
    return ordered


def _iter_global_step_dirs(root: Path) -> Sequence[tuple[int, int, Path]]:
    rows: list[tuple[int, int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _GLOBAL_STEP_PATTERN.match(path.name)
        if match is None:
            continue
        step_num = int(match.group(1))
        try:
            mtime_ns = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        rows.append((step_num, mtime_ns, path))
    return rows


def _print_dry_run_plan(
    config: RFTLoopConfig,
    *,
    collector_max_in_flight_tasks: int,
) -> None:
    preview_steps = min(config.rft_steps, 2)
    print(
        "# [dry-run] planned RFT loop",
        f"steps={config.rft_steps}",
        f"samples_per_task={config.samples_per_task}",
        f"task_batch_size={config.task_batch_size}",
        f"eval_split_fraction={config.eval_split_fraction}",
        f"eval_min_rows={config.eval_min_rows}",
        f"collector_max_in_flight_tasks={collector_max_in_flight_tasks}",
        "collector_max_turns_per_attempt="
        f"{config.collector_max_turns_per_attempt or 'default'}",
        f"checkpoint_keep_last={config.checkpoint_keep_last}",
    )
    if config.manage_vllm:
        initial_vllm = build_vllm_server_command(
            python_bin=config.python_bin,
            launch_module=config.vllm_launch_module,
            base_url=config.vllm_base_url,
            model_path=config.initial_model,
            served_model_name=config.vllm_served_model,
            extra_args=config.vllm_extra_args,
        )
        print(shlex.join(initial_vllm))

    for step_index in range(preview_steps):
        step_dir = config.output_dir / f"rft_step_{step_index:05d}"
        train_parquet_path = step_dir / "accepted_trajectories.parquet"
        eval_parquet_path = step_dir / "accepted_trajectories_eval.parquet"
        checkpoint_root = step_dir / "trainer_checkpoints"
        print(
            "# [dry-run] step="
            f"{step_index} collect selected trajectories -> train:{train_parquet_path} "
            f"eval:{eval_parquet_path}"
        )
        trainer_command = build_trainer_step_command(
            python_bin=config.python_bin,
            nnodes=config.nnodes,
            nproc_per_node=config.nproc_per_node,
            trainer_module=config.trainer_module,
            config_name=config.config_name,
            config_dir=config.config_dir,
            model_path=config.initial_model,
            train_parquet_path=train_parquet_path,
            val_parquet_path=eval_parquet_path,
            trainer_output_dir=checkpoint_root,
            train_batch_size=config.train_batch_size,
            sft_num_epoch_per_batch=config.sft_num_epoch_per_batch,
            trainer_overrides=config.trainer_overrides,
        )
        print(shlex.join(trainer_command))
        if config.manage_vllm:
            refreshed_vllm = build_vllm_server_command(
                python_bin=config.python_bin,
                launch_module=config.vllm_launch_module,
                base_url=config.vllm_base_url,
                model_path=str(checkpoint_root / "global_step_<n>" / "huggingface"),
                served_model_name=config.vllm_served_model,
                extra_args=config.vllm_extra_args,
            )
            print(shlex.join(refreshed_vllm))
    if config.rft_steps > preview_steps:
        print(f"# [dry-run] ... repeated for remaining {config.rft_steps - preview_steps} steps")


def _run_command(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
    )


def _load_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - train-only dependency
        raise RuntimeError(
            "RFT runtime loop requires transformers. Install training extras (`pip install -e \".[train]\"`)."
        ) from exc

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)


def _build_models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def _is_http_endpoint_ready(url: str, *, api_key: str | None = None) -> bool:
    headers = {}
    if api_key is not None and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=2.0) as response:
            return 200 <= int(response.status) < 300
    except HTTPError:
        return False
    except (URLError, TimeoutError, OSError):
        return False


def _resolve_vllm_api_key() -> str | None:
    for name in ("SMALL_SWE_VLLM_API_KEY", "OPENAI_API_KEY"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return value
    return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")


def _coerce_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def resolve_effective_train_batch_size(
    *,
    requested: int,
    selected_count: int,
    world_size: int,
    micro_batch_size_per_gpu: int = 1,
) -> int | None:
    """Clamp global train batch size to selected rows and DP/micro-batch divisibility."""
    if requested < 1:
        raise ValueError("requested train batch size must be >= 1.")
    if selected_count < 1:
        raise ValueError("selected_count must be >= 1.")
    if world_size < 1:
        raise ValueError("world_size must be >= 1.")
    if micro_batch_size_per_gpu < 1:
        raise ValueError("micro_batch_size_per_gpu must be >= 1.")

    max_global = min(requested, selected_count)
    divisor = world_size * micro_batch_size_per_gpu
    if max_global < divisor:
        return None

    divisible = (max_global // divisor) * divisor
    if divisible < 1:
        return None
    return divisible


def upsample_selected_rows_to_batch_multiple(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    global_batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """Repeat rows so dataset size is divisible by global train batch size."""
    if global_batch_size < 1:
        raise ValueError("global_batch_size must be >= 1.")

    rows: list[dict[str, Any]] = [dict(row) for row in selected_rows]
    if not rows:
        raise ValueError("selected_rows must be non-empty for upsampling.")

    remainder = len(rows) % global_batch_size
    if remainder == 0:
        return rows, 0

    needed = global_batch_size - remainder
    base_rows = list(rows)
    for index in range(needed):
        rows.append(dict(base_rows[index % len(base_rows)]))
    return rows, needed


def split_selected_rows_for_eval(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    eval_split_fraction: float,
    min_eval_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split selected rows into deterministic train/eval partitions."""
    if eval_split_fraction < 0.0 or eval_split_fraction >= 1.0:
        raise ValueError("eval_split_fraction must be in [0.0, 1.0).")
    if min_eval_rows < 0:
        raise ValueError("min_eval_rows must be >= 0.")

    rows = [dict(row) for row in selected_rows]
    total = len(rows)
    if total < 1:
        raise ValueError("selected_rows must be non-empty.")

    max_eval_rows = total - 1
    if max_eval_rows < 1 or eval_split_fraction <= 0.0:
        return rows, []

    eval_rows_target = int(total * eval_split_fraction)
    eval_rows_target = max(eval_rows_target, min_eval_rows)
    eval_rows_target = min(eval_rows_target, max_eval_rows)
    if eval_rows_target < 1:
        return rows, []

    ranked_indexes = sorted(
        range(total),
        key=lambda row_index: _stable_split_rank(rows[row_index], row_index=row_index),
    )
    eval_indexes = set(ranked_indexes[:eval_rows_target])

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if row_index in eval_indexes:
            eval_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows:
        train_rows.append(eval_rows.pop())
    return train_rows, eval_rows


def filter_selected_rows_by_token_length(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_sequence_length: int,
) -> tuple[list[dict[str, Any]], int]:
    """Drop selected rows whose multiturn transcript exceeds trainer max token length."""
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be >= 1.")

    rows = [dict(row) for row in selected_rows]
    if not rows:
        return [], 0

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError(
            "Tokenizer must expose apply_chat_template so selected rows can be length-filtered "
            "before writing multiturn parquet."
        )

    kept_rows: list[dict[str, Any]] = []
    dropped_count = 0
    for row_index, row in enumerate(rows):
        messages = build_multiturn_messages(row, row_index=row_index)
        try:
            token_count = _multiturn_token_count(messages=messages, tokenizer=tokenizer)
        except Exception as exc:
            task_id = str(row.get("task_id", "")).strip() or "<unknown>"
            raise RuntimeError(
                f"Failed to compute multiturn token count for selected_rows[{row_index}] "
                f"(task_id={task_id!r})."
            ) from exc

        if token_count > max_sequence_length:
            dropped_count += 1
            continue
        kept_rows.append(row)

    return kept_rows, dropped_count


def _multiturn_token_count(*, messages: Sequence[Mapping[str, Any]], tokenizer: Any) -> int:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("tokenizer must define callable apply_chat_template for multiturn token counting.")

    try:
        payload = apply_chat_template(
            list(messages),
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
        )
    except TypeError:
        payload = apply_chat_template(
            list(messages),
            add_generation_prompt=False,
            tokenize=True,
        )

    input_ids = _extract_chat_template_input_ids(payload)
    return len(input_ids)


def _extract_chat_template_input_ids(payload: Any) -> list[int]:
    if isinstance(payload, Mapping):
        raw_input_ids = payload.get("input_ids")
    else:
        raw_input_ids = payload

    if hasattr(raw_input_ids, "tolist"):
        raw_input_ids = raw_input_ids.tolist()

    if isinstance(raw_input_ids, Sequence) and not isinstance(raw_input_ids, (str, bytes)):
        if raw_input_ids and isinstance(raw_input_ids[0], Sequence) and not isinstance(
            raw_input_ids[0],
            (str, bytes),
        ):
            raw_input_ids = raw_input_ids[0]
        return [int(token_id) for token_id in raw_input_ids]

    raise ValueError("chat template tokenization payload did not provide sequence `input_ids`.")


def _stable_split_rank(row: Mapping[str, Any], *, row_index: int) -> str:
    task_id = str(row.get("task_id", "")).strip()
    step_index = row.get("step_index", "")
    attempt_index = row.get("attempt_index", "")
    turn_index = row.get("turn_index", "")
    token = f"{task_id}|{step_index}|{attempt_index}|{turn_index}|{row_index}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def resolve_micro_batch_size_per_gpu(
    *,
    config_dir: Path,
    config_name: str,
    trainer_overrides: Sequence[str],
) -> int:
    """Resolve micro-batch size from config file with optional override precedence."""
    resolved = _load_default_micro_batch_size_per_gpu(config_dir=config_dir, config_name=config_name)
    for override in trainer_overrides:
        parsed = _parse_positive_int_override(override, key=_MICRO_BATCH_SIZE_KEY)
        if parsed is not None:
            resolved = parsed
    return resolved


def resolve_data_max_length(
    *,
    config_dir: Path,
    config_name: str,
    trainer_overrides: Sequence[str],
) -> int:
    """Resolve trainer data.max_length with override precedence."""
    resolved = _load_default_data_max_length(config_dir=config_dir, config_name=config_name)
    for override in trainer_overrides:
        parsed = _parse_positive_int_override(override, key=_DATA_MAX_LENGTH_KEY)
        if parsed is not None:
            resolved = parsed
    return resolved


def _load_default_micro_batch_size_per_gpu(*, config_dir: Path, config_name: str) -> int:
    config_path = config_dir / f"{config_name}.yaml"
    if not config_path.is_file():
        return 1

    pattern = re.compile(r"^\s*micro_batch_size_per_gpu\s*:\s*([0-9]+)\s*(?:#.*)?$")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        value = int(match.group(1))
        if value >= 1:
            return value
        break
    return 1


def _load_default_data_max_length(*, config_dir: Path, config_name: str) -> int:
    config_path = config_dir / f"{config_name}.yaml"
    if not config_path.is_file():
        return 1024

    pattern = re.compile(r"^\s*max_length\s*:\s*([0-9]+)\s*(?:#.*)?$")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        value = int(match.group(1))
        if value >= 1:
            return value
        break
    return 1024


def _parse_positive_int_override(override: str, *, key: str) -> int | None:
    normalized = override.strip()
    while normalized.startswith("+"):
        normalized = normalized[1:]
    prefix = f"{key}="
    if not normalized.startswith(prefix):
        return None

    value_raw = normalized[len(prefix) :].strip()
    try:
        value = int(value_raw)
    except ValueError as exc:
        raise ValueError(f"{key} override must be an integer >= 1 (got {value_raw!r}).") from exc
    if value < 1:
        raise ValueError(f"{key} override must be >= 1 (got {value}).")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _parse_args(argv: Sequence[str] | None = None) -> RFTLoopConfig:
    parser = argparse.ArgumentParser(
        description="Run the on-policy RFT collector/trainer loop with checkpoint-driven vLLM restarts.",
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--config-name", default="rft_swe")
    parser.add_argument(
        "--trainer-module",
        default="verl_integration.fsdp_sft_trainer_entry",
        help=(
            "trainer module (project wrapper around verl entrypoint; "
            f"upstream source: {_VERL_SFT_TRAINER_DOC})"
        ),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--rft-steps", type=int, required=True)
    parser.add_argument("--samples-per-task", type=int, required=True)
    parser.add_argument("--task-batch-size", type=int, required=True)
    parser.add_argument(
        "--collector-max-in-flight-tasks",
        type=int,
        default=None,
        help=(
            "optional override for collector task-dispatch concurrency; "
            "defaults to centralized runtime policy (clamped by task-batch-size)."
        ),
    )
    parser.add_argument(
        "--collector-max-turns-per-attempt",
        type=int,
        default=None,
        help=(
            "optional override for collector max turns per trajectory attempt; "
            "defaults to centralized on-policy runtime config."
        ),
    )
    parser.add_argument("--sft-num-epoch-per-batch", type=int, required=True)
    parser.add_argument("--checkpoint-keep-last", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument(
        "--eval-split-fraction",
        type=float,
        default=0.1,
        help=(
            "fraction of selected rows to reserve for val/eval parquet in each step; "
            "must satisfy 0.0 <= value < 1.0."
        ),
    )
    parser.add_argument(
        "--eval-min-rows",
        type=int,
        default=1,
        help=(
            "minimum number of held-out eval rows when split fraction is positive and "
            "at least two selected rows are available."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-config-name", default="on_policy_swe_smith")
    parser.add_argument("--turn-generator-mode", default="default")
    parser.add_argument("--initial-model", required=True)
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--vllm-served-model", required=True)
    parser.add_argument(
        "--vllm-launch-module",
        default="trainer.vllm_api_server_entry",
        help=(
            "vLLM OpenAI server module "
            f"(see {_VLLM_OPENAI_SERVER_DOC}, source: {_VLLM_OPENAI_SERVER_SOURCE})"
        ),
    )
    parser.add_argument("--vllm-ready-timeout-sec", type=int, default=180)
    parser.add_argument("--vllm-stop-timeout-sec", type=int, default=30)
    parser.add_argument("--vllm-extra-args", default="")
    parser.add_argument("--skip-vllm-management", action="store_true")
    parser.add_argument("--trainer-override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    if args.rft_steps < 1:
        raise ValueError("--rft-steps must be >= 1.")
    if args.samples_per_task < 1:
        raise ValueError("--samples-per-task must be >= 1.")
    if args.task_batch_size < 1:
        raise ValueError("--task-batch-size must be >= 1.")
    if args.collector_max_in_flight_tasks is not None and args.collector_max_in_flight_tasks < 1:
        raise ValueError("--collector-max-in-flight-tasks must be >= 1 when provided.")
    if (
        args.collector_max_turns_per_attempt is not None
        and args.collector_max_turns_per_attempt < 1
    ):
        raise ValueError("--collector-max-turns-per-attempt must be >= 1 when provided.")
    if args.sft_num_epoch_per_batch < 1:
        raise ValueError("--sft-num-epoch-per-batch must be >= 1.")
    if args.checkpoint_keep_last < 1:
        raise ValueError("--checkpoint-keep-last must be >= 1.")
    if args.train_batch_size < 1:
        raise ValueError("--train-batch-size must be >= 1.")
    if args.eval_split_fraction < 0.0 or args.eval_split_fraction >= 1.0:
        raise ValueError("--eval-split-fraction must satisfy 0.0 <= value < 1.0.")
    if args.eval_min_rows < 0:
        raise ValueError("--eval-min-rows must be >= 0.")
    if args.nnodes < 1:
        raise ValueError("--nnodes must be >= 1.")
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be >= 1.")

    return RFTLoopConfig(
        project_root=Path(args.project_root).resolve(),
        config_dir=Path(args.config_dir).resolve(),
        config_name=str(args.config_name),
        trainer_module=str(args.trainer_module),
        python_bin=str(args.python_bin),
        nnodes=int(args.nnodes),
        nproc_per_node=int(args.nproc_per_node),
        rft_steps=int(args.rft_steps),
        samples_per_task=int(args.samples_per_task),
        task_batch_size=int(args.task_batch_size),
        sft_num_epoch_per_batch=int(args.sft_num_epoch_per_batch),
        checkpoint_keep_last=int(args.checkpoint_keep_last),
        train_batch_size=int(args.train_batch_size),
        output_dir=Path(args.output_dir).resolve(),
        data_config_name=str(args.data_config_name),
        turn_generator_mode=str(args.turn_generator_mode),
        initial_model=str(args.initial_model),
        vllm_base_url=str(args.vllm_base_url),
        vllm_served_model=str(args.vllm_served_model),
        manage_vllm=not bool(args.skip_vllm_management),
        vllm_launch_module=str(args.vllm_launch_module),
        vllm_ready_timeout_sec=int(args.vllm_ready_timeout_sec),
        vllm_stop_timeout_sec=int(args.vllm_stop_timeout_sec),
        vllm_extra_args=tuple(shlex.split(str(args.vllm_extra_args))),
        trainer_overrides=tuple(str(item) for item in args.trainer_override),
        dry_run=bool(args.dry_run),
        collector_max_in_flight_tasks=(
            int(args.collector_max_in_flight_tasks)
            if args.collector_max_in_flight_tasks is not None
            else None
        ),
        collector_max_turns_per_attempt=(
            int(args.collector_max_turns_per_attempt)
            if args.collector_max_turns_per_attempt is not None
            else None
        ),
        eval_split_fraction=float(args.eval_split_fraction),
        eval_min_rows=int(args.eval_min_rows),
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    run_rft_runtime_loop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
