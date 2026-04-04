"""End-to-end RFT loop orchestration for live rollout -> train -> checkpoint refresh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
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

from config import resolve_rft_collector_max_in_flight_default, resolve_rft_handoff_settings
from trainer.rft_multiturn_dataset import (
    build_multiturn_messages,
    write_selected_rows_to_multiturn_parquet,
)
from trainer.rft_runtime import OnPolicyRFTRuntimeRequest, collect_onpolicy_rft_runtime_batch

_GLOBAL_STEP_PATTERN = re.compile(r"^global_step_(\d+)$")
_STEP_DIR_PATTERN = re.compile(r"^rft_step_(\d+)$")
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
# Keep periodic saves effectively disabled for inner SFT loops while still
# allowing the trainer's end-of-run checkpoint export to materialize.
_INNER_SFT_CHECKPOINT_DISABLED_SAVE_FREQ = 2_147_483_647
_MAX_MODEL_LEN_KEY = "max_model_len"
_DEFAULT_LORA_RANK = 16
_DEFAULT_LORA_ALPHA = 32
_DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_DEFAULT_RFT_WANDB_PROJECT = "small-swe-rft"
_DEFAULT_RFT_WANDB_GROUP = "small-swe-rft"
_TOOL_RESPONSE_PREFIX = "<tool_response>"
_TRAIN_LOSS_PATTERN = re.compile(r"step:(\d+)\s*-\s*train/loss:([0-9eE+\-.]+)")
_VAL_LOSS_PATTERN = re.compile(r"step:(\d+)\s*-\s*val/loss:([0-9eE+\-.]+)")
_FORMAT_RFT_STAGE_NAME = "format_rft"
_POSITIVE_RFT_STAGE_NAME = "positive_rft"
_DEFAULT_PROCESS_GROUP_CLEANUP_TIMEOUT_SEC = 5.0
_DEFAULT_DIAGNOSTIC_COMMAND_TIMEOUT_SEC = 5.0
_MODEL_ARTIFACT_FILE_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}
_RFT_RUNTIME_LOOP_MANIFEST_FILE_NAME = "rft_runtime_loop_manifest.json"
_RFT_LATEST_COMMITTED_CHECKPOINT_FILE_NAME = "rft_latest_committed_checkpoint.json"


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
    stage_name: str = _FORMAT_RFT_STAGE_NAME


@dataclass(frozen=True)
class LoraMergeSpec:
    rank: int
    alpha: int
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class LatestCommittedCheckpoint:
    stage: str
    committed_step_index: int
    latest_hf_checkpoint: Path
    latest_vllm_checkpoint: Path
    resume_model_path: Path
    selection_contract: Mapping[str, Any]
    correctness_contract: str
    committed_utc: str


def resolve_rft_stage_name(stage_name: str) -> str:
    normalized = stage_name.strip().lower()
    if normalized in {"", "default", "format", _FORMAT_RFT_STAGE_NAME}:
        return _FORMAT_RFT_STAGE_NAME
    if normalized in {"positive", _POSITIVE_RFT_STAGE_NAME}:
        return _POSITIVE_RFT_STAGE_NAME
    raise ValueError("stage_name must be one of: format_rft, positive_rft.")


def resolve_rft_stage_handoff_overrides(stage_name: str) -> dict[str, Any]:
    resolved_stage_name = resolve_rft_stage_name(stage_name)
    if resolved_stage_name == _POSITIVE_RFT_STAGE_NAME:
        return {
            "selection": {
                "require_terminal": False,
                "require_format_valid": False,
                "require_resolved": True,
                "reject_on_invalid_final_submit": False,
            }
        }
    return {}


def resolve_rft_stage_verify_submissions(stage_name: str) -> bool:
    resolved_stage_name = resolve_rft_stage_name(stage_name)
    return resolved_stage_name == _POSITIVE_RFT_STAGE_NAME


class VLLMServerController:
    """Manage an OpenAI-compatible vLLM server process for the RFT loop.

    Grounding: vLLM serves OpenAI-compatible chat/completions via
    `python -m vllm.entrypoints.openai.api_server` as documented in:
    https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

    def __init__(self, *, config: RFTLoopConfig, log_path: Path) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._process_group_id: int | None = None
        self._log_path = log_path
        self._models_url = _build_models_url(config.vllm_base_url)
        self._api_key = _resolve_vllm_api_key()

    def start(self, *, model_path: str) -> None:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("vLLM server is already running; stop it before starting a new model.")
        if _is_http_endpoint_ready(self._models_url, api_key=self._api_key):
            raise RuntimeError(
                "Managed vLLM launch target already has a ready endpoint at "
                f"{self._models_url}. Refusing to start a new server on an occupied address."
            )

        command = build_vllm_server_command(
            python_bin=self._config.python_bin,
            launch_module=self._config.vllm_launch_module,
            base_url=self._config.vllm_base_url,
            model_path=model_path,
            served_model_name=self._config.vllm_served_model,
            extra_args=self._config.vllm_extra_args,
        )

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        _append_vllm_debug_snapshot(
            log_path=self._log_path,
            label=f"pre-launch GPU snapshot for model={model_path}",
        )
        with self._log_path.open("a", encoding="utf-8") as log_handle:
            self._process = subprocess.Popen(
                command,
                cwd=self._config.project_root,
                stdout=log_handle,
                stderr=log_handle,
                text=True,
                start_new_session=True,
            )
        self._process_group_id = self._process.pid
        try:
            self._wait_until_ready()
        except Exception:
            _cleanup_process_group(
                self._process_group_id,
                timeout_sec=self._config.vllm_stop_timeout_sec,
            )
            if self._process is not None:
                try:
                    self._process.wait(timeout=self._config.vllm_stop_timeout_sec)
                except subprocess.TimeoutExpired:
                    pass
            self._process = None
            self._process_group_id = None
            raise

    def stop(self) -> None:
        process = self._process
        self._process = None
        process_group_id = self._process_group_id
        self._process_group_id = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self._config.vllm_stop_timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=self._config.vllm_stop_timeout_sec)
        _cleanup_process_group(
            process_group_id,
            timeout_sec=self._config.vllm_stop_timeout_sec,
        )

    def _wait_until_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self._config.vllm_ready_timeout_sec
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                _append_vllm_debug_snapshot(
                    log_path=self._log_path,
                    label="startup failure GPU snapshot",
                )
                raise RuntimeError(
                    f"vLLM server exited early with code {self._process.returncode}. "
                    f"Inspect logs at {self._log_path}."
                )
            if _is_http_endpoint_ready(
                self._models_url,
                api_key=self._api_key,
                expected_model_name=self._config.vllm_served_model,
            ):
                return
            time.sleep(1.0)
        observed_models = _query_http_endpoint_models(self._models_url, api_key=self._api_key)
        observed_models_hint = ""
        if observed_models:
            observed_models_hint = (
                " Observed served models: "
                + ", ".join(repr(model_name) for model_name in observed_models)
                + f"; expected {self._config.vllm_served_model!r}."
            )
        raise RuntimeError(
            "Timed out waiting for vLLM readiness at "
            f"{self._models_url}.{observed_models_hint} Inspect logs at {self._log_path}."
        )


def _process_group_exists(process_group_id: int | None) -> bool:
    if process_group_id is None or process_group_id <= 0:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(process_group_id: int | None, sig: signal.Signals) -> None:
    if process_group_id is None or process_group_id <= 0:
        return
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return


def _wait_for_process_group_exit(process_group_id: int | None, *, timeout_sec: float) -> bool:
    if process_group_id is None or process_group_id <= 0:
        return True
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while True:
        if not _process_group_exists(process_group_id):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _cleanup_process_group(process_group_id: int | None, *, timeout_sec: float) -> None:
    if process_group_id is None or process_group_id <= 0:
        return
    if _wait_for_process_group_exit(process_group_id, timeout_sec=0.0):
        return
    _signal_process_group(process_group_id, signal.SIGTERM)
    if _wait_for_process_group_exit(process_group_id, timeout_sec=timeout_sec):
        return
    _signal_process_group(process_group_id, signal.SIGKILL)
    _wait_for_process_group_exit(process_group_id, timeout_sec=timeout_sec)


def _run_diagnostic_command(command: Sequence[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_DIAGNOSTIC_COMMAND_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"timeout after {exc.timeout}s"
    except OSError as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout, completed.stderr


def _append_vllm_debug_snapshot(*, log_path: Path, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"[small-swe] {label} @ {_utc_now()}"]
    gpu_query = (
        "index,name,memory.used,memory.free,memory.total"
    )
    compute_query = "gpu_uuid,pid,process_name,used_gpu_memory"
    diagnostic_commands: list[tuple[str, list[str]]] = [
        (
            "gpu memory",
            [
                "nvidia-smi",
                f"--query-gpu={gpu_query}",
                "--format=csv,noheader,nounits",
            ],
        ),
        (
            "compute apps",
            [
                "nvidia-smi",
                f"--query-compute-apps={compute_query}",
                "--format=csv,noheader,nounits",
            ],
        ),
    ]

    compute_app_output = ""
    for title, command in diagnostic_commands:
        if shutil.which(command[0]) is None:
            lines.append(f"[small-swe] {title}: command not found: {command[0]}")
            continue
        return_code, stdout, stderr = _run_diagnostic_command(command)
        stdout = stdout.strip()
        stderr = stderr.strip()
        lines.append(f"[small-swe] {title}: rc={return_code}")
        if stdout:
            lines.append(stdout)
        if stderr:
            lines.append(f"[small-swe] {title} stderr: {stderr}")
        if title == "compute apps":
            compute_app_output = stdout

    gpu_pids: list[str] = []
    for row in compute_app_output.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 2:
            continue
        pid = parts[1]
        if pid.isdigit():
            gpu_pids.append(pid)
    if gpu_pids and shutil.which("ps") is not None:
        return_code, stdout, stderr = _run_diagnostic_command(
            [
                "ps",
                "-o",
                "pid=,ppid=,pgid=,user=,stat=,etime=,cmd=",
                "-p",
                ",".join(sorted(set(gpu_pids))),
            ]
        )
        lines.append(f"[small-swe] ps for gpu pids: rc={return_code}")
        stdout = stdout.strip()
        stderr = stderr.strip()
        if stdout:
            lines.append(stdout)
        if stderr:
            lines.append(f"[small-swe] ps stderr: {stderr}")

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def _runtime_loop_manifest_path(output_dir: Path) -> Path:
    return output_dir / _RFT_RUNTIME_LOOP_MANIFEST_FILE_NAME


def _latest_committed_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / _RFT_LATEST_COMMITTED_CHECKPOINT_FILE_NAME


def _selection_contract_for_rft() -> dict[str, Any]:
    return {
        "mode": "format_first_rft",
        "require_terminal": True,
        "require_format_valid": True,
    }


def resolve_rft_stage_selection_contract(stage_name: str) -> dict[str, Any]:
    resolved_stage_name = resolve_rft_stage_name(stage_name)
    if resolved_stage_name == _POSITIVE_RFT_STAGE_NAME:
        return {
            "mode": "positive_rft",
            "require_terminal": False,
            "require_format_valid": False,
            "require_resolved": True,
            "reject_on_invalid_final_submit": False,
        }
    return _selection_contract_for_rft()


def resolve_rft_stage_correctness_contract(stage_name: str) -> str:
    resolved_stage_name = resolve_rft_stage_name(stage_name)
    if resolved_stage_name == _POSITIVE_RFT_STAGE_NAME:
        return "verifier"
    return "heuristic"


def _load_existing_step_summaries(output_dir: Path, *, committed_step_index: int) -> list[dict[str, Any]]:
    if committed_step_index < 0:
        return []

    summaries: list[dict[str, Any]] = []
    for step_index in range(committed_step_index + 1):
        summary_path = output_dir / f"rft_step_{step_index:05d}" / "rft_step_summary.json"
        payload = _load_json_mapping(summary_path)
        if payload is None:
            continue
        summaries.append(payload)
    return summaries


def _load_existing_runtime_manifest(
    *,
    output_dir: Path,
    default_config: Mapping[str, Any],
    committed_step_index: int,
) -> dict[str, Any]:
    if committed_step_index < 0:
        return {
            "generated_utc": _utc_now(),
            "config": dict(default_config),
            "steps": [],
        }

    manifest_path = _runtime_loop_manifest_path(output_dir)
    payload = _load_json_mapping(manifest_path)
    if payload is None:
        return {
            "generated_utc": _utc_now(),
            "config": dict(default_config),
            "steps": _load_existing_step_summaries(
                output_dir,
                committed_step_index=committed_step_index,
            ),
        }

    steps = payload.get("steps")
    if not isinstance(steps, list):
        steps = []
    if committed_step_index >= 0:
        steps = steps[: committed_step_index + 1]

    normalized_steps: list[dict[str, Any]] = []
    seen_step_indexes: set[int] = set()
    for raw_step in steps:
        if not isinstance(raw_step, Mapping):
            continue
        parsed_step = dict(raw_step)
        step_index = parsed_step.get("step_index")
        if not isinstance(step_index, int):
            continue
        if step_index < 0 or step_index > committed_step_index or step_index in seen_step_indexes:
            continue
        seen_step_indexes.add(step_index)
        normalized_steps.append(parsed_step)

    for step_index in range(committed_step_index + 1):
        if step_index in seen_step_indexes:
            continue
        step_summary_path = output_dir / f"rft_step_{step_index:05d}" / "rft_step_summary.json"
        recovered = _load_json_mapping(step_summary_path)
        if recovered is None:
            continue
        recovered_step_index = recovered.get("step_index")
        if recovered_step_index != step_index:
            continue
        normalized_steps.append(recovered)

    normalized_steps.sort(key=lambda item: int(item.get("step_index", 0)))

    payload["config"] = dict(default_config)
    payload["steps"] = normalized_steps
    return dict(payload)


def _load_latest_committed_checkpoint(output_dir: Path) -> LatestCommittedCheckpoint | None:
    payload = _load_json_mapping(_latest_committed_checkpoint_path(output_dir))
    if payload is None:
        return None

    stage = str(payload.get("stage", "")).strip() or "format_rft"
    committed_step_index = _require_non_negative_int(
        payload.get("committed_step_index"),
        label="committed_step_index",
    )
    latest_hf_checkpoint = _require_existing_path(
        payload.get("latest_hf_checkpoint"),
        label="latest_hf_checkpoint",
    )
    latest_vllm_checkpoint = _require_existing_path(
        payload.get("latest_vllm_checkpoint"),
        label="latest_vllm_checkpoint",
    )
    resume_model_path = _require_existing_path(
        payload.get("resume_model_path"),
        label="resume_model_path",
    )
    selection_contract_raw = payload.get("selection_contract")
    selection_contract = (
        dict(selection_contract_raw)
        if isinstance(selection_contract_raw, Mapping)
        else resolve_rft_stage_selection_contract(stage)
    )
    correctness_contract = (
        str(payload.get("correctness_contract", "")).strip()
        or resolve_rft_stage_correctness_contract(stage)
    )
    committed_utc = str(payload.get("committed_utc", "")).strip() or _utc_now()
    return LatestCommittedCheckpoint(
        stage=stage,
        committed_step_index=committed_step_index,
        latest_hf_checkpoint=latest_hf_checkpoint,
        latest_vllm_checkpoint=latest_vllm_checkpoint,
        resume_model_path=resume_model_path,
        selection_contract=selection_contract,
        correctness_contract=correctness_contract,
        committed_utc=committed_utc,
    )


def _build_latest_committed_checkpoint_payload(
    *,
    stage_name: str,
    step_index: int,
    latest_hf_checkpoint: Path,
    latest_vllm_checkpoint: Path,
) -> dict[str, Any]:
    resolved_stage_name = resolve_rft_stage_name(stage_name)
    return {
        "stage": resolved_stage_name,
        "committed_step_index": int(step_index),
        "latest_hf_checkpoint": str(latest_hf_checkpoint),
        "latest_vllm_checkpoint": str(latest_vllm_checkpoint),
        "resume_model_path": str(latest_vllm_checkpoint),
        "selection_contract": resolve_rft_stage_selection_contract(resolved_stage_name),
        "correctness_contract": resolve_rft_stage_correctness_contract(resolved_stage_name),
        "committed_utc": _utc_now(),
    }


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Expected mapping payload in {path}.")
    return dict(payload)


def _require_existing_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Latest committed checkpoint is missing `{label}`.")
    path = Path(value).resolve()
    if not path.exists():
        raise RuntimeError(
            "Latest committed checkpoint is incomplete: "
            f"`{label}` does not exist at {path}."
        )
    return path


def _require_non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"Latest committed checkpoint `{label}` must be a non-negative integer.")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise RuntimeError(
                f"Latest committed checkpoint `{label}` must be a non-negative integer."
            ) from exc
        if parsed >= 0:
            return parsed
    raise RuntimeError(f"Latest committed checkpoint `{label}` must be a non-negative integer.")


def _discover_existing_step_dirs(
    output_dir: Path,
    *,
    max_step_index: int | None = None,
) -> list[Path]:
    if not output_dir.is_dir():
        return []

    discovered: list[tuple[int, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        match = _STEP_DIR_PATTERN.match(path.name)
        if match is None:
            continue
        step_index = int(match.group(1))
        if max_step_index is not None and step_index > max_step_index:
            continue
        discovered.append((step_index, path))
    discovered.sort(key=lambda item: item[0])
    return [path for _, path in discovered]


def _append_unique_path(paths: list[Path], path: Path) -> None:
    if path not in paths:
        paths.append(path)


def run_rft_runtime_loop(config: RFTLoopConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_stage_name = resolve_rft_stage_name(config.stage_name)
    stage_handoff_overrides = resolve_rft_stage_handoff_overrides(resolved_stage_name)
    stage_verify_submissions = resolve_rft_stage_verify_submissions(resolved_stage_name)
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
    default_manifest_config = {
        "rft_steps": config.rft_steps,
        "samples_per_task": config.samples_per_task,
        "task_batch_size": config.task_batch_size,
        "stage_name": resolved_stage_name,
        "verify_submissions": stage_verify_submissions,
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
    }
    latest_committed_checkpoint = _load_latest_committed_checkpoint(config.output_dir)
    committed_step_index = (
        latest_committed_checkpoint.committed_step_index
        if latest_committed_checkpoint is not None
        else -1
    )
    runtime_manifest = _load_existing_runtime_manifest(
        output_dir=config.output_dir,
        default_config=default_manifest_config,
        committed_step_index=committed_step_index,
    )
    runtime_manifest["latest_committed_checkpoint_path"] = str(
        _latest_committed_checkpoint_path(config.output_dir)
    )
    runtime_manifest["resume_mode"] = (
        "latest_committed_outer_loop_checkpoint"
        if latest_committed_checkpoint is not None
        else "fresh_run"
    )
    runtime_manifest["resume_start_step_index"] = committed_step_index + 1
    runtime_manifest["latest_committed_step_index"] = (
        latest_committed_checkpoint.committed_step_index
        if latest_committed_checkpoint is not None
        else None
    )
    if latest_committed_checkpoint is not None:
        resumed_stage_name = resolve_rft_stage_name(latest_committed_checkpoint.stage)
        if resumed_stage_name != resolved_stage_name:
            raise ValueError(
                "Refusing to resume a committed run with a different stage_name: "
                f"latest checkpoint stage={resumed_stage_name!r}, "
                f"requested stage_name={resolved_stage_name!r}. "
                "Use a fresh output directory for a new stage."
            )

    if config.dry_run:
        _print_dry_run_plan(
            config=config,
            collector_max_in_flight_tasks=collector_max_in_flight_tasks,
        )
        return

    tokenizer = _load_tokenizer(config.initial_model)
    current_model_path = (
        str(latest_committed_checkpoint.resume_model_path)
        if latest_committed_checkpoint is not None
        else config.initial_model
    )
    start_step_index = committed_step_index + 1
    if start_step_index >= config.rft_steps:
        runtime_manifest["final_model_path"] = current_model_path
        runtime_manifest["completed_utc"] = _utc_now()
        _write_json(_runtime_loop_manifest_path(config.output_dir), runtime_manifest)
        return
    vllm_controller = VLLMServerController(config=config, log_path=vllm_logs)
    existing_step_dirs = _discover_existing_step_dirs(
        config.output_dir,
        max_step_index=committed_step_index,
    )
    run_step_dirs: list[Path] = list(existing_step_dirs)
    checkpoint_step_dirs: list[Path] = [
        step_dir for step_dir in existing_step_dirs if (step_dir / "trainer_checkpoints").is_dir()
    ]
    wandb_run = _init_rft_runtime_loop_wandb_run(config=config)

    try:
        if config.manage_vllm:
            vllm_controller.start(model_path=current_model_path)

        for step_index in range(start_step_index, config.rft_steps):
            step_start = time.monotonic()
            step_dir = config.output_dir / f"rft_step_{step_index:05d}"
            collector_dir = step_dir / "collector_artifacts"
            collector_train_dir = collector_dir / "train"
            collector_eval_dir = collector_dir / "eval"
            train_parquet_path = step_dir / "accepted_trajectories.parquet"
            eval_parquet_path = step_dir / "accepted_trajectories_eval.parquet"
            trainer_checkpoint_root = step_dir / "trainer_checkpoints"
            reset_step_artifacts(step_dir)
            step_dir.mkdir(parents=True, exist_ok=True)
            _append_unique_path(run_step_dirs, step_dir)

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
                handoff_overrides=stage_handoff_overrides,
                output_dir=str(collector_train_dir),
                task_partition=(
                    "train" if config.eval_split_fraction > 0.0 else "all"
                ),
                task_eval_split_fraction=config.eval_split_fraction,
                task_eval_min_rows=config.eval_min_rows,
                verify_submissions=stage_verify_submissions,
                stage_name=resolved_stage_name,
            )
            collect_train_start = time.monotonic()
            train_handoff = collect_onpolicy_rft_runtime_batch(
                request=request,
                tokenizer=tokenizer,
            )
            train_collect_duration_sec = time.monotonic() - collect_train_start
            train_selected_rows = _coerce_rows(train_handoff.get("selected_rows"))
            train_rejected_rows = _coerce_rows(train_handoff.get("rejected_rows"))
            selected_count_raw = len(train_selected_rows)
            avg_generation_length_raw = compute_average_generation_length(
                selected_rows=train_selected_rows,
                tokenizer=tokenizer,
            )
            effective_keep_budget = min(
                trainer_data_max_length,
                resolve_rft_handoff_settings(overrides=stage_handoff_overrides).max_sequence_length,
            )
            (
                train_selected_rows,
                selected_rows_over_max_length_dropped,
            ) = filter_selected_rows_by_token_length(
                selected_rows=train_selected_rows,
                tokenizer=tokenizer,
                max_sequence_length=effective_keep_budget,
            )
            selected_count_after_max_length_filter = len(train_selected_rows)
            avg_generation_length = compute_average_generation_length(
                selected_rows=train_selected_rows,
                tokenizer=tokenizer,
            )
            selected_count_for_train_raw = 0
            selected_count_for_train = 0
            eval_selected_count_raw = 0
            eval_selected_count_after_length_filter = 0
            selected_count_for_eval_raw = 0
            selected_count_for_eval = 0
            selected_rows_upsampled = 0
            selected_rows_eval_upsampled = 0
            eval_selected_rows_over_max_length_dropped = 0
            eval_split_fallback_to_train = False
            resolved_val_parquet_path = train_parquet_path
            effective_train_batch_size: int | None = None
            effective_eval_batch_size: int | None = None
            trainer_command: list[str] | None = None
            latest_hf_checkpoint: Path | None = None
            latest_vllm_checkpoint: Path | None = None
            pruned_global_step_checkpoints: list[Path] = []
            trainer_duration_sec: float | None = None
            trainer_metrics: dict[str, float | int] = {}
            trainer_skipped = False
            skip_reason: str | None = None
            pending_latest_committed_checkpoint_payload: dict[str, Any] | None = None
            pending_latest_committed_step_index = (
                latest_committed_checkpoint.committed_step_index
                if latest_committed_checkpoint is not None
                else None
            )
            eval_collect_duration_sec = 0.0
            eval_selected_rows: list[dict[str, Any]] = []
            eval_rejected_rows: list[dict[str, Any]] = []

            if config.eval_split_fraction > 0.0:
                eval_request = OnPolicyRFTRuntimeRequest(
                    data_config_name=config.data_config_name,
                    turn_generator_mode=config.turn_generator_mode,
                    total_steps=1,
                    start_step_index=step_index,
                    runtime_overrides=runtime_overrides,
                    handoff_overrides=stage_handoff_overrides,
                    output_dir=str(collector_eval_dir),
                    task_partition="eval",
                    task_eval_split_fraction=config.eval_split_fraction,
                    task_eval_min_rows=config.eval_min_rows,
                    verify_submissions=stage_verify_submissions,
                    stage_name=resolved_stage_name,
                )
                collect_eval_start = time.monotonic()
                eval_handoff = collect_onpolicy_rft_runtime_batch(
                    request=eval_request,
                    tokenizer=tokenizer,
                )
                eval_collect_duration_sec = time.monotonic() - collect_eval_start
                eval_selected_rows = _coerce_rows(eval_handoff.get("selected_rows"))
                eval_rejected_rows = _coerce_rows(eval_handoff.get("rejected_rows"))
                eval_selected_count_raw = len(eval_selected_rows)
                (
                    eval_selected_rows,
                    eval_selected_rows_over_max_length_dropped,
                ) = filter_selected_rows_by_token_length(
                    selected_rows=eval_selected_rows,
                    tokenizer=tokenizer,
                    max_sequence_length=effective_keep_budget,
                )
                eval_selected_count_after_length_filter = len(eval_selected_rows)

            if selected_count_after_max_length_filter < 1:
                trainer_skipped = True
                skip_reason = "no_selected_rows_after_length_filter"
            else:
                selected_rows_for_train = list(train_selected_rows)
                selected_rows_for_eval = list(eval_selected_rows)
                selected_count_for_train_raw = len(selected_rows_for_train)
                selected_count_for_eval_raw = len(selected_rows_for_eval)
                selected_count_for_eval = selected_count_for_eval_raw
                if selected_count_for_train_raw < 1:
                    trainer_skipped = True
                    skip_reason = "empty_train_split"

                world_size = config.nnodes * config.nproc_per_node
                if not trainer_skipped:
                    effective_eval_batch_size = world_size * micro_batch_size_per_gpu
                    selected_count_for_train = write_selected_rows_to_multiturn_parquet(
                        selected_rows_for_train,
                        train_parquet_path,
                    )
                    if selected_rows_for_eval:
                        selected_rows_for_eval, selected_rows_eval_upsampled = (
                            upsample_selected_rows_to_batch_multiple(
                                selected_rows_for_eval,
                                global_batch_size=effective_eval_batch_size,
                            )
                        )
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
                        raw_trainer_metrics = _run_command(trainer_command, cwd=config.project_root)
                        if isinstance(raw_trainer_metrics, Mapping):
                            trainer_metrics = {
                                key: value
                                for key, value in dict(raw_trainer_metrics).items()
                                if isinstance(value, (int, float))
                            }
                        else:
                            trainer_metrics = {}
                        trainer_duration_sec = time.monotonic() - trainer_start

                        try:
                            latest_hf_checkpoint = resolve_latest_hf_checkpoint(
                                trainer_checkpoint_root
                            )
                        except FileNotFoundError as exc:
                            raise RuntimeError(
                                "Trainer step completed but produced no checkpoint under "
                                f"{trainer_checkpoint_root}. Outer RFT requires one checkpoint "
                                "per non-skipped step."
                            ) from exc
                        latest_vllm_checkpoint = materialize_vllm_compatible_checkpoint(
                            checkpoint_dir=latest_hf_checkpoint,
                            trainer_overrides=config.trainer_overrides,
                        )
                        pending_latest_committed_checkpoint_payload = (
                            _build_latest_committed_checkpoint_payload(
                                stage_name=resolved_stage_name,
                                step_index=step_index,
                                latest_hf_checkpoint=latest_hf_checkpoint,
                                latest_vllm_checkpoint=latest_vllm_checkpoint,
                            )
                        )
                        pending_latest_committed_step_index = step_index
                        runtime_manifest["latest_committed_step_index"] = (
                            pending_latest_committed_step_index
                        )
                        pruned_global_step_checkpoints = prune_old_global_step_checkpoints(
                            checkpoint_root=trainer_checkpoint_root,
                            keep_last=config.checkpoint_keep_last,
                        )
                        current_model_path = str(latest_vllm_checkpoint)
                        _append_unique_path(checkpoint_step_dirs, step_dir)

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
                    protected_step_dirs=(
                        [config.output_dir / f"rft_step_{pending_latest_committed_step_index:05d}"]
                        if pending_latest_committed_step_index is not None
                        else ()
                    ),
                )
            pruned_step_payloads = prune_old_step_payloads(
                step_dirs=run_step_dirs,
                keep_last=config.checkpoint_keep_last,
                protected_step_dirs=(
                    [config.output_dir / f"rft_step_{pending_latest_committed_step_index:05d}"]
                    if pending_latest_committed_step_index is not None
                    else ()
                ),
            )
            step_summary = {
                "step_index": step_index,
                "stage": resolved_stage_name,
                "selected_count": selected_count_raw,
                "selected_count_raw": selected_count_raw,
                "selected_count_after_length_filter": selected_count_after_max_length_filter,
                "avg_generation_length_raw": avg_generation_length_raw,
                "avg_generation_length": avg_generation_length,
                "selected_rows_over_max_length_dropped": selected_rows_over_max_length_dropped,
                "eval_selected_count_raw": eval_selected_count_raw,
                "eval_selected_count_after_length_filter": eval_selected_count_after_length_filter,
                "eval_selected_rows_over_max_length_dropped": (
                    eval_selected_rows_over_max_length_dropped
                ),
                "selected_count_for_train_raw": selected_count_for_train_raw,
                "selected_count_for_train": selected_count_for_train,
                "selected_count_for_eval_raw": selected_count_for_eval_raw,
                "selected_count_for_eval": selected_count_for_eval,
                "selected_rows_upsampled": selected_rows_upsampled,
                "selected_rows_eval_upsampled": selected_rows_eval_upsampled,
                "eval_split_fallback_to_train": eval_split_fallback_to_train,
                "selected_task_family_counts": _count_rows_by_text_field(
                    train_selected_rows,
                    field_name="task_family",
                    default_label="unknown",
                ),
                "selected_difficulty_band_counts": _count_rows_by_text_field(
                    train_selected_rows,
                    field_name="difficulty_band",
                    default_label="unbanded",
                ),
                "eval_selected_task_family_counts": _count_rows_by_text_field(
                    eval_selected_rows,
                    field_name="task_family",
                    default_label="unknown",
                ),
                "eval_selected_difficulty_band_counts": _count_rows_by_text_field(
                    eval_selected_rows,
                    field_name="difficulty_band",
                    default_label="unbanded",
                ),
                "rejected_count": len(train_rejected_rows),
                "train_rejected_count": len(train_rejected_rows),
                "eval_rejected_count": len(eval_rejected_rows),
                "trainer_skipped": trainer_skipped,
                "skip_reason": skip_reason,
                "effective_train_batch_size": effective_train_batch_size,
                "effective_eval_batch_size": effective_eval_batch_size,
                "collector_duration_sec": (
                    train_collect_duration_sec + eval_collect_duration_sec
                ),
                "collector_train_duration_sec": train_collect_duration_sec,
                "collector_eval_duration_sec": eval_collect_duration_sec,
                "trainer_duration_sec": trainer_duration_sec,
                "inner_train_step_first": trainer_metrics.get("train_step_first"),
                "inner_train_loss_first": trainer_metrics.get("train_loss_first"),
                "inner_train_step_last": trainer_metrics.get("train_step_last"),
                "inner_train_loss_last": trainer_metrics.get("train_loss_last"),
                "inner_train_loss_min": trainer_metrics.get("train_loss_min"),
                "inner_train_loss_min_step": trainer_metrics.get("train_loss_min_step"),
                "inner_train_loss_delta": trainer_metrics.get("train_loss_delta"),
                "inner_val_step_first": trainer_metrics.get("val_step_first"),
                "inner_val_loss_first": trainer_metrics.get("val_loss_first"),
                "inner_val_step_last": trainer_metrics.get("val_step_last"),
                "inner_val_loss_last": trainer_metrics.get("val_loss_last"),
                "inner_val_loss_min": trainer_metrics.get("val_loss_min"),
                "inner_val_loss_min_step": trainer_metrics.get("val_loss_min_step"),
                "inner_val_loss_delta": trainer_metrics.get("val_loss_delta"),
                "step_duration_sec": time.monotonic() - step_start,
                "train_parquet": str(train_parquet_path),
                "eval_parquet": str(resolved_val_parquet_path),
                "trainer_checkpoint_root": str(trainer_checkpoint_root),
                "latest_hf_checkpoint": str(latest_hf_checkpoint) if latest_hf_checkpoint else None,
                "latest_vllm_checkpoint": (
                    str(latest_vllm_checkpoint) if latest_vllm_checkpoint else None
                ),
                "resume_model_path": current_model_path,
                "selection_contract": resolve_rft_stage_selection_contract(resolved_stage_name),
                "correctness_contract": resolve_rft_stage_correctness_contract(resolved_stage_name),
                "trainer_command": trainer_command,
                "pruned_global_step_checkpoints": [
                    str(path) for path in pruned_global_step_checkpoints
                ],
                "pruned_checkpoint_roots": [str(path) for path in pruned_checkpoint_roots],
                "pruned_step_payloads": [str(path) for path in pruned_step_payloads],
                "latest_committed_step_index": (
                    pending_latest_committed_step_index
                ),
            }
            runtime_manifest["steps"].append(step_summary)
            _write_json(step_dir / "rft_step_summary.json", step_summary)
            _write_json(_runtime_loop_manifest_path(config.output_dir), runtime_manifest)
            if pending_latest_committed_checkpoint_payload is not None:
                _write_json(
                    _latest_committed_checkpoint_path(config.output_dir),
                    pending_latest_committed_checkpoint_payload,
                )
                latest_committed_checkpoint = _load_latest_committed_checkpoint(
                    config.output_dir
                )
            _log_rft_runtime_step_to_wandb(wandb_run=wandb_run, step_summary=step_summary)
    finally:
        if config.manage_vllm:
            vllm_controller.stop()
        _finish_rft_runtime_loop_wandb_run(
            wandb_run=wandb_run,
            runtime_manifest=runtime_manifest,
            final_model_path=current_model_path,
        )

    runtime_manifest["final_model_path"] = current_model_path
    runtime_manifest["completed_utc"] = _utc_now()
    runtime_manifest["latest_committed_checkpoint_path"] = str(
        _latest_committed_checkpoint_path(config.output_dir)
    )
    _write_json(_runtime_loop_manifest_path(config.output_dir), runtime_manifest)


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
    inner_trainer_logger = "[console]"
    if _coerce_bool_env("SMALL_SWE_RFT_INNER_TRAINER_WANDB_ENABLE", default=False):
        inner_trainer_logger = "[console,wandb]"

    required_overrides = [
        f"trainer.total_epochs={sft_num_epoch_per_batch}",
        f"trainer.n_gpus_per_node={nproc_per_node}",
        "trainer.resume_mode=disable",
        f"trainer.logger={inner_trainer_logger}",
        f"trainer.default_local_dir={trainer_output_dir}",
        f"trainer.save_freq={_INNER_SFT_CHECKPOINT_DISABLED_SAVE_FREQ}",
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


def materialize_vllm_compatible_checkpoint(
    *,
    checkpoint_dir: str | Path,
    trainer_overrides: Sequence[str],
) -> Path:
    """Return a dense checkpoint path that vLLM can load.

    verl+LoRA exports can carry PEFT wrapper keys (`base_model.model.*`), which
    vLLM rejects for plain model classes. When detected, this function rebuilds
    and merges the LoRA adapters into a dense HuggingFace checkpoint.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        return checkpoint_path

    weight_names = _list_checkpoint_weight_names(checkpoint_path)
    if not _checkpoint_requires_lora_merge(weight_names):
        return checkpoint_path

    merge_spec = _resolve_lora_merge_spec(
        trainer_overrides=trainer_overrides,
        checkpoint_dir=checkpoint_path,
        checkpoint_weight_names=weight_names,
    )
    merged_path = checkpoint_path.parent / "huggingface_vllm_merged"
    _merge_lora_checkpoint_to_dense(
        checkpoint_dir=checkpoint_path,
        merged_dir=merged_path,
        merge_spec=merge_spec,
    )
    return merged_path


def _list_checkpoint_weight_names(checkpoint_dir: Path) -> tuple[str, ...]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, Mapping):
            return tuple(str(name) for name in weight_map)
        return ()

    single_safetensors_path = checkpoint_dir / "model.safetensors"
    if single_safetensors_path.is_file():
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as exc:  # pragma: no cover - train-only dependency
            raise RuntimeError(
                "LoRA checkpoint conversion requires safetensors. "
                "Install training extras (`pip install -e \".[train]\"`)."
            ) from exc
        with safe_open(str(single_safetensors_path), framework="pt", device="cpu") as handle:
            return tuple(str(name) for name in handle.keys())

    return ()


def _checkpoint_requires_lora_merge(weight_names: Sequence[str]) -> bool:
    has_peft_prefix = any(name.startswith("base_model.model.") for name in weight_names)
    has_lora_weights = any(".lora_A." in name or ".lora_B." in name for name in weight_names)
    return has_peft_prefix and has_lora_weights


def _resolve_lora_merge_spec(
    *,
    trainer_overrides: Sequence[str],
    checkpoint_dir: Path,
    checkpoint_weight_names: Sequence[str],
) -> LoraMergeSpec:
    inferred_targets = _infer_lora_target_modules_from_weight_names(checkpoint_weight_names)
    configured_targets = _parse_hydra_list_override(
        _find_override_value(
            trainer_overrides,
            keys=("model.target_modules", "actor_rollout_ref.model.lora.target_modules"),
        )
        or ""
    )
    target_modules = inferred_targets or configured_targets or _DEFAULT_LORA_TARGET_MODULES

    inferred_rank = _infer_lora_rank_from_checkpoint(checkpoint_dir)
    configured_rank = _resolve_positive_int_override_value(
        trainer_overrides,
        keys=("model.lora_rank", "actor_rollout_ref.model.lora.rank"),
    )
    rank = configured_rank or inferred_rank or _DEFAULT_LORA_RANK

    configured_alpha = _resolve_positive_int_override_value(
        trainer_overrides,
        keys=("model.lora_alpha", "actor_rollout_ref.model.lora.alpha"),
    )
    alpha = configured_alpha or _DEFAULT_LORA_ALPHA

    if rank < 1:
        raise ValueError(f"Invalid LoRA rank for merge: {rank}")
    if alpha < 1:
        raise ValueError(f"Invalid LoRA alpha for merge: {alpha}")
    if not target_modules:
        raise ValueError("Unable to resolve LoRA target modules for checkpoint merge.")
    return LoraMergeSpec(rank=rank, alpha=alpha, target_modules=tuple(target_modules))


def _merge_lora_checkpoint_to_dense(
    *,
    checkpoint_dir: Path,
    merged_dir: Path,
    merge_spec: LoraMergeSpec,
) -> None:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.modeling_utils import load_sharded_checkpoint
    except ModuleNotFoundError as exc:  # pragma: no cover - train-only dependency
        raise RuntimeError(
            "LoRA checkpoint conversion requires train dependencies "
            "(torch/transformers/peft/safetensors). Install with `pip install -e \".[train]\"`."
        ) from exc

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    model_config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=False)
    model_kwargs: dict[str, Any] = {}
    resolved_dtype = _resolve_torch_dtype_from_config(model_config=model_config, torch_module=torch)
    if resolved_dtype is not None:
        model_kwargs["dtype"] = resolved_dtype

    base_model = _load_model_from_config_with_dtype_fallback(
        auto_model_cls=AutoModelForCausalLM,
        model_config=model_config,
        model_kwargs=model_kwargs,
    )
    lora_config = LoraConfig(
        r=merge_spec.rank,
        lora_alpha=merge_spec.alpha,
        target_modules=list(merge_spec.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(base_model, lora_config)

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        load_sharded_checkpoint(lora_model, str(checkpoint_dir), strict=False)
    else:
        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.is_file():
            state_dict = load_file(str(safetensors_path))
            lora_model.load_state_dict(state_dict, strict=False)
        else:
            bin_path = checkpoint_dir / "pytorch_model.bin"
            if not bin_path.is_file():
                raise FileNotFoundError(
                    f"Unable to find model weights in checkpoint directory: {checkpoint_dir}"
                )
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            lora_model.load_state_dict(state_dict, strict=False)

    with torch.no_grad():
        merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    _copy_non_model_hf_artifacts(source_dir=checkpoint_dir, destination_dir=merged_dir)


def _resolve_torch_dtype_from_config(*, model_config: Any, torch_module: Any) -> Any:
    raw = getattr(model_config, "torch_dtype", None)
    if raw is None:
        raw = getattr(model_config, "dtype", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        normalized = raw.strip().lower().replace("torch.", "")
        alias_map = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
        }
        resolved = alias_map.get(normalized, normalized)
        return getattr(torch_module, resolved, None)
    dtype_type = getattr(torch_module, "dtype", None)
    if dtype_type is not None and isinstance(raw, dtype_type):
        return raw
    return None


def _load_model_from_config_with_dtype_fallback(
    *,
    auto_model_cls: Any,
    model_config: Any,
    model_kwargs: Mapping[str, Any],
) -> Any:
    kwargs = dict(model_kwargs)
    try:
        return auto_model_cls.from_config(model_config, trust_remote_code=False, **kwargs)
    except TypeError as exc:
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            message = str(exc)
            if "unexpected keyword argument 'dtype'" in message:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
                return auto_model_cls.from_config(
                    model_config,
                    trust_remote_code=False,
                    **fallback_kwargs,
                )
        raise


def _copy_non_model_hf_artifacts(*, source_dir: Path, destination_dir: Path) -> None:
    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in _MODEL_ARTIFACT_FILE_NAMES:
            continue
        if path.name.startswith("model-") and path.name.endswith(".safetensors"):
            continue
        shutil.copy2(path, destination_dir / path.name)


def _infer_lora_target_modules_from_weight_names(weight_names: Sequence[str]) -> tuple[str, ...]:
    pattern = re.compile(r"\.([^.]+)\.lora_A\.default\.weight$")
    modules: set[str] = set()
    for name in weight_names:
        match = pattern.search(name)
        if match is not None:
            modules.add(match.group(1))
    return tuple(sorted(modules))


def _infer_lora_rank_from_checkpoint(checkpoint_dir: Path) -> int | None:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, Mapping):
            lora_a_keys = [
                str(name)
                for name in weight_map
                if str(name).endswith(".lora_A.default.weight")
            ]
            if lora_a_keys:
                first_key = lora_a_keys[0]
                shard_name = str(weight_map[first_key])
                shard_path = checkpoint_dir / shard_name
                if shard_path.is_file():
                    try:
                        from safetensors import safe_open
                    except ModuleNotFoundError:
                        return None
                    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                        tensor = handle.get_tensor(first_key)
                    if getattr(tensor, "ndim", 0) >= 1:
                        return int(tensor.shape[0])
        return None

    single_safetensors_path = checkpoint_dir / "model.safetensors"
    if single_safetensors_path.is_file():
        try:
            from safetensors import safe_open
        except ModuleNotFoundError:
            return None
        with safe_open(str(single_safetensors_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not str(key).endswith(".lora_A.default.weight"):
                    continue
                tensor = handle.get_tensor(str(key))
                if getattr(tensor, "ndim", 0) >= 1:
                    return int(tensor.shape[0])
                break
    return None


def _resolve_positive_int_override_value(
    overrides: Sequence[str],
    *,
    keys: Sequence[str],
) -> int | None:
    resolved: int | None = None
    for key in keys:
        for override in overrides:
            parsed = _parse_positive_int_override(override, key=key)
            if parsed is not None:
                resolved = parsed
    return resolved


def _find_override_value(overrides: Sequence[str], *, keys: Sequence[str]) -> str | None:
    resolved: str | None = None
    for override in overrides:
        normalized = override.strip()
        while normalized.startswith("+"):
            normalized = normalized[1:]
        if "=" not in normalized:
            continue
        raw_key, raw_value = normalized.split("=", 1)
        if raw_key.strip() in keys:
            resolved = raw_value.strip()
    return resolved


def _parse_hydra_list_override(raw_value: str) -> tuple[str, ...]:
    normalized = raw_value.strip()
    if not normalized:
        return ()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if not normalized.strip():
        return ()
    items: list[str] = []
    for item in normalized.split(","):
        token = item.strip().strip("\"'")
        if token:
            items.append(token)
    return tuple(items)


def prune_old_step_checkpoints(
    *,
    step_dirs: Sequence[str | Path],
    keep_last: int,
    protected_step_dirs: Sequence[str | Path] = (),
) -> list[Path]:
    """Delete old per-step trainer checkpoint trees beyond the keep-last window."""
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1 to preserve the current model checkpoint.")

    ordered_step_dirs = _coerce_step_dirs_in_order(step_dirs)
    if len(ordered_step_dirs) <= keep_last:
        return []

    protected = {Path(item) for item in protected_step_dirs}
    to_prune = [
        step_dir
        for step_dir in ordered_step_dirs[: len(ordered_step_dirs) - keep_last]
        if step_dir not in protected
    ]
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


def prune_old_step_payloads(
    *,
    step_dirs: Sequence[str | Path],
    keep_last: int,
    protected_step_dirs: Sequence[str | Path] = (),
) -> list[Path]:
    """Delete old per-step rollout payloads beyond the keep-last window.

    Retained summaries (`rft_step_summary.json`) remain in each step directory for
    lightweight auditability while bulky artifacts are pruned.
    """
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1 to preserve current step payload artifacts.")

    ordered_step_dirs = _coerce_step_dirs_in_order(step_dirs)
    if len(ordered_step_dirs) <= keep_last:
        return []

    protected = {Path(item) for item in protected_step_dirs}
    to_prune = [
        step_dir
        for step_dir in ordered_step_dirs[: len(ordered_step_dirs) - keep_last]
        if step_dir not in protected
    ]
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
    resolved_stage_name = resolve_rft_stage_name(config.stage_name)
    preview_steps = min(config.rft_steps, 2)
    print(
        "# [dry-run] planned RFT loop",
        f"stage_name={resolved_stage_name}",
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


def _run_command(command: Sequence[str], *, cwd: Path) -> dict[str, float | int]:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    process_group_id = process.pid

    train_step_first: int | None = None
    train_loss_first: float | None = None
    train_step_last: int | None = None
    train_loss_last: float | None = None
    train_loss_min: float | None = None
    train_loss_min_step: int | None = None
    val_step_first: int | None = None
    val_loss_first: float | None = None
    val_step_last: int | None = None
    val_loss_last: float | None = None
    val_loss_min: float | None = None
    val_loss_min_step: int | None = None
    command_error: subprocess.CalledProcessError | None = None
    cleanup_error: Exception | None = None

    try:
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
                train_match = _TRAIN_LOSS_PATTERN.search(line)
                if train_match is not None:
                    train_step = int(train_match.group(1))
                    train_loss = float(train_match.group(2))
                    if train_step_first is None:
                        train_step_first = train_step
                        train_loss_first = train_loss
                    train_step_last = train_step
                    train_loss_last = train_loss
                    if train_loss_min is None or train_loss < train_loss_min:
                        train_loss_min = train_loss
                        train_loss_min_step = train_step
                val_match = _VAL_LOSS_PATTERN.search(line)
                if val_match is not None:
                    val_step = int(val_match.group(1))
                    val_loss = float(val_match.group(2))
                    if val_step_first is None:
                        val_step_first = val_step
                        val_loss_first = val_loss
                    val_step_last = val_step
                    val_loss_last = val_loss
                    if val_loss_min is None or val_loss < val_loss_min:
                        val_loss_min = val_loss
                        val_loss_min_step = val_step

        return_code = process.wait()
        if return_code != 0:
            command_error = subprocess.CalledProcessError(return_code, list(command))
    finally:
        try:
            _cleanup_process_group(
                process_group_id,
                timeout_sec=_DEFAULT_PROCESS_GROUP_CLEANUP_TIMEOUT_SEC,
            )
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            cleanup_error = exc

    if command_error is not None:
        if cleanup_error is not None:
            command_error.add_note(
                f"trainer subprocess cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise command_error
    if cleanup_error is not None:
        raise cleanup_error

    metrics: dict[str, float | int] = {}
    if train_step_first is not None:
        metrics["train_step_first"] = train_step_first
    if train_loss_first is not None:
        metrics["train_loss_first"] = train_loss_first
    if train_step_last is not None:
        metrics["train_step_last"] = train_step_last
    if train_loss_last is not None:
        metrics["train_loss_last"] = train_loss_last
    if train_loss_min is not None:
        metrics["train_loss_min"] = train_loss_min
    if train_loss_min_step is not None:
        metrics["train_loss_min_step"] = train_loss_min_step
    if train_loss_first is not None and train_loss_last is not None:
        metrics["train_loss_delta"] = train_loss_last - train_loss_first
    if val_step_first is not None:
        metrics["val_step_first"] = val_step_first
    if val_loss_first is not None:
        metrics["val_loss_first"] = val_loss_first
    if val_step_last is not None:
        metrics["val_step_last"] = val_step_last
    if val_loss_last is not None:
        metrics["val_loss_last"] = val_loss_last
    if val_loss_min is not None:
        metrics["val_loss_min"] = val_loss_min
    if val_loss_min_step is not None:
        metrics["val_loss_min_step"] = val_loss_min_step
    if val_loss_first is not None and val_loss_last is not None:
        metrics["val_loss_delta"] = val_loss_last - val_loss_first
    return metrics


def _load_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - train-only dependency
        raise RuntimeError(
            "RFT runtime loop requires transformers. Install training extras (`pip install -e \".[train]\"`)."
        ) from exc

    kwargs = {
        "trust_remote_code": False,
        "fix_mistral_regex": True,
    }
    try:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(model_path, **kwargs)


def _build_models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def _query_http_endpoint_models(
    url: str,
    *,
    api_key: str | None = None,
) -> tuple[str, ...] | None:
    headers = {}
    if api_key is not None and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=2.0) as response:
            if not (200 <= int(response.status) < 300):
                return None
            response_body = response.read().decode("utf-8", errors="replace")
    except HTTPError:
        return None
    except (URLError, TimeoutError, OSError):
        return None
    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        return ()
    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str):
            continue
        normalized = model_id.strip()
        if normalized:
            model_ids.append(normalized)
    return tuple(model_ids)


def _is_http_endpoint_ready(
    url: str,
    *,
    api_key: str | None = None,
    expected_model_name: str | None = None,
) -> bool:
    model_ids = _query_http_endpoint_models(url, api_key=api_key)
    if model_ids is None:
        return False
    if expected_model_name is None or not expected_model_name.strip():
        return True
    return expected_model_name.strip() in model_ids


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
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def _coerce_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def compute_average_generation_length(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> float | None:
    if not selected_rows:
        return None

    lengths: list[float] = []
    for row in selected_rows:
        length = _generation_length_from_selected_row(row=row, tokenizer=tokenizer)
        if length is None:
            continue
        lengths.append(length)

    if not lengths:
        return None
    return float(sum(lengths) / len(lengths))


def _generation_length_from_selected_row(*, row: Mapping[str, Any], tokenizer: Any) -> float | None:
    action_mask_length = _generation_length_from_action_mask(row.get("action_mask_rft"))
    if action_mask_length is not None:
        return action_mask_length

    token_labels_length = _generation_length_from_token_labels(row.get("token_labels"))
    if token_labels_length is not None:
        return token_labels_length

    return _generation_length_from_assistant_text(row=row, tokenizer=tokenizer)


def _generation_length_from_action_mask(raw_mask: Any) -> float | None:
    if not isinstance(raw_mask, Sequence) or isinstance(raw_mask, (str, bytes)):
        return None

    count = 0
    has_numeric = False
    for item in raw_mask:
        parsed = _coerce_numeric(item)
        if parsed is None:
            continue
        has_numeric = True
        if parsed > 0.0:
            count += 1
    if not has_numeric:
        return None
    return float(count)


def _generation_length_from_token_labels(raw_labels: Any) -> float | None:
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes)):
        return None

    count = 0
    has_numeric = False
    for item in raw_labels:
        parsed = _coerce_numeric(item)
        if parsed is None:
            continue
        has_numeric = True
        if parsed != -100.0:
            count += 1
    if not has_numeric:
        return None
    return float(count)


def _generation_length_from_assistant_text(*, row: Mapping[str, Any], tokenizer: Any) -> float | None:
    assistant_turns: list[str] = []
    trajectory_history = row.get("trajectory_history")
    if isinstance(trajectory_history, Sequence) and not isinstance(trajectory_history, (str, bytes)):
        for entry in trajectory_history:
            text = str(entry).strip()
            if not text:
                continue
            if text.startswith(_TOOL_RESPONSE_PREFIX):
                continue
            assistant_turns.append(text)

    if not assistant_turns:
        assistant_response = str(row.get("assistant_response", "")).strip()
        if assistant_response:
            assistant_turns.append(assistant_response)

    if not assistant_turns:
        return None

    total = 0
    for text in assistant_turns:
        total += _token_count_for_generation_text(tokenizer=tokenizer, text=text)
    return float(total)


def _token_count_for_generation_text(*, tokenizer: Any, text: str) -> int:
    encoded_count = _token_count_with_encode(tokenizer=tokenizer, text=text)
    if encoded_count is not None:
        return encoded_count

    call_count = _token_count_with_tokenizer_call(tokenizer=tokenizer, text=text)
    if call_count is not None:
        return call_count

    if callable(getattr(tokenizer, "apply_chat_template", None)):
        try:
            return _multiturn_token_count(
                messages=[{"role": "assistant", "content": text}],
                tokenizer=tokenizer,
            )
        except Exception:
            pass

    return len(text)


def _token_count_with_encode(*, tokenizer: Any, text: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None
    try:
        payload = encode(text, add_special_tokens=False)
    except TypeError:
        payload = encode(text)
    except Exception:
        return None
    return _sequence_length(payload)


def _token_count_with_tokenizer_call(*, tokenizer: Any, text: str) -> int | None:
    if not callable(tokenizer):
        return None
    try:
        payload = tokenizer(text, add_special_tokens=False)
    except TypeError:
        try:
            payload = tokenizer(text)
        except Exception:
            return None
    except Exception:
        return None

    if isinstance(payload, Mapping):
        input_ids = payload.get("input_ids")
    else:
        input_ids = payload
    return _sequence_length(input_ids)


def _sequence_length(payload: Any) -> int | None:
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if payload and isinstance(payload[0], Sequence) and not isinstance(payload[0], (str, bytes)):
            payload = payload[0]
        return len(payload)
    return None


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def _init_rft_runtime_loop_wandb_run(*, config: RFTLoopConfig) -> Any | None:
    if not _coerce_bool_env("SMALL_SWE_RFT_LOOP_WANDB_ENABLE", default=False):
        return None
    if os.environ.get("WANDB_MODE", "").strip().lower() == "disabled":
        return None

    try:
        import wandb
    except ModuleNotFoundError:
        return None

    project_override = _find_override_value(config.trainer_overrides, keys=("trainer.project_name",))
    group_override = _find_override_value(config.trainer_overrides, keys=("trainer.group_name",))
    experiment_override = _find_override_value(
        config.trainer_overrides,
        keys=("trainer.experiment_name",),
    )
    project = (
        os.environ.get("SMALL_SWE_RFT_LOOP_WANDB_PROJECT")
        or os.environ.get("WANDB_PROJECT")
        or project_override
        or _DEFAULT_RFT_WANDB_PROJECT
    )
    group = (
        os.environ.get("SMALL_SWE_RFT_LOOP_WANDB_GROUP")
        or os.environ.get("WANDB_GROUP")
        or group_override
        or _DEFAULT_RFT_WANDB_GROUP
    )
    experiment = experiment_override or os.environ.get("EXPERIMENT") or "rft"
    default_name = f"{experiment}-runtime-loop-{config.output_dir.name}"
    run_name = os.environ.get("SMALL_SWE_RFT_LOOP_WANDB_RUN_NAME", default_name)

    try:
        return wandb.init(
            project=project,
            group=group,
            name=run_name,
            job_type="rft_runtime_loop",
            config={
                "rft_steps": config.rft_steps,
                "samples_per_task": config.samples_per_task,
                "task_batch_size": config.task_batch_size,
                "sft_num_epoch_per_batch": config.sft_num_epoch_per_batch,
                "train_batch_size": config.train_batch_size,
                "eval_split_fraction": config.eval_split_fraction,
                "eval_min_rows": config.eval_min_rows,
                "output_dir": str(config.output_dir),
                "initial_model": config.initial_model,
            },
            reinit=True,
        )
    except Exception as exc:
        print(
            "[rft-runtime-loop] wandb init failed; step-level metrics will not be logged: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _log_rft_runtime_step_to_wandb(*, wandb_run: Any | None, step_summary: Mapping[str, Any]) -> None:
    if wandb_run is None:
        return

    step_value = step_summary.get("step_index")
    if not isinstance(step_value, int):
        return

    metrics: dict[str, Any] = {
        "rft/step_index": step_value,
        "rft/selected_count": step_summary.get("selected_count"),
        "rft/selected_count_after_length_filter": step_summary.get(
            "selected_count_after_length_filter"
        ),
        "rft/avg_generation_length_raw": step_summary.get("avg_generation_length_raw"),
        "rft/avg_generation_length": step_summary.get("avg_generation_length"),
        "rft/rejected_count": step_summary.get("rejected_count"),
        "rft/eval_selected_count_after_length_filter": step_summary.get(
            "eval_selected_count_after_length_filter"
        ),
        "rft/eval_rejected_count": step_summary.get("eval_rejected_count"),
        "rft/trainer_skipped": int(bool(step_summary.get("trainer_skipped", False))),
        "rft/collector_duration_sec": step_summary.get("collector_duration_sec"),
        "rft/trainer_duration_sec": step_summary.get("trainer_duration_sec"),
        "rft/inner_train_loss_first": step_summary.get("inner_train_loss_first"),
        "rft/inner_train_step_last": step_summary.get("inner_train_step_last"),
        "rft/inner_train_loss_last": step_summary.get("inner_train_loss_last"),
        "rft/inner_train_loss_min": step_summary.get("inner_train_loss_min"),
        "rft/inner_train_loss_delta": step_summary.get("inner_train_loss_delta"),
        "rft/inner_val_loss_first": step_summary.get("inner_val_loss_first"),
        "rft/inner_val_step_last": step_summary.get("inner_val_step_last"),
        "rft/inner_val_loss_last": step_summary.get("inner_val_loss_last"),
        "rft/inner_val_loss_min": step_summary.get("inner_val_loss_min"),
        "rft/inner_val_loss_delta": step_summary.get("inner_val_loss_delta"),
        "rft/step_duration_sec": step_summary.get("step_duration_sec"),
    }
    sanitized_metrics: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            sanitized_metrics[key] = int(value)
            continue
        if isinstance(value, int):
            sanitized_metrics[key] = value
            continue
        if isinstance(value, float):
            sanitized_metrics[key] = value
            continue
    if not sanitized_metrics:
        return

    try:
        wandb_run.log(sanitized_metrics, step=step_value)
    except Exception as exc:
        print(
            "[rft-runtime-loop] wandb step log failed; continuing without W&B step metrics: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _finish_rft_runtime_loop_wandb_run(
    *,
    wandb_run: Any | None,
    runtime_manifest: Mapping[str, Any],
    final_model_path: str,
) -> None:
    if wandb_run is None:
        return

    try:
        steps = runtime_manifest.get("steps")
        if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
            wandb_run.summary["rft/steps_completed"] = len(steps)
        wandb_run.summary["rft/final_model_path"] = str(final_model_path)
        wandb_run.finish()
    except Exception as exc:
        print(
            "[rft-runtime-loop] wandb finish failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _coerce_bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


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
        annotated_row = dict(row)
        annotated_row["selected_token_count"] = int(token_count)
        annotated_row["selected_over_budget"] = False
        kept_rows.append(annotated_row)

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


def _count_rows_by_text_field(
    rows: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    default_label: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(field_name, "")).strip() or default_label
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


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
        parsed = _parse_positive_int_override(override, key=_MAX_MODEL_LEN_KEY)
        if parsed is not None:
            resolved = parsed
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

    max_length_pattern = re.compile(r"^\s*max_length\s*:\s*([0-9]+)\s*(?:#.*)?$")
    top_level_max_model_len_pattern = re.compile(r"^max_model_len\s*:\s*([0-9]+)\s*(?:#.*)?$")
    lines = config_path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        match = max_length_pattern.match(line)
        if match is None:
            continue
        value = int(match.group(1))
        if value >= 1:
            return value
        break

    for line in lines:
        match = top_level_max_model_len_pattern.match(line)
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
            "fraction of tasks to reserve for a deterministic held-out eval partition; "
            "must satisfy 0.0 <= value < 1.0."
        ),
    )
    parser.add_argument(
        "--eval-min-rows",
        type=int,
        default=1,
        help=(
            "minimum number of held-out eval tasks when split fraction is positive and "
            "at least two valid tasks are available."
        ),
    )
    parser.add_argument("--stage-name", default=_FORMAT_RFT_STAGE_NAME)
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
    resolved_stage_name = resolve_rft_stage_name(str(args.stage_name))

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
        stage_name=resolved_stage_name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    run_rft_runtime_loop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
