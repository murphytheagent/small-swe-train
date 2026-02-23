"""End-to-end RFT loop orchestration for live rollout -> train -> checkpoint refresh."""

from __future__ import annotations

import argparse
import json
import re
import shlex
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

from trainer.rft_multiturn_dataset import write_selected_rows_to_multiturn_parquet
from trainer.rft_runtime import OnPolicyRFTRuntimeRequest, collect_onpolicy_rft_runtime_batch

_GLOBAL_STEP_PATTERN = re.compile(r"^global_step_(\d+)$")
_VERL_SFT_TRAINER_DOC = (
    "https://github.com/lasgroup/SDPO/blob/main/verl/trainer/fsdp_sft_trainer.py"
)
_VLLM_OPENAI_SERVER_DOC = (
    "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html"
)


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
            if _is_http_endpoint_ready(self._models_url):
                return
            time.sleep(1.0)
        raise RuntimeError(
            "Timed out waiting for vLLM readiness at "
            f"{self._models_url}. Inspect logs at {self._log_path}."
        )


def run_rft_runtime_loop(config: RFTLoopConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    vllm_logs = config.output_dir / "vllm_server.log"
    runtime_manifest: dict[str, Any] = {
        "generated_utc": _utc_now(),
        "config": {
            "rft_steps": config.rft_steps,
            "samples_per_task": config.samples_per_task,
            "task_batch_size": config.task_batch_size,
            "sft_num_epoch_per_batch": config.sft_num_epoch_per_batch,
            "train_batch_size": config.train_batch_size,
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
        _print_dry_run_plan(config=config)
        return

    tokenizer = _load_tokenizer(config.initial_model)
    current_model_path = config.initial_model
    vllm_controller = VLLMServerController(config=config, log_path=vllm_logs)

    try:
        if config.manage_vllm:
            vllm_controller.start(model_path=current_model_path)

        for step_index in range(config.rft_steps):
            step_dir = config.output_dir / f"rft_step_{step_index:05d}"
            collector_dir = step_dir / "collector_artifacts"
            parquet_path = step_dir / "accepted_trajectories.parquet"
            trainer_checkpoint_root = step_dir / "trainer_checkpoints"
            step_dir.mkdir(parents=True, exist_ok=True)

            request = OnPolicyRFTRuntimeRequest(
                data_config_name=config.data_config_name,
                turn_generator_mode=config.turn_generator_mode,
                total_steps=1,
                runtime_overrides={
                    "task_batch_size": config.task_batch_size,
                    "attempts_per_task": config.samples_per_task,
                    "env_pool_size": config.task_batch_size,
                },
                output_dir=str(collector_dir),
            )
            handoff = collect_onpolicy_rft_runtime_batch(
                request=request,
                tokenizer=tokenizer,
            )
            selected_rows = _coerce_rows(handoff.get("selected_rows"))
            rejected_rows = _coerce_rows(handoff.get("rejected_rows"))
            selected_count = write_selected_rows_to_multiturn_parquet(selected_rows, parquet_path)

            trainer_command = build_trainer_step_command(
                nnodes=config.nnodes,
                nproc_per_node=config.nproc_per_node,
                trainer_module=config.trainer_module,
                config_name=config.config_name,
                config_dir=config.config_dir,
                model_path=current_model_path,
                train_parquet_path=parquet_path,
                val_parquet_path=parquet_path,
                trainer_output_dir=trainer_checkpoint_root,
                train_batch_size=config.train_batch_size,
                sft_num_epoch_per_batch=config.sft_num_epoch_per_batch,
                trainer_overrides=config.trainer_overrides,
            )

            if config.manage_vllm:
                vllm_controller.stop()
            _run_command(trainer_command, cwd=config.project_root)

            latest_hf_checkpoint = resolve_latest_hf_checkpoint(trainer_checkpoint_root)
            current_model_path = str(latest_hf_checkpoint)

            if config.manage_vllm:
                vllm_controller.start(model_path=current_model_path)

            step_summary = {
                "step_index": step_index,
                "selected_count": selected_count,
                "rejected_count": len(rejected_rows),
                "train_parquet": str(parquet_path),
                "trainer_checkpoint_root": str(trainer_checkpoint_root),
                "latest_hf_checkpoint": str(latest_hf_checkpoint),
                "trainer_command": trainer_command,
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
        "trainer.checkpoint.save_contents=[model,hf_model,extra]",
        "trainer.checkpoint.load_contents=[model,hf_model,extra]",
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
        "torchrun",
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

    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _GLOBAL_STEP_PATTERN.match(path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(f"No global_step_* checkpoint directories found in {root}")

    candidates.sort(key=lambda item: item[0])
    _, latest_step_dir = candidates[-1]
    huggingface_dir = latest_step_dir / "huggingface"
    if not huggingface_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint {latest_step_dir} is missing huggingface export directory."
        )
    return huggingface_dir


def _print_dry_run_plan(config: RFTLoopConfig) -> None:
    preview_steps = min(config.rft_steps, 2)
    print(
        "# [dry-run] planned RFT loop",
        f"steps={config.rft_steps}",
        f"samples_per_task={config.samples_per_task}",
        f"task_batch_size={config.task_batch_size}",
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
        parquet_path = step_dir / "accepted_trajectories.parquet"
        checkpoint_root = step_dir / "trainer_checkpoints"
        print(
            f"# [dry-run] step={step_index} collect selected trajectories -> {parquet_path}"
        )
        trainer_command = build_trainer_step_command(
            nnodes=config.nnodes,
            nproc_per_node=config.nproc_per_node,
            trainer_module=config.trainer_module,
            config_name=config.config_name,
            config_dir=config.config_dir,
            model_path=config.initial_model,
            train_parquet_path=parquet_path,
            val_parquet_path=parquet_path,
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


def _is_http_endpoint_ready(url: str) -> bool:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=2.0) as response:
            return 200 <= int(response.status) < 500
    except HTTPError as exc:
        return 200 <= int(exc.code) < 500
    except (URLError, TimeoutError, OSError):
        return False


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
        default="verl.trainer.fsdp_sft_trainer",
        help=f"verl trainer module (see {_VERL_SFT_TRAINER_DOC})",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--rft-steps", type=int, required=True)
    parser.add_argument("--samples-per-task", type=int, required=True)
    parser.add_argument("--task-batch-size", type=int, required=True)
    parser.add_argument("--sft-num-epoch-per-batch", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-config-name", default="on_policy_swe_smith")
    parser.add_argument("--turn-generator-mode", default="default")
    parser.add_argument("--initial-model", required=True)
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--vllm-served-model", required=True)
    parser.add_argument(
        "--vllm-launch-module",
        default="vllm.entrypoints.openai.api_server",
        help=f"vLLM OpenAI server module (see {_VLLM_OPENAI_SERVER_DOC})",
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
    if args.sft_num_epoch_per_batch < 1:
        raise ValueError("--sft-num-epoch-per-batch must be >= 1.")
    if args.train_batch_size < 1:
        raise ValueError("--train-batch-size must be >= 1.")
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
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    run_rft_runtime_loop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
