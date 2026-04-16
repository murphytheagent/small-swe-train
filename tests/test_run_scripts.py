from __future__ import annotations

import json
import os
import socket
import signal
import shutil
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Mapping

import config
import pytest

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_script(
    script_name: str,
    *args: str,
    env_overrides: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    script_path = _repo_root() / "scripts" / script_name
    env = os.environ.copy()
    if script_name == "run_sdpo.sh":
        env.setdefault("SDPO_RFT_CHECKPOINT", "/tmp/rft-checkpoint")
    if env_overrides is not None:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", str(script_path), "--dry-run", *args],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def _convert_torchrun_command_to_hydra_preflight(command_line: str) -> list[str]:
    tokens = shlex.split(command_line)
    if len(tokens) < 3:
        raise ValueError(f"Unexpected command line: {command_line!r}")
    try:
        config_name_idx = tokens.index("--config-name")
        config_dir_idx = tokens.index("--config-dir")
    except ValueError as exc:
        raise ValueError(f"Missing Hydra config flags in command: {command_line!r}") from exc

    module_flag_idx = None
    for idx, token in enumerate(tokens[: config_name_idx + 1]):
        if token == "-m":
            module_flag_idx = idx
    if module_flag_idx is None or module_flag_idx + 1 >= len(tokens):
        raise ValueError(f"Unable to resolve trainer module from command: {command_line!r}")

    overrides_start_idx = max(config_name_idx + 2, config_dir_idx + 2)
    trainer_module = tokens[module_flag_idx + 1]
    return [
        tokens[0],
        "-m",
        trainer_module,
        "--config-name",
        tokens[config_name_idx + 1],
        "--config-dir",
        tokens[config_dir_idx + 1],
        *tokens[overrides_start_idx:],
        "--cfg",
        "job",
    ]


def _run_hydra_preflight_for_torchrun_command(command_line: str) -> subprocess.CompletedProcess[str]:
    preflight_cmd = _convert_torchrun_command_to_hydra_preflight(command_line)
    env = os.environ.copy()
    repo_root = _repo_root()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    return subprocess.run(
        preflight_cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        env=env,
    )


def _write_python_defaults_stub(tmp_path: Path, defaults_line: str) -> Path:
    stub_path = tmp_path / "python-defaults-stub.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"-\" ]]; then\n"
        "  payload=\"$(cat)\"\n"
        "  if [[ \"${payload}\" == *\"torch.cuda.device_count\"* ]]; then\n"
        "    printf '%s\\n' \"${STUB_GPU_COUNT:-0}\"\n"
        "  else\n"
        f"    printf '%s\\n' '{defaults_line}'\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "exec python3 \"$@\"\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_python_no_preload_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "python-no-preload-stub.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"env.preload_sdpo_dataset\" ]]; then\n"
        "  echo \"unexpected preload helper invocation\" >&2\n"
        "  exit 91\n"
        "fi\n"
        "exec python3 \"$@\"\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_python_env_probe_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "python-env-probe-stub.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"-c\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "printf 'EXPERIMENT=%s\\n' \"${EXPERIMENT:-}\"\n"
        "printf 'TASK=%s\\n' \"${TASK:-}\"\n"
        "printf 'CUDA_VISIBLE_DEVICES=%s\\n' \"${CUDA_VISIBLE_DEVICES:-}\"\n"
        "printf 'ROCR_VISIBLE_DEVICES=%s\\n' \"${ROCR_VISIBLE_DEVICES:-}\"\n"
        "printf 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=%s\\n' \"${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-}\"\n"
        "printf 'TVM_FFI_DISABLE_TORCH_C_DLPACK=%s\\n' \"${TVM_FFI_DISABLE_TORCH_C_DLPACK:-}\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_python_run_rft_probe_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "python-run-rft-probe-stub.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == \"-c\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"-\" ]]; then\n"
        f"  exec {shlex.quote(sys.executable)} \"$@\"\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"-m\" ]]; then\n"
        "  if [[ -n \"${STUB_RFT_CAPTURE_FILE:-}\" ]]; then\n"
        "    printf '%s\\n' \"$@\" >\"${STUB_RFT_CAPTURE_FILE}\"\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_docker_cleanup_probe_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "docker"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "log_file=\"${FAKE_DOCKER_LOG_FILE:?}\"\n"
        "cmd=\"$*\"\n"
        "printf '%s\\n' \"${cmd}\" >>\"${log_file}\"\n"
        "if [[ \"${1:-}\" == \"ps\" ]]; then\n"
        "  if [[ \"${cmd}\" == *\"label=small_swe.slurm_job_id=4242\"* ]]; then\n"
        "    exit 0\n"
        "  fi\n"
        "  if [[ \"${cmd}\" == *\"label=small_swe.run_label=run-fallback-4242\"* ]]; then\n"
        "    printf '%s\\n' 'container-from-run-label'\n"
        "    exit 0\n"
        "  fi\n"
        "  if [[ \"${cmd}\" == *\"name=sdpo-swe-bridge-\"* ]]; then\n"
        "    printf '%s\\n' 'container-from-name-fallback'\n"
        "    exit 0\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"rm\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_pilot_docker_cleanup_probe_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "docker"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "log_file=\"${FAKE_DOCKER_LOG_FILE:?}\"\n"
        "cmd=\"$*\"\n"
        "printf '%s\\n' \"${cmd}\" >>\"${log_file}\"\n"
        "if [[ \"${1:-}\" == \"ps\" ]]; then\n"
        "  if [[ \"${cmd}\" == *\"label=small_swe.pool_name=onpolicy-task\"* ]]; then\n"
        "    printf '%s\\n' 'live-container 987654'\n"
        "    printf '%s\\n' 'other-live-container 555555'\n"
        "    printf '%s\\n' 'stale-container-1 4242'\n"
        "    printf '%s\\n' 'stale-container-2 4243'\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"rm\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_squeue_probe_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "squeue"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' '987654'\n"
        "printf '%s\\n' '555555'\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_python_wandb_repair_probe_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "python-wandb-repair-probe.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == \"-c\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"-m\" ]]; then\n"
        "  if [[ -n \"${STUB_TRAINER_STDOUT:-}\" ]]; then\n"
        "    printf '%b\\n' \"${STUB_TRAINER_STDOUT}\"\n"
        "  fi\n"
        "  if [[ -n \"${STUB_TRAINER_SLEEP_SEC:-}\" ]]; then\n"
        "    sleep \"${STUB_TRAINER_SLEEP_SEC}\"\n"
        "  fi\n"
        "  exit ${STUB_TRAINER_EXIT_CODE:-0}\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"-\" ]]; then\n"
        "  payload=\"$(cat)\"\n"
        "  if [[ \"${payload}\" == *\"run_sdpo.sh wandb-repair\"* ]]; then\n"
        "    if [[ -n \"${STUB_REPAIR_STDOUT_MARKER:-}\" ]]; then\n"
        "      printf '%s\\n' \"${STUB_REPAIR_STDOUT_MARKER}\"\n"
        "    fi\n"
        "    if [[ -n \"${STUB_REPAIR_LOG_FILE:-}\" ]]; then\n"
        "      printf '%s\\n' \"$*\" >>\"${STUB_REPAIR_LOG_FILE}\"\n"
        "    fi\n"
        "    exit 0\n"
        "  fi\n"
        f"  printf '%s' \"${{payload}}\" | {shlex.quote(sys.executable)} \"$@\"\n"
        "  exit $?\n"
        "fi\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_python_cleanup_probe_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "python-cleanup-probe-stub.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == \"-c\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"-\" ]]; then\n"
        "  payload=\"$(cat)\"\n"
        "  if [[ \"${payload}\" == *\"resolve_sdpo_task_cache_dir\"* ]]; then\n"
        "    printf '%s\\n' \"${STUB_SDPO_CACHE_DIR:-/tmp/sdpo_task_cache}\"\n"
        "    exit 0\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"-m\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "exec python3 \"$@\"\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_teacher_pilot_python_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "python-teacher-pilot-stub.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"trainer.vllm_api_server_entry\" ]]; then\n"
        "  exec python3 - <<'PY'\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "import json\n"
        "\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        if self.path != '/v1/models':\n"
        "            self.send_error(404)\n"
        "            return\n"
        "        payload = json.dumps({'data': [{'id': 'stub-model'}]}).encode('utf-8')\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Type', 'application/json')\n"
        "        self.send_header('Content-Length', str(len(payload)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(payload)\n"
        "\n"
        "    def log_message(self, format, *args):\n"
        "        return\n"
        "\n"
        "HTTPServer(('127.0.0.1', 8000), Handler).serve_forever()\n"
        "PY\n"
        "fi\n"
        "if [[ \"${1:-}\" == */run_teacher_reprompt_pilot.py || \"${1:-}\" == \"scripts/run_teacher_reprompt_pilot.py\" ]]; then\n"
        "  if [[ \" ${*} \" == *\" --print-resolved-rft-checkpoint \"* ]]; then\n"
        "    shift\n"
        "    while [[ $# -gt 0 ]]; do\n"
        "      case \"$1\" in\n"
        "        --rft-checkpoint)\n"
        "          printf '%s\\n' \"${2:-}\"\n"
        "          exit 0\n"
        "          ;;\n"
        "        --rft-checkpoint=*)\n"
        "          printf '%s\\n' \"${1#*=}\"\n"
        "          exit 0\n"
        "          ;;\n"
        "      esac\n"
        "      shift\n"
        "    done\n"
        "    printf '\\n'\n"
        "    exit 0\n"
        "  fi\n"
        "  : \"${TEACHER_PILOT_CAPTURE:?}\"\n"
        "  printf '%s\\n' \"$@\" >\"${TEACHER_PILOT_CAPTURE}\"\n"
        "  exit 0\n"
        "fi\n"
        "exec python3 \"$@\"\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def _write_onpolicy_difficulty_probe_python_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "python-difficulty-probe-stub.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"trainer.vllm_api_server_entry\" ]]; then\n"
        "  if [[ \"${STUB_VLLM_FAIL_FAST:-0}\" == \"1\" ]]; then\n"
        "    echo 'stub vLLM startup failure' >&2\n"
        "    exit 23\n"
        "  fi\n"
        "  exec python3 - \"$@\" <<'PY'\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "import json\n"
        "import sys\n"
        "\n"
        "port = 8000\n"
        "for index, token in enumerate(sys.argv):\n"
        "    if token == '--port' and index + 1 < len(sys.argv):\n"
        "        port = int(sys.argv[index + 1])\n"
        "        break\n"
        "\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        if self.path != '/v1/models':\n"
        "            self.send_error(404)\n"
        "            return\n"
        "        payload = json.dumps({'data': [{'id': 'stub-model'}]}).encode('utf-8')\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Type', 'application/json')\n"
        "        self.send_header('Content-Length', str(len(payload)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(payload)\n"
        "\n"
        "    def log_message(self, format, *args):\n"
        "        return\n"
        "\n"
        "HTTPServer(('127.0.0.1', port), Handler).serve_forever()\n"
        "PY\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"env.preload_onpolicy_difficulty_bands\" ]]; then\n"
        "  : \"${DIFFICULTY_PROBE_CAPTURE:?}\"\n"
        "  printf '%s\\n' \"$@\" >\"${DIFFICULTY_PROBE_CAPTURE}\"\n"
        "  cache_dir=''\n"
        "  probe_label='positive_rft_probe'\n"
        "  while [[ $# -gt 0 ]]; do\n"
        "    case \"$1\" in\n"
        "      --cache-dir)\n"
        "        cache_dir=\"${2:-}\"\n"
        "        shift 2\n"
        "        ;;\n"
        "      --probe-label)\n"
        "        probe_label=\"${2:-}\"\n"
        "        shift 2\n"
        "        ;;\n"
        "      *)\n"
        "        shift\n"
        "        ;;\n"
        "    esac\n"
        "  done\n"
        "  mkdir -p \"${cache_dir}\"\n"
        "  cache_path=\"${cache_dir}/difficulty_bands_SWE_bench_SWE_smith_py_train_${probe_label}.json\"\n"
        "  cat >\"${cache_path}\" <<JSON\n"
        "{\n"
        "  \"task_count\": 1,\n"
        "  \"records\": [\n"
        "    {\n"
        "      \"task_id\": \"task-1\",\n"
        "      \"difficulty_band\": \"learnable\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "JSON\n"
        "  printf '%s\\n' \"${cache_path}\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"-\" ]]; then\n"
        f"  exec {shlex.quote(sys.executable)} \"$@\"\n"
        "fi\n"
        "exec python3 \"$@\"\n",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)
    return stub_path


def test_run_rft_script_dry_run_prints_verl_command() -> None:
    result = _run_script("run_rft.sh", "trainer.total_training_steps=1")
    assert "-m torch.distributed.run" in result.stdout
    assert "-m verl_integration.fsdp_sft_trainer_entry" in result.stdout
    assert "--config-name rft_swe" in result.stdout
    assert "trainer.total_training_steps=1" in result.stdout
    assert "actor_rollout_ref.model.path=" not in result.stdout
    assert "++data.apply_chat_template_kwargs.enable_thinking=false" in result.stdout
    assert "~data.apply_chat_template_kwargs.enable_thinking" not in result.stdout


def test_run_rft_script_direct_mode_default_overrides_hydra_resolve() -> None:
    if subprocess.run(
        [sys.executable, "-c", "import verl"],
        check=False,
        capture_output=True,
        text=True,
    ).returncode != 0:
        pytest.skip("verl is not installed in the active test environment")

    result = _run_script(
        "run_rft.sh",
        env_overrides={
            "RFT_RUNTIME_MODE": "direct",
            "NPROC_PER_NODE": "1",
        },
    )
    command_line = next(
        (line.strip() for line in result.stdout.splitlines() if "-m torch.distributed.run" in line),
        "",
    )
    assert command_line, f"unable to find direct-mode trainer command in output:\n{result.stdout}"

    preflight_result = _run_hydra_preflight_for_torchrun_command(command_line)
    assert preflight_result.returncode == 0, (
        "Hydra preflight failed for direct-mode RFT command.\n"
        f"Command: {command_line}\n"
        f"Stdout:\n{preflight_result.stdout}\n"
        f"Stderr:\n{preflight_result.stderr}"
    )


def test_run_rft_script_runtime_loop_trainer_overrides_hydra_resolve() -> None:
    if subprocess.run(
        [sys.executable, "-c", "import verl"],
        check=False,
        capture_output=True,
        text=True,
    ).returncode != 0:
        pytest.skip("verl is not installed in the active test environment")

    result = _run_script(
        "run_rft.sh",
        env_overrides={
            "NPROC_PER_NODE": "1",
        },
    )
    command_line = next(
        (
            line.strip()
            for line in result.stdout.splitlines()
            if "-m torch.distributed.run" in line and "--config-name rft_swe" in line
        ),
        "",
    )
    assert command_line, f"unable to find loop-mode trainer command in output:\n{result.stdout}"

    preflight_result = _run_hydra_preflight_for_torchrun_command(command_line)
    assert preflight_result.returncode == 0, (
        "Hydra preflight failed for runtime-loop RFT trainer command.\n"
        f"Command: {command_line}\n"
        f"Stdout:\n{preflight_result.stdout}\n"
        f"Stderr:\n{preflight_result.stderr}"
    )


def test_run_rft_script_dry_run_propagates_positive_stage_to_loop_runtime() -> None:
    result = _run_script(
        "run_rft.sh",
        env_overrides={"RFT_STAGE_NAME": "positive_rft"},
    )
    assert "stage_name=positive_rft" in result.stdout


def test_run_rft_script_dry_run_direct_mode_wires_positive_selection_overrides() -> None:
    result = _run_script(
        "run_rft.sh",
        env_overrides={
            "RFT_RUNTIME_MODE": "direct",
            "RFT_STAGE_NAME": "positive_rft",
            "NPROC_PER_NODE": "1",
        },
    )
    assert "+data.on_policy.stage_name=positive_rft" in result.stdout
    assert "+data.on_policy.runtime_overrides.verify_submissions=true" in result.stdout
    assert "+data.on_policy.rft_handoff_overrides.selection.require_resolved=true" in result.stdout
    assert "+data.on_policy.rft_handoff_overrides.selection.require_format_valid=false" in result.stdout


def test_run_rft_script_dry_run_direct_mode_propagates_task_holdout_settings() -> None:
    result = _run_script(
        "run_rft.sh",
        env_overrides={
            "RFT_RUNTIME_MODE": "direct",
            "RFT_EVAL_SPLIT_FRACTION": "0.25",
            "RFT_EVAL_MIN_ROWS": "2",
            "NPROC_PER_NODE": "1",
        },
    )
    assert "+data.on_policy.task_eval_split_fraction=0.25" in result.stdout
    assert "+data.on_policy.task_eval_min_rows=2" in result.stdout


def test_run_rft_script_dry_run_defaults_vllm_tp_dp_for_eight_gpus() -> None:
    expected_tp, expected_dp = config.resolve_rft_vllm_parallel_defaults(nproc_per_node=8)
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={"NPROC_PER_NODE": "8"},
    )
    assert f"--tensor-parallel-size {expected_tp}" in result.stdout
    assert f"--data-parallel-size {expected_dp}" in result.stdout
    assert "--gpu-memory-utilization 0.8" in result.stdout


def test_run_rft_script_dry_run_honors_centralized_default_dp_for_divisible_topology(
    tmp_path: Path,
) -> None:
    fake_python = _write_python_defaults_stub(
        tmp_path,
        (
            "100 8 64 32 1 1 512 0.1 1 2 2 "
            "http://127.0.0.1:8000/v1 "
            f"{config.DEFAULT_TRAINING_MODEL_NAME} 90 1024 0.0 1.0 8 12288 "
            "lora bf16 16 32 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
    )
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "PYTHON_BIN": str(fake_python),
        },
    )
    assert "--tensor-parallel-size 2" in result.stdout
    assert "--data-parallel-size 2" in result.stdout


def test_run_rft_script_dry_run_defaults_nproc_to_detected_gpu_count(
    tmp_path: Path,
) -> None:
    fake_python = _write_python_defaults_stub(
        tmp_path,
        (
            "100 8 64 32 1 1 512 0.1 1 2 4 "
            "http://127.0.0.1:8000/v1 "
            f"{config.DEFAULT_TRAINING_MODEL_NAME} 90 1024 0.0 1.0 8 12288 "
            "lora bf16 16 32 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
    )
    result = _run_script(
        "run_rft.sh",
        env_overrides={
            "PYTHON_BIN": str(fake_python),
            "STUB_GPU_COUNT": "8",
            "NPROC_PER_NODE": "",
        },
    )
    assert "--nproc_per_node 8" in result.stdout


def test_run_rft_script_dry_run_uses_centralized_collector_in_flight_default() -> None:
    runtime_defaults = config.rft_runtime_defaults()
    task_batch_size = int(runtime_defaults["loop"]["task_batch_size"])
    expected_in_flight = config.resolve_rft_collector_max_in_flight_default(task_batch_size=task_batch_size)
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
    )
    assert f"collector_max_in_flight_tasks={expected_in_flight}" in result.stdout


def test_run_rft_script_dry_run_allows_explicit_tp_override() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "RFT_VLLM_TP_SIZE": "8",
        },
    )
    assert "--tensor-parallel-size 8" in result.stdout
    assert "--data-parallel-size" not in result.stdout


def test_run_rft_script_dry_run_nondivisible_tp_override_falls_back_to_dp_one() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "RFT_VLLM_TP_SIZE": "3",
        },
    )
    assert "--tensor-parallel-size 3" in result.stdout
    assert "--data-parallel-size" not in result.stdout


def test_run_rft_script_dry_run_propagates_collector_in_flight_override() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "RFT_COLLECTOR_MAX_IN_FLIGHT_TASKS": "6",
        },
    )
    assert "collector_max_in_flight_tasks=6" in result.stdout


def test_run_rft_script_dry_run_propagates_collector_max_turns_override() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT": "16",
        },
    )
    assert "collector_max_turns_per_attempt=16" in result.stdout


def test_run_rft_script_dry_run_respects_explicit_vllm_extra_args() -> None:
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "RFT_VLLM_EXTRA_ARGS": "--tensor-parallel-size 2 --max-num-seqs 16",
        },
    )
    assert "--tensor-parallel-size 2" in result.stdout
    assert "--max-num-seqs 16" in result.stdout
    assert "--gpu-memory-utilization 0.8" not in result.stdout


def test_run_rft_script_dry_run_uses_centralized_sequence_length_overrides() -> None:
    expected_max_length = config.resolve_rft_handoff_settings().max_sequence_length
    result = _run_script(
        "run_rft.sh",
        "trainer.total_training_steps=1",
    )
    assert f"max_model_len={expected_max_length}" in result.stdout


def test_run_rft_script_dry_run_direct_mode_uses_centralized_runtime_overrides() -> None:
    on_policy_defaults = config.on_policy_runtime_defaults()
    expected_max_turns = on_policy_defaults["max_turns_per_attempt"]
    expected_max_length = config.resolve_rft_handoff_settings().max_sequence_length
    result = _run_script(
        "run_rft.sh",
        env_overrides={
            "NPROC_PER_NODE": "8",
            "RFT_RUNTIME_MODE": "direct",
            "RFT_COLLECTOR_MAX_TURNS_PER_ATTEMPT": "",
        },
    )
    assert (
        "actor_rollout_ref.model.lora.target_modules="
        "\\[q_proj\\,k_proj\\,v_proj\\,o_proj\\,gate_proj\\,up_proj\\,down_proj\\]"
    ) in result.stdout
    assert f"+data.on_policy.runtime_overrides.max_turns_per_attempt={expected_max_turns}" in result.stdout
    assert f"max_model_len={expected_max_length}" in result.stdout


def test_run_rft_script_non_dry_run_fails_fast_when_managed_vllm_port_is_occupied(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_rft.sh"
    python_stub = _write_python_run_rft_probe_stub(tmp_path)
    capture_path = tmp_path / "rft-loop-capture.txt"
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(python_stub),
            "NPROC_PER_NODE": "1",
            "SMALL_SWE_VLLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE": "0",
            "STUB_RFT_CAPTURE_FILE": str(capture_path),
        }
    )

    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            env=env,
            timeout=30,
        )
    finally:
        listener.close()

    assert result.returncode != 0
    assert "Managed vLLM launch target is already in use" in result.stderr
    assert not capture_path.exists()


def test_run_rft_script_non_dry_run_allows_occupied_vllm_port_when_management_is_disabled(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_rft.sh"
    python_stub = _write_python_run_rft_probe_stub(tmp_path)
    capture_path = tmp_path / "rft-loop-capture.txt"
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(python_stub),
            "NPROC_PER_NODE": "1",
            "RFT_MANAGE_VLLM": "0",
            "SMALL_SWE_VLLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE": "0",
            "STUB_RFT_CAPTURE_FILE": str(capture_path),
        }
    )

    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            env=env,
            timeout=30,
        )
    finally:
        listener.close()

    assert result.returncode == 0, result.stderr
    captured_args = capture_path.read_text(encoding="utf-8")
    assert "trainer.rft_runtime_loop" in captured_args
    assert "--skip-vllm-management" in captured_args


def test_teacher_reprompt_pilot_slurm_script_dry_run_uses_all_visible_gpus_for_tp() -> None:
    result = _run_script(
        "run_teacher_reprompt_pilot_slurm.sh",
        env_overrides={
            "SLURM_GPUS_ON_NODE": "8",
            "PILOT_MODEL_PATH": "/tmp/nonexistent-model-ok-for-dry-run",
        },
    )
    assert "--tensor-parallel-size 8" in result.stdout
    assert "--max-in-flight-tasks 128" in result.stdout


def test_teacher_reprompt_pilot_slurm_script_dry_run_accepts_fixed_kv_cache_overrides() -> None:
    result = _run_script(
        "run_teacher_reprompt_pilot_slurm.sh",
        env_overrides={
            "SLURM_GPUS_ON_NODE": "8",
            "PILOT_MODEL_PATH": "/tmp/nonexistent-model-ok-for-dry-run",
            "PILOT_VLLM_KV_CACHE_MEMORY_BYTES": "17179869184",
            "PILOT_VLLM_NUM_GPU_BLOCKS_OVERRIDE": "8192",
        },
    )
    assert "--kv-cache-memory-bytes 17179869184" in result.stdout
    assert "--num-gpu-blocks-override 8192" in result.stdout


def test_teacher_reprompt_pilot_slurm_script_dry_run_succeeds_without_user_env() -> None:
    script_path = _repo_root() / "scripts" / "run_teacher_reprompt_pilot_slurm.sh"
    env = {"PATH": os.environ["PATH"], "SLURM_GPUS_ON_NODE": "8", "PILOT_MODEL_PATH": "/tmp/nonexistent-model-ok-for-dry-run"}
    result = subprocess.run(
        ["bash", str(script_path), "--dry-run"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--tensor-parallel-size 8" in result.stdout
    assert "--max-in-flight-tasks 128" in result.stdout


def test_teacher_reprompt_pilot_slurm_script_dry_run_normalizes_negative_index_to_dynamic_middle() -> None:
    result = _run_script(
        "run_teacher_reprompt_pilot_slurm.sh",
        env_overrides={
            "SLURM_GPUS_ON_NODE": "8",
            "PILOT_MODEL_PATH": "/tmp/nonexistent-model-ok-for-dry-run",
            "PILOT_TEACHER_TURN_INDEX": "-1",
            "PILOT_TEACHER_TURN_INDEX_MODE": "fixed",
        },
    )
    assert "--teacher-reprompt-turn-index -1" in result.stdout
    assert "--teacher-reprompt-turn-index-mode dynamic_middle" in result.stdout


def test_teacher_reprompt_pilot_slurm_script_dry_run_uses_larger_default_task_batch_size() -> None:
    result = _run_script(
        "run_teacher_reprompt_pilot_slurm.sh",
        env_overrides={
            "SLURM_GPUS_ON_NODE": "8",
            "PILOT_MODEL_PATH": "/tmp/nonexistent-model-ok-for-dry-run",
        },
    )
    assert "--task-batch-size 1024" in result.stdout
    assert "--attempts-per-task 4" in result.stdout


def test_teacher_reprompt_pilot_slurm_script_dry_run_load_latest_rft_checkpoint_overrides_model_everywhere(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    checkpoint_path = tmp_path / "resolved-rft-checkpoint"
    checkpoint_path.mkdir(parents=True)

    run_dir = (
        repo_root
        / "outputs"
        / "slurm"
        / "rft_runtime"
        / f"pytest-teacher-pilot-manifest-{os.getpid()}-{time.time_ns()}"
    )
    manifest_path = run_dir / "rft_runtime_loop_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"final_model_path": str(checkpoint_path)}), encoding="utf-8")
    future_epoch = 4_102_444_800  # 2100-01-01T00:00:00Z
    os.utime(manifest_path, (future_epoch, future_epoch))

    try:
        result = _run_script(
            "run_teacher_reprompt_pilot_slurm.sh",
            "--load-latest-rft-checkpoint",
            env_overrides={
                "SLURM_GPUS_ON_NODE": "8",
                "PILOT_MODEL_PATH": "/tmp/nonexistent-model-ok-for-dry-run",
                "PILOT_SERVED_MODEL": "Qwen/Qwen3-4B-Instruct-2507",
            },
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    assert f"--model {checkpoint_path}" in result.stdout
    assert f"--served-model-name {checkpoint_path}" in result.stdout
    assert f"--rft-checkpoint {checkpoint_path}" in result.stdout


def test_teacher_reprompt_pilot_slurm_script_dry_run_accepts_hf_repo_id_via_pilot_model_path() -> None:
    result = _run_script(
        "run_teacher_reprompt_pilot_slurm.sh",
        env_overrides={
            "SLURM_GPUS_ON_NODE": "2",
            "PILOT_MODEL_PATH": "Qwen/Qwen3.5-9B",
            "PILOT_SERVED_MODEL": "Qwen/Qwen3.5-9B",
        },
    )
    assert "--model Qwen/Qwen3.5-9B" in result.stdout
    assert "--served-model-name Qwen/Qwen3.5-9B" in result.stdout
    assert "--rft-checkpoint Qwen/Qwen3.5-9B" not in result.stdout


def test_teacher_reprompt_pilot_slurm_script_non_dry_run_accepts_hf_repo_id_via_pilot_model_path(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_teacher_reprompt_pilot_slurm.sh"
    python_stub = _write_teacher_pilot_python_stub(tmp_path)
    capture_path = tmp_path / "teacher-pilot-args.txt"
    output_dir = repo_root / "outputs" / "teacher_reprompt_pilot" / "job987654"
    vllm_log = repo_root / "outputs" / "slurm" / "teacher-pilot-vllm-987654.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(python_stub),
            "SLURM_GPUS_ON_NODE": "2",
            "SLURM_JOB_ID": "987654",
            "TEACHER_PILOT_CAPTURE": str(capture_path),
            "HF_HOME": str(tmp_path / "hf_home"),
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf_home" / "hub"),
            "TRANSFORMERS_CACHE": str(tmp_path / "hf_home" / "transformers"),
            "VLLM_CACHE_ROOT": str(tmp_path / "vllm_cache"),
            "TORCH_HOME": str(tmp_path / "torch_home"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
        }
    )

    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            env={
                **env,
                "PILOT_MODEL_PATH": "Qwen/Qwen3.5-9B",
                "PILOT_SERVED_MODEL": "Qwen/Qwen3.5-9B",
            },
            timeout=30,
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        if vllm_log.exists():
            vllm_log.unlink()

    assert result.returncode == 0, result.stderr
    captured_args = capture_path.read_text(encoding="utf-8").splitlines()
    assert any(item.endswith("scripts/run_teacher_reprompt_pilot.py") for item in captured_args)
    assert "--rft-checkpoint" not in captured_args


def test_teacher_reprompt_pilot_slurm_script_non_dry_run_preflight_sweeps_stale_managed_containers(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_teacher_reprompt_pilot_slurm.sh"
    python_stub = _write_teacher_pilot_python_stub(tmp_path)
    _write_pilot_docker_cleanup_probe_stub(tmp_path)
    _write_squeue_probe_stub(tmp_path)
    capture_path = tmp_path / "teacher-pilot-args.txt"
    docker_log_path = tmp_path / "docker-invocations.log"
    output_dir = repo_root / "outputs" / "teacher_reprompt_pilot" / "job987654"
    vllm_log = repo_root / "outputs" / "slurm" / "teacher-pilot-vllm-987654.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}{os.pathsep}{env.get('PATH', '')}",
            "FAKE_DOCKER_LOG_FILE": str(docker_log_path),
            "PYTHON_BIN": str(python_stub),
            "SLURM_GPUS_ON_NODE": "2",
            "SLURM_JOB_ID": "987654",
            "TEACHER_PILOT_CAPTURE": str(capture_path),
            "HF_HOME": str(tmp_path / "hf_home"),
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf_home" / "hub"),
            "TRANSFORMERS_CACHE": str(tmp_path / "hf_home" / "transformers"),
            "VLLM_CACHE_ROOT": str(tmp_path / "vllm_cache"),
            "TORCH_HOME": str(tmp_path / "torch_home"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
            "PILOT_MODEL_PATH": "Qwen/Qwen3.5-9B",
            "PILOT_SERVED_MODEL": "Qwen/Qwen3.5-9B",
        }
    )

    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            env=env,
            timeout=30,
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        if vllm_log.exists():
            vllm_log.unlink()

    assert result.returncode == 0, result.stderr
    docker_invocations = docker_log_path.read_text(encoding="utf-8").splitlines()
    assert any("label=small_swe.pool_name=onpolicy-task" in line for line in docker_invocations)
    assert any("rm -f stale-container-1 stale-container-2" in line for line in docker_invocations)
    assert not any("rm -f live-container" in line for line in docker_invocations)


def test_teacher_reprompt_pilot_slurm_script_non_dry_run_rejects_hf_repo_id_via_rft_checkpoint(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_teacher_reprompt_pilot_slurm.sh"
    python_stub = _write_teacher_pilot_python_stub(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(python_stub),
            "SLURM_GPUS_ON_NODE": "2",
            "HF_HOME": str(tmp_path / "hf_home"),
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf_home" / "hub"),
            "TRANSFORMERS_CACHE": str(tmp_path / "hf_home" / "transformers"),
            "VLLM_CACHE_ROOT": str(tmp_path / "vllm_cache"),
            "TORCH_HOME": str(tmp_path / "torch_home"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
        }
    )

    result = subprocess.run(
        ["bash", str(script_path), "--rft-checkpoint", "Qwen/Qwen3.5-9B"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )

    assert result.returncode != 0
    assert "Checkpoint path does not exist" in result.stderr


def test_onpolicy_difficulty_probe_slurm_script_dry_run_uses_visible_gpus_for_tp() -> None:
    result = _run_script(
        "run_onpolicy_difficulty_probe_slurm.sh",
        env_overrides={
            "SLURM_GPUS_ON_NODE": "2",
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
        },
    )
    assert "--tensor-parallel-size 2" in result.stdout
    assert "--initial-model Qwen/Qwen3.5-9B" in result.stdout
    assert "--attempts-per-task 4" in result.stdout
    assert "--task-batch-size 1024" in result.stdout
    assert "--env-pool-size 128" in result.stdout
    assert "--max-in-flight-tasks 128" in result.stdout


def test_onpolicy_difficulty_probe_slurm_script_dry_run_passes_parallel_probe_overrides() -> None:
    result = _run_script(
        "run_onpolicy_difficulty_probe_slurm.sh",
        env_overrides={
            "SLURM_GPUS_ON_NODE": "8",
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "PROBE_VLLM_TP_SIZE": "2",
            "PROBE_VLLM_DP_SIZE": "4",
            "PROBE_TASK_BATCH_SIZE": "4",
            "PROBE_ENV_POOL_SIZE": "4",
            "PROBE_MAX_IN_FLIGHT_TASKS": "4",
        },
    )
    assert "--tensor-parallel-size 2" in result.stdout
    assert "--data-parallel-size 4" in result.stdout
    assert "--task-batch-size 4" in result.stdout
    assert "--env-pool-size 4" in result.stdout
    assert "--max-in-flight-tasks 4" in result.stdout


def test_onpolicy_difficulty_probe_slurm_script_dry_run_derives_tp_from_dp() -> None:
    result = _run_script(
        "run_onpolicy_difficulty_probe_slurm.sh",
        env_overrides={
            "SLURM_GPUS_ON_NODE": "8",
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "PROBE_VLLM_DP_SIZE": "4",
        },
    )
    assert "--tensor-parallel-size 2" in result.stdout
    assert "--data-parallel-size 4" in result.stdout


def test_onpolicy_difficulty_probe_slurm_script_rejects_tp_larger_than_visible_gpus() -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_onpolicy_difficulty_probe_slurm.sh"
    result = subprocess.run(
        ["bash", str(script_path), "--dry-run"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "SLURM_GPUS_ON_NODE": "8",
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "PROBE_VLLM_TP_SIZE": "9",
        },
    )

    assert result.returncode != 0
    assert "Requested PROBE_VLLM_TP_SIZE=9 exceeds visible GPU count 8." in result.stderr


def test_onpolicy_difficulty_probe_slurm_script_dry_run_uses_overridden_base_url_port() -> None:
    result = _run_script(
        "run_onpolicy_difficulty_probe_slurm.sh",
        env_overrides={
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "SMALL_SWE_VLLM_BASE_URL": "http://127.0.0.1:19191",
        },
    )
    assert "--port 19191" in result.stdout


def test_onpolicy_difficulty_probe_slurm_script_non_dry_run_materializes_cache(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_onpolicy_difficulty_probe_slurm.sh"
    python_stub = _write_onpolicy_difficulty_probe_python_stub(tmp_path)
    capture_path = tmp_path / "difficulty-probe-args.txt"
    cache_dir = tmp_path / "difficulty-cache"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(python_stub),
            "SLURM_GPUS_ON_NODE": "2",
            "SLURM_JOB_ID": "24680",
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "PROBE_CACHE_DIR": str(cache_dir),
            "DIFFICULTY_PROBE_CAPTURE": str(capture_path),
            "HF_HOME": str(tmp_path / "hf_home"),
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf_home" / "hub"),
            "TRANSFORMERS_CACHE": str(tmp_path / "hf_home" / "transformers"),
            "VLLM_CACHE_ROOT": str(tmp_path / "vllm_cache"),
            "TORCH_HOME": str(tmp_path / "torch_home"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
            "SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    captured_args = capture_path.read_text(encoding="utf-8").splitlines()
    assert captured_args[:2] == ["-m", "env.preload_onpolicy_difficulty_bands"]
    cache_path = Path(result.stdout.strip())
    assert cache_path.is_file()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["task_count"] == 1
    assert payload["records"][0]["task_id"] == "task-1"


def test_onpolicy_difficulty_probe_slurm_script_non_dry_run_preflight_sweeps_stale_managed_containers(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_onpolicy_difficulty_probe_slurm.sh"
    python_stub = _write_onpolicy_difficulty_probe_python_stub(tmp_path)
    _write_pilot_docker_cleanup_probe_stub(tmp_path)
    _write_squeue_probe_stub(tmp_path)
    capture_path = tmp_path / "difficulty-probe-args.txt"
    docker_log_path = tmp_path / "docker-invocations.log"
    cache_dir = tmp_path / "difficulty-cache-preflight"
    vllm_log = repo_root / "outputs" / "slurm" / "difficulty-probe-vllm-987654.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}{os.pathsep}{env.get('PATH', '')}",
            "FAKE_DOCKER_LOG_FILE": str(docker_log_path),
            "PYTHON_BIN": str(python_stub),
            "SLURM_GPUS_ON_NODE": "2",
            "SLURM_JOB_ID": "987654",
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "PROBE_CACHE_DIR": str(cache_dir),
            "DIFFICULTY_PROBE_CAPTURE": str(capture_path),
            "HF_HOME": str(tmp_path / "hf_home"),
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf_home" / "hub"),
            "TRANSFORMERS_CACHE": str(tmp_path / "hf_home" / "transformers"),
            "VLLM_CACHE_ROOT": str(tmp_path / "vllm_cache"),
            "TORCH_HOME": str(tmp_path / "torch_home"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
        }
    )

    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            env=env,
            timeout=30,
        )
    finally:
        if vllm_log.exists():
            vllm_log.unlink()

    assert result.returncode == 0, result.stderr
    docker_invocations = docker_log_path.read_text(encoding="utf-8").splitlines()
    assert any("label=small_swe.pool_name=onpolicy-task" in line for line in docker_invocations)
    assert any("rm -f stale-container-1 stale-container-2" in line for line in docker_invocations)
    assert not any("rm -f live-container" in line for line in docker_invocations)


def test_onpolicy_difficulty_probe_slurm_script_accepts_models_endpoint_base_url(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_onpolicy_difficulty_probe_slurm.sh"
    python_stub = _write_onpolicy_difficulty_probe_python_stub(tmp_path)
    capture_path = tmp_path / "difficulty-probe-args-models-url.txt"
    cache_dir = tmp_path / "difficulty-cache-models-url"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(python_stub),
            "SLURM_GPUS_ON_NODE": "2",
            "SLURM_JOB_ID": "24681",
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "PROBE_CACHE_DIR": str(cache_dir),
            "DIFFICULTY_PROBE_CAPTURE": str(capture_path),
            "HF_HOME": str(tmp_path / "hf_home"),
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf_home" / "hub"),
            "TRANSFORMERS_CACHE": str(tmp_path / "hf_home" / "transformers"),
            "VLLM_CACHE_ROOT": str(tmp_path / "vllm_cache"),
            "TORCH_HOME": str(tmp_path / "torch_home"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
            "SMALL_SWE_VLLM_BASE_URL": "http://127.0.0.1:19191/v1/models",
            "SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    cache_path = Path(result.stdout.strip())
    assert cache_path.is_file()


def test_onpolicy_difficulty_probe_slurm_script_fails_fast_when_vllm_exits_early(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_onpolicy_difficulty_probe_slurm.sh"
    python_stub = _write_onpolicy_difficulty_probe_python_stub(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(python_stub),
            "SLURM_GPUS_ON_NODE": "2",
            "SLURM_JOB_ID": "24682",
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "PROBE_CACHE_DIR": str(tmp_path / "difficulty-cache-fail-fast"),
            "DIFFICULTY_PROBE_CAPTURE": str(tmp_path / "difficulty-probe-capture.txt"),
            "HF_HOME": str(tmp_path / "hf_home"),
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "hf_home" / "hub"),
            "TRANSFORMERS_CACHE": str(tmp_path / "hf_home" / "transformers"),
            "VLLM_CACHE_ROOT": str(tmp_path / "vllm_cache"),
            "TORCH_HOME": str(tmp_path / "torch_home"),
            "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
            "SMALL_SWE_PREFLIGHT_CONTAINER_SWEEP_ENABLE": "0",
            "PROBE_VLLM_READY_TIMEOUT_SEC": "30",
            "STUB_VLLM_FAIL_FAST": "1",
        }
    )

    start = time.monotonic()
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
    )
    elapsed = time.monotonic() - start

    assert result.returncode != 0
    assert elapsed < 10
    assert "vLLM process exited before readiness probe succeeded." in result.stderr
    assert "stub vLLM startup failure" in result.stderr


def test_onpolicy_difficulty_probe_slurm_script_rejects_non_local_base_url() -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_onpolicy_difficulty_probe_slurm.sh"
    result = subprocess.run(
        ["bash", str(script_path), "--dry-run"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "SMALL_SWE_VLLM_BASE_URL": "http://example.com:19191/v1/models",
        },
        timeout=30,
    )

    assert result.returncode != 0
    assert "must point at a local loopback host" in result.stderr


def test_onpolicy_difficulty_probe_slurm_script_rejects_https_base_url() -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_onpolicy_difficulty_probe_slurm.sh"
    result = subprocess.run(
        ["bash", str(script_path), "--dry-run"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "SMALL_SWE_VLLM_BASE_URL": "https://127.0.0.1:19191/v1",
        },
        timeout=30,
    )

    assert result.returncode != 0
    assert "Invalid SMALL_SWE_VLLM_BASE_URL" in result.stderr


def test_onpolicy_difficulty_probe_slurm_script_rejects_prefixed_local_base_url() -> None:
    repo_root = _repo_root()
    script_path = repo_root / "scripts" / "run_onpolicy_difficulty_probe_slurm.sh"
    result = subprocess.run(
        ["bash", str(script_path), "--dry-run"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PROBE_INITIAL_MODEL": "Qwen/Qwen3.5-9B",
            "SMALL_SWE_VLLM_BASE_URL": "http://127.0.0.1:19191/foo/v1",
        },
        timeout=30,
    )

    assert result.returncode != 0
    assert "must be empty or point at the local /v1" in result.stderr


def test_run_sdpo_script_dry_run_prints_sdpo_config() -> None:
    result = _run_script("run_sdpo.sh", "data.train_batch_size=4")
    assert "-m verl_integration.main_ppo_entry" in result.stdout
    assert "--config-name sdpo_swe" in result.stdout
    assert "trainer.logger=\\[console\\,wandb\\,file\\]" in result.stdout
    assert "actor_rollout_ref.model.path=/tmp/rft-checkpoint" in result.stdout
    assert "data.filter_overlong_prompts=false" in result.stdout
    assert "data.train_files=" in result.stdout
    assert "data.val_files=" in result.stdout
    assert "data.train_batch_size=4" in result.stdout


def test_run_sdpo_script_dry_run_manifest_resolution_prefers_existing_candidate(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "trainer_checkpoints" / "global_step_42"
    existing_hf = checkpoint_root / "huggingface"
    existing_hf.mkdir(parents=True)
    missing_merged = checkpoint_root / "huggingface_vllm_merged"
    manifest_path = tmp_path / "rft_runtime_loop_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "final_model_path": str(missing_merged),
                "latest_vllm_checkpoint": str(missing_merged),
                "latest_hf_checkpoint": str(existing_hf),
            }
        ),
        encoding="utf-8",
    )

    result = _run_script(
        "run_sdpo.sh",
        env_overrides={
            "SDPO_RFT_CHECKPOINT": "",
            "SDPO_RFT_MANIFEST": str(manifest_path),
        },
    )
    assert f"actor_rollout_ref.model.path={existing_hf}" in result.stdout


def test_run_sdpo_script_dry_run_discovers_manifest_from_slurm_rft_runtime(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    checkpoint_path = tmp_path / "resolved-rft-checkpoint"
    checkpoint_path.mkdir(parents=True)

    run_dir = (
        repo_root
        / "outputs"
        / "slurm"
        / "rft_runtime"
        / f"pytest-run-script-manifest-{os.getpid()}-{time.time_ns()}"
    )
    manifest_path = run_dir / "rft_runtime_loop_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"final_model_path": str(checkpoint_path)}),
        encoding="utf-8",
    )

    future_epoch = 4_102_444_800  # 2100-01-01T00:00:00Z
    os.utime(manifest_path, (future_epoch, future_epoch))

    try:
        result = _run_script(
            "run_sdpo.sh",
            env_overrides={
                "SDPO_RFT_CHECKPOINT": "",
                "SDPO_RFT_MANIFEST": "",
            },
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    assert f"actor_rollout_ref.model.path={checkpoint_path}" in result.stdout


def test_run_sdpo_script_dry_run_allows_entrypoint_override() -> None:
    result = _run_script(
        "run_sdpo.sh",
        env_overrides={"SDPO_TRAINER_MODULE": "verl.trainer.main_ppo"},
    )
    assert "-m verl.trainer.main_ppo" in result.stdout


def test_run_sdpo_script_defaults_experiment_with_slurm_job_suffix(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_env_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SLURM_JOB_ID": "4242",
            "SDPO_RUN_TIMESTAMP": "20260226T123456Z",
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_PRELOADED_TASK_PARQUET": str(fake_parquet),
            "SDPO_TRAINER_MODULE": "dummy.module",
        }
    )
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "EXPERIMENT=small-swe-sdpo_20260226T123456Z_job4242" in result.stdout
    assert "TASK=small-swe-sdpo" in result.stdout
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1" in result.stdout
    assert "TVM_FFI_DISABLE_TORCH_C_DLPACK=1" in result.stdout


def test_run_sdpo_script_allows_disabling_ray_noset_visible_devices(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_env_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_PRELOADED_TASK_PARQUET": str(fake_parquet),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_RAY_FORCE_NOSET_VISIBLE_DEVICES": "0",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "",
        }
    )
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=" in result.stdout
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1" not in result.stdout


def test_run_sdpo_script_unsets_rocr_visible_devices_when_cuda_visible(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_env_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_PRELOADED_TASK_PARQUET": str(fake_parquet),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "ROCR_VISIBLE_DEVICES": "0,1",
        }
    )
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "CUDA_VISIBLE_DEVICES=0,1" in result.stdout
    assert "ROCR_VISIBLE_DEVICES=" in result.stdout
    assert "ROCR_VISIBLE_DEVICES=0,1" not in result.stdout


def test_run_sdpo_script_allows_disabling_watchdog_monitor(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_env_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_PRELOADED_TASK_PARQUET": str(fake_parquet),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "run_sdpo.sh watchdog: disabled (SDPO_MONITOR_ENABLE=0)" in result.stdout


def test_run_sdpo_script_skips_wandb_repair_when_run_id_unresolved(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 1, "data": {"training/global_step": 1}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("trainer started\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run ",
            "WANDB_RUN_ID": "",
            "SDPO_WANDB_RUN_ID": "",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "unable to resolve run id from trainer log/env; skipping." in result.stdout
    assert not repair_log_path.exists()


def test_run_sdpo_script_wandb_repair_propagates_trainer_exit_code(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
            "STUB_TRAINER_EXIT_CODE": "17",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
        ],
        cwd=_repo_root(),
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 17
    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[0] == "-"
    assert repair_argv[1] == "testRun123"
    assert repair_argv[-1] == "17"


def test_run_sdpo_script_wandb_repair_parses_non_alnum_run_id_from_trainer_log(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run run-with_dash_123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run run-with_dash_123",
            "WANDB_RUN_ID": "",
            "SDPO_WANDB_RUN_ID": "",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[1] == "run-with_dash_123"


def test_run_sdpo_script_wandb_repair_captures_run_id_when_monitor_disabled(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run monitorOff123",
            "WANDB_RUN_ID": "",
            "SDPO_WANDB_RUN_ID": "",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[1] == "monitorOff123"
    assert "wandb: setting up run monitorOff123" in trainer_log_path.read_text(encoding="utf-8")


def test_run_sdpo_script_wandb_repair_uses_env_run_id_without_log_banner(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "WANDB_RUN_ID": "envRun123",
            "SDPO_WANDB_RUN_ID": "",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[1] == "envRun123"


def test_run_sdpo_script_wandb_repair_uses_trainer_project_override(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
            "trainer.project_name=custom-sdpo-project",
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "custom-sdpo-project"


def test_run_sdpo_script_wandb_repair_uses_double_plus_trainer_project_override(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
            "++trainer.project_name=custom-plusplus-project",
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "custom-plusplus-project"


def test_run_sdpo_script_wandb_repair_uses_last_trainer_project_override(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
            "trainer.project_name=first-project",
            "trainer.project_name=second-project",
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "second-project"


def test_run_sdpo_script_wandb_repair_defaults_to_config_project_name(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    broken_python3 = tmp_path / "python3"
    broken_python3.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == \"-\" ]]; then\n"
        "  payload=\"$(cat)\"\n"
        "  if [[ \"${payload}\" == *\"trainer_indent = None\"* ]]; then\n"
        "    exit 127\n"
        "  fi\n"
        f"  printf '%s' \"${{payload}}\" | {shlex.quote(sys.executable)} \"$@\"\n"
        "  exit $?\n"
        "fi\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    broken_python3.chmod(0o755)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
            "SDPO_TASK_NAME": "custom-task-name",
            "SDPO_WANDB_PROJECT_NAME": "",
            "SDPO_WANDB_PROJECT": "",
            "WANDB_PROJECT": "",
            "PATH": f"{tmp_path}{os.pathsep}{env.get('PATH', '')}",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "small-swe-sdpo"


def test_run_sdpo_script_wandb_repair_prefers_config_project_over_wandb_project_env(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
            "WANDB_PROJECT": "global-wandb-project",
            "SDPO_WANDB_PROJECT_NAME": "",
            "SDPO_WANDB_PROJECT": "",
            "SDPO_TASK_NAME": "custom-task-name",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "small-swe-sdpo"


def test_run_sdpo_script_wandb_repair_uses_project_from_overridden_hydra_config(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    custom_config_dir = tmp_path / "custom-configs"
    custom_config_dir.mkdir()
    (custom_config_dir / "custom_sdpo.yaml").write_text(
        "trainer:\n  project_name: custom-config-project\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
            "WANDB_PROJECT": "global-wandb-project",
            "SDPO_WANDB_PROJECT_NAME": "",
            "SDPO_WANDB_PROJECT": "",
            "SDPO_TASK_NAME": "custom-task-name",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
            "--config-name",
            "custom_sdpo",
            "--config-dir",
            str(custom_config_dir),
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "custom-config-project"


def test_run_sdpo_script_wandb_repair_config_parser_ignores_nested_project_name(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    custom_config_dir = tmp_path / "custom-configs-nested"
    custom_config_dir.mkdir()
    (custom_config_dir / "nested_project.yaml").write_text(
        "trainer:\n"
        "  nested_block:\n"
        "    project_name: nested-project-name\n"
        "  project_name: direct-project-name\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
            "WANDB_PROJECT": "global-wandb-project",
            "SDPO_WANDB_PROJECT_NAME": "",
            "SDPO_WANDB_PROJECT": "",
            "SDPO_TASK_NAME": "custom-task-name",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
            "--config-name",
            "nested_project",
            "--config-dir",
            str(custom_config_dir),
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "direct-project-name"


def test_run_sdpo_script_wandb_repair_config_parser_accepts_trainer_header_comment(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    custom_config_dir = tmp_path / "custom-configs-commented"
    custom_config_dir.mkdir()
    (custom_config_dir / "commented_header.yaml").write_text(
        "trainer: # inline comment\n"
        "  project_name: commented-header-project\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
            "WANDB_PROJECT": "global-wandb-project",
            "SDPO_WANDB_PROJECT_NAME": "",
            "SDPO_WANDB_PROJECT": "",
            "SDPO_TASK_NAME": "custom-task-name",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
            "--config-name",
            "commented_header",
            "--config-dir",
            str(custom_config_dir),
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "commented-header-project"


def test_run_sdpo_script_wandb_repair_resolves_project_from_defaults_tree(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run defaultsTree123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    custom_config_dir = tmp_path / "custom-configs-defaults"
    (custom_config_dir / "trainer").mkdir(parents=True)
    (custom_config_dir / "defaults_tree.yaml").write_text(
        "defaults:\n  - trainer: replay_project\n",
        encoding="utf-8",
    )
    (custom_config_dir / "trainer" / "replay_project.yaml").write_text(
        "project_name: defaults-tree-project\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run defaultsTree123",
            "WANDB_PROJECT": "global-wandb-project",
            "SDPO_WANDB_PROJECT_NAME": "",
            "SDPO_WANDB_PROJECT": "",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
            "--config-name",
            "defaults_tree",
            "--config-dir",
            str(custom_config_dir),
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "defaults-tree-project"


def test_run_sdpo_script_wandb_repair_skips_when_trainer_launch_never_starts(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "2",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "WANDB_RUN_ID": "staleRun123",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
        ],
        cwd=_repo_root(),
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert not repair_log_path.exists()


def test_run_sdpo_script_wandb_repair_skips_on_early_trainer_startup_failure(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run staleOldRun\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_EXIT_CODE": "17",
            "WANDB_RUN_ID": "staleRun123",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
        ],
        cwd=_repo_root(),
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 17
    assert "trainer launch never started; skipping." in result.stdout
    assert not repair_log_path.exists()


def test_run_sdpo_script_wandb_repair_resolves_oc_env_project_interpolation(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run testRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"
    custom_config_dir = tmp_path / "custom-configs-oc-env"
    custom_config_dir.mkdir()
    (custom_config_dir / "oc_env_project.yaml").write_text(
        "trainer:\n  project_name: ${oc.env:SDPO_TEST_WANDB_PROJECT,config-fallback-project}\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run testRun123",
            "SDPO_TEST_WANDB_PROJECT": "resolved-oc-env-project",
            "WANDB_PROJECT": "",
            "SDPO_WANDB_PROJECT_NAME": "",
            "SDPO_WANDB_PROJECT": "",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
            "--config-name",
            "oc_env_project",
            "--config-dir",
            str(custom_config_dir),
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "resolved-oc-env-project"


def test_run_sdpo_script_wandb_repair_resolves_simple_hydra_project_interpolation(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run interpRef123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"
    custom_config_dir = tmp_path / "custom-configs-interp-ref"
    custom_config_dir.mkdir()
    (custom_config_dir / "interp_ref.yaml").write_text(
        "project_name: interpolated-project\n"
        "trainer:\n"
        "  project_name: ${project_name}\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SDPO_CLEANUP_ON_EXIT": "0",
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "STUB_TRAINER_STDOUT": "wandb: setting up run interpRef123",
            "WANDB_PROJECT": "global-wandb-project",
            "SDPO_WANDB_PROJECT_NAME": "",
            "SDPO_WANDB_PROJECT": "",
        }
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            f"data.train_files={fake_parquet}",
            f"data.val_files={fake_parquet}",
            "--config-name",
            "interp_ref",
            "--config-dir",
            str(custom_config_dir),
        ],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    logged_invocations = repair_log_path.read_text(encoding="utf-8").splitlines()
    assert len(logged_invocations) == 1
    repair_argv = shlex.split(logged_invocations[0])
    assert repair_argv[3] == "interpolated-project"


def test_run_sdpo_script_dry_run_does_not_trigger_wandb_repair(tmp_path: Path) -> None:
    fake_python = _write_python_wandb_repair_probe_stub(tmp_path)
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 2, "data": {"training/global_step": 2, "_step": 2}}) + "\n",
        encoding="utf-8",
    )
    trainer_log_path = tmp_path / "trainer.log"
    trainer_log_path.write_text("wandb: setting up run dryRun123\n", encoding="utf-8")
    repair_log_path = tmp_path / "repair-calls.log"

    _run_script(
        "run_sdpo.sh",
        env_overrides={
            "PYTHON_BIN": str(fake_python),
            "VERL_FILE_LOGGER_PATH": str(metrics_path),
            "SDPO_TRAINER_LOG_PATH": str(trainer_log_path),
            "STUB_REPAIR_LOG_FILE": str(repair_log_path),
            "SDPO_WANDB_RUN_ID": "dryRun123",
        },
    )

    assert not repair_log_path.exists()


def test_run_sdpo_script_cleanup_falls_back_to_run_label_when_slurm_label_lookup_empty(
    tmp_path: Path,
) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_env_probe_stub(tmp_path)
    _write_docker_cleanup_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    docker_log_path = tmp_path / "docker-invocations.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "PATH": f"{tmp_path}{os.pathsep}{env.get('PATH', '')}",
            "FAKE_DOCKER_LOG_FILE": str(docker_log_path),
            "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
            "SDPO_PRELOADED_TASK_PARQUET": str(fake_parquet),
            "SDPO_TRAINER_MODULE": "dummy.module",
            "SDPO_MONITOR_ENABLE": "0",
            "SLURM_JOB_ID": "",
            "SLURM_JOBID": "4242",
            "SDPO_RUN_LABEL": "run-fallback-4242",
        }
    )
    subprocess.run(
        ["bash", str(script_path)],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    invocations = docker_log_path.read_text(encoding="utf-8").splitlines()
    assert any("label=small_swe.slurm_job_id=4242" in line for line in invocations)
    assert any("label=small_swe.run_label=run-fallback-4242" in line for line in invocations)
    assert not any("name=sdpo-swe-bridge-" in line for line in invocations)
    assert any("rm -f container-from-run-label" in line for line in invocations)


def test_run_sdpo_script_cleanup_revalidates_slurm_job_before_signaling(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_cleanup_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    fake_proc_root = tmp_path / "fake-proc"
    fake_proc_root.mkdir()
    cache_dir = tmp_path / "cache-dir"
    cache_dir.mkdir()

    ray_proc = subprocess.Popen(
        ["bash", "-lc", "exec -a raylet sleep 120"],
        env={
            **os.environ,
            "SLURM_JOB_ID": "4242",
        },
    )

    pid_dir = fake_proc_root / str(ray_proc.pid)
    pid_dir.mkdir(parents=True)
    environ_path = pid_dir / "environ"
    environ_path.write_bytes(b"SLURM_JOB_ID=4242\0")

    try:
        def _flip_job_id_after_drain_start() -> None:
            time.sleep(0.5)
            environ_path.write_bytes(b"SLURM_JOB_ID=9999\0")

        mutator = threading.Thread(target=_flip_job_id_after_drain_start, daemon=True)
        mutator.start()

        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": str(fake_python),
                "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
                "SDPO_TRAINER_MODULE": "dummy.module",
                "SDPO_MONITOR_ENABLE": "0",
                "SDPO_WANDB_REPAIR_ON_EXIT": "0",
                "SLURM_JOB_ID": "4242",
                "SLURM_JOBID": "4242",
                "SDPO_PROC_ROOT": str(fake_proc_root),
                "SDPO_CLEANUP_DRAIN_SEC": "2",
                "SDPO_CLEANUP_GRACE_SEC": "0",
                "STUB_SDPO_CACHE_DIR": str(cache_dir),
            }
        )
        result = subprocess.run(
            [
                "bash",
                str(script_path),
                f"data.train_files={fake_parquet}",
                f"data.val_files={fake_parquet}",
            ],
            cwd=_repo_root(),
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )

        assert ray_proc.poll() is None
        assert "no matching runtime processes remained after drain" in result.stdout
    finally:
        if ray_proc.poll() is None:
            ray_proc.terminate()
        try:
            ray_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ray_proc.kill()
            ray_proc.wait(timeout=5)


def test_run_sdpo_script_cleanup_accepts_zero_padded_drain_seconds(tmp_path: Path) -> None:
    script_path = _repo_root() / "scripts" / "run_sdpo.sh"
    fake_python = _write_python_cleanup_probe_stub(tmp_path)
    fake_checkpoint = tmp_path / "rft-checkpoint"
    fake_checkpoint.mkdir()
    fake_parquet = tmp_path / "sdpo_tasks.parquet"
    fake_parquet.write_text("stub", encoding="utf-8")
    fake_proc_root = tmp_path / "fake-proc"
    fake_proc_root.mkdir()
    cache_dir = tmp_path / "cache-dir"
    cache_dir.mkdir()

    ray_proc = subprocess.Popen(
        ["bash", "-lc", "exec -a raylet sleep 120"],
        env={
            **os.environ,
            "SLURM_JOB_ID": "4242",
        },
    )

    pid_dir = fake_proc_root / str(ray_proc.pid)
    pid_dir.mkdir(parents=True)
    environ_path = pid_dir / "environ"
    environ_path.write_bytes(b"SLURM_JOB_ID=4242\0")

    try:
        def _flip_job_id_after_drain_start() -> None:
            time.sleep(0.5)
            environ_path.write_bytes(b"SLURM_JOB_ID=9999\0")

        mutator = threading.Thread(target=_flip_job_id_after_drain_start, daemon=True)
        mutator.start()

        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": str(fake_python),
                "SDPO_RFT_CHECKPOINT": str(fake_checkpoint),
                "SDPO_TRAINER_MODULE": "dummy.module",
                "SDPO_MONITOR_ENABLE": "0",
                "SDPO_WANDB_REPAIR_ON_EXIT": "0",
                "SLURM_JOB_ID": "4242",
                "SLURM_JOBID": "4242",
                "SDPO_PROC_ROOT": str(fake_proc_root),
                "SDPO_CLEANUP_DRAIN_SEC": "08",
                "SDPO_CLEANUP_GRACE_SEC": "0",
                "STUB_SDPO_CACHE_DIR": str(cache_dir),
            }
        )
        result = subprocess.run(
            [
                "bash",
                str(script_path),
                f"data.train_files={fake_parquet}",
                f"data.val_files={fake_parquet}",
            ],
            cwd=_repo_root(),
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )

        assert ray_proc.poll() is None
        assert "no matching runtime processes remained after drain" in result.stdout
    finally:
        if ray_proc.poll() is None:
            ray_proc.terminate()
        try:
            ray_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ray_proc.kill()
            ray_proc.wait(timeout=5)


def test_run_sdpo_script_dry_run_sets_ray_num_cpus_from_slurm_cpus_per_task() -> None:
    result = _run_script(
        "run_sdpo.sh",
        env_overrides={"SLURM_CPUS_PER_TASK": "64"},
    )
    assert "ray_kwargs.ray_init.num_cpus=64" in result.stdout


def test_run_sdpo_script_dry_run_sets_ray_num_cpus_from_slurm_cpus_per_gpu_times_visible_gpus() -> None:
    result = _run_script(
        "run_sdpo.sh",
        env_overrides={
            "SLURM_CPUS_PER_TASK": "",
            "SLURM_CPUS_PER_GPU": "12",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        },
    )
    assert "ray_kwargs.ray_init.num_cpus=48" in result.stdout


def test_run_sdpo_script_dry_run_prefers_sdpo_ray_num_cpus_env() -> None:
    result = _run_script(
        "run_sdpo.sh",
        env_overrides={
            "SDPO_RAY_NUM_CPUS": "72",
            "SLURM_CPUS_PER_TASK": "64",
        },
    )
    assert "ray_kwargs.ray_init.num_cpus=72" in result.stdout
    assert "ray_kwargs.ray_init.num_cpus=64" not in result.stdout


def test_run_sdpo_script_dry_run_respects_explicit_ray_num_cpus_override() -> None:
    result = _run_script(
        "run_sdpo.sh",
        "ray_kwargs.ray_init.num_cpus=7",
        env_overrides={"SLURM_CPUS_PER_TASK": "64"},
    )
    assert "ray_kwargs.ray_init.num_cpus=7" in result.stdout
    assert "ray_kwargs.ray_init.num_cpus=64" not in result.stdout


def test_run_sdpo_script_dry_run_rollout_only_disables_validation_rollouts_by_default() -> None:
    result = _run_script(
        "run_sdpo.sh",
        env_overrides={"SDPO_ROLLOUT_ONLY_E2E": "1"},
    )
    assert "trainer.test_freq=0" in result.stdout
    assert "trainer.val_before_train=false" in result.stdout


def test_run_sdpo_script_dry_run_rollout_only_respects_explicit_validation_overrides() -> None:
    result = _run_script(
        "run_sdpo.sh",
        "trainer.test_freq=7",
        "trainer.val_before_train=true",
        env_overrides={"SDPO_ROLLOUT_ONLY_E2E": "1"},
    )
    assert "trainer.test_freq=7" in result.stdout
    assert "trainer.val_before_train=true" in result.stdout
    assert "trainer.test_freq=0" not in result.stdout
    assert "trainer.val_before_train=false" not in result.stdout


def test_run_sdpo_script_dry_run_respects_explicit_prompt_filter_override() -> None:
    result = _run_script(
        "run_sdpo.sh",
        "data.filter_overlong_prompts=true",
    )
    assert "data.filter_overlong_prompts=true" in result.stdout
    assert "data.filter_overlong_prompts=false" not in result.stdout


def test_run_sdpo_script_dry_run_respects_explicit_logger_override() -> None:
    result = _run_script(
        "run_sdpo.sh",
        "trainer.logger=[console,wandb]",
    )
    assert "trainer.logger=\\[console\\,wandb\\]" in result.stdout
    assert "trainer.logger=\\[console\\,wandb\\,file\\]" not in result.stdout


def test_run_sdpo_script_dry_run_uses_task_sdpo_cache_defaults() -> None:
    result = _run_script("run_sdpo.sh")
    assert "data.train_files=" in result.stdout
    assert "data.val_files=" in result.stdout
    assert "/data/sdpo_task_cache/" in result.stdout


def test_run_sdpo_script_dry_run_prefers_preloaded_train_val_pair_in_cache(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "sdpo_tasks_pair_train.parquet"
    val_path = tmp_path / "sdpo_tasks_pair_val.parquet"
    train_path.write_text("train", encoding="utf-8")
    val_path.write_text("val", encoding="utf-8")

    result = _run_script(
        "run_sdpo.sh",
        env_overrides={"SDPO_TASK_CACHE_DIR": str(tmp_path)},
    )
    assert f"data.train_files={train_path}" in result.stdout
    assert f"data.val_files={val_path}" in result.stdout


def test_run_sdpo_script_dry_run_mirrors_explicit_train_files_to_val_files() -> None:
    result = _run_script(
        "run_sdpo.sh",
        "data.train_files=/tmp/custom_sdpo_tasks.parquet",
    )
    assert "data.train_files=/tmp/custom_sdpo_tasks.parquet" in result.stdout
    assert "data.val_files=/tmp/custom_sdpo_tasks.parquet" in result.stdout


def test_run_sdpo_script_dry_run_preloaded_task_parquet_overrides_split_defaults() -> None:
    result = _run_script(
        "run_sdpo.sh",
        env_overrides={"SDPO_PRELOADED_TASK_PARQUET": "/tmp/preloaded.parquet"},
    )
    assert "data.train_files=/tmp/preloaded.parquet" in result.stdout
    assert "data.val_files=/tmp/preloaded.parquet" in result.stdout


def test_run_sdpo_script_dry_run_does_not_invoke_preload_helper_module(tmp_path: Path) -> None:
    fake_python = _write_python_no_preload_stub(tmp_path)
    result = _run_script(
        "run_sdpo.sh",
        env_overrides={"PYTHON_BIN": str(fake_python)},
    )
    assert "data.train_files=" in result.stdout
    assert "data.val_files=" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_sets_onpolicy_overrides() -> None:
    result = _run_script("run_rft_onpolicy_rollout_proof.sh")
    assert "--config-name rft_swe" in result.stdout
    assert "-m verl_integration.fsdp_sft_trainer_entry" in result.stdout
    assert "trainer.logger=\\[console\\,wandb\\]" in result.stdout
    assert "trainer.default_local_dir=" in result.stdout
    assert "data.on_policy.enabled=true" in result.stdout
    assert "data.on_policy.turn_generator_mode=proof_tool_chain" in result.stdout
    assert "data.on_policy.total_steps=1" in result.stdout
    assert "+data.on_policy.runtime_overrides.task_batch_size=" in result.stdout
    assert "model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_honors_model_override() -> None:
    result = _run_script(
        "run_rft_onpolicy_rollout_proof.sh",
        env_overrides={"ON_POLICY_PROOF_MODEL_PATH": "Qwen/Qwen2.5-1.5B-Instruct"},
    )
    assert "model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_propagates_steps() -> None:
    result = _run_script(
        "run_rft_onpolicy_rollout_proof.sh",
        env_overrides={"ON_POLICY_PROOF_STEPS": "3"},
    )
    assert "trainer.total_training_steps=3" in result.stdout
    assert "data.on_policy.total_steps=3" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_defaults_train_batch_to_world_size() -> None:
    result = _run_script(
        "run_rft_onpolicy_rollout_proof.sh",
        env_overrides={"ON_POLICY_PROOF_NPROC_PER_NODE": "8"},
    )
    assert "trainer.n_gpus_per_node=8" in result.stdout
    assert "data.train_batch_size=8" in result.stdout
    assert "data.micro_batch_size_per_gpu=1" in result.stdout
    assert "+data.on_policy.runtime_overrides.task_batch_size=8" in result.stdout


def test_run_rft_onpolicy_rollout_proof_script_honors_explicit_batch_overrides() -> None:
    result = _run_script(
        "run_rft_onpolicy_rollout_proof.sh",
        env_overrides={
            "ON_POLICY_PROOF_NPROC_PER_NODE": "8",
            "ON_POLICY_TRAIN_BATCH_SIZE": "16",
            "ON_POLICY_MICRO_BATCH_SIZE_PER_GPU": "2",
        },
    )
    assert "data.train_batch_size=16" in result.stdout
    assert "data.micro_batch_size_per_gpu=2" in result.stdout


def test_run_flash_attn_rebuild_script_dry_run_uses_safe_defaults() -> None:
    result = _run_script("run_flash_attn_rebuild.sh")
    assert "--partition gpu" in result.stdout
    assert "--gres" not in result.stdout
    assert "--cpus-per-task 8" in result.stdout
    assert "--mem 128G" in result.stdout
    assert "rebuild-flash-attn" in result.stdout
    assert "CORES=8" in result.stdout
    assert "FLASH_ATTN_CUDA_ARCHS=120" in result.stdout
    assert "pipefail" not in result.stdout


def test_run_flash_attn_rebuild_script_dry_run_has_no_log_dir_side_effect(tmp_path: Path) -> None:
    log_dir = tmp_path / "flash-attn-rebuild-logs"
    result = _run_script(
        "run_flash_attn_rebuild.sh",
        env_overrides={"FLASH_ATTN_BUILD_LOG_DIR": str(log_dir)},
    )
    assert str(log_dir / "slurm-%j.out") in result.stdout
    assert str(log_dir / "slurm-%j.err") in result.stdout
    assert not log_dir.exists()


def test_run_flash_attn_rebuild_script_honors_env_overrides() -> None:
    result = _run_script(
        "run_flash_attn_rebuild.sh",
        env_overrides={
            "FLASH_ATTN_BUILD_PARTITION": "gpu",
            "FLASH_ATTN_BUILD_GRES": "gpu:2",
            "FLASH_ATTN_BUILD_CPUS_PER_TASK": "12",
            "FLASH_ATTN_BUILD_MEM": "96G",
            "FLASH_ATTN_BUILD_TIME": "02:30:00",
            "FLASH_ATTN_BUILD_MAX_JOBS": "4",
            "FLASH_ATTN_CUDA_ARCHS": "90",
        },
    )
    assert "--partition gpu" in result.stdout
    assert "--gres gpu:2" in result.stdout
    assert "--cpus-per-task 12" in result.stdout
    assert "--mem 96G" in result.stdout
    assert "--time 02:30:00" in result.stdout
    assert "rebuild-flash-attn" in result.stdout
    assert "CORES=4" in result.stdout
    assert "FLASH_ATTN_CUDA_ARCHS=90" in result.stdout
