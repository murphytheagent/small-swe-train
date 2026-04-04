"""CLI helper that probes on-policy task images for verifier prerequisites."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import (
    DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
    resolve_on_policy_bad_task_cache_dir,
    resolve_on_policy_settings,
)
from env.container_pool import BatchContainerPool
from env.docker_executor import DockerToolExecutor
from env.runtime_protocol import ToolRequest
from env.shell_helpers import (
    build_executable_resolver_shell,
    build_python_interpreter_resolver_shell,
)
from env.task_dataset import (
    ON_POLICY_BAD_TASK_CACHE_SCHEMA_VERSION,
    TaskSample,
    _coerce_task_row,
    load_hf_dataset,
    resolve_on_policy_bad_task_cache_path,
)
from verifier_utils import normalize_verifier_kind

_DEFAULT_PROBE_TIMEOUT_SEC = 120
_DEFAULT_MAX_WORKERS = 8
_TASK_SCOPED_BAD_REASONS = {"selector_invalid", "selector_probe_failed"}


def _build_probe_command(
    *,
    verifier_kind: str,
    selectors: Sequence[str] | None = None,
) -> str:
    selectors_json = json.dumps([str(selector) for selector in (selectors or ())], ensure_ascii=True)
    go_resolver = ""
    go_prefix = ""
    if verifier_kind == "go_test":
        go_resolver = build_executable_resolver_shell(
            var_name="gobin",
            command_names=("go",),
            fallback_paths=("/usr/local/go/bin/go", "/usr/lib/go/bin/go", "/opt/go/bin/go"),
            not_found_message="Go executable missing in task container.",
        )
        go_prefix = 'SMALL_SWE_PREFLIGHT_GO_BIN="${gobin}" '
    node_resolver = ""
    node_prefix = ""
    if verifier_kind == "node_test":
        node_resolver = (
            build_executable_resolver_shell(
                var_name="nodebin",
                command_names=("node",),
                fallback_paths=("/usr/local/bin/node", "/usr/bin/node", "/opt/node/bin/node"),
                not_found_message="Node executable missing in task container.",
            )
            + build_executable_resolver_shell(
                var_name="npmbin",
                command_names=("npm",),
                fallback_paths=("/usr/local/bin/npm", "/usr/bin/npm", "/opt/node/bin/npm"),
                not_found_message="npm executable missing in task container.",
            )
        )
        node_prefix = 'SMALL_SWE_PREFLIGHT_NODE_BIN="${nodebin}" SMALL_SWE_PREFLIGHT_NPM_BIN="${npmbin}" '
    return (
        "set -eu; "
        + build_python_interpreter_resolver_shell(var_name="pybin")
        + go_resolver
        + node_resolver
        + f"SMALL_SWE_PREFLIGHT_VERIFIER_KIND={json.dumps(verifier_kind)} "
        + f"SMALL_SWE_PREFLIGHT_SELECTORS_JSON={json.dumps(selectors_json)} "
        + go_prefix
        + node_prefix
        + '''"${pybin}" - <<'PY'
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import re

repo_root = os.environ.get("TASK_REPO_ROOT", "")
verifier_kind = os.environ.get("SMALL_SWE_PREFLIGHT_VERIFIER_KIND", "pytest").strip().lower()
selectors = json.loads(os.environ.get("SMALL_SWE_PREFLIGHT_SELECTORS_JSON", "[]"))
gobin = os.environ.get("SMALL_SWE_PREFLIGHT_GO_BIN", "").strip()
nodebin = os.environ.get("SMALL_SWE_PREFLIGHT_NODE_BIN", "").strip()
npmbin = os.environ.get("SMALL_SWE_PREFLIGHT_NPM_BIN", "").strip()
try:
    selectors = json.loads(selectors) if isinstance(selectors, str) else selectors
except json.JSONDecodeError:
    selectors = []
selectors = [str(selector).strip() for selector in selectors if str(selector).strip()]
runner_available = False
runner_label = ""
selector_valid = True
selector_error = ""
missing_selectors: list[str] = []


def _resolve_executable(env_value: str, command_names: tuple[str, ...], fallback_paths: tuple[str, ...]) -> str:
    if env_value:
        return env_value
    for command_name in command_names:
        resolved = shutil.which(command_name)
        if resolved:
            return resolved
    for candidate in fallback_paths:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""

if verifier_kind == "pytest":
    runner_available = importlib.util.find_spec("pytest") is not None
    runner_label = "pytest"
elif verifier_kind == "go_test":
    resolved_go = _resolve_executable(
        gobin,
        ("go",),
        ("/usr/local/go/bin/go", "/usr/lib/go/bin/go", "/opt/go/bin/go"),
    )
    runner_available = bool(resolved_go)
    runner_label = resolved_go or "go"
elif verifier_kind == "node_test":
    resolved_node = _resolve_executable(
        nodebin,
        ("node",),
        ("/usr/local/bin/node", "/usr/bin/node", "/opt/node/bin/node"),
    )
    resolved_npm = _resolve_executable(
        npmbin,
        ("npm",),
        ("/usr/local/bin/npm", "/usr/bin/npm", "/opt/node/bin/npm"),
    )
    runner_available = bool(resolved_node and resolved_npm)
    runner_label = f"{resolved_node or 'node'}+{resolved_npm or 'npm'}"
elif verifier_kind == "command":
    runner_available = True
    runner_label = "shell"
else:
    selector_valid = False
    selector_error = f"unsupported verifier kind: {verifier_kind}"

if repo_root and os.path.exists(repo_root) and runner_available and selectors:
    try:
        if verifier_kind == "pytest":
            command = [os.environ.get("SMALL_SWE_PYBIN", sys.executable), "-m", "pytest", "--collect-only", "-q", *selectors]
            completed = subprocess.run(
                command,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            selector_valid = completed.returncode == 0
            if not selector_valid:
                selector_error = completed.stderr.strip() or completed.stdout.strip() or f"pytest collect failed with exit code {completed.returncode}"
        elif verifier_kind == "go_test":
            regex = "^(?:" + "|".join(re.escape(selector) for selector in selectors) + ")$" if selectors else "^$"
            command = [resolved_go, "test", "./...", "-run", "^$", "-list", regex]
            completed = subprocess.run(
                command,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            if completed.returncode != 0:
                selector_valid = False
                selector_error = completed.stderr.strip() or completed.stdout.strip() or f"go test -list failed with exit code {completed.returncode}"
            else:
                listed = {line.strip() for line in completed.stdout.splitlines() if line.strip() in selectors}
                missing_selectors = [selector for selector in selectors if selector not in listed]
                selector_valid = not missing_selectors
                if missing_selectors:
                    selector_error = "missing selectors: " + ", ".join(missing_selectors[:10])
        else:
            selector_valid = False
            selector_error = f"selector probe unsupported for verifier kind: {verifier_kind}"
    except Exception as exc:
        selector_valid = False
        selector_error = str(exc)

payload = {
    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    "repo_root": repo_root,
    "repo_root_exists": bool(repo_root) and os.path.exists(repo_root),
    "sys_executable": sys.executable,
    "pytest_importable": importlib.util.find_spec("pytest") is not None,
    "verifier_kind": verifier_kind,
    "runner_available": bool(runner_available),
    "runner_label": runner_label,
    "selector_count": len(selectors),
    "selector_valid": bool(selector_valid),
    "selector_error": selector_error,
    "missing_selectors": missing_selectors,
}
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
PY'''
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe on-policy task images and cache entries with broken verifier prerequisites.",
    )
    parser.add_argument(
        "--data-config-name",
        default=DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
        help="Named config from configs/data/<name>.yaml (default: on_policy_swe_smith).",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Directory for the deterministic bad-task cache file (default: data/on_policy_bad_task_cache).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild cache even when the resolved file already exists.",
    )
    parser.add_argument(
        "--print-path-only",
        action="store_true",
        help="Print the resolved cache path without probing containers.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Probe at most this many unique task images in deterministic order.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=_DEFAULT_MAX_WORKERS,
        help=f"Concurrent image probes to run (default: {_DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--probe-timeout-sec",
        type=int,
        default=_DEFAULT_PROBE_TIMEOUT_SEC,
        help=f"Per-image probe timeout in seconds (default: {_DEFAULT_PROBE_TIMEOUT_SEC}).",
    )
    return parser


def _decode_probe_payload(stdout_text: str) -> dict[str, Any] | None:
    stripped = str(stdout_text or "").strip()
    if not stripped:
        return None
    for line in reversed(stripped.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return None


def _iter_dataset_tasks(
    *,
    config: Any,
    dataset_loader: Any | None = None,
) -> list[TaskSample]:
    loader = dataset_loader or load_hf_dataset
    dataset = loader(config.dataset_id, config.dataset_split)
    tasks: list[TaskSample] = []
    for row_index in range(len(dataset)):
        row = dataset[row_index]
        if not isinstance(row, Mapping):
            continue
        try:
            tasks.append(_coerce_task_row(row, config=config, row_index=row_index))
        except ValueError:
            continue
    return tasks


def _classify_probe_record(record: Mapping[str, Any]) -> tuple[str, str]:
    if not bool(record.get("image_probe_ok", False)):
        return "bad", "probe_command_failed"
    payload = record.get("image_probe_payload")
    if not isinstance(payload, Mapping):
        return "bad", "invalid_probe_payload"
    if not bool(payload.get("repo_root_exists", False)):
        return "bad", "repo_root_missing"
    if not bool(payload.get("runner_available", False)):
        return "bad", "runner_unavailable"
    if record.get("task_probe_exit_code") is not None and not bool(record.get("task_probe_ok", False)):
        return "bad", "selector_probe_failed"
    task_payload = record.get("task_probe_payload")
    if isinstance(task_payload, Mapping) and not bool(task_payload.get("selector_valid", True)):
        return "bad", "selector_invalid"
    return "ok", ""


def _probe_task(
    *,
    task: TaskSample,
    verifier_kind: str,
    probe_timeout_sec: int,
) -> dict[str, Any]:
    pool = BatchContainerPool(
        env_pool_size=1,
        container_start_timeout_sec=max(int(probe_timeout_sec), 1),
    )
    record: dict[str, Any] = {
        "task_id": task.task_id,
        "image_name": task.image_name,
        "image_probe_ok": False,
        "image_probe_payload": None,
        "image_probe_exit_code": None,
        "image_probe_stderr": "",
        "task_probe_ok": False,
        "task_probe_payload": None,
        "task_probe_exit_code": None,
        "task_probe_stderr": "",
    }
    try:
        handles = pool.acquire([task])
        handle = handles[0]
        executor = DockerToolExecutor(
            container_id=handle.container_id,
            tool_timeout_sec=max(int(probe_timeout_sec), 1),
        )
        response = executor.run(
            ToolRequest(
                tool="bash",
                args={
                    "command": _build_probe_command(verifier_kind=verifier_kind),
                    "timeout_sec": max(int(probe_timeout_sec), 1),
                },
            )
        )
        record["image_probe_exit_code"] = int(response.exit_code)
        record["image_probe_stderr"] = str(response.stderr or "").strip()
        if response.exit_code == 0:
            payload = _decode_probe_payload(str(response.stdout or ""))
            record["image_probe_payload"] = payload
            record["image_probe_ok"] = payload is not None
            if payload is not None and bool(payload.get("runner_available", False)):
                selector_response = executor.run(
                    ToolRequest(
                        tool="bash",
                        args={
                            "command": _build_probe_command(
                                verifier_kind=verifier_kind,
                                selectors=[*task.fail_to_pass, *task.pass_to_pass],
                            ),
                            "timeout_sec": max(int(probe_timeout_sec), 1),
                        },
                    )
                )
                record["task_probe_exit_code"] = int(selector_response.exit_code)
                record["task_probe_stderr"] = str(selector_response.stderr or "").strip()
                if selector_response.exit_code == 0:
                    task_payload = _decode_probe_payload(str(selector_response.stdout or ""))
                    record["task_probe_payload"] = task_payload
                    record["task_probe_ok"] = task_payload is not None
    except Exception as exc:
        record["image_probe_stderr"] = str(exc)
        record["image_probe_exit_code"] = 1
        record["image_probe_ok"] = False
    finally:
        pool.release_all()

    status, reason = _classify_probe_record(record)
    record["status"] = status
    record["reason"] = reason
    return record


def scan_dataset_for_bad_verifier_tasks(
    *,
    config: Any,
    dataset_loader: Any | None = None,
    probe_timeout_sec: int = _DEFAULT_PROBE_TIMEOUT_SEC,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    max_images: int | None = None,
) -> dict[str, Any]:
    tasks = _iter_dataset_tasks(config=config, dataset_loader=dataset_loader)
    selected_tasks = list(tasks)
    if max_images is not None:
        max_unique_images = max(int(max_images), 0)
        if max_unique_images == 0:
            selected_tasks = []
        else:
            selected_image_names = []
            seen_images: set[str] = set()
            for task in selected_tasks:
                if task.image_name in seen_images:
                    continue
                if len(selected_image_names) >= max_unique_images:
                    break
                seen_images.add(task.image_name)
                selected_image_names.append(task.image_name)
            selected_image_name_set = set(selected_image_names)
            selected_tasks = [task for task in selected_tasks if task.image_name in selected_image_name_set]

    records: list[dict[str, Any]] = []
    if max_images is not None:
        probed_image_count = len({task.image_name for task in selected_tasks})
    else:
        probed_image_count = len({task.image_name for task in tasks})
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {
            executor.submit(
                _probe_task,
                task=task,
                verifier_kind=normalize_verifier_kind(config.verifier_kind),
                probe_timeout_sec=probe_timeout_sec,
            ): task.task_id
            for task in selected_tasks
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)

    ordered_records = sorted(records, key=lambda item: (str(item.get("image_name", "")), str(item.get("task_id", ""))))
    bad_image_names = sorted(
        {
            str(record.get("image_name", "")).strip()
            for record in ordered_records
            if str(record.get("status", "")).strip().lower() == "bad"
            and str(record.get("reason", "")).strip() not in _TASK_SCOPED_BAD_REASONS
        }
    )
    bad_image_name_set = set(bad_image_names)
    bad_task_id_set = {
        str(record.get("task_id", "")).strip()
        for record in ordered_records
        if str(record.get("status", "")).strip().lower() == "bad"
        and str(record.get("reason", "")).strip() in _TASK_SCOPED_BAD_REASONS
        and str(record.get("task_id", "")).strip()
    }
    bad_task_id_set.update(
        task.task_id
        for task in selected_tasks
        if task.image_name in bad_image_name_set
    )
    bad_task_ids = sorted(bad_task_id_set)
    return {
        "schema_version": ON_POLICY_BAD_TASK_CACHE_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_split": config.dataset_split,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_timeout_sec": int(probe_timeout_sec),
        "scanned_task_count": len(selected_tasks),
        "probed_image_count": probed_image_count,
        "bad_task_ids": bad_task_ids,
        "bad_image_names": bad_image_names,
        "records": ordered_records,
    }


def _cache_scope_suffix(*, max_images: int | None, probe_timeout_sec: int) -> str:
    parts: list[str] = []
    if max_images is not None:
        parts.append(f"max_images_{max(int(max_images), 0)}")
    resolved_timeout_sec = max(int(probe_timeout_sec), 1)
    if resolved_timeout_sec != _DEFAULT_PROBE_TIMEOUT_SEC:
        parts.append(f"probe_timeout_{resolved_timeout_sec}s")
    if not parts:
        return ""
    return "__" + "__".join(parts)


def _resolve_cache_path(
    *,
    settings: Any,
    cache_dir: str,
    max_images: int | None,
    probe_timeout_sec: int,
) -> Path:
    configured_dir = str(cache_dir or "").strip()
    if configured_dir:
        target_dir = Path(configured_dir)
    else:
        project_root = Path(__file__).resolve().parents[2]
        target_dir = resolve_on_policy_bad_task_cache_dir(project_root=project_root)
    base_path = resolve_on_policy_bad_task_cache_path(config=settings.data, cache_dir=target_dir)
    suffix = _cache_scope_suffix(
        max_images=max_images,
        probe_timeout_sec=probe_timeout_sec,
    )
    if not suffix:
        return base_path
    return base_path.with_name(base_path.stem + suffix + base_path.suffix)


def _write_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    settings = resolve_on_policy_settings(data_config_name=args.data_config_name)
    target_path = _resolve_cache_path(
        settings=settings,
        cache_dir=args.cache_dir,
        max_images=args.max_images,
        probe_timeout_sec=int(args.probe_timeout_sec),
    )
    if args.print_path_only:
        print(target_path)
        return 0
    if target_path.is_file() and not bool(args.force_refresh):
        print(target_path)
        return 0

    payload = scan_dataset_for_bad_verifier_tasks(
        config=settings.data,
        probe_timeout_sec=int(args.probe_timeout_sec),
        max_workers=int(args.max_workers),
        max_images=args.max_images,
    )
    _write_cache(target_path, payload)
    print(target_path)
    print(f"bad_images={len(payload['bad_image_names'])}")
    print(f"bad_tasks={len(payload['bad_task_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
