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
from env.shell_helpers import build_python_interpreter_resolver_shell
from env.task_dataset import (
    ON_POLICY_BAD_TASK_CACHE_SCHEMA_VERSION,
    TaskSample,
    _coerce_task_row,
    load_hf_dataset,
    resolve_on_policy_bad_task_cache_path,
)

_DEFAULT_PROBE_TIMEOUT_SEC = 120
_DEFAULT_MAX_WORKERS = 8


def _build_probe_command() -> str:
    return (
        "set -eu; "
        + build_python_interpreter_resolver_shell(var_name="pybin")
        + '''"${pybin}" - <<'PY'
import importlib.util
import json
import os
import sys

repo_root = os.environ.get("TASK_REPO_ROOT", "")
payload = {
    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    "repo_root": repo_root,
    "repo_root_exists": bool(repo_root) and os.path.exists(repo_root),
    "sys_executable": sys.executable,
    "pytest_importable": importlib.util.find_spec("pytest") is not None,
}
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
PY'''
    )


_PROBE_COMMAND = _build_probe_command()


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
    if not bool(record.get("probe_ok", False)):
        return "bad", "probe_command_failed"
    payload = record.get("probe_payload")
    if not isinstance(payload, Mapping):
        return "bad", "invalid_probe_payload"
    if not bool(payload.get("repo_root_exists", False)):
        return "bad", "repo_root_missing"
    if not bool(payload.get("pytest_importable", False)):
        return "bad", "pytest_unavailable"
    return "ok", ""


def _probe_image_task(
    *,
    task: TaskSample,
    probe_timeout_sec: int,
) -> dict[str, Any]:
    pool = BatchContainerPool(
        env_pool_size=1,
        container_start_timeout_sec=max(int(probe_timeout_sec), 1),
    )
    record: dict[str, Any] = {
        "task_id": task.task_id,
        "image_name": task.image_name,
        "probe_ok": False,
        "probe_payload": None,
        "probe_exit_code": None,
        "probe_stderr": "",
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
                    "command": _PROBE_COMMAND,
                    "timeout_sec": max(int(probe_timeout_sec), 1),
                },
            )
        )
        record["probe_exit_code"] = int(response.exit_code)
        record["probe_stderr"] = str(response.stderr or "").strip()
        if response.exit_code == 0:
            payload = _decode_probe_payload(str(response.stdout or ""))
            record["probe_payload"] = payload
            record["probe_ok"] = payload is not None
    except Exception as exc:
        record["probe_stderr"] = str(exc)
        record["probe_exit_code"] = 1
        record["probe_ok"] = False
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
    representatives_by_image: dict[str, TaskSample] = {}
    for task in tasks:
        if task.image_name not in representatives_by_image:
            representatives_by_image[task.image_name] = task

    representative_tasks = list(representatives_by_image.values())
    if max_images is not None:
        representative_tasks = representative_tasks[: max(int(max_images), 0)]
    selected_images = {task.image_name for task in representative_tasks}
    selected_tasks = [task for task in tasks if task.image_name in selected_images]

    records_by_image: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {
            executor.submit(
                _probe_image_task,
                task=task,
                probe_timeout_sec=probe_timeout_sec,
            ): task.image_name
            for task in representative_tasks
        }
        for future in as_completed(futures):
            record = future.result()
            records_by_image[str(record["image_name"])] = record

    ordered_records = [records_by_image[task.image_name] for task in representative_tasks]
    bad_image_names = sorted(
        image_name
        for image_name, record in records_by_image.items()
        if str(record.get("status", "")).strip().lower() == "bad"
    )
    bad_image_name_set = set(bad_image_names)
    bad_task_ids = sorted(
        task.task_id
        for task in selected_tasks
        if task.image_name in bad_image_name_set
    )
    return {
        "schema_version": ON_POLICY_BAD_TASK_CACHE_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_split": config.dataset_split,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_timeout_sec": int(probe_timeout_sec),
        "scanned_task_count": len(selected_tasks),
        "probed_image_count": len(representative_tasks),
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
