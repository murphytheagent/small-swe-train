"""Lightweight profiler metrics shared by RFT and SDPO runtimes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


def token_profile_metrics(
    *,
    attention_mask: Any,
    loss_mask: Any | None = None,
    elapsed_sec: float | None = None,
    num_gpus: int | None = None,
    prefix: str = "profiler",
) -> dict[str, float]:
    """Compute token-throughput and padding metrics from tensor-like masks."""
    counts = token_profile_counts(attention_mask=attention_mask, loss_mask=loss_mask)
    return token_profile_metrics_from_counts(
        total_tokens=counts["total_tokens"],
        non_padding_tokens=counts["non_padding_tokens"],
        loss_tokens=counts["loss_tokens"],
        elapsed_sec=elapsed_sec,
        num_gpus=num_gpus,
        prefix=prefix,
    )


def token_profile_counts(*, attention_mask: Any, loss_mask: Any | None = None) -> dict[str, float]:
    """Return token counts before rate computation or distributed aggregation."""
    total_positions = float(_numel(attention_mask))
    non_padding_tokens = float(_sum_tensor_like(attention_mask))
    loss_tokens = float(_sum_tensor_like(loss_mask)) if loss_mask is not None else non_padding_tokens
    return {
        "total_tokens": total_positions,
        "non_padding_tokens": non_padding_tokens,
        "loss_tokens": loss_tokens,
    }


def token_profile_metrics_from_counts(
    *,
    total_tokens: float,
    non_padding_tokens: float,
    loss_tokens: float | None = None,
    elapsed_sec: float | None = None,
    num_gpus: int | None = None,
    prefix: str = "profiler",
) -> dict[str, float]:
    """Compute profiler metrics from pre-aggregated token counts."""
    total_positions = float(total_tokens)
    non_padding_tokens = float(non_padding_tokens)
    loss_tokens = non_padding_tokens if loss_tokens is None else float(loss_tokens)
    metrics = {
        f"{prefix}/total_tokens": total_positions,
        f"{prefix}/non_padding_tokens": non_padding_tokens,
        f"{prefix}/loss_tokens": loss_tokens,
        f"{prefix}/non_padding_ratio": _safe_div(non_padding_tokens, total_positions),
        f"{prefix}/loss_tokens_per_total_tokens": _safe_div(loss_tokens, total_positions),
    }
    if elapsed_sec is not None and elapsed_sec > 0:
        metrics[f"{prefix}/global_tokens_per_sec"] = non_padding_tokens / float(elapsed_sec)
        metrics[f"{prefix}/loss_tokens_per_sec"] = loss_tokens / float(elapsed_sec)
        if num_gpus is not None and num_gpus > 0:
            metrics[f"{prefix}/tokens_per_sec_per_gpu"] = non_padding_tokens / float(elapsed_sec) / float(num_gpus)
    return metrics


def cuda_memory_metrics(*, prefix: str = "profiler") -> dict[str, float]:
    """Return torch CUDA memory metrics when available; missing telemetry is non-fatal."""
    try:
        import torch
    except Exception:
        return {}
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return {}
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total = float(getattr(props, "total_memory", 0) or 0)
        reserved = float(torch.cuda.memory_reserved(device))
        allocated = float(torch.cuda.memory_allocated(device))
        max_reserved = float(torch.cuda.max_memory_reserved(device))
        max_allocated = float(torch.cuda.max_memory_allocated(device))
    except Exception:
        return {}
    metrics = {
        f"{prefix}/gpu_memory_allocated_bytes": allocated,
        f"{prefix}/gpu_memory_reserved_bytes": reserved,
        f"{prefix}/gpu_memory_max_allocated_bytes": max_allocated,
        f"{prefix}/gpu_memory_max_reserved_bytes": max_reserved,
    }
    if total > 0:
        metrics[f"{prefix}/gpu_memory_total_bytes"] = total
        metrics[f"{prefix}/oom_margin"] = max(0.0, (total - max_reserved) / total)
    return metrics


def reset_cuda_peak_memory_stats() -> None:
    try:
        import torch
    except Exception:
        return
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def nvidia_smi_utilization() -> dict[str, float]:
    """Sample GPU utilization via nvidia-smi at coarse boundaries only."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return {}
    gpu_utils: list[float] = []
    memory_used: list[float] = []
    memory_total: list[float] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_utils.append(float(parts[0]))
            memory_used.append(float(parts[1]) * 1024 * 1024)
            memory_total.append(float(parts[2]) * 1024 * 1024)
        except ValueError:
            continue
    if not gpu_utils:
        return {}
    total = sum(memory_total)
    return {
        "profiler/gpu_utilization_percent_mean": sum(gpu_utils) / len(gpu_utils),
        "profiler/gpu_memory_used_bytes_sum": sum(memory_used),
        "profiler/gpu_memory_total_bytes_sum": total,
        "profiler/oom_margin_nvidia_smi": max(0.0, (total - sum(memory_used)) / total) if total > 0 else 0.0,
    }


def append_profiler_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def _sum_tensor_like(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        summed = value.sum()
        if hasattr(summed, "item"):
            return float(summed.item())
        return float(summed)
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return float(sum(_sum_tensor_like(item) for item in value))
    try:
        return float(value)
    except Exception:
        return 0.0


def _numel(value: Any) -> int:
    if value is None:
        return 0
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except Exception:
            return 0
    shape = getattr(value, "shape", None)
    if shape is not None:
        total = 1
        for dim in shape:
            total *= int(dim)
        return total
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        if isinstance(value[0], (list, tuple)):
            return sum(_numel(item) for item in value)
        return len(value)
    return 0


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
