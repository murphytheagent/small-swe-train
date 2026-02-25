"""Runtime orchestration helpers for on-policy RFT handoff into verl SFT training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from config import DEFAULT_ON_POLICY_DATA_CONFIG_NAME
from data.tokenization import SupportsOffsetsTokenizer
from rollout.vllm_turn_generator import build_vllm_turn_generator
from trainer.rft_handoff import (
    build_onpolicy_collector,
    collect_rft_sft_batch_for_steps,
)


@dataclass(frozen=True)
class OnPolicyRFTRuntimeRequest:
    data_config_name: str = DEFAULT_ON_POLICY_DATA_CONFIG_NAME
    turn_generator_mode: str = "default"
    total_steps: int = 1
    start_step_index: int = 0
    runtime_overrides: Mapping[str, Any] | None = None
    data_overrides: Mapping[str, Any] | None = None
    handoff_overrides: Mapping[str, Any] | None = None
    output_dir: str | None = None


def collect_onpolicy_rft_runtime_batch(
    *,
    request: OnPolicyRFTRuntimeRequest,
    tokenizer: SupportsOffsetsTokenizer,
) -> dict[str, Any]:
    """Collect rollouts and build one RFT SFT batch via the live runtime path."""
    collector = build_onpolicy_collector(
        data_config_name=request.data_config_name,
        runtime_overrides=request.runtime_overrides,
        data_overrides=request.data_overrides,
        turn_generator=_resolve_turn_generator(request.turn_generator_mode),
    )

    resolved_output_dir = _normalized_output_dir(request.output_dir)
    if resolved_output_dir is not None:
        Path(resolved_output_dir).mkdir(parents=True, exist_ok=True)
    if request.start_step_index < 0:
        raise ValueError("start_step_index must be >= 0")

    result = collect_rft_sft_batch_for_steps(
        total_steps=request.total_steps,
        start_step_index=request.start_step_index,
        collector=collector,
        tokenizer=tokenizer,
        handoff_overrides=request.handoff_overrides,
        output_dir=resolved_output_dir,
    )

    if resolved_output_dir is not None:
        manifest_path = Path(resolved_output_dir) / "rft_runtime_manifest.json"
        _write_json(
            manifest_path,
            _build_runtime_manifest_payload(
                request=request,
                result=result,
            ),
        )
    return result


def _normalized_output_dir(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _build_runtime_manifest_payload(
    *,
    request: OnPolicyRFTRuntimeRequest,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    rollout_rows = _coerce_rows(result.get("rollout_rows"))
    selected_rows = _coerce_rows(result.get("selected_rows"))
    rejected_rows = _coerce_rows(result.get("rejected_rows"))
    dataproto_payload = _as_mapping(result.get("dataproto_payload"))
    meta_info = _as_mapping(dataproto_payload.get("meta_info"))

    rejection_reason_counts: dict[str, int] = {}
    for row in rejected_rows:
        reason_text = str(row.get("rft_rejection_reason", "")).strip()
        if not reason_text:
            parsed_reasons = ["unknown"]
        else:
            parsed_reasons = [item.strip() for item in reason_text.split(",") if item.strip()]
            if not parsed_reasons:
                parsed_reasons = ["unknown"]

        for reason in parsed_reasons:
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1

    return {
        "data_config_name": request.data_config_name,
        "turn_generator_mode": request.turn_generator_mode,
        "total_steps": int(request.total_steps),
        "start_step_index": int(request.start_step_index),
        "rollout_count": len(rollout_rows),
        "selected_count": len(selected_rows),
        "rejected_count": len(rejected_rows),
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "dataproto_meta_info": dict(meta_info),
    }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _coerce_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")


def _resolve_turn_generator(mode: str):
    normalized_mode = mode.strip().lower()
    if normalized_mode == "default":
        return build_vllm_turn_generator()
    if normalized_mode == "vllm_live":
        return build_vllm_turn_generator()
    if normalized_mode == "proof_tool_chain":
        return _proof_tool_chain_turn_generator
    raise ValueError(
        "data.on_policy.turn_generator_mode must be one of: "
        "'default', 'vllm_live', 'proof_tool_chain'."
    )


def _proof_tool_chain_turn_generator(
    *,
    task,
    attempt_index: int,
    turn_index: int,
    step_index: int,
    history,
) -> str:
    del task, step_index, history
    path = f"/tmp/rft_proof_attempt_{attempt_index}.txt"
    if turn_index == 0:
        return (
            "<tool_call>"
            '{"tool":"bash","args":{"command":"printf \'proof_seed\\n\' > '
            + path
            + "\"}}"
            "</tool_call>"
        )
    if turn_index == 1:
        return (
            "<tool_call>"
            '{"tool":"search","args":{"query":"proof_seed","path_hint":"/tmp","top_k":5}}'
            "</tool_call>"
        )
    if turn_index == 2:
        return (
            "<tool_call>"
            '{"tool":"apply_patch","args":{"path":"'
            + path
            + '","patch":"proof_patch"}}'
            "</tool_call>"
        )
    if turn_index == 3:
        return (
            "<tool_call>"
            '{"tool":"search","args":{"query":"proof_patch","path_hint":"/tmp","top_k":5}}'
            "</tool_call>"
        )
    return (
        "<tool_call>"
        '{"tool":"submit","args":{"final_response":"proof terminal submit"}}'
        "</tool_call>"
    )
