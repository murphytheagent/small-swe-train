"""Custom verl SFT dataset that pulls on-policy rollouts and builds RFT batches in-memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import DEFAULT_ON_POLICY_DATA_CONFIG_NAME
from verl_integration.onpolicy_rollout_adapter import (
    build_onpolicy_collector,
    collect_rft_sft_batch_for_steps,
)

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", ""}
_ONPOLICY_RFT_CACHE: dict[str, dict[str, Any]] = {}


class OnPolicyRFTDataset:
    """verl-compatible Dataset that materializes one on-policy RFT batch."""

    def __init__(
        self,
        parquet_files: str | Sequence[str],
        tokenizer: Any,
        config: Mapping[str, Any],
        processor: Any | None = None,
        max_samples: int = -1,
    ) -> None:
        del parquet_files, processor  # This dataset is generated on-policy, not from parquet.

        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - train-only dependency
            raise RuntimeError(
                "OnPolicyRFTDataset requires torch. Install training extras (`pip install -e \".[train]\"`)."
            ) from exc

        config_mapping = _as_mapping(config)
        on_policy_cfg = _as_mapping(config_mapping.get("on_policy", {}))
        if not _coerce_bool(on_policy_cfg.get("enabled", True), fallback=True):
            raise ValueError("data.on_policy.enabled must be true when using OnPolicyRFTDataset.")

        data_config_name = str(
            on_policy_cfg.get("data_config_name", DEFAULT_ON_POLICY_DATA_CONFIG_NAME)
        ).strip() or DEFAULT_ON_POLICY_DATA_CONFIG_NAME
        turn_generator_mode = str(on_policy_cfg.get("turn_generator_mode", "default")).strip().lower()
        total_steps = _coerce_positive_int(on_policy_cfg.get("total_steps", 1), label="data.on_policy.total_steps")
        runtime_overrides = _as_mapping(on_policy_cfg.get("runtime_overrides", {}))
        data_overrides = _as_mapping(on_policy_cfg.get("data_overrides", {}))
        handoff_overrides = _as_mapping(on_policy_cfg.get("rft_handoff_overrides", {}))
        output_dir_raw = on_policy_cfg.get("output_dir")
        output_dir = str(output_dir_raw).strip() if isinstance(output_dir_raw, str) and output_dir_raw.strip() else None

        cache_key = _cache_key(
            data_config_name=data_config_name,
            turn_generator_mode=turn_generator_mode,
            total_steps=total_steps,
            runtime_overrides=runtime_overrides,
            data_overrides=data_overrides,
            handoff_overrides=handoff_overrides,
            tokenizer=tokenizer,
        )
        cached_result = _ONPOLICY_RFT_CACHE.get(cache_key)
        if cached_result is None:
            collector = build_onpolicy_collector(
                data_config_name=data_config_name,
                runtime_overrides=runtime_overrides,
                data_overrides=data_overrides,
                turn_generator=_resolve_turn_generator(turn_generator_mode),
            )
            resolved_output_dir = output_dir
            if resolved_output_dir is not None:
                Path(resolved_output_dir).mkdir(parents=True, exist_ok=True)
            cached_result = collect_rft_sft_batch_for_steps(
                total_steps=total_steps,
                collector=collector,
                tokenizer=tokenizer,
                handoff_overrides=handoff_overrides,
                output_dir=resolved_output_dir,
            )
            _ONPOLICY_RFT_CACHE[cache_key] = cached_result

        sft_batch = _as_mapping(cached_result["sft_batch"])
        tensors = _as_mapping(sft_batch["tensors"])
        grouping = _as_mapping(sft_batch["grouping_metadata"])
        meta_info = dict(_as_mapping(sft_batch["meta_info"]))
        rejected_rows = tuple(cached_result["rejected_rows"])

        input_ids_rows = _as_rows(tensors.get("input_ids"), label="sft_batch.tensors.input_ids")
        sample_count = len(input_ids_rows)
        if max_samples > 0:
            sample_count = min(sample_count, max_samples)
        if sample_count < 1:
            raise ValueError("OnPolicyRFTDataset produced zero selected rows for training.")

        self._samples: list[dict[str, Any]] = []
        for index in range(sample_count):
            self._samples.append(
                {
                    "input_ids": torch.tensor(tensors["input_ids"][index], dtype=torch.long),
                    "attention_mask": torch.tensor(tensors["attention_mask"][index], dtype=torch.long),
                    "position_ids": torch.tensor(tensors["position_ids"][index], dtype=torch.long),
                    "loss_mask": torch.tensor(tensors["loss_mask"][index], dtype=torch.long),
                }
            )

        self.grouping_metadata = {
            key: list(value[:sample_count]) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else value
            for key, value in grouping.items()
        }
        self.meta_info = {
            **meta_info,
            "selected_count_after_max_samples": sample_count,
            "rejected_count": len(rejected_rows),
        }
        self.rejected_rows = rejected_rows

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError("Expected mapping-like configuration payload.")


def _as_rows(value: Any, *, label: str) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of rows.")
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"{label} must contain sequence rows.")
        rows.append([int(item) for item in row])
    return rows


def _coerce_bool(value: Any, *, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return value != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return fallback


def _coerce_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= 1.")
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                parsed = int(stripped)
            except ValueError as exc:
                raise ValueError(f"{label} must be an integer >= 1.") from exc
            if parsed >= 1:
                return parsed
    raise ValueError(f"{label} must be an integer >= 1.")


def _cache_key(
    *,
    data_config_name: str,
    turn_generator_mode: str,
    total_steps: int,
    runtime_overrides: Mapping[str, Any],
    data_overrides: Mapping[str, Any],
    handoff_overrides: Mapping[str, Any],
    tokenizer: Any,
) -> str:
    payload = {
        "data_config_name": data_config_name,
        "turn_generator_mode": turn_generator_mode,
        "total_steps": total_steps,
        "runtime_overrides": _normalize_mapping(runtime_overrides),
        "data_overrides": _normalize_mapping(data_overrides),
        "handoff_overrides": _normalize_mapping(handoff_overrides),
        "tokenizer_fingerprint": _tokenizer_cache_fingerprint(tokenizer),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _normalize_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_json(payload)
    if isinstance(normalized, dict):
        return normalized
    return {}


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _tokenizer_cache_fingerprint(tokenizer: Any) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
    }
    for attr_name in ("name_or_path", "vocab_size", "model_max_length", "padding_side", "truncation_side"):
        attr_value = getattr(tokenizer, attr_name, None)
        if attr_value is not None and attr_value != "":
            fingerprint[attr_name] = attr_value

    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        try:
            added_vocab = get_added_vocab()
        except Exception:  # pragma: no cover - defensive only
            added_vocab = None
        if isinstance(added_vocab, Mapping):
            fingerprint["added_vocab"] = {
                str(token): int(token_id) if isinstance(token_id, int) else repr(token_id)
                for token, token_id in sorted(added_vocab.items(), key=lambda item: str(item[0]))
            }

    special_tokens_map = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special_tokens_map, Mapping):
        fingerprint["special_tokens_map"] = _normalize_mapping(special_tokens_map)

    # If tokenizer metadata is unavailable, isolate cache entries by instance to avoid collisions.
    if len(fingerprint) == 1:
        fingerprint["instance_id"] = id(tokenizer)

    return _normalize_mapping(fingerprint)


def _resolve_turn_generator(mode: str):
    if mode == "default":
        return None
    if mode == "proof_tool_chain":
        return _proof_tool_chain_turn_generator
    raise ValueError(
        "data.on_policy.turn_generator_mode must be one of: "
        "'default', 'proof_tool_chain'."
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
            '{"tool":"edit","args":{"path":"'
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
