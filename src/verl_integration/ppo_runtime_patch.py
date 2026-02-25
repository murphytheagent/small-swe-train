"""Runtime monkeypatches for SWE-specific SDPO wiring in verl PPO trainer."""

from __future__ import annotations

import importlib
import logging
import numbers
from typing import Any, Mapping, Sequence

from verl_integration.reprompt_adapter import build_self_distillation_batch
from verl_integration.reward_adapter import dataproto_to_rows, rows_to_reward_tensor

LOGGER = logging.getLogger(__name__)

_PATCH_MARKER_ATTR = "_small_swe_sdpo_runtime_patch_applied"
_ORIGINAL_REWARD_ATTR = "_small_swe_original_compute_or_extract_reward"
_ORIGINAL_DISTILL_ATTR = "_small_swe_original_maybe_build_self_distillation_batch"

try:  # pragma: no cover - exercised in train runtime
    import torch
except ModuleNotFoundError:  # pragma: no cover - unit-test environments without train deps
    torch = None  # type: ignore[assignment]


def apply_small_swe_sdpo_runtime_patch(ray_trainer_module: Any | None = None) -> bool:
    """Patch RayPPOTrainer hooks for SWE-bridge SDPO reward + reprompt wiring."""
    if ray_trainer_module is None:
        try:
            ray_trainer_module = importlib.import_module("verl.trainer.ppo.ray_trainer")
        except ModuleNotFoundError:
            LOGGER.warning("Skipping SDPO runtime patch: verl.trainer.ppo.ray_trainer is unavailable.")
            return False

    trainer_cls = getattr(ray_trainer_module, "RayPPOTrainer", None)
    if trainer_cls is None:
        LOGGER.warning("Skipping SDPO runtime patch: RayPPOTrainer was not found.")
        return False

    if getattr(trainer_cls, _PATCH_MARKER_ATTR, False):
        return True

    original_reward = getattr(trainer_cls, "_compute_or_extract_reward", None)
    original_distill = getattr(trainer_cls, "_maybe_build_self_distillation_batch", None)
    if not callable(original_reward) or not callable(original_distill):
        LOGGER.warning("Skipping SDPO runtime patch: expected RayPPOTrainer hooks are missing.")
        return False

    setattr(trainer_cls, _ORIGINAL_REWARD_ATTR, original_reward)
    setattr(trainer_cls, _ORIGINAL_DISTILL_ATTR, original_distill)

    DataProto = getattr(ray_trainer_module, "DataProto")
    compute_position_id_with_mask = getattr(ray_trainer_module, "compute_position_id_with_mask")

    def _patched_compute_or_extract_reward(
        self: Any,
        batch: Any,
        reward_fn: Any = None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> Any:
        original = getattr(type(self), _ORIGINAL_REWARD_ATTR)
        if not (_is_swe_bridge_loop_enabled(self) or _batch_looks_like_swe_bridge(batch)):
            return original(self, batch, reward_fn=reward_fn, return_dict=return_dict, sum_reward=sum_reward)

        batch_tensors = getattr(batch, "batch", {})
        batch_keys = set(batch_tensors.keys()) if hasattr(batch_tensors, "keys") else set()
        if "rm_scores" in batch_keys:
            return original(self, batch, reward_fn=reward_fn, return_dict=return_dict, sum_reward=sum_reward)

        try:
            rows = dataproto_to_rows(batch=batch, tokenizer=getattr(self, "tokenizer", None))
            responses = batch_tensors.get("responses")
            response_width = _resolve_response_width(responses)
            device = getattr(responses, "device", None)
            reward_tensor, reward_extra_infos_dict = rows_to_reward_tensor(
                rows,
                response_width=response_width,
                device=device,
            )

            reward_output = reward_tensor.sum(dim=-1) if sum_reward else reward_tensor
            if return_dict:
                return {
                    "reward_tensor": reward_output,
                    "reward_extra_info": reward_extra_infos_dict,
                }
            if sum_reward:
                return reward_output
            return reward_output, reward_extra_infos_dict
        except Exception as exc:  # pragma: no cover - fallback path
            LOGGER.warning(
                "SWE reward-adapter path failed; falling back to upstream reward computation: %s",
                exc,
                exc_info=True,
            )
            return original(self, batch, reward_fn=reward_fn, return_dict=return_dict, sum_reward=sum_reward)

    def _patched_maybe_build_self_distillation_batch(
        self: Any,
        batch: Any,
        reward_tensor: Any,
        reward_extra_infos_dict: dict[str, list[Any]] | None = None,
    ) -> Any:
        original = getattr(type(self), _ORIGINAL_DISTILL_ATTR)
        if not (_is_swe_bridge_loop_enabled(self) or _batch_looks_like_swe_bridge(batch)):
            return original(self, batch, reward_tensor, reward_extra_infos_dict)

        self_distillation_cfg = _resolve_self_distillation_cfg(self)
        if self_distillation_cfg is None:
            return None

        try:
            if torch is None:
                raise RuntimeError("torch is required for self-distillation runtime patch.")

            rows = dataproto_to_rows(batch=batch, tokenizer=getattr(self, "tokenizer", None))
            if not rows:
                return None

            success_reward_threshold = float(_cfg_get(self_distillation_cfg, "success_reward_threshold", 1.0))
            reward_sums = reward_tensor.sum(dim=-1)
            resolved_flags = _to_bool_list(reward_sums >= success_reward_threshold)
            for index, row in enumerate(rows):
                row["resolved"] = resolved_flags[index] if index < len(resolved_flags) else False

            include_student_attempt = bool(
                _cfg_get(self_distillation_cfg, "include_student_attempt_for_teacher", True)
            )
            max_reprompt_len = int(_cfg_get(self_distillation_cfg, "max_reprompt_len", 10240))
            reprompt_batch = build_self_distillation_batch(
                rows,
                include_student_attempt_for_teacher=include_student_attempt,
                max_reprompt_len=max_reprompt_len,
            )
            teacher_prompts = [str(item) for item in reprompt_batch.get("teacher_prompts", [])]
            if not teacher_prompts:
                return None

            responses = batch.batch["responses"]
            response_mask = batch.batch["response_mask"]
            response_device = getattr(responses, "device", None)
            enable_thinking = _resolve_enable_thinking(self)
            teacher_prompt_tensors = _tokenize_teacher_prompts(
                tokenizer=getattr(self, "tokenizer", None),
                rows=rows,
                teacher_prompts=teacher_prompts,
                max_reprompt_len=max_reprompt_len,
                enable_thinking=enable_thinking,
                device=response_device,
            )

            response_mask_tensor = _ensure_tensor_like(
                response_mask,
                device=response_device,
                dtype=teacher_prompt_tensors["attention_mask"].dtype,
            )
            teacher_input_ids = torch.cat([teacher_prompt_tensors["input_ids"], responses], dim=1)
            teacher_attention_mask = torch.cat(
                [teacher_prompt_tensors["attention_mask"], response_mask_tensor],
                dim=1,
            )
            teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

            self_distillation_mask_values = [bool(item) for item in reprompt_batch["self_distillation_mask"]]
            self_distillation_mask_values = _pad_or_trim_bools(self_distillation_mask_values, target_len=len(rows))
            self_distillation_mask = torch.tensor(
                self_distillation_mask_values,
                dtype=torch.float32,
                device=response_device,
            )

            prompt_truncated = _pad_or_trim_bools(
                [bool(item) for item in reprompt_batch.get("prompt_truncated", [])],
                target_len=len(rows),
            )
            feedback_count = _count_non_empty_feedback(reward_extra_infos_dict, batch_size=len(rows))
            resolved_count = sum(bool(row.get("resolved")) for row in rows)
            active_count = int(sum(self_distillation_mask_values))
            batch_size = max(len(rows), 1)

            metrics = {
                "self_distillation/success_sample_fraction": resolved_count / batch_size,
                "self_distillation/feedback_available_fraction": feedback_count / batch_size,
                "self_distillation/reprompt_sample_fraction": active_count / batch_size,
                "self_distillation/prompt_truncated_fraction": sum(prompt_truncated) / batch_size,
                "self_distillation/empty_target_batch": 1.0 if active_count == 0 else 0.0,
            }
            return DataProto.from_dict(
                tensors={
                    "teacher_input_ids": teacher_input_ids,
                    "teacher_attention_mask": teacher_attention_mask,
                    "teacher_position_ids": teacher_position_ids,
                    "self_distillation_mask": self_distillation_mask,
                }
            ), metrics
        except Exception as exc:  # pragma: no cover - fallback path
            LOGGER.warning(
                "SWE self-distillation patch failed; falling back to upstream hook: %s",
                exc,
                exc_info=True,
            )
            return original(self, batch, reward_tensor, reward_extra_infos_dict)

    setattr(trainer_cls, "_compute_or_extract_reward", _patched_compute_or_extract_reward)
    setattr(trainer_cls, "_maybe_build_self_distillation_batch", _patched_maybe_build_self_distillation_batch)
    setattr(trainer_cls, _PATCH_MARKER_ATTR, True)
    return True


def _resolve_self_distillation_cfg(trainer: Any) -> Any | None:
    actor_cfg = _cfg_get(_cfg_get(_cfg_get(trainer, "config"), "actor_rollout_ref"), "actor")
    if actor_cfg is None:
        return None
    self_distillation_cfg = _cfg_get(actor_cfg, "self_distillation")
    loss_mode = _cfg_get(_cfg_get(actor_cfg, "policy_loss"), "loss_mode", "vanilla")
    if self_distillation_cfg is None or str(loss_mode).strip().lower() != "sdpo":
        return None
    return self_distillation_cfg


def _resolve_enable_thinking(trainer: Any) -> bool:
    data_cfg = _cfg_get(_cfg_get(trainer, "config"), "data")
    apply_chat_template_kwargs = _cfg_get(data_cfg, "apply_chat_template_kwargs")
    enable_thinking = _cfg_get(apply_chat_template_kwargs, "enable_thinking", True)
    return bool(enable_thinking)


def _is_swe_bridge_loop_enabled(trainer: Any) -> bool:
    rollout_cfg = _cfg_get(_cfg_get(_cfg_get(trainer, "config"), "actor_rollout_ref"), "rollout")
    agent_cfg = _cfg_get(rollout_cfg, "agent")
    default_agent_loop = _cfg_get(agent_cfg, "default_agent_loop", "")
    return str(default_agent_loop).strip() == "swe_bridge_agent"


def _batch_looks_like_swe_bridge(batch: Any) -> bool:
    non_tensor_batch = getattr(batch, "non_tensor_batch", {}) or {}
    if not hasattr(non_tensor_batch, "keys"):
        return False
    keys = set(non_tensor_batch.keys())
    swe_keys = {
        "trajectory_steps",
        "tool_response_blocks",
        "loop_exit_reason",
        "trajectory_tool_validation_errors",
    }
    return bool(keys.intersection(swe_keys))


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                result = getter(key)
            except Exception:
                return default
            return default if result is None else result
    if hasattr(container, key):
        return getattr(container, key)
    return default


def _resolve_response_width(responses: Any) -> int | None:
    shape = getattr(responses, "shape", None)
    if isinstance(shape, Sequence) and len(shape) >= 2:
        try:
            return int(shape[1])
        except (TypeError, ValueError):
            return None
    return None


def _tokenize_teacher_prompts(
    *,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    teacher_prompts: Sequence[str],
    max_reprompt_len: int,
    enable_thinking: bool,
    device: Any = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for teacher prompt tokenization.")

    if tokenizer is None:
        raise RuntimeError("tokenizer is unavailable for SDPO runtime patch.")

    tokenized: Mapping[str, Any]
    if hasattr(tokenizer, "apply_chat_template"):
        messages: list[list[dict[str, str]]] = []
        for index, prompt in enumerate(teacher_prompts):
            raw_prompt_messages = rows[index].get("_raw_prompt_messages")
            if isinstance(raw_prompt_messages, Sequence):
                system_messages = [
                    {"role": _coerce_role(item.get("role")), "content": str(item.get("content", ""))}
                    for item in raw_prompt_messages[:-1]
                    if isinstance(item, Mapping) and _coerce_role(item.get("role"))
                ]
            else:
                system_messages = []
            messages.append(system_messages + [{"role": "user", "content": prompt}])

        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=max_reprompt_len,
            padding=True,
            truncation=True,
        )
    else:
        tokenized = tokenizer(
            list(teacher_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_reprompt_len,
        )

    input_ids = _ensure_tensor_like(tokenized["input_ids"], dtype=torch.long, device=device)
    attention_mask_raw = tokenized.get("attention_mask")
    if attention_mask_raw is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        attention_mask = _ensure_tensor_like(attention_mask_raw, dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _ensure_tensor_like(value: Any, *, dtype: Any, device: Any = None) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for tensor conversion.")
    tensor_value = value
    if not hasattr(tensor_value, "to"):
        tensor_value = torch.tensor(value, dtype=dtype)
    else:
        tensor_value = tensor_value.to(dtype=dtype)
    if device is not None:
        tensor_value = tensor_value.to(device=device)
    return tensor_value


def _pad_or_trim_bools(values: Sequence[bool], *, target_len: int) -> list[bool]:
    if target_len <= 0:
        return []
    normalized = [bool(item) for item in values]
    if len(normalized) < target_len:
        normalized.extend([False] * (target_len - len(normalized)))
    elif len(normalized) > target_len:
        normalized = normalized[:target_len]
    return normalized


def _to_bool_list(value: Any) -> list[bool]:
    if value is None:
        return []
    detached = value
    if hasattr(detached, "detach"):
        detached = detached.detach()
    if hasattr(detached, "cpu"):
        detached = detached.cpu()
    if hasattr(detached, "tolist"):
        raw_list = detached.tolist()
    elif isinstance(detached, Sequence) and not isinstance(detached, (str, bytes)):
        raw_list = list(detached)
    else:
        raw_list = [detached]
    return [_coerce_bool(item) for item in raw_list]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value) != 0
    if isinstance(value, float):
        return value != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"system", "user", "assistant"}:
        return role
    return ""


def _count_non_empty_feedback(
    reward_extra_infos_dict: Mapping[str, Any] | None,
    *,
    batch_size: int,
) -> int:
    if reward_extra_infos_dict is None:
        return 0
    raw_feedback = reward_extra_infos_dict.get("feedback")
    if not isinstance(raw_feedback, Sequence) or isinstance(raw_feedback, (str, bytes)):
        return 0
    count = 0
    for index in range(min(len(raw_feedback), batch_size)):
        value = raw_feedback[index]
        if isinstance(value, str) and value.strip():
            count += 1
    return count
