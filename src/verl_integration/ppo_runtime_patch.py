"""Runtime monkeypatches for SWE-specific SDPO wiring in verl PPO trainer."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import logging
import numbers
import os
import time
from typing import Any, Mapping, Sequence

from metrics.profiler import cuda_memory_metrics, reset_cuda_peak_memory_stats, token_profile_metrics
from verl_integration.reprompt_adapter import (
    DEFAULT_MAX_REPROMPT_LEN,
    build_self_distillation_batch,
)
from verl_integration.reward_adapter import dataproto_to_rows, rows_to_reward_tensor

LOGGER = logging.getLogger(__name__)

_PATCH_MARKER_ATTR = "_small_swe_sdpo_runtime_patch_applied"
_ORIGINAL_REWARD_ATTR = "_small_swe_original_compute_or_extract_reward"
_ORIGINAL_DISTILL_ATTR = "_small_swe_original_maybe_build_self_distillation_batch"
_ORIGINAL_VAL_METRICS_UPDATE_ATTR = "_small_swe_original_val_metrics_update"
_VAL_METRICS_PATCH_MARKER_ATTR = "_small_swe_val_metrics_patch_applied"
_ACTOR_PATCH_MARKER_ATTR = "_small_swe_turn_level_actor_patch_applied"
_ORIGINAL_ACTOR_UPDATE_ATTR = "_small_swe_original_update_policy"
_AGENT_LOOP_PATCH_MARKER_ATTR = "_small_swe_agent_loop_queue_patch_applied"
_ORIGINAL_AGENT_LOOP_GENERATE_ATTR = "_small_swe_original_agent_loop_generate_sequences"
_AGENT_LOOP_SERVER_PATCH_MARKER_ATTR = "_small_swe_agent_loop_server_patch_applied"
_ORIGINAL_AGENT_LOOP_CHOOSE_SERVER_ATTR = "_small_swe_original_agent_loop_choose_server"
_DISTRIBUTED_TURN_LEVEL_EXPANSION_ENV = "SMALL_SWE_ENABLE_DISTRIBUTED_TURN_LEVEL_EXPANSION"
_TURN_SUPERVISION_NEXT = "next_turn"
_TURN_SUPERVISION_CURRENT = "current_turn"
_TURN_SUPERVISION_MODES = {_TURN_SUPERVISION_NEXT, _TURN_SUPERVISION_CURRENT}
_AGENT_LOOP_MAX_IN_FLIGHT_ENV = "SMALL_SWE_SDPO_AGENT_LOOP_MAX_IN_FLIGHT"
_VERIFIER_FEEDBACK_NONE = "none"
_VERIFIER_FEEDBACK_FINAL_TURN_ONLY = "final_turn_only"
_VERIFIER_FEEDBACK_ALL_TURNS = "all_turns"
_VERIFIER_FEEDBACK_MODES = {
    _VERIFIER_FEEDBACK_NONE,
    _VERIFIER_FEEDBACK_FINAL_TURN_ONLY,
    _VERIFIER_FEEDBACK_ALL_TURNS,
}
_LEGACY_GATING_RESOLVED_ONLY = "resolved_only"
_LEGACY_GATING_FEEDBACK_PRESENT = "feedback_present"
_LEGACY_GATING_ALWAYS = "always"
_LEGACY_GATING_POLICIES = {
    _LEGACY_GATING_RESOLVED_ONLY,
    _LEGACY_GATING_FEEDBACK_PRESENT,
    _LEGACY_GATING_ALWAYS,
}
_TURN_LEVEL_REQUIRED_KEYS = {
    "turn_teacher_input_ids",
    "turn_teacher_attention_mask",
    "turn_teacher_position_ids",
    "turn_response_mask",
    "turn_self_distillation_mask",
}
_NON_RECOVERABLE_REWARD_ERROR_FRAGMENTS = {
    "SWE rows require non-empty _response_mask",
}

try:  # pragma: no cover - exercised in train runtime
    import torch
except ModuleNotFoundError:  # pragma: no cover - unit-test environments without train deps
    torch = None  # type: ignore[assignment]


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _is_non_recoverable_reward_adapter_error(exc: Exception) -> bool:
    message = str(exc)
    return any(fragment in message for fragment in _NON_RECOVERABLE_REWARD_ERROR_FRAGMENTS)


def _should_skip_turn_level_expansion_for_distributed() -> bool:
    if _env_flag_enabled(_DISTRIBUTED_TURN_LEVEL_EXPANSION_ENV, default=False):
        return False
    if torch is None:
        return False
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return False
    if not distributed.is_available() or not distributed.is_initialized():
        return False
    try:
        return distributed.get_world_size() > 1
    except Exception:
        return True


def _normalize_turn_supervision_mode(value: Any) -> str:
    if value is None:
        return _TURN_SUPERVISION_CURRENT
    normalized = str(value).strip().lower()
    if not normalized:
        return _TURN_SUPERVISION_CURRENT
    if normalized not in _TURN_SUPERVISION_MODES:
        supported = ", ".join(sorted(_TURN_SUPERVISION_MODES))
        raise ValueError(f"turn_supervision_mode must be one of: {supported}")
    return normalized


def _normalize_verifier_feedback_mode(value: Any) -> str:
    if value is None:
        return _VERIFIER_FEEDBACK_NONE
    normalized = str(value).strip().lower()
    if not normalized:
        return _VERIFIER_FEEDBACK_NONE
    if normalized not in _VERIFIER_FEEDBACK_MODES:
        supported = ", ".join(sorted(_VERIFIER_FEEDBACK_MODES))
        raise ValueError(f"verifier_feedback_mode must be one of: {supported}")
    return normalized


def _normalize_legacy_gating_policy(value: Any) -> str:
    if value is None:
        return _LEGACY_GATING_RESOLVED_ONLY
    normalized = str(value).strip().lower()
    if not normalized:
        return _LEGACY_GATING_RESOLVED_ONLY
    if normalized not in _LEGACY_GATING_POLICIES:
        supported = ", ".join(sorted(_LEGACY_GATING_POLICIES))
        raise ValueError(f"legacy_distillation_gating_policy must be one of: {supported}")
    return normalized


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
        _apply_validation_metrics_update_patch(trainer_cls)
        _apply_turn_level_actor_update_patch()
        _apply_agent_loop_server_routing_patch()
        _apply_agent_loop_queue_patch()
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
        responses = batch_tensors.get("responses")
        expected_len = _resolve_response_batch_size(responses)
        if "rm_scores" in batch_keys:
            reward_output = original(self, batch, reward_fn=reward_fn, return_dict=return_dict, sum_reward=sum_reward)
            return _sanitize_reward_result_payload(
                reward_output,
                return_dict=return_dict,
                sum_reward=sum_reward,
                expected_len=expected_len,
            )

        try:
            rows = dataproto_to_rows(batch=batch, tokenizer=getattr(self, "tokenizer", None))
            response_width = _resolve_response_width(responses)
            device = getattr(responses, "device", None)
            reward_tensor, reward_extra_infos_dict = rows_to_reward_tensor(
                rows,
                response_width=response_width,
                device=device,
            )

            reward_output = reward_tensor.sum(dim=-1) if sum_reward else reward_tensor
            if return_dict:
                return _sanitize_reward_result_payload(
                    {
                        "reward_tensor": reward_output,
                        "reward_extra_info": reward_extra_infos_dict,
                    },
                    return_dict=return_dict,
                    sum_reward=sum_reward,
                    expected_len=expected_len,
                )
            if sum_reward:
                return reward_output
            return _sanitize_reward_result_payload(
                (reward_output, reward_extra_infos_dict),
                return_dict=return_dict,
                sum_reward=sum_reward,
                expected_len=expected_len,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            if _is_non_recoverable_reward_adapter_error(exc):
                raise
            LOGGER.warning(
                "SWE reward-adapter path failed; falling back to upstream reward computation: %s",
                exc,
                exc_info=True,
            )
            reward_output = original(self, batch, reward_fn=reward_fn, return_dict=return_dict, sum_reward=sum_reward)
            return _sanitize_reward_result_payload(
                reward_output,
                return_dict=return_dict,
                sum_reward=sum_reward,
                expected_len=expected_len,
            )

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

        turn_supervision_mode = _normalize_turn_supervision_mode(
            _cfg_get(self_distillation_cfg, "turn_supervision_mode", _TURN_SUPERVISION_CURRENT)
        )
        verifier_feedback_mode = _normalize_verifier_feedback_mode(
            _cfg_get(self_distillation_cfg, "verifier_feedback_mode", _VERIFIER_FEEDBACK_ALL_TURNS)
        )
        legacy_distillation_gating_policy = _normalize_legacy_gating_policy(
            _cfg_get(
                self_distillation_cfg,
                "legacy_distillation_gating_policy",
                _LEGACY_GATING_RESOLVED_ONLY,
            )
        )

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
            include_teacher_memory_blocks = _coerce_bool(
                _cfg_get(self_distillation_cfg, "include_teacher_memory_blocks", True)
            )
            max_reprompt_len = int(_cfg_get(self_distillation_cfg, "max_reprompt_len", DEFAULT_MAX_REPROMPT_LEN))
            num_recent_raw_blocks = int(_cfg_get(self_distillation_cfg, "num_recent_raw_blocks", 3))
            responses = batch.batch["responses"]
            response_width = _resolve_response_width(responses) or 0
            actor_cfg = _cfg_get(_cfg_get(_cfg_get(self, "config"), "actor_rollout_ref"), "actor")
            ppo_max_token_len_per_gpu = int(_cfg_get(actor_cfg, "ppo_max_token_len_per_gpu", 0) or 0)
            ulysses_sequence_parallel_size = int(_cfg_get(actor_cfg, "ulysses_sequence_parallel_size", 1) or 1)
            max_token_len = ppo_max_token_len_per_gpu * max(ulysses_sequence_parallel_size, 1)
            if response_width > 0 and max_token_len > 0:
                safe_max_reprompt_len = max_token_len - response_width
                if safe_max_reprompt_len <= 0:
                    raise ValueError(
                        "Invalid SDPO token budget: response width exceeds or equals actor max token length "
                        f"(response_width={response_width}, max_token_len={max_token_len})."
                    )
                if max_reprompt_len > safe_max_reprompt_len:
                    LOGGER.warning(
                        "Clipping self_distillation.max_reprompt_len from %s to %s to satisfy "
                        "teacher sequence budget (response_width=%s, max_token_len=%s).",
                        max_reprompt_len,
                        safe_max_reprompt_len,
                        response_width,
                        max_token_len,
                    )
                    max_reprompt_len = safe_max_reprompt_len
            reprompt_batch = build_self_distillation_batch(
                rows,
                include_student_attempt_for_teacher=include_student_attempt,
                include_teacher_memory_blocks=include_teacher_memory_blocks,
                max_reprompt_len=max_reprompt_len,
                num_recent_raw_blocks=num_recent_raw_blocks,
                turn_supervision_mode=turn_supervision_mode,
                verifier_feedback_mode=verifier_feedback_mode,
                legacy_distillation_gating_policy=legacy_distillation_gating_policy,
            )
            teacher_prompts = [str(item) for item in reprompt_batch.get("teacher_prompts", [])]
            if not teacher_prompts:
                return None

            response_mask = batch.batch["response_mask"]
            response_device = getattr(responses, "device", None)
            teacher_prompt_tensors = _tokenize_teacher_prompts(
                tokenizer=getattr(self, "tokenizer", None),
                rows=rows,
                teacher_prompts=teacher_prompts,
                max_reprompt_len=max_reprompt_len,
                device=response_device,
            )

            response_mask_tensor = _ensure_tensor_like(
                response_mask,
                device=response_device,
                dtype=teacher_prompt_tensors["attention_mask"].dtype,
            )
            teacher_response_attention_mask = _build_teacher_response_attention_mask(
                responses=responses,
                response_mask=response_mask_tensor,
                tokenizer=getattr(self, "tokenizer", None),
                device=response_device,
                dtype=teacher_prompt_tensors["attention_mask"].dtype,
            )
            teacher_input_ids = torch.cat([teacher_prompt_tensors["input_ids"], responses], dim=1)
            teacher_attention_mask = torch.cat(
                [teacher_prompt_tensors["attention_mask"], teacher_response_attention_mask],
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
            (
                turn_teacher_input_ids,
                turn_teacher_attention_mask,
                turn_teacher_position_ids,
                turn_response_mask,
                turn_self_distillation_mask,
                turn_pair_count_per_sample,
            ) = _build_turn_level_teacher_tensors(
                tokenizer=getattr(self, "tokenizer", None),
                rows=rows,
                turn_teacher_prompts=reprompt_batch.get("turn_teacher_prompts"),
                turn_response_masks=reprompt_batch.get("turn_response_masks"),
                turn_distillation_mask=reprompt_batch.get("turn_distillation_mask"),
                responses=responses,
                response_mask=response_mask_tensor,
                max_reprompt_len=max_reprompt_len,
                compute_position_id_with_mask=compute_position_id_with_mask,
                device=response_device,
            )
            feedback_count = _count_non_empty_feedback(reward_extra_infos_dict, batch_size=len(rows))
            resolved_count = sum(bool(row.get("resolved")) for row in rows)
            active_count = int(sum(self_distillation_mask_values))
            batch_size = max(len(rows), 1)
            turn_pair_count = int(sum(turn_pair_count_per_sample))
            supervised_token_ratio = float(response_mask_tensor.float().mean().item())
            teacher_attention_valid_token_ratio = float(
                teacher_response_attention_mask.float().mean().item()
            )
            invalid_supervised_overlap_count = int(
                ((response_mask_tensor > 0) & (teacher_response_attention_mask == 0))
                .sum()
                .item()
            )
            response_width = _resolve_response_width(responses) or 0
            if (
                response_width > 0
                and turn_teacher_attention_mask is not None
                and turn_response_mask is not None
            ):
                turn_teacher_response_attention_mask = turn_teacher_attention_mask[..., -response_width:]
                invalid_supervised_overlap_count += int(
                    ((turn_response_mask > 0) & (turn_teacher_response_attention_mask == 0))
                    .sum()
                    .item()
                )

            metrics = {
                "self_distillation/success_sample_fraction": resolved_count / batch_size,
                "self_distillation/feedback_available_fraction": feedback_count / batch_size,
                "self_distillation/reprompt_sample_fraction": active_count / batch_size,
                "self_distillation/prompt_truncated_fraction": sum(prompt_truncated) / batch_size,
                "self_distillation/empty_target_batch": 1.0 if active_count == 0 else 0.0,
                "self_distillation/turn_pair_count_per_sample": turn_pair_count / batch_size,
                "self_distillation/teacher_attention_valid_token_ratio": (
                    teacher_attention_valid_token_ratio
                ),
                "self_distillation/supervised_token_ratio": supervised_token_ratio,
                "self_distillation/invalid_supervised_overlap_count": float(
                    invalid_supervised_overlap_count
                ),
                "self_distillation/turn_supervision_mode_next_turn": (
                    1.0 if turn_supervision_mode == _TURN_SUPERVISION_NEXT else 0.0
                ),
                "self_distillation/turn_supervision_mode_current_turn": (
                    1.0 if turn_supervision_mode == _TURN_SUPERVISION_CURRENT else 0.0
                ),
                "self_distillation/verifier_feedback_mode_none": (
                    1.0 if verifier_feedback_mode == _VERIFIER_FEEDBACK_NONE else 0.0
                ),
                "self_distillation/verifier_feedback_mode_final_turn_only": (
                    1.0 if verifier_feedback_mode == _VERIFIER_FEEDBACK_FINAL_TURN_ONLY else 0.0
                ),
                "self_distillation/verifier_feedback_mode_all_turns": (
                    1.0 if verifier_feedback_mode == _VERIFIER_FEEDBACK_ALL_TURNS else 0.0
                ),
                "self_distillation/legacy_distillation_gating_policy_resolved_only": (
                    1.0 if legacy_distillation_gating_policy == _LEGACY_GATING_RESOLVED_ONLY else 0.0
                ),
                "self_distillation/legacy_distillation_gating_policy_feedback_present": (
                    1.0 if legacy_distillation_gating_policy == _LEGACY_GATING_FEEDBACK_PRESENT else 0.0
                ),
                "self_distillation/legacy_distillation_gating_policy_always": (
                    1.0 if legacy_distillation_gating_policy == _LEGACY_GATING_ALWAYS else 0.0
                ),
            }
            LOGGER.debug(
                (
                    "Built self-distillation batch with turn_supervision_mode=%s, "
                    "verifier_feedback_mode=%s, legacy_gating=%s "
                    "(active=%s/%s, turn_pairs=%s, supervised_ratio=%.4f, attention_ratio=%.4f, invalid_overlap=%s)."
                ),
                turn_supervision_mode,
                verifier_feedback_mode,
                legacy_distillation_gating_policy,
                active_count,
                len(rows),
                turn_pair_count,
                supervised_token_ratio,
                teacher_attention_valid_token_ratio,
                invalid_supervised_overlap_count,
            )
            tensors = {
                "teacher_input_ids": teacher_input_ids,
                "teacher_attention_mask": teacher_attention_mask,
                "teacher_position_ids": teacher_position_ids,
                "self_distillation_mask": self_distillation_mask,
            }
            if turn_teacher_input_ids is not None:
                tensors.update(
                    {
                        "turn_teacher_input_ids": turn_teacher_input_ids,
                        "turn_teacher_attention_mask": turn_teacher_attention_mask,
                        "turn_teacher_position_ids": turn_teacher_position_ids,
                        "turn_response_mask": turn_response_mask,
                        "turn_self_distillation_mask": turn_self_distillation_mask,
                    }
                )
            return DataProto.from_dict(tensors=tensors), metrics
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
    _apply_validation_metrics_update_patch(trainer_cls)
    _apply_turn_level_actor_update_patch()
    _apply_agent_loop_server_routing_patch()
    _apply_agent_loop_queue_patch()
    return True


def _apply_agent_loop_server_routing_patch(agent_loop_module: Any | None = None) -> bool:
    if agent_loop_module is None:
        try:
            agent_loop_module = importlib.import_module("verl.experimental.agent_loop.agent_loop")
        except ModuleNotFoundError:
            return False

    manager_cls = getattr(agent_loop_module, "AsyncLLMServerManager", None)
    if manager_cls is None:
        return False
    if getattr(manager_cls, _AGENT_LOOP_SERVER_PATCH_MARKER_ATTR, False):
        return True

    original_choose_server = getattr(manager_cls, "_choose_server", None)
    if not callable(original_choose_server):
        return False

    setattr(manager_cls, _ORIGINAL_AGENT_LOOP_CHOOSE_SERVER_ATTR, original_choose_server)

    def _patched_choose_server(self: Any, request_id: str) -> Any:
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        handles = getattr(self, "server_handles", None)
        if not isinstance(handles, Sequence) or len(handles) == 0:
            original_impl = getattr(type(self), _ORIGINAL_AGENT_LOOP_CHOOSE_SERVER_ATTR)
            return original_impl(self, request_id)

        # Spread requests uniformly across all rollout replicas while preserving
        # sticky routing for multi-turn conversations on the same request_id.
        request_digest = hashlib.sha1(str(request_id).encode("utf-8")).digest()
        server_index = int.from_bytes(request_digest[:8], byteorder="big", signed=False) % len(handles)
        server = handles[server_index]
        self.request_id_to_server[request_id] = server
        return server

    setattr(manager_cls, "_choose_server", _patched_choose_server)
    setattr(manager_cls, _AGENT_LOOP_SERVER_PATCH_MARKER_ATTR, True)
    return True


def _apply_agent_loop_queue_patch(agent_loop_module: Any | None = None) -> bool:
    if agent_loop_module is None:
        try:
            agent_loop_module = importlib.import_module("verl.experimental.agent_loop.agent_loop")
        except ModuleNotFoundError:
            return False

    worker_cls = getattr(agent_loop_module, "AgentLoopWorker", None)
    if worker_cls is None:
        return False
    if getattr(worker_cls, _AGENT_LOOP_PATCH_MARKER_ATTR, False):
        return True

    original_generate = getattr(worker_cls, "generate_sequences", None)
    np_module = getattr(agent_loop_module, "np", None)
    rollout_trace_config_cls = getattr(agent_loop_module, "RolloutTraceConfig", None)
    get_trajectory_info = getattr(agent_loop_module, "get_trajectory_info", None)
    if (
        not callable(original_generate)
        or np_module is None
        or rollout_trace_config_cls is None
        or not callable(get_trajectory_info)
    ):
        return False

    setattr(worker_cls, _ORIGINAL_AGENT_LOOP_GENERATE_ATTR, original_generate)
    agent_loop_registry = getattr(agent_loop_module, "_agent_loop_registry", {})

    async def _patched_generate_sequences(self: Any, batch: Any) -> Any:
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np_module.array([default_agent_loop] * len(batch), dtype=object)

        if not _batch_uses_swe_bridge_agent(batch.non_tensor_batch.get("agent_name")):
            original_impl = getattr(type(self), _ORIGINAL_AGENT_LOOP_GENERATE_ATTR)
            return await original_impl(self, batch)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np_module.arange(len(batch))

        max_samples_per_worker = rollout_trace_config_cls.get_instance().max_samples_per_step_per_worker
        if max_samples_per_worker is not None:
            unique_sample_indices = np_module.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np_module.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        batch_size = len(batch)
        max_in_flight = _resolve_swe_agent_loop_max_in_flight(
            worker=self,
            batch_size=batch_size,
            agent_loop_registry=agent_loop_registry,
        )
        max_in_flight = max(1, min(max_in_flight, batch_size))

        if max_in_flight >= batch_size:
            tasks = []
            for sample_index in range(batch_size):
                trace_this_sample = sample_index in traced_indices
                kwargs = {k: v[sample_index] for k, v in batch.non_tensor_batch.items()}
                tasks.append(
                    asyncio.create_task(
                        self._run_agent_loop(
                            sampling_params,
                            trajectory_info[sample_index],
                            trace=trace_this_sample,
                            **kwargs,
                        )
                    )
                )
            outputs = await asyncio.gather(*tasks)
            return self._postprocess(outputs)

        outputs: list[Any | None] = [None] * batch_size
        in_flight: set[asyncio.Task[tuple[int, Any]]] = set()
        next_sample_index = 0

        def _launch(sample_index: int) -> asyncio.Task[tuple[int, Any]]:
            trace_this_sample = sample_index in traced_indices
            kwargs = {k: v[sample_index] for k, v in batch.non_tensor_batch.items()}

            async def _run_one() -> tuple[int, Any]:
                result = await self._run_agent_loop(
                    sampling_params,
                    trajectory_info[sample_index],
                    trace=trace_this_sample,
                    **kwargs,
                )
                return sample_index, result

            return asyncio.create_task(_run_one())

        try:
            while next_sample_index < batch_size and len(in_flight) < max_in_flight:
                in_flight.add(_launch(next_sample_index))
                next_sample_index += 1

            while in_flight:
                done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                in_flight = set(pending)
                for finished in done:
                    sample_index, sample_output = finished.result()
                    outputs[sample_index] = sample_output
                    if next_sample_index < batch_size:
                        in_flight.add(_launch(next_sample_index))
                        next_sample_index += 1
        except Exception:
            for task in in_flight:
                task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
            raise

        if any(sample_output is None for sample_output in outputs):
            raise RuntimeError("Bounded swe_bridge agent loop scheduler produced incomplete outputs.")
        return self._postprocess(outputs)

    tqbridge = getattr(agent_loop_module, "tqbridge", None)
    patched_generate: Any = _patched_generate_sequences
    if callable(tqbridge):
        try:
            patched_generate = tqbridge()(_patched_generate_sequences)
        except Exception:
            patched_generate = _patched_generate_sequences

    setattr(worker_cls, "generate_sequences", patched_generate)
    setattr(worker_cls, _AGENT_LOOP_PATCH_MARKER_ATTR, True)
    return True


def _apply_validation_metrics_update_patch(trainer_cls: Any) -> bool:
    if getattr(trainer_cls, _VAL_METRICS_PATCH_MARKER_ATTR, False):
        return True

    original = getattr(trainer_cls, "_val_metrics_update", None)
    if not callable(original):
        return False

    setattr(trainer_cls, _ORIGINAL_VAL_METRICS_UPDATE_ATTR, original)

    def _patched_val_metrics_update(
        self: Any,
        data_sources: Any,
        sample_uids: Any,
        reward_extra_infos_dict: Any,
        sample_turns: Any,
    ) -> Any:
        original_impl = getattr(type(self), _ORIGINAL_VAL_METRICS_UPDATE_ATTR)
        sanitized_reward_infos = _sanitize_validation_reward_extra_infos(
            reward_extra_infos_dict,
            expected_len=_safe_len(sample_uids),
        )
        return original_impl(self, data_sources, sample_uids, sanitized_reward_infos, sample_turns)

    setattr(trainer_cls, "_val_metrics_update", _patched_val_metrics_update)
    setattr(trainer_cls, _VAL_METRICS_PATCH_MARKER_ATTR, True)
    return True


def _safe_len(value: Any) -> int | None:
    try:
        return int(len(value))
    except Exception:
        return None


def _coerce_optional_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        candidate = int(value)
        return candidate if candidate > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            candidate = int(stripped)
        except ValueError:
            return None
        return candidate if candidate > 0 else None
    return None


def _batch_uses_swe_bridge_agent(agent_names: Any) -> bool:
    if agent_names is None:
        return False
    if isinstance(agent_names, (str, bytes)):
        return str(agent_names).strip() == "swe_bridge_agent"
    if hasattr(agent_names, "tolist"):
        try:
            return _batch_uses_swe_bridge_agent(agent_names.tolist())
        except Exception:
            pass
    if isinstance(agent_names, Sequence):
        for name in agent_names:
            if _batch_uses_swe_bridge_agent(name):
                return True
        return False
    iterator = getattr(agent_names, "__iter__", None)
    if callable(iterator):
        try:
            for name in iterator():
                if _batch_uses_swe_bridge_agent(name):
                    return True
            return False
        except Exception:
            return False
    return str(agent_names).strip() == "swe_bridge_agent"


def _resolve_swe_agent_loop_max_in_flight(
    *,
    worker: Any,
    batch_size: int,
    agent_loop_registry: Mapping[str, Any] | None,
) -> int:
    env_override = _coerce_optional_positive_int(os.environ.get(_AGENT_LOOP_MAX_IN_FLIGHT_ENV))
    if env_override is not None:
        return env_override

    rollout_cfg = _cfg_get(_cfg_get(_cfg_get(worker, "config"), "actor_rollout_ref"), "rollout")
    agent_cfg = _cfg_get(rollout_cfg, "agent")

    configured = _coerce_optional_positive_int(_cfg_get(agent_cfg, "max_in_flight_tasks"))
    if configured is not None:
        return configured

    default_agent_loop = str(_cfg_get(agent_cfg, "default_agent_loop", "")).strip()
    agent_loop_config = None
    if default_agent_loop and isinstance(agent_loop_registry, Mapping):
        agent_loop_config = agent_loop_registry.get(default_agent_loop)
    env_pool_size = _coerce_optional_positive_int(_cfg_get(agent_loop_config, "env_pool_size"))
    if env_pool_size is None:
        return max(1, batch_size)

    num_workers = _coerce_optional_positive_int(_cfg_get(agent_cfg, "num_workers")) or 1
    return max(1, (env_pool_size + num_workers - 1) // num_workers)


def _resolve_response_batch_size(responses: Any) -> int | None:
    shape = getattr(responses, "shape", None)
    if isinstance(shape, Sequence) and len(shape) >= 1:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None
    try:
        return int(len(responses))
    except Exception:
        return None


def _sanitize_reward_result_payload(
    payload: Any,
    *,
    return_dict: bool,
    sum_reward: bool,
    expected_len: int | None,
) -> Any:
    if return_dict:
        if isinstance(payload, Mapping):
            output = dict(payload)
            output["reward_extra_info"] = _sanitize_validation_reward_extra_infos(
                output.get("reward_extra_info"),
                expected_len=expected_len,
            )
            return output
        return payload
    if sum_reward:
        return payload
    if isinstance(payload, tuple) and len(payload) == 2:
        reward_tensor, reward_extra_infos = payload
        sanitized_extra_infos = _sanitize_validation_reward_extra_infos(
            reward_extra_infos,
            expected_len=expected_len,
        )
        return reward_tensor, sanitized_extra_infos
    return payload


def _sanitize_validation_reward_extra_infos(
    reward_extra_infos_dict: Mapping[str, Any] | None,
    *,
    expected_len: int | None = None,
) -> dict[str, list[Any]]:
    if reward_extra_infos_dict is None or not hasattr(reward_extra_infos_dict, "items"):
        return {}

    sanitized: dict[str, list[Any]] = {}
    for raw_key, raw_values in reward_extra_infos_dict.items():
        key = str(raw_key)
        values = _coerce_validation_values_list(raw_values)
        if expected_len is not None and expected_len >= 0 and len(values) not in {0, expected_len}:
            continue

        if not values:
            sanitized[key] = []
            continue

        if key == "pred":
            sanitized[key] = [_to_validation_text(item) for item in values]
            continue

        non_none_values = [item for item in values if item is not None]
        should_treat_as_numeric = bool(non_none_values) and all(
            _is_numeric_validation_value(item) for item in non_none_values
        )
        if should_treat_as_numeric:
            sanitized[key] = [_to_validation_numeric_or_nan(item) for item in values]
        else:
            sanitized[key] = [_to_validation_text(item) for item in values]

    return sanitized


def _coerce_validation_values_list(value: Any) -> list[Any]:
    current = value
    if hasattr(current, "tolist"):
        try:
            current = current.tolist()
        except Exception:
            pass
    if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        return list(current)
    return [current]


def _unwrap_validation_scalar(value: Any) -> Any:
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            return item_fn()
        except Exception:
            return value
    return value


def _is_numeric_validation_value(value: Any) -> bool:
    scalar = _unwrap_validation_scalar(value)
    return isinstance(scalar, numbers.Real)


def _to_validation_numeric_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    scalar = _unwrap_validation_scalar(value)
    if isinstance(scalar, numbers.Real):
        return float(scalar)
    return float("nan")


def _to_validation_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    return str(value)


def _resolve_self_distillation_cfg(trainer: Any) -> Any | None:
    actor_cfg = _cfg_get(_cfg_get(_cfg_get(trainer, "config"), "actor_rollout_ref"), "actor")
    if actor_cfg is None:
        return None
    self_distillation_cfg = _cfg_get(actor_cfg, "self_distillation")
    loss_mode = _cfg_get(_cfg_get(actor_cfg, "policy_loss"), "loss_mode", "vanilla")
    if self_distillation_cfg is None or str(loss_mode).strip().lower() != "sdpo":
        return None
    return self_distillation_cfg


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


def _build_teacher_response_attention_mask(
    *,
    responses: Any,
    response_mask: Any,
    tokenizer: Any,
    device: Any,
    dtype: Any,
) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for teacher attention-mask construction.")
    response_mask_tensor = _ensure_tensor_like(response_mask, dtype=dtype, device=device)
    responses_tensor = _ensure_tensor_like(responses, dtype=torch.long, device=device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return response_mask_tensor

    non_pad_mask = (responses_tensor != int(pad_token_id)).to(dtype=response_mask_tensor.dtype)
    return ((response_mask_tensor > 0) | (non_pad_mask > 0)).to(dtype=response_mask_tensor.dtype)


def _tokenize_teacher_prompts(
    *,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    teacher_prompts: Sequence[str],
    max_reprompt_len: int,
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


def _build_turn_level_teacher_tensors(
    *,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    turn_teacher_prompts: Any,
    turn_response_masks: Any,
    turn_distillation_mask: Any,
    responses: Any,
    response_mask: Any,
    max_reprompt_len: int,
    compute_position_id_with_mask: Any,
    device: Any,
) -> tuple[Any | None, Any | None, Any | None, Any | None, Any | None, list[int]]:
    if torch is None:
        return None, None, None, None, None, []
    if not rows:
        return None, None, None, None, None, []

    batch_size = len(rows)
    response_width = _resolve_response_width(responses) or 0
    if response_width <= 0:
        return None, None, None, None, None, [0 for _ in range(batch_size)]

    normalized_prompts: list[list[str]] = []
    max_turn_pairs = 0
    for row_index in range(batch_size):
        prompts_for_row = []
        if isinstance(turn_teacher_prompts, Sequence) and not isinstance(turn_teacher_prompts, (str, bytes)):
            if row_index < len(turn_teacher_prompts):
                prompts_for_row = [
                    str(item)
                    for item in (turn_teacher_prompts[row_index] or [])
                ]
        normalized_prompts.append(prompts_for_row)
        max_turn_pairs = max(max_turn_pairs, len(prompts_for_row))

    if max_turn_pairs <= 0:
        return None, None, None, None, None, [0 for _ in range(batch_size)]

    flat_prompts: list[str] = []
    flat_rows: list[Mapping[str, Any]] = []
    normalized_turn_response_masks: list[list[list[int]]] = []
    normalized_turn_distillation_mask: list[list[bool]] = []
    turn_pair_count_per_sample: list[int] = []

    for row_index in range(batch_size):
        prompts_for_row = normalized_prompts[row_index]
        masks_for_row_raw: Sequence[Any] = []
        distill_for_row_raw: Sequence[Any] = []
        if isinstance(turn_response_masks, Sequence) and not isinstance(turn_response_masks, (str, bytes)):
            if row_index < len(turn_response_masks):
                candidate = turn_response_masks[row_index]
                if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                    masks_for_row_raw = candidate
        if isinstance(turn_distillation_mask, Sequence) and not isinstance(turn_distillation_mask, (str, bytes)):
            if row_index < len(turn_distillation_mask):
                candidate = turn_distillation_mask[row_index]
                if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                    distill_for_row_raw = candidate

        row_turn_masks: list[list[int]] = []
        row_turn_distillation_mask: list[bool] = []
        row_pair_count = 0
        for turn_index in range(max_turn_pairs):
            prompt = prompts_for_row[turn_index] if turn_index < len(prompts_for_row) else ""
            flat_prompts.append(prompt)
            flat_rows.append(rows[row_index])

            raw_mask = masks_for_row_raw[turn_index] if turn_index < len(masks_for_row_raw) else []
            mask_vector = _coerce_binary_mask(raw_mask, length_hint=response_width)
            row_turn_masks.append(mask_vector)

            is_active = bool(distill_for_row_raw[turn_index]) if turn_index < len(distill_for_row_raw) else False
            is_active = is_active and any(mask_vector)
            if is_active:
                row_pair_count += 1
            row_turn_distillation_mask.append(is_active)

        normalized_turn_response_masks.append(row_turn_masks)
        normalized_turn_distillation_mask.append(row_turn_distillation_mask)
        turn_pair_count_per_sample.append(row_pair_count)

    tokenized = _tokenize_teacher_prompts(
        tokenizer=tokenizer,
        rows=flat_rows,
        teacher_prompts=flat_prompts,
        max_reprompt_len=max_reprompt_len,
        device=device,
    )
    prompt_input_ids = tokenized["input_ids"]
    prompt_attention_mask = tokenized["attention_mask"]
    prompt_width = int(prompt_input_ids.shape[1])

    responses_tensor = _ensure_tensor_like(responses, dtype=torch.long, device=device)
    response_mask_tensor = _ensure_tensor_like(response_mask, dtype=torch.long, device=device)
    response_attention_mask_tensor = _build_teacher_response_attention_mask(
        responses=responses_tensor,
        response_mask=response_mask_tensor,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.long,
    )

    responses_expanded = responses_tensor.unsqueeze(1).expand(batch_size, max_turn_pairs, response_width)
    responses_flat = responses_expanded.reshape(batch_size * max_turn_pairs, response_width)
    response_attention_mask_expanded = response_attention_mask_tensor.unsqueeze(1).expand(
        batch_size,
        max_turn_pairs,
        response_width,
    )
    response_attention_mask_flat = response_attention_mask_expanded.reshape(
        batch_size * max_turn_pairs,
        response_width,
    )

    teacher_input_ids_flat = torch.cat([prompt_input_ids, responses_flat], dim=1)
    teacher_attention_mask_flat = torch.cat([prompt_attention_mask, response_attention_mask_flat], dim=1)
    teacher_position_ids_flat = compute_position_id_with_mask(teacher_attention_mask_flat)

    full_width = prompt_width + response_width
    teacher_input_ids = teacher_input_ids_flat.reshape(batch_size, max_turn_pairs, full_width)
    teacher_attention_mask = teacher_attention_mask_flat.reshape(batch_size, max_turn_pairs, full_width)
    teacher_position_ids = teacher_position_ids_flat.reshape(batch_size, max_turn_pairs, full_width)

    turn_response_mask_tensor = torch.tensor(
        normalized_turn_response_masks,
        dtype=response_mask_tensor.dtype,
        device=device,
    )
    turn_self_distillation_mask_tensor = torch.tensor(
        normalized_turn_distillation_mask,
        dtype=torch.float32,
        device=device,
    )
    # Enforce token-level subset semantics against the actual runtime response mask.
    response_mask_expanded = response_mask_tensor.unsqueeze(1).to(dtype=turn_response_mask_tensor.dtype)
    turn_response_mask_tensor = turn_response_mask_tensor * (response_mask_expanded > 0).to(
        dtype=turn_response_mask_tensor.dtype
    )
    active_turn_pairs = (turn_response_mask_tensor.sum(dim=-1) > 0).to(dtype=turn_self_distillation_mask_tensor.dtype)
    turn_self_distillation_mask_tensor = turn_self_distillation_mask_tensor * active_turn_pairs
    turn_pair_count_per_sample = [
        int(value)
        for value in active_turn_pairs.sum(dim=-1).detach().cpu().tolist()
    ]
    return (
        teacher_input_ids,
        teacher_attention_mask,
        teacher_position_ids,
        turn_response_mask_tensor,
        turn_self_distillation_mask_tensor,
        turn_pair_count_per_sample,
    )


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


def _coerce_binary_mask(value: Any, *, length_hint: int | None = None) -> list[int]:
    values = _coerce_int_list(value)
    if length_hint is not None and length_hint >= 1:
        if len(values) < length_hint:
            values = values + [0] * (length_hint - len(values))
        elif len(values) > length_hint:
            values = values[:length_hint]
    return [1 if item else 0 for item in values]


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    detached = value
    if hasattr(detached, "detach"):
        detached = detached.detach()
    if hasattr(detached, "cpu"):
        detached = detached.cpu()
    if hasattr(detached, "tolist"):
        detached = detached.tolist()
    if not isinstance(detached, Sequence) or isinstance(detached, (str, bytes)):
        return []

    rows: list[int] = []
    for item in detached:
        if isinstance(item, bool):
            rows.append(int(item))
        elif isinstance(item, numbers.Integral):
            rows.append(int(item))
        elif isinstance(item, float) and item.is_integer():
            rows.append(int(item))
        else:
            rows.append(0)
    return rows


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


def _apply_turn_level_actor_update_patch() -> bool:
    try:
        dp_actor_module = importlib.import_module("verl.workers.actor.dp_actor")
    except (ImportError, ModuleNotFoundError) as exc:
        LOGGER.warning("Skipping turn-level actor patch: failed to import dp_actor: %s", exc)
        return False

    actor_cls = getattr(dp_actor_module, "DataParallelPPOActor", None)
    if actor_cls is None:
        return False
    if getattr(actor_cls, _ACTOR_PATCH_MARKER_ATTR, False):
        return True

    original_update = getattr(actor_cls, "update_policy", None)
    if not callable(original_update):
        LOGGER.warning("Skipping turn-level actor patch: DataParallelPPOActor.update_policy is missing.")
        return False

    setattr(actor_cls, _ORIGINAL_ACTOR_UPDATE_ATTR, original_update)

    def _patched_update_policy(self: Any, data: Any) -> Any:
        original = getattr(type(self), _ORIGINAL_ACTOR_UPDATE_ATTR)
        update_start = time.monotonic()
        reset_cuda_peak_memory_stats()
        if _has_turn_level_distillation_tensors(getattr(data, "batch", None)):
            try:
                result = _run_turn_level_sequential_update_policy(
                    self,
                    data,
                    dp_actor_module=dp_actor_module,
                )
                return _attach_sdpo_actor_profiler_metrics(
                    result,
                    data=data,
                    elapsed_sec=time.monotonic() - update_start,
                )
            except Exception as exc:  # pragma: no cover - fallback path
                LOGGER.warning(
                    "Sequential turn-level SDPO update failed; falling back to upstream update: %s",
                    exc,
                    exc_info=True,
                )
                result = original(self, data)
                return _attach_sdpo_actor_profiler_metrics(
                    result,
                    data=data,
                    elapsed_sec=time.monotonic() - update_start,
                )

        if _should_skip_turn_level_expansion_for_distributed():
            result = original(self, data)
            return _attach_sdpo_actor_profiler_metrics(
                result,
                data=data,
                elapsed_sec=time.monotonic() - update_start,
            )
        try:
            expanded = _maybe_expand_turn_level_distillation_data(data)
        except Exception as exc:  # pragma: no cover - fallback path
            LOGGER.warning(
                "Turn-level SDPO actor expansion failed; falling back to row-level update: %s",
                exc,
                exc_info=True,
            )
            expanded = None
        if expanded is not None:
            data = expanded
        result = original(self, data)
        return _attach_sdpo_actor_profiler_metrics(
            result,
            data=data,
            elapsed_sec=time.monotonic() - update_start,
        )

    setattr(actor_cls, "update_policy", _patched_update_policy)
    setattr(actor_cls, _ACTOR_PATCH_MARKER_ATTR, True)
    return True


def _attach_sdpo_actor_profiler_metrics(
    result: Any,
    *,
    data: Any,
    elapsed_sec: float,
) -> Any:
    if not isinstance(result, dict):
        return result
    batch = getattr(data, "batch", None)
    if batch is None or not hasattr(batch, "get"):
        return result
    profiler_metrics = {
        "profiler/sdpo_actor_update_sec": float(elapsed_sec),
    }
    profiler_metrics.update(
        token_profile_metrics(
            attention_mask=batch.get("response_mask"),
            loss_mask=batch.get("response_mask"),
            elapsed_sec=elapsed_sec,
            prefix="profiler",
        )
    )
    profiler_metrics.update(cuda_memory_metrics())
    result.update(profiler_metrics)
    return result


def _has_turn_level_distillation_tensors(batch: Any) -> bool:
    if batch is None or not hasattr(batch, "keys"):
        return False
    batch_keys = set(batch.keys())
    return _TURN_LEVEL_REQUIRED_KEYS.issubset(batch_keys)


def _compute_turn_level_self_distillation_pg_loss(
    actor: Any,
    *,
    model_inputs: Mapping[str, Any],
    temperature: float,
    self_distillation_cfg: Any,
    teacher_regularization: str,
    return_all_logps: bool,
    distill_topk: int | None,
    student_topk_indices: Any,
    log_prob: Any,
    old_log_prob: Any,
    student_all_logps: Any,
    student_topk_logps: Any,
    loss_agg_mode: str,
    rollout_is_weights: Any,
    compute_self_distillation_loss_fn: Any,
) -> tuple[Any, dict[str, float]]:
    if torch is None:
        raise RuntimeError("torch is required for sequential turn-level SDPO updates.")

    turn_teacher_input_ids = model_inputs["turn_teacher_input_ids"]
    turn_teacher_attention_mask = model_inputs["turn_teacher_attention_mask"]
    turn_teacher_position_ids = model_inputs["turn_teacher_position_ids"]
    turn_response_masks = model_inputs["turn_response_mask"]
    turn_self_distillation_mask = model_inputs["turn_self_distillation_mask"]

    if len(getattr(turn_teacher_input_ids, "shape", ())) != 3:
        raise ValueError("turn_teacher_input_ids must have shape [batch, turns, seq].")

    turn_count = int(turn_teacher_input_ids.shape[1])
    if turn_count <= 0:
        zero_pg_loss = (log_prob * 0.0).sum()
        return zero_pg_loss, {"self_distillation/empty_target_batch": 1.0}

    teacher_model = actor.teacher_module or actor.actor_module
    if teacher_regularization == "trust-region" and (
        actor.teacher_module is None or actor.teacher_module is actor.actor_module
    ):
        raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")

    weighted_pg_loss = log_prob.new_zeros(())
    total_target_tokens = log_prob.new_zeros(())
    total_active_turn_pairs = log_prob.new_zeros(())

    for turn_index in range(turn_count):
        teacher_inputs = {
            "responses": model_inputs["responses"],
            "input_ids": turn_teacher_input_ids[:, turn_index, :],
            "attention_mask": turn_teacher_attention_mask[:, turn_index, :],
            "position_ids": turn_teacher_position_ids[:, turn_index, :],
        }
        with torch.no_grad():
            teacher_outputs = actor._forward_micro_batch(
                teacher_inputs,
                temperature=temperature,
                calculate_entropy=False,
                return_all_logps=return_all_logps,
                distill_topk=distill_topk,
                topk_indices=student_topk_indices,
                module=teacher_model,
            )

        turn_teacher_log_prob = teacher_outputs["log_probs"]
        turn_teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
        turn_teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None

        turn_pair_mask = turn_self_distillation_mask[:, turn_index].to(dtype=torch.float32, device=log_prob.device)
        turn_response_mask = turn_response_masks[:, turn_index, :].to(
            dtype=model_inputs["response_mask"].dtype,
            device=log_prob.device,
        )
        turn_pg_loss, _ = compute_self_distillation_loss_fn(
            student_log_probs=log_prob,
            teacher_log_probs=turn_teacher_log_prob,
            response_mask=turn_response_mask,
            self_distillation_config=self_distillation_cfg,
            old_log_probs=old_log_prob,
            student_all_log_probs=student_all_logps,
            teacher_all_log_probs=turn_teacher_all_logps,
            student_topk_log_probs=student_topk_logps,
            teacher_topk_log_probs=turn_teacher_topk_logps,
            self_distillation_mask=turn_pair_mask,
            loss_agg_mode=loss_agg_mode,
            rollout_is_weights=rollout_is_weights,
        )
        turn_token_count = (turn_response_mask * turn_pair_mask.unsqueeze(1)).sum()
        weighted_pg_loss = weighted_pg_loss + turn_pg_loss * turn_token_count
        total_target_tokens = total_target_tokens + turn_token_count
        total_active_turn_pairs = total_active_turn_pairs + turn_pair_mask.sum()

    pg_loss = weighted_pg_loss / total_target_tokens.clamp(min=1.0)
    pg_metrics = {
        "self_distillation/empty_target_batch": 1.0 if total_target_tokens.item() == 0 else 0.0,
        "self_distillation/active_turn_pairs_in_micro_batch": float(total_active_turn_pairs.item()),
    }
    return pg_loss, pg_metrics


def _run_turn_level_sequential_update_policy(
    actor: Any,
    data: Any,
    *,
    dp_actor_module: Any,
) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for sequential turn-level SDPO updates.")

    prepare_dynamic_batch = getattr(dp_actor_module, "prepare_dynamic_batch")
    get_device_id = getattr(dp_actor_module, "get_device_id")
    get_policy_loss_fn = getattr(dp_actor_module, "get_policy_loss_fn")
    compute_self_distillation_loss_fn = getattr(dp_actor_module, "compute_self_distillation_loss")
    append_to_dict = getattr(dp_actor_module, "append_to_dict")
    agg_loss = getattr(dp_actor_module, "agg_loss")
    kl_penalty = getattr(dp_actor_module, "kl_penalty")

    actor.actor_module.train()

    temperature = data.meta_info["temperature"]
    pad_token_id = data.meta_info.get("pad_token_id", 0)
    loss_mode = actor.config.policy_loss.get("loss_mode", "vanilla")

    self_distillation_enabled = loss_mode == "sdpo"
    self_distillation_cfg = getattr(actor.config, "self_distillation", None)
    if self_distillation_enabled and self_distillation_cfg is None:
        raise ValueError("SDPO update requires self_distillation config.")
    self_distillation_required_keys = {
        "teacher_input_ids",
        "teacher_attention_mask",
        "teacher_position_ids",
        "self_distillation_mask",
    }
    if self_distillation_enabled and not self_distillation_required_keys.issubset(set(data.batch.keys())):
        missing = self_distillation_required_keys - set(data.batch.keys())
        raise ValueError(f"Missing required self-distillation keys: {missing}")

    turn_level_enabled = self_distillation_enabled and _has_turn_level_distillation_tensors(data.batch)

    select_keys = [
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "old_log_probs",
        "advantages",
    ]
    if actor.use_prefix_grouper and "prompts" in data.batch.keys():
        select_keys.append("prompts")
    if actor.config.use_kl_loss:
        select_keys.append("ref_log_prob")
    if self_distillation_enabled:
        select_keys.extend(list(self_distillation_required_keys))
    if turn_level_enabled:
        select_keys.extend(list(_TURN_LEVEL_REQUIRED_KEYS))
    if "rollout_is_weights" in data.batch.keys():
        select_keys.append("rollout_is_weights")
    if "rollout_log_probs" in data.batch.keys():
        select_keys.append("rollout_log_probs")

    has_multi_modal_inputs = actor._has_non_empty_multi_modal_inputs(data.non_tensor_batch.get("multi_modal_inputs"))
    non_tensor_select_keys: list[str] = []
    if has_multi_modal_inputs:
        non_tensor_select_keys.append("multi_modal_inputs")
    if actor.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
        non_tensor_select_keys.append("uid")

    data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
    mini_batches = data.split(actor.config.ppo_mini_batch_size)
    on_policy = len(mini_batches) == 1 and actor.config.ppo_epochs == 1

    metrics: dict[str, Any] = {
        "actor/pg_loss": 0.0,
        "actor/kl_loss": 0.0,
    }
    did_update = False
    for _ in range(actor.config.ppo_epochs):
        for mini_batch in mini_batches:
            if actor.config.use_dynamic_bsz:
                max_token_len = actor.config.ppo_max_token_len_per_gpu * actor.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(actor.config.ppo_micro_batch_size_per_gpu)
            actor.gradient_accumulation = max(len(micro_batches), 1)

            actor.actor_optimizer.zero_grad()

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                micro_batch_metrics: dict[str, Any] = {}
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                response_mask = model_inputs["response_mask"]

                entropy_coeff = actor.config.entropy_coeff
                loss_agg_mode = actor.config.loss_agg_mode
                calculate_entropy = actor.config.calculate_entropy or (entropy_coeff != 0)
                if self_distillation_enabled:
                    assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for distillation"

                if actor.config.use_dynamic_bsz:
                    loss_scale_factor = response_mask.shape[0] / actor.config.ppo_mini_batch_size
                else:
                    loss_scale_factor = 1.0 / float(actor.gradient_accumulation)

                teacher_regularization = "ema"
                return_all_logps = False
                distill_topk = None
                if self_distillation_enabled:
                    teacher_regularization = str(
                        _cfg_get(self_distillation_cfg, "teacher_regularization", "ema")
                    ).strip().lower() or "ema"
                    if teacher_regularization == "trust-region" and actor.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    full_logit_distillation = bool(_cfg_get(self_distillation_cfg, "full_logit_distillation", False))
                    distillation_topk = _cfg_get(self_distillation_cfg, "distillation_topk", None)
                    return_all_logps = (
                        full_logit_distillation and not distillation_topk
                    )
                    distill_topk = (
                        distillation_topk if full_logit_distillation else None
                    )
                outputs = actor._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_all_logps=return_all_logps,
                    distill_topk=distill_topk,
                )
                log_prob = outputs["log_probs"]
                entropy = outputs["entropys"] if calculate_entropy else None
                student_all_logps = outputs.get("all_logps") if return_all_logps else None
                student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                if hasattr(actor.config, "use_rollout_log_probs") and actor.config.use_rollout_log_probs:
                    old_log_prob = model_inputs["old_log_probs"]
                else:
                    old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

                rollout_is_weights = model_inputs.get("rollout_is_weights")

                if self_distillation_enabled:
                    if turn_level_enabled and _has_turn_level_distillation_tensors(model_inputs):
                        pg_loss, pg_metrics = _compute_turn_level_self_distillation_pg_loss(
                            actor,
                            model_inputs=model_inputs,
                            temperature=temperature,
                            self_distillation_cfg=self_distillation_cfg,
                            teacher_regularization=teacher_regularization,
                            return_all_logps=return_all_logps,
                            distill_topk=distill_topk,
                            student_topk_indices=student_topk_indices,
                            log_prob=log_prob,
                            old_log_prob=old_log_prob,
                            student_all_logps=student_all_logps,
                            student_topk_logps=student_topk_logps,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                            compute_self_distillation_loss_fn=compute_self_distillation_loss_fn,
                        )
                    else:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        teacher_model = actor.teacher_module or actor.actor_module
                        if teacher_regularization == "trust-region" and (
                            actor.teacher_module is None or actor.teacher_module is actor.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                        with torch.no_grad():
                            teacher_outputs = actor._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=teacher_model,
                            )
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                        self_distillation_mask = model_inputs.get("self_distillation_mask")
                        pg_loss, pg_metrics = compute_self_distillation_loss_fn(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                        )
                        pg_metrics["self_distillation/empty_target_batch"] = (
                            1.0 if self_distillation_mask is None or self_distillation_mask.sum().item() == 0 else 0.0
                        )
                    micro_batch_metrics.update(pg_metrics)
                else:
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=model_inputs["advantages"],
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=actor.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                rollout_log_prob = model_inputs.get("rollout_log_probs")
                if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                    rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                        log_prob=log_prob,
                        rollout_log_prob=rollout_log_prob,
                        response_mask=response_mask,
                    )
                    micro_batch_metrics.update(rollout_corr_metrics)

                policy_loss = pg_loss
                if calculate_entropy and entropy is not None:
                    entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                    if entropy_coeff != 0:
                        policy_loss -= entropy_agg * entropy_coeff

                if actor.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=actor.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    policy_loss = policy_loss + kl_loss * actor.config.kl_loss_coef
                    metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                    micro_batch_metrics["actor/kl_coef"] = actor.config.kl_loss_coef

                loss = policy_loss * loss_scale_factor
                if actor.scaler is not None:
                    actor.scaler.scale(loss).backward()
                else:
                    loss.backward()

                metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                append_to_dict(metrics, micro_batch_metrics)

            grad_norm = actor._optimizer_step()
            if torch.isfinite(grad_norm).item():
                did_update = True
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

    actor.actor_optimizer.zero_grad()
    if did_update:
        actor._update_teacher()
    return metrics


def _maybe_expand_turn_level_distillation_data(data: Any) -> Any | None:
    if torch is None:
        return None
    batch = getattr(data, "batch", None)
    if batch is None or not hasattr(batch, "keys"):
        return None

    batch_keys = set(batch.keys())
    if not _TURN_LEVEL_REQUIRED_KEYS.issubset(batch_keys):
        return None

    turn_mask = _ensure_tensor_like(batch["turn_self_distillation_mask"], dtype=torch.float32)
    if len(getattr(turn_mask, "shape", ())) != 2:
        return None

    active_pairs = torch.nonzero(turn_mask > 0.0, as_tuple=False)
    if active_pairs.numel() == 0:
        return None

    sample_index = active_pairs[:, 0].long()
    turn_index = active_pairs[:, 1].long()

    tensors: dict[str, Any] = {}
    for key in batch.keys():
        if key in _TURN_LEVEL_REQUIRED_KEYS:
            continue
        if key in {"teacher_input_ids", "teacher_attention_mask", "teacher_position_ids", "self_distillation_mask"}:
            continue
        tensors[str(key)] = batch[key][sample_index]

    tensors["teacher_input_ids"] = batch["turn_teacher_input_ids"][sample_index, turn_index, :]
    tensors["teacher_attention_mask"] = batch["turn_teacher_attention_mask"][sample_index, turn_index, :]
    tensors["teacher_position_ids"] = batch["turn_teacher_position_ids"][sample_index, turn_index, :]
    response_mask_dtype = batch["response_mask"].dtype if "response_mask" in batch_keys else torch.long
    tensors["response_mask"] = batch["turn_response_mask"][sample_index, turn_index, :].to(dtype=response_mask_dtype)
    tensors["self_distillation_mask"] = torch.ones(
        (sample_index.shape[0],),
        dtype=torch.float32,
        device=tensors["teacher_input_ids"].device,
    )

    non_tensors: dict[str, Any] = {}
    non_tensor_batch = getattr(data, "non_tensor_batch", {}) or {}
    sample_index_cpu = sample_index.detach().cpu().numpy()
    if hasattr(non_tensor_batch, "items"):
        for key, value in non_tensor_batch.items():
            try:
                non_tensors[str(key)] = value[sample_index_cpu]
            except Exception:
                non_tensors[str(key)] = [value[int(index)] for index in sample_index_cpu]
    non_tensors["distillation_turn_index"] = turn_index.detach().cpu().tolist()

    meta_info = dict(getattr(data, "meta_info", {}) or {})
    return type(data).from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=meta_info,
    )


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
