from __future__ import annotations

import asyncio
import importlib
import math
from types import SimpleNamespace
from typing import Any

import pytest

import verl_integration.ppo_runtime_patch as runtime_patch
from verl_integration.ppo_runtime_patch import (
    _ORIGINAL_DISTILL_ATTR,
    _ORIGINAL_REWARD_ATTR,
    _PATCH_MARKER_ATTR,
    apply_small_swe_sdpo_runtime_patch,
)


class _ConfigNode(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(item)


def _build_non_swe_trainer_class():
    config = _ConfigNode(
        actor_rollout_ref=_ConfigNode(
            rollout=_ConfigNode(agent=_ConfigNode(default_agent_loop="tool_agent")),
            actor=_ConfigNode(
                policy_loss=_ConfigNode(loss_mode="sdpo"),
                self_distillation=_ConfigNode(max_reprompt_len=128),
            ),
        ),
    )

    class _Trainer:
        def __init__(self) -> None:
            self.config = config
            self.tokenizer = None

        def _compute_or_extract_reward(self, batch, reward_fn=None, return_dict=False, sum_reward=False):
            return ("original_reward", reward_fn, return_dict, sum_reward, batch)

        def _maybe_build_self_distillation_batch(
            self,
            batch,
            reward_tensor,
            reward_extra_infos_dict=None,
        ):
            return ("original_distill", batch, reward_tensor, reward_extra_infos_dict)

    return _Trainer


def _build_swe_trainer_class():
    config = _ConfigNode(
        actor_rollout_ref=_ConfigNode(
            rollout=_ConfigNode(agent=_ConfigNode(default_agent_loop="swe_bridge_agent")),
            actor=_ConfigNode(
                policy_loss=_ConfigNode(loss_mode="sdpo"),
                self_distillation=_ConfigNode(
                    max_reprompt_len=128,
                    success_reward_threshold=1.0,
                    include_student_attempt_for_teacher=True,
                ),
            ),
        ),
    )

    class _Trainer:
        def __init__(self) -> None:
            self.config = config
            self.tokenizer = object()

        def _compute_or_extract_reward(self, batch, reward_fn=None, return_dict=False, sum_reward=False):
            return ("original_reward", reward_fn, return_dict, sum_reward, batch)

        def _maybe_build_self_distillation_batch(
            self,
            batch,
            reward_tensor,
            reward_extra_infos_dict=None,
        ):
            return ("original_distill", batch, reward_tensor, reward_extra_infos_dict)

    return _Trainer


def _build_swe_trainer_with_val_metrics_class():
    base_cls = _build_swe_trainer_class()

    class _Trainer(base_cls):
        def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
            _ = data_sources, sample_uids, sample_turns
            return reward_extra_infos_dict

    return _Trainer


def _build_swe_trainer_with_rm_scores_reward_class():
    base_cls = _build_swe_trainer_class()

    class _Trainer(base_cls):
        def _compute_or_extract_reward(self, batch, reward_fn=None, return_dict=False, sum_reward=False):
            _ = batch, reward_fn
            if return_dict:
                return {
                    "reward_tensor": "reward_tensor",
                    "reward_extra_info": {
                        "reward": [1.0, None],
                        "feedback": [None, "ready"],
                    },
                }
            if sum_reward:
                return "summed_reward"
            return "reward_tensor", {
                "reward": [1.0, None],
                "feedback": [None, "ready"],
            }

    return _Trainer


def test_apply_small_swe_sdpo_runtime_patch_noops_when_verl_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_module_not_found(_name: str):
        raise ModuleNotFoundError("verl unavailable")

    monkeypatch.setattr(importlib, "import_module", _raise_module_not_found)

    assert apply_small_swe_sdpo_runtime_patch() is False


def test_apply_small_swe_sdpo_runtime_patch_is_idempotent_on_fake_module() -> None:
    trainer_cls = _build_non_swe_trainer_class()
    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=object,
        compute_position_id_with_mask=lambda mask: mask,
    )

    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True
    first_reward_impl = trainer_cls._compute_or_extract_reward
    first_distill_impl = trainer_cls._maybe_build_self_distillation_batch

    assert getattr(trainer_cls, _PATCH_MARKER_ATTR) is True
    assert callable(getattr(trainer_cls, _ORIGINAL_REWARD_ATTR))
    assert callable(getattr(trainer_cls, _ORIGINAL_DISTILL_ATTR))

    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True
    assert trainer_cls._compute_or_extract_reward is first_reward_impl
    assert trainer_cls._maybe_build_self_distillation_batch is first_distill_impl


def test_patched_hooks_fallback_to_original_for_non_swe_loops() -> None:
    trainer_cls = _build_non_swe_trainer_class()
    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=object,
        compute_position_id_with_mask=lambda mask: mask,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True

    trainer = trainer_cls()
    reward_out = trainer._compute_or_extract_reward(batch="batch", reward_fn="reward_fn", return_dict=True)
    distill_out = trainer._maybe_build_self_distillation_batch(
        batch="batch",
        reward_tensor="reward",
        reward_extra_infos_dict={"feedback": ["x"]},
    )

    assert reward_out[0] == "original_reward"
    assert reward_out[2] is True
    assert distill_out[0] == "original_distill"


def test_teacher_attention_mask_uses_pad_token_to_include_non_pad_tokens() -> None:
    torch = pytest.importorskip("torch")
    responses = torch.tensor([[5, 0, 7]], dtype=torch.long)
    response_mask = torch.tensor([[1, 0, 0]], dtype=torch.long)
    tokenizer = SimpleNamespace(pad_token_id=0)

    mask = runtime_patch._build_teacher_response_attention_mask(
        responses=responses,
        response_mask=response_mask,
        tokenizer=tokenizer,
        device=responses.device,
        dtype=torch.long,
    )

    assert mask.tolist() == [[1, 0, 1]]


def test_patched_distillation_hook_raises_for_invalid_turn_supervision_mode() -> None:
    trainer_cls = _build_swe_trainer_class()
    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=object,
        compute_position_id_with_mask=lambda mask: mask,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True

    trainer = trainer_cls()
    trainer.config.actor_rollout_ref.actor.self_distillation["turn_supervision_mode"] = "curr_turn"

    with pytest.raises(ValueError, match="turn_supervision_mode"):
        trainer._maybe_build_self_distillation_batch(
            batch="batch",
            reward_tensor="reward",
            reward_extra_infos_dict=None,
        )


def test_patched_distillation_hook_rejects_verifier_feedback_all_turns_without_override() -> None:
    trainer_cls = _build_swe_trainer_class()
    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=object,
        compute_position_id_with_mask=lambda mask: mask,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True

    trainer = trainer_cls()
    trainer.config.actor_rollout_ref.actor.self_distillation["verifier_feedback_mode"] = "all_turns"

    with pytest.raises(ValueError, match="all_turns"):
        trainer._maybe_build_self_distillation_batch(
            batch="batch",
            reward_tensor="reward",
            reward_extra_infos_dict=None,
        )


def test_patched_distillation_hook_allows_verifier_feedback_all_turns_with_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer_cls = _build_swe_trainer_class()
    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=object,
        compute_position_id_with_mask=lambda mask: mask,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True
    monkeypatch.setenv("SMALL_SWE_SDPO_ALLOW_VERIFIER_FEEDBACK_ALL_TURNS", "1")

    trainer = trainer_cls()
    trainer.config.actor_rollout_ref.actor.self_distillation["verifier_feedback_mode"] = "all_turns"

    output = trainer._maybe_build_self_distillation_batch(
        batch="batch",
        reward_tensor="reward",
        reward_extra_infos_dict=None,
    )
    assert output[0] == "original_distill"


def test_patched_reward_hook_uses_local_adapter_for_swe_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    trainer_cls = _build_swe_trainer_class()
    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=object,
        compute_position_id_with_mask=lambda mask: mask,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True

    rows_seen = {"count": 0}

    def _fake_rows(*, batch, tokenizer):
        _ = batch, tokenizer
        rows_seen["count"] += 1
        return [{"_response_mask": [1, 1]}, {"_response_mask": [1, 0]}]

    def _fake_reward_tensor(rows, *, response_width, device=None):
        assert len(rows) == 2
        assert response_width == 2
        reward_tensor = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32, device=device)
        return reward_tensor, {"feedback": ["first", ""]}

    monkeypatch.setattr(runtime_patch, "dataproto_to_rows", _fake_rows)
    monkeypatch.setattr(runtime_patch, "rows_to_reward_tensor", _fake_reward_tensor)

    trainer = trainer_cls()
    batch = SimpleNamespace(
        batch={"responses": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)},
        non_tensor_batch={"trajectory_steps": [[], []]},
    )

    reward_pair = trainer._compute_or_extract_reward(batch=batch, return_dict=False, sum_reward=False)
    assert rows_seen["count"] == 1
    reward_tensor, reward_info = reward_pair
    assert reward_tensor.shape == (2, 2)
    assert reward_info["feedback"] == ["first", ""]

    reward_dict = trainer._compute_or_extract_reward(batch=batch, return_dict=True, sum_reward=False)
    assert reward_dict["reward_tensor"].shape == (2, 2)
    assert reward_dict["reward_extra_info"]["feedback"] == ["first", ""]

    summed = trainer._compute_or_extract_reward(batch=batch, return_dict=False, sum_reward=True)
    assert list(summed.tolist()) == [1.0, 1.0]


def test_patched_distillation_hook_builds_teacher_tensors_on_swe_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    trainer_cls = _build_swe_trainer_class()

    class _FakeDataProto:
        def __init__(self, *, tensors):
            self.tensors = tensors

        @classmethod
        def from_dict(cls, *, tensors):
            return cls(tensors=tensors)

    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=_FakeDataProto,
        compute_position_id_with_mask=lambda mask: mask.cumsum(dim=1) - 1,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True

    captured = {"resolved": None, "turn_supervision_mode": None}

    def _fake_rows(batch, tokenizer):
        _ = batch, tokenizer
        return [
            {"_raw_prompt_messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "u1"}]},
            {"_raw_prompt_messages": [{"role": "user", "content": "u2"}]},
        ]

    def _fake_build_self_distillation_batch(
        rows,
        *,
        include_student_attempt_for_teacher,
        max_reprompt_len,
        num_recent_raw_blocks,
        turn_supervision_mode,
        verifier_feedback_mode,
        legacy_distillation_gating_policy,
    ):
        _ = (
            include_student_attempt_for_teacher,
            max_reprompt_len,
            num_recent_raw_blocks,
            verifier_feedback_mode,
            legacy_distillation_gating_policy,
        )
        captured["resolved"] = [bool(row.get("resolved")) for row in rows]
        captured["turn_supervision_mode"] = turn_supervision_mode
        return {
            "teacher_prompts": ["fix one", "fix two"],
            "self_distillation_mask": [True, False],
            "prompt_truncated": [False, True],
        }

    monkeypatch.setattr(runtime_patch, "dataproto_to_rows", _fake_rows)
    monkeypatch.setattr(runtime_patch, "build_self_distillation_batch", _fake_build_self_distillation_batch)

    trainer = trainer_cls()
    class _FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            return_tensors,
            return_dict,
            continue_final_message,
            add_generation_prompt,
            max_length,
            padding,
            truncation,
        ):
            _ = (
                continue_final_message,
                add_generation_prompt,
                max_length,
                padding,
                truncation,
            )
            assert tokenize is True
            assert return_tensors == "pt"
            assert return_dict is True
            assert len(messages) == 2
            return {
                "input_ids": torch.tensor([[10, 11, 0], [20, 21, 22]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
            }

    trainer.tokenizer = _FakeTokenizer()
    responses = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.long)
    batch = SimpleNamespace(
        batch={"responses": responses, "response_mask": response_mask},
        non_tensor_batch={"trajectory_steps": [[], []]},
    )
    reward_tensor = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    reward_extra = {"feedback": ["has-feedback", ""]}

    output = trainer._maybe_build_self_distillation_batch(batch, reward_tensor, reward_extra)
    assert output is not None

    distill_batch, metrics = output
    assert captured["resolved"] == [True, False]
    assert captured["turn_supervision_mode"] == "current_turn"
    assert list(distill_batch.tensors["self_distillation_mask"].tolist()) == [1.0, 0.0]
    assert distill_batch.tensors["teacher_input_ids"].tolist() == [
        [10, 11, 0, 1, 2],
        [20, 21, 22, 3, 4],
    ]
    assert distill_batch.tensors["teacher_attention_mask"].tolist() == [
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0],
    ]
    assert distill_batch.tensors["teacher_position_ids"].tolist() == [
        [0, 1, 1, 2, 3],
        [0, 1, 2, 3, 3],
    ]

    assert metrics["self_distillation/success_sample_fraction"] == pytest.approx(0.5)
    assert metrics["self_distillation/feedback_available_fraction"] == pytest.approx(0.5)
    assert metrics["self_distillation/reprompt_sample_fraction"] == pytest.approx(0.5)
    assert metrics["self_distillation/prompt_truncated_fraction"] == pytest.approx(0.5)
    assert metrics["self_distillation/empty_target_batch"] == pytest.approx(0.0)
    assert metrics["self_distillation/turn_supervision_mode_next_turn"] == pytest.approx(0.0)
    assert metrics["self_distillation/turn_supervision_mode_current_turn"] == pytest.approx(1.0)
    assert "self_distillation/teacher_attention_valid_token_ratio" in metrics
    assert "self_distillation/supervised_token_ratio" in metrics
    assert "self_distillation/invalid_supervised_overlap_count" in metrics


def test_patched_distillation_hook_clamps_max_reprompt_len_to_sequence_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    trainer_cls = _build_swe_trainer_class()

    class _FakeDataProto:
        def __init__(self, *, tensors):
            self.tensors = tensors

        @classmethod
        def from_dict(cls, *, tensors):
            return cls(tensors=tensors)

    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=_FakeDataProto,
        compute_position_id_with_mask=lambda mask: mask.cumsum(dim=1) - 1,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True

    captured: dict[str, int] = {}

    def _fake_rows(batch, tokenizer):
        _ = batch, tokenizer
        return [{"_raw_prompt_messages": [{"role": "user", "content": "u"}], "_response_mask": [1] * 40}]

    def _fake_build_self_distillation_batch(
        rows,
        *,
        include_student_attempt_for_teacher,
        max_reprompt_len,
        num_recent_raw_blocks,
        turn_supervision_mode,
        verifier_feedback_mode,
        legacy_distillation_gating_policy,
    ):
        _ = (
            rows,
            include_student_attempt_for_teacher,
            num_recent_raw_blocks,
            turn_supervision_mode,
            verifier_feedback_mode,
            legacy_distillation_gating_policy,
        )
        captured["max_reprompt_len"] = int(max_reprompt_len)
        return {
            "teacher_prompts": ["teacher"],
            "self_distillation_mask": [True],
            "prompt_truncated": [False],
        }

    monkeypatch.setattr(runtime_patch, "dataproto_to_rows", _fake_rows)
    monkeypatch.setattr(runtime_patch, "build_self_distillation_batch", _fake_build_self_distillation_batch)

    trainer = trainer_cls()
    trainer.config.actor_rollout_ref.actor["ppo_max_token_len_per_gpu"] = 32
    trainer.config.actor_rollout_ref.actor["ulysses_sequence_parallel_size"] = 2
    trainer.config.actor_rollout_ref.actor.self_distillation["max_reprompt_len"] = 100

    class _FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            return_tensors,
            return_dict,
            continue_final_message,
            add_generation_prompt,
            max_length,
            padding,
            truncation,
        ):
            _ = (
                messages,
                tokenize,
                return_tensors,
                return_dict,
                continue_final_message,
                add_generation_prompt,
                max_length,
                padding,
                truncation,
            )
            return {
                "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    trainer.tokenizer = _FakeTokenizer()
    batch = SimpleNamespace(
        batch={
            "responses": torch.arange(40, dtype=torch.long).reshape(1, 40),
            "response_mask": torch.ones((1, 40), dtype=torch.long),
        },
        non_tensor_batch={"trajectory_steps": [[]]},
    )
    reward_tensor = torch.zeros((1, 40), dtype=torch.float32)
    reward_tensor[0, 0] = 1.0

    output = trainer._maybe_build_self_distillation_batch(batch, reward_tensor, {"feedback": ["x"]})
    assert output is not None
    assert captured["max_reprompt_len"] == 24


def test_turn_level_actor_expansion_builds_per_turn_rows() -> None:
    torch = pytest.importorskip("torch")

    class _FakeDataProto:
        def __init__(self, *, batch, non_tensor_batch=None, meta_info=None):
            self.batch = batch
            self.non_tensor_batch = non_tensor_batch or {}
            self.meta_info = meta_info or {}

        @classmethod
        def from_dict(cls, *, tensors=None, non_tensors=None, meta_info=None):
            return cls(
                batch=tensors or {},
                non_tensor_batch=non_tensors or {},
                meta_info=meta_info or {},
            )

    data = _FakeDataProto(
        batch={
            "responses": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "input_ids": torch.tensor([[10, 11, 1, 2], [20, 21, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
            "old_log_probs": torch.zeros((2, 2), dtype=torch.float32),
            "advantages": torch.zeros((2, 2), dtype=torch.float32),
            "teacher_input_ids": torch.tensor([[90, 91, 1, 2], [92, 93, 3, 4]], dtype=torch.long),
            "teacher_attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            "teacher_position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
            "self_distillation_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "turn_teacher_input_ids": torch.tensor(
                [
                    [[100, 101, 1, 2], [110, 111, 1, 2]],
                    [[200, 201, 3, 4], [210, 211, 3, 4]],
                ],
                dtype=torch.long,
            ),
            "turn_teacher_attention_mask": torch.tensor(
                [
                    [[1, 1, 1, 1], [1, 1, 1, 1]],
                    [[1, 1, 1, 1], [1, 1, 1, 1]],
                ],
                dtype=torch.long,
            ),
            "turn_teacher_position_ids": torch.tensor(
                [
                    [[0, 1, 2, 3], [0, 1, 2, 3]],
                    [[0, 1, 2, 3], [0, 1, 2, 3]],
                ],
                dtype=torch.long,
            ),
            "turn_response_mask": torch.tensor(
                [
                    [[1, 0], [0, 1]],
                    [[1, 1], [0, 0]],
                ],
                dtype=torch.long,
            ),
            "turn_self_distillation_mask": torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        },
        non_tensor_batch={"uid": ["a", "b"]},
        meta_info={"temperature": 1.0},
    )

    expanded = runtime_patch._maybe_expand_turn_level_distillation_data(data)
    assert expanded is not None

    assert expanded.batch["teacher_input_ids"].tolist() == [
        [100, 101, 1, 2],
        [200, 201, 3, 4],
    ]
    assert expanded.batch["response_mask"].tolist() == [
        [1, 0],
        [1, 1],
    ]
    assert expanded.batch["self_distillation_mask"].tolist() == [1.0, 1.0]
    assert list(expanded.non_tensor_batch["uid"]) == ["a", "b"]
    assert list(expanded.non_tensor_batch["distillation_turn_index"]) == [0, 0]


def test_turn_level_expansion_guard_skips_distributed_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDistributed:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_world_size() -> int:
            return 8

    class _FakeTorch:
        distributed = _FakeDistributed()

    monkeypatch.setattr(runtime_patch, "torch", _FakeTorch())
    monkeypatch.delenv("SMALL_SWE_ENABLE_DISTRIBUTED_TURN_LEVEL_EXPANSION", raising=False)

    assert runtime_patch._should_skip_turn_level_expansion_for_distributed() is True


def test_turn_level_expansion_guard_allows_distributed_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDistributed:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_world_size() -> int:
            return 8

    class _FakeTorch:
        distributed = _FakeDistributed()

    monkeypatch.setattr(runtime_patch, "torch", _FakeTorch())
    monkeypatch.setenv("SMALL_SWE_ENABLE_DISTRIBUTED_TURN_LEVEL_EXPANSION", "1")

    assert runtime_patch._should_skip_turn_level_expansion_for_distributed() is False


def test_turn_level_sequential_loss_accumulates_without_row_filtering() -> None:
    torch = pytest.importorskip("torch")

    class _FakeActor:
        def __init__(self) -> None:
            self.teacher_module = object()
            self.actor_module = object()

        def _forward_micro_batch(
            self,
            _inputs,
            *,
            temperature,
            calculate_entropy,
            return_all_logps,
            distill_topk,
            topk_indices=None,
            module=None,
        ):
            _ = (
                temperature,
                calculate_entropy,
                return_all_logps,
                distill_topk,
                topk_indices,
                module,
            )
            return {"log_probs": torch.zeros((1, 4), dtype=torch.float32)}

    actor = _FakeActor()
    model_inputs = {
        "responses": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "response_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "turn_teacher_input_ids": torch.tensor(
            [[[10, 11, 1, 2, 3], [20, 21, 1, 2, 3], [30, 31, 1, 2, 3]]],
            dtype=torch.long,
        ),
        "turn_teacher_attention_mask": torch.ones((1, 3, 5), dtype=torch.long),
        "turn_teacher_position_ids": torch.tensor(
            [[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]],
            dtype=torch.long,
        ),
        "turn_response_mask": torch.tensor(
            [[[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]]],
            dtype=torch.long,
        ),
        "turn_self_distillation_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
    }

    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _fake_compute_self_distillation_loss(**kwargs):
        calls.append(
            (
                kwargs["response_mask"].detach().cpu(),
                kwargs["self_distillation_mask"].detach().cpu(),
            )
        )
        turn_id = float(len(calls))
        return torch.tensor(turn_id, dtype=torch.float32), {}

    pg_loss, metrics = runtime_patch._compute_turn_level_self_distillation_pg_loss(
        actor,
        model_inputs=model_inputs,
        temperature=1.0,
        self_distillation_cfg=SimpleNamespace(full_logit_distillation=False, distillation_topk=None),
        teacher_regularization="ema",
        return_all_logps=False,
        distill_topk=None,
        student_topk_indices=None,
        log_prob=torch.zeros((1, 4), dtype=torch.float32),
        old_log_prob=torch.zeros((1, 4), dtype=torch.float32),
        student_all_logps=None,
        student_topk_logps=None,
        loss_agg_mode="token-mean",
        rollout_is_weights=None,
        compute_self_distillation_loss_fn=_fake_compute_self_distillation_loss,
    )

    assert len(calls) == 3
    assert pg_loss.item() == pytest.approx(2.0)
    assert metrics["self_distillation/empty_target_batch"] == pytest.approx(0.0)
    assert metrics["self_distillation/active_turn_pairs_in_micro_batch"] == pytest.approx(2.0)


def test_patched_distillation_hook_emits_turn_level_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    trainer_cls = _build_swe_trainer_class()

    class _FakeDataProto:
        def __init__(self, *, tensors):
            self.tensors = tensors

        @classmethod
        def from_dict(cls, *, tensors):
            return cls(tensors=tensors)

    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=_FakeDataProto,
        compute_position_id_with_mask=lambda mask: mask.cumsum(dim=1) - 1,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True

    def _fake_rows(batch, tokenizer):
        _ = batch, tokenizer
        return [{"_raw_prompt_messages": [{"role": "user", "content": "u"}], "_response_mask": [1, 1]}]

    def _fake_build_self_distillation_batch(
        rows,
        *,
        include_student_attempt_for_teacher,
        max_reprompt_len,
        num_recent_raw_blocks,
        turn_supervision_mode,
        verifier_feedback_mode,
        legacy_distillation_gating_policy,
    ):
        _ = (
            rows,
            include_student_attempt_for_teacher,
            max_reprompt_len,
            num_recent_raw_blocks,
            turn_supervision_mode,
            verifier_feedback_mode,
            legacy_distillation_gating_policy,
        )
        return {
            "teacher_prompts": ["row-level"],
            "self_distillation_mask": [True],
            "prompt_truncated": [False],
            "turn_teacher_prompts": [["turn-0", "turn-1"]],
            "turn_response_masks": [[[1, 0], [0, 1]]],
            "turn_distillation_mask": [[True, True]],
        }

    monkeypatch.setattr(runtime_patch, "dataproto_to_rows", _fake_rows)
    monkeypatch.setattr(runtime_patch, "build_self_distillation_batch", _fake_build_self_distillation_batch)

    trainer = trainer_cls()

    class _FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            return_tensors,
            return_dict,
            continue_final_message,
            add_generation_prompt,
            max_length,
            padding,
            truncation,
        ):
            _ = (
                tokenize,
                return_tensors,
                return_dict,
                continue_final_message,
                add_generation_prompt,
                max_length,
                padding,
                truncation,
            )
            rows = len(messages)
            input_ids = torch.arange(rows * 3, dtype=torch.long).reshape(rows, 3)
            attention_mask = torch.ones((rows, 3), dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    trainer.tokenizer = _FakeTokenizer()
    batch = SimpleNamespace(
        batch={
            "responses": torch.tensor([[1, 2]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        non_tensor_batch={"trajectory_steps": [[]]},
    )
    reward_tensor = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    output = trainer._maybe_build_self_distillation_batch(batch, reward_tensor, {"feedback": [""]})
    assert output is not None
    distill_batch, _metrics = output

    assert "turn_teacher_input_ids" in distill_batch.tensors
    assert distill_batch.tensors["turn_teacher_input_ids"].shape[0] == 1
    assert distill_batch.tensors["turn_teacher_input_ids"].shape[1] == 2
    assert distill_batch.tensors["turn_response_mask"].tolist() == [[[1, 0], [0, 1]]]


def test_turn_level_teacher_tensors_intersect_masks_with_runtime_response_mask() -> None:
    torch = pytest.importorskip("torch")

    class _FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            return_tensors,
            return_dict,
            continue_final_message,
            add_generation_prompt,
            max_length,
            padding,
            truncation,
        ):
            _ = (
                messages,
                tokenize,
                return_tensors,
                return_dict,
                continue_final_message,
                add_generation_prompt,
                max_length,
                padding,
                truncation,
            )
            return {
                "input_ids": torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            }

    (
        _turn_teacher_input_ids,
        _turn_teacher_attention_mask,
        _turn_teacher_position_ids,
        turn_response_mask,
        turn_self_distillation_mask,
        turn_pair_count_per_sample,
    ) = runtime_patch._build_turn_level_teacher_tensors(
        tokenizer=_FakeTokenizer(),
        rows=[{"_raw_prompt_messages": [{"role": "user", "content": "u"}]}],
        turn_teacher_prompts=[["turn-0", "turn-1"]],
        turn_response_masks=[[[1, 1], [0, 1]]],
        turn_distillation_mask=[[True, True]],
        responses=torch.tensor([[1, 2]], dtype=torch.long),
        response_mask=torch.tensor([[1, 0]], dtype=torch.long),
        max_reprompt_len=64,
        compute_position_id_with_mask=lambda mask: mask.cumsum(dim=1) - 1,
        device=None,
    )

    assert turn_response_mask.tolist() == [[[1, 0], [0, 0]]]
    assert turn_self_distillation_mask.tolist() == [[1.0, 0.0]]
    assert turn_pair_count_per_sample == [1]


def test_sanitize_validation_reward_extra_infos_handles_none_and_non_numeric_values() -> None:
    sanitized = runtime_patch._sanitize_validation_reward_extra_infos(
        {
            "reward": [1.0, None, 0.0],
            "pred": ["yes", None, "no"],
            "metadata": [{"k": "v"}, None, {"k": "w"}],
            "bad_len": [1.0],
        },
        expected_len=3,
    )

    assert sanitized["reward"][0] == pytest.approx(1.0)
    assert math.isnan(sanitized["reward"][1])
    assert sanitized["reward"][2] == pytest.approx(0.0)
    assert sanitized["pred"] == ["yes", "", "no"]
    assert sanitized["metadata"][0].startswith("{")
    assert sanitized["metadata"][1] == ""
    assert "bad_len" not in sanitized


def test_patched_val_metrics_update_sanitizes_reward_infos_before_upstream_call() -> None:
    trainer_cls = _build_swe_trainer_with_val_metrics_class()
    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=object,
        compute_position_id_with_mask=lambda mask: mask,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True

    trainer = trainer_cls()
    sanitized = trainer._val_metrics_update(
        data_sources=["source", "source"],
        sample_uids=["uid-1", "uid-2"],
        reward_extra_infos_dict={"reward": [1.0, None], "pred": ["a", "b"]},
        sample_turns=[],
    )

    assert sanitized["reward"][0] == pytest.approx(1.0)
    assert math.isnan(sanitized["reward"][1])
    assert sanitized["pred"] == ["a", "b"]


def test_patched_reward_hook_sanitizes_rm_scores_reward_info_for_train_and_val_paths() -> None:
    trainer_cls = _build_swe_trainer_with_rm_scores_reward_class()
    fake_module = SimpleNamespace(
        RayPPOTrainer=trainer_cls,
        DataProto=object,
        compute_position_id_with_mask=lambda mask: mask,
    )
    assert apply_small_swe_sdpo_runtime_patch(ray_trainer_module=fake_module) is True

    trainer = trainer_cls()
    batch = SimpleNamespace(
        batch={"rm_scores": [[0.0], [0.0]], "responses": [[1], [2]]},
        non_tensor_batch={"trajectory_steps": [[], []]},
    )

    reward_tensor, reward_info = trainer._compute_or_extract_reward(batch=batch, return_dict=False, sum_reward=False)
    assert reward_tensor == "reward_tensor"
    assert reward_info["reward"][0] == pytest.approx(1.0)
    assert math.isnan(reward_info["reward"][1])
    assert reward_info["feedback"] == ["", "ready"]

    reward_dict = trainer._compute_or_extract_reward(batch=batch, return_dict=True, sum_reward=False)
    assert reward_dict["reward_tensor"] == "reward_tensor"
    assert reward_dict["reward_extra_info"]["reward"][0] == pytest.approx(1.0)
    assert math.isnan(reward_dict["reward_extra_info"]["reward"][1])
    assert reward_dict["reward_extra_info"]["feedback"] == ["", "ready"]

    summed = trainer._compute_or_extract_reward(batch=batch, return_dict=False, sum_reward=True)
    assert summed == "summed_reward"


def test_resolve_swe_agent_loop_max_in_flight_uses_env_pool_and_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = SimpleNamespace(
        config=_ConfigNode(
            actor_rollout_ref=_ConfigNode(
                rollout=_ConfigNode(
                    agent=_ConfigNode(default_agent_loop="swe_bridge_agent", num_workers=8),
                ),
            )
        )
    )

    resolved = runtime_patch._resolve_swe_agent_loop_max_in_flight(
        worker=worker,
        batch_size=3892,
        agent_loop_registry={"swe_bridge_agent": {"env_pool_size": 32}},
    )
    assert resolved == 4

    monkeypatch.setenv("SMALL_SWE_SDPO_AGENT_LOOP_MAX_IN_FLIGHT", "6")
    overridden = runtime_patch._resolve_swe_agent_loop_max_in_flight(
        worker=worker,
        batch_size=3892,
        agent_loop_registry={"swe_bridge_agent": {"env_pool_size": 32}},
    )
    assert overridden == 6


def test_agent_loop_queue_patch_caps_swe_in_flight_and_preserves_order() -> None:
    np = pytest.importorskip("numpy")

    class _FakeRolloutTraceConfig:
        @staticmethod
        def get_instance() -> Any:
            return SimpleNamespace(max_samples_per_step_per_worker=None)

    async def _fake_get_trajectory_info(step: int, indices: list[int], validate: bool) -> list[dict[str, Any]]:
        return [
            {"step": step, "sample_index": int(index), "rollout_n": 0, "validate": validate}
            for index in indices
        ]

    class _FakeBatch:
        def __init__(self, size: int) -> None:
            self.non_tensor_batch = {
                "agent_name": np.array(["swe_bridge_agent"] * size, dtype=object),
                "sample_index": np.arange(size),
                "raw_prompt": np.array(["prompt"] * size, dtype=object),
            }
            self.meta_info = {"global_steps": 17, "validate": False}

        def __len__(self) -> int:
            return int(len(self.non_tensor_batch["sample_index"]))

    class _FakeWorker:
        def __init__(self) -> None:
            self.config = _ConfigNode(
                actor_rollout_ref=_ConfigNode(
                    rollout=_ConfigNode(
                        temperature=0.7,
                        top_p=0.9,
                        calculate_log_probs=False,
                        val_kwargs=_ConfigNode(top_p=1.0, temperature=0.0),
                        agent=_ConfigNode(
                            default_agent_loop="swe_bridge_agent",
                            max_in_flight_tasks=3,
                        ),
                    )
                )
            )
            self.active_calls = 0
            self.max_active_calls = 0

        async def generate_sequences(self, batch: Any) -> Any:
            return ("original", len(batch))

        async def _run_agent_loop(
            self,
            sampling_params: dict[str, Any],
            trajectory: dict[str, Any],
            *,
            agent_name: str,
            trace: bool = True,
            **kwargs: Any,
        ) -> str:
            _ = sampling_params, trajectory, agent_name, trace
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            try:
                await asyncio.sleep(0.01)
                return f"out-{int(kwargs['sample_index'])}"
            finally:
                self.active_calls -= 1

        def _postprocess(self, outputs: list[str]) -> list[str]:
            return outputs

    fake_module = SimpleNamespace(
        AgentLoopWorker=_FakeWorker,
        np=np,
        RolloutTraceConfig=_FakeRolloutTraceConfig,
        get_trajectory_info=_fake_get_trajectory_info,
        _agent_loop_registry={"swe_bridge_agent": {"env_pool_size": 32}},
        tqbridge=lambda: (lambda fn: fn),
    )

    assert runtime_patch._apply_agent_loop_queue_patch(fake_module) is True

    worker = _FakeWorker()
    result = asyncio.run(worker.generate_sequences(_FakeBatch(size=12)))
    assert result == [f"out-{i}" for i in range(12)]
    assert worker.max_active_calls <= 3


def test_agent_loop_queue_patch_falls_back_to_original_for_non_swe() -> None:
    np = pytest.importorskip("numpy")

    class _FakeRolloutTraceConfig:
        @staticmethod
        def get_instance() -> Any:
            return SimpleNamespace(max_samples_per_step_per_worker=None)

    async def _fake_get_trajectory_info(step: int, indices: list[int], validate: bool) -> list[dict[str, Any]]:
        return [
            {"step": step, "sample_index": int(index), "rollout_n": 0, "validate": validate}
            for index in indices
        ]

    class _FakeBatch:
        def __init__(self, size: int) -> None:
            self.non_tensor_batch = {
                "agent_name": np.array(["tool_agent"] * size, dtype=object),
                "sample_index": np.arange(size),
                "raw_prompt": np.array(["prompt"] * size, dtype=object),
            }
            self.meta_info = {"global_steps": 9, "validate": False}

        def __len__(self) -> int:
            return int(len(self.non_tensor_batch["sample_index"]))

    class _FakeWorker:
        def __init__(self) -> None:
            self.config = _ConfigNode(
                actor_rollout_ref=_ConfigNode(
                    rollout=_ConfigNode(
                        temperature=0.7,
                        top_p=0.9,
                        calculate_log_probs=False,
                        val_kwargs=_ConfigNode(top_p=1.0, temperature=0.0),
                        agent=_ConfigNode(default_agent_loop="tool_agent", max_in_flight_tasks=2),
                    )
                )
            )
            self.original_called = 0

        async def generate_sequences(self, batch: Any) -> Any:
            self.original_called += 1
            return ("original", len(batch))

        async def _run_agent_loop(
            self,
            sampling_params: dict[str, Any],
            trajectory: dict[str, Any],
            *,
            agent_name: str,
            trace: bool = True,
            **kwargs: Any,
        ) -> Any:
            _ = sampling_params, trajectory, agent_name, trace, kwargs
            raise AssertionError("_run_agent_loop should not be called for non-swe agents")

        def _postprocess(self, outputs: list[Any]) -> list[Any]:
            return outputs

    fake_module = SimpleNamespace(
        AgentLoopWorker=_FakeWorker,
        np=np,
        RolloutTraceConfig=_FakeRolloutTraceConfig,
        get_trajectory_info=_fake_get_trajectory_info,
        _agent_loop_registry={"tool_agent": {"env_pool_size": 4}},
        tqbridge=lambda: (lambda fn: fn),
    )

    assert runtime_patch._apply_agent_loop_queue_patch(fake_module) is True

    worker = _FakeWorker()
    result = asyncio.run(worker.generate_sequences(_FakeBatch(size=4)))
    assert result == ("original", 4)
    assert worker.original_called == 1


def test_agent_loop_server_routing_patch_spreads_and_sticks() -> None:
    class _FakeManager:
        def __init__(self, handles: list[Any]) -> None:
            self.server_handles = handles
            self.request_id_to_server: dict[str, Any] = {}

        def _choose_server(self, request_id: str) -> Any:
            # Original fallback behavior: always first server.
            return self.server_handles[0]

    fake_module = SimpleNamespace(AsyncLLMServerManager=_FakeManager)
    assert runtime_patch._apply_agent_loop_server_routing_patch(fake_module) is True

    manager = _FakeManager(handles=[f"server-{idx}" for idx in range(8)])
    seen_counts: dict[str, int] = {}
    for idx in range(512):
        req_id = f"req-{idx}"
        server = manager._choose_server(req_id)
        seen_counts[server] = seen_counts.get(server, 0) + 1

    # Hash routing should spread across all servers for a moderate request set.
    assert len(seen_counts) == 8

    sticky = manager._choose_server("sticky-request")
    assert sticky == manager._choose_server("sticky-request")


def test_agent_loop_server_routing_patch_falls_back_when_handles_missing() -> None:
    class _FakeManager:
        def __init__(self) -> None:
            self.server_handles = None
            self.request_id_to_server: dict[str, Any] = {}

        def _choose_server(self, request_id: str) -> Any:
            return f"orig-{request_id}"

    fake_module = SimpleNamespace(AsyncLLMServerManager=_FakeManager)
    assert runtime_patch._apply_agent_loop_server_routing_patch(fake_module) is True

    manager = _FakeManager()
    assert manager._choose_server("x") == "orig-x"
