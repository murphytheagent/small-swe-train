from __future__ import annotations

import importlib
from types import SimpleNamespace

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
        data=_ConfigNode(apply_chat_template_kwargs=_ConfigNode(enable_thinking=True)),
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
        data=_ConfigNode(apply_chat_template_kwargs=_ConfigNode(enable_thinking=False)),
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

    captured = {"resolved": None}

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
    ):
        _ = include_student_attempt_for_teacher, max_reprompt_len
        captured["resolved"] = [bool(row.get("resolved")) for row in rows]
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
            enable_thinking,
            max_length,
            padding,
            truncation,
        ):
            _ = (
                continue_final_message,
                add_generation_prompt,
                enable_thinking,
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
