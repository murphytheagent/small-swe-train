from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

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
