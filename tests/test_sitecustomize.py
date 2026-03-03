from __future__ import annotations

import asyncio
import builtins
import importlib.util
import os
import sys
import types
import warnings
from importlib.machinery import ModuleSpec

import pytest

import small_swe_runtime_patches as sitecustomize


def test_sitecustomize_can_hide_external_flash_attn(monkeypatch) -> None:
    marker = ModuleSpec(name="marker_mod", loader=None)

    def _fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "flash_attn":
            return ModuleSpec(name="flash_attn", loader=None)
        return marker

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setenv("SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN", "1")
    original_import = builtins.__import__
    original_root = sys.modules.get("flash_attn")
    original_child = sys.modules.get("flash_attn.flash_attn_interface")
    sys.modules["flash_attn"] = object()
    sys.modules["flash_attn.flash_attn_interface"] = object()

    try:
        sitecustomize.apply_small_swe_runtime_patches()

        assert importlib.util.find_spec("flash_attn") is None
        assert importlib.util.find_spec("another_mod") is marker
        assert "flash_attn" not in sys.modules
        assert "flash_attn.flash_attn_interface" not in sys.modules
        with pytest.raises(ModuleNotFoundError):
            builtins.__import__("flash_attn")
    finally:
        builtins.__import__ = original_import
        if original_root is None:
            sys.modules.pop("flash_attn", None)
        else:
            sys.modules["flash_attn"] = original_root
        if original_child is None:
            sys.modules.pop("flash_attn.flash_attn_interface", None)
        else:
            sys.modules["flash_attn.flash_attn_interface"] = original_child


def test_sitecustomize_preserves_existing_import_wrapper(monkeypatch) -> None:
    calls = {"count": 0}
    original_import = builtins.__import__

    def _wrapped_import(name, globals=None, locals=None, fromlist=(), level=0):
        calls["count"] += 1
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setenv("SMALL_SWE_HIDE_EXTERNAL_FLASH_ATTN", "1")
    builtins.__import__ = _wrapped_import
    try:
        sitecustomize.apply_small_swe_runtime_patches()
        with pytest.raises(ModuleNotFoundError):
            builtins.__import__("flash_attn")
        builtins.__import__("math")
    finally:
        builtins.__import__ = original_import

    assert calls["count"] >= 1


def test_sitecustomize_can_install_sdpo_patch_import_guard(monkeypatch) -> None:
    calls = {"count": 0, "module": None}

    fake_patch_module = types.ModuleType("verl_integration.ppo_runtime_patch")

    def _fake_apply_small_swe_sdpo_runtime_patch(module=None):
        calls["count"] += 1
        calls["module"] = module
        return True

    fake_patch_module.apply_small_swe_sdpo_runtime_patch = _fake_apply_small_swe_sdpo_runtime_patch

    fake_verl_pkg = types.ModuleType("verl")
    fake_trainer_pkg = types.ModuleType("verl.trainer")
    fake_ppo_pkg = types.ModuleType("verl.trainer.ppo")
    fake_ray_trainer = types.ModuleType("verl.trainer.ppo.ray_trainer")
    fake_ray_trainer.RayPPOTrainer = object

    monkeypatch.setitem(sys.modules, "verl_integration.ppo_runtime_patch", fake_patch_module)
    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer", fake_trainer_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo", fake_ppo_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.ray_trainer", fake_ray_trainer)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    original_import = builtins.__import__
    try:
        sitecustomize.apply_small_swe_runtime_patches()
        builtins.__import__("verl.trainer.ppo.ray_trainer")
    finally:
        builtins.__import__ = original_import

    assert calls["count"] >= 1
    assert calls["module"] is fake_ray_trainer


def test_sitecustomize_skips_sdpo_patch_until_ray_trainer_class_ready(monkeypatch) -> None:
    calls = {"count": 0}

    fake_patch_module = types.ModuleType("verl_integration.ppo_runtime_patch")

    def _fake_apply_small_swe_sdpo_runtime_patch(module=None):
        calls["count"] += 1
        _ = module
        return True

    fake_patch_module.apply_small_swe_sdpo_runtime_patch = _fake_apply_small_swe_sdpo_runtime_patch

    fake_verl_pkg = types.ModuleType("verl")
    fake_trainer_pkg = types.ModuleType("verl.trainer")
    fake_ppo_pkg = types.ModuleType("verl.trainer.ppo")
    # Simulate partially initialized import where RayPPOTrainer is not yet set.
    fake_ray_trainer = types.ModuleType("verl.trainer.ppo.ray_trainer")

    monkeypatch.setitem(sys.modules, "verl_integration.ppo_runtime_patch", fake_patch_module)
    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer", fake_trainer_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo", fake_ppo_pkg)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.ray_trainer", fake_ray_trainer)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    original_import = builtins.__import__
    try:
        sitecustomize.apply_small_swe_runtime_patches()
        builtins.__import__("verl.trainer.ppo.ray_trainer")
    finally:
        builtins.__import__ = original_import

    assert calls["count"] == 0


def test_sitecustomize_accepts_self_distillation_compat_fields_on_older_verl_config(monkeypatch) -> None:
    class _FakeSelfDistillationConfig:
        def __init__(self, alpha: float = 0.0) -> None:
            self.alpha = alpha

    fake_verl_pkg = types.ModuleType("verl")
    fake_workers_pkg = types.ModuleType("verl.workers")
    fake_config_pkg = types.ModuleType("verl.workers.config")
    fake_actor_module = types.ModuleType("verl.workers.config.actor")
    fake_actor_module.SelfDistillationConfig = _FakeSelfDistillationConfig

    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.workers", fake_workers_pkg)
    monkeypatch.setitem(sys.modules, "verl.workers.config", fake_config_pkg)
    monkeypatch.setitem(sys.modules, "verl.workers.config.actor", fake_actor_module)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    original_import = builtins.__import__
    try:
        sitecustomize.apply_small_swe_runtime_patches()
        cfg = _FakeSelfDistillationConfig(
            alpha=0.25,
            num_recent_raw_blocks=7,
            turn_supervision_mode="current_turn",
        )
    finally:
        builtins.__import__ = original_import

    assert cfg.alpha == 0.25
    assert getattr(cfg, "num_recent_raw_blocks") == 7
    assert getattr(cfg, "turn_supervision_mode") == "current_turn"


def test_sitecustomize_rejects_invalid_turn_supervision_mode_on_older_verl_config(monkeypatch) -> None:
    class _FakeSelfDistillationConfig:
        def __init__(self, alpha: float = 0.0) -> None:
            self.alpha = alpha

    fake_verl_pkg = types.ModuleType("verl")
    fake_workers_pkg = types.ModuleType("verl.workers")
    fake_config_pkg = types.ModuleType("verl.workers.config")
    fake_actor_module = types.ModuleType("verl.workers.config.actor")
    fake_actor_module.SelfDistillationConfig = _FakeSelfDistillationConfig

    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.workers", fake_workers_pkg)
    monkeypatch.setitem(sys.modules, "verl.workers.config", fake_config_pkg)
    monkeypatch.setitem(sys.modules, "verl.workers.config.actor", fake_actor_module)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    original_import = builtins.__import__
    try:
        sitecustomize.apply_small_swe_runtime_patches()
        with pytest.raises(ValueError, match="turn_supervision_mode"):
            _FakeSelfDistillationConfig(alpha=0.25, turn_supervision_mode="bad_mode")
    finally:
        builtins.__import__ = original_import


def test_sitecustomize_sets_local_rank_from_rank_in_ray_noset_mode(monkeypatch) -> None:
    calls = {"base_called": 0, "set_device": []}

    class _FakeWorker:
        def _setup_env_cuda_visible_devices(self) -> None:
            calls["base_called"] += 1
            os.environ["LOCAL_RANK"] = "0"

    class _FakeTorchDevice:
        def set_device(self, device: int) -> None:
            calls["set_device"].append(device)

    fake_verl_pkg = types.ModuleType("verl")
    fake_single_controller_pkg = types.ModuleType("verl.single_controller")
    fake_single_controller_base_pkg = types.ModuleType("verl.single_controller.base")
    fake_worker_module = types.ModuleType("verl.single_controller.base.worker")
    fake_worker_module.Worker = _FakeWorker

    fake_utils_pkg = types.ModuleType("verl.utils")
    fake_device_module = types.ModuleType("verl.utils.device")
    fake_device_module.get_torch_device = lambda: _FakeTorchDevice()
    fake_ray_utils_module = types.ModuleType("verl.utils.ray_utils")
    fake_ray_utils_module.ray_noset_visible_devices = lambda env_vars=os.environ: True

    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.single_controller", fake_single_controller_pkg)
    monkeypatch.setitem(sys.modules, "verl.single_controller.base", fake_single_controller_base_pkg)
    monkeypatch.setitem(sys.modules, "verl.single_controller.base.worker", fake_worker_module)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_utils_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils.device", fake_device_module)
    monkeypatch.setitem(sys.modules, "verl.utils.ray_utils", fake_ray_utils_module)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "8")

    sitecustomize.apply_small_swe_runtime_patches()
    worker = _FakeWorker()
    worker._setup_env_cuda_visible_devices()

    assert calls["base_called"] == 1
    assert calls["set_device"] == [5]
    assert os.environ["LOCAL_RANK"] == "5"


def test_sitecustomize_installs_model_type_aware_mistral_regex_default(monkeypatch, tmp_path) -> None:
    calls: dict[str, list[dict[str, object]]] = {"tokenizer": []}

    fake_verl_pkg = types.ModuleType("verl")
    fake_verl_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_utils_pkg = types.ModuleType("verl.utils")
    fake_utils_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_tokenizer_module = types.ModuleType("verl.utils.tokenizer")

    def _fake_hf_tokenizer(name_or_path, *args, **kwargs):
        _ = name_or_path, args
        calls["tokenizer"].append(dict(kwargs))
        return kwargs

    def _fake_hf_processor(name_or_path, *args, **kwargs):
        _ = name_or_path, args, kwargs
        return None

    fake_tokenizer_module.hf_tokenizer = _fake_hf_tokenizer
    fake_tokenizer_module.hf_processor = _fake_hf_processor
    fake_utils_pkg.tokenizer = fake_tokenizer_module
    fake_utils_pkg.hf_tokenizer = _fake_hf_tokenizer
    fake_utils_pkg.hf_processor = _fake_hf_processor

    qwen_dir = tmp_path / "qwen"
    qwen_dir.mkdir()
    (qwen_dir / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    mistral_dir = tmp_path / "mistral"
    mistral_dir.mkdir()
    (mistral_dir / "config.json").write_text('{"model_type":"mistral"}', encoding="utf-8")

    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_utils_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils.tokenizer", fake_tokenizer_module)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    sitecustomize.apply_small_swe_runtime_patches()

    fake_tokenizer_module.hf_tokenizer(str(qwen_dir))
    fake_tokenizer_module.hf_tokenizer(str(mistral_dir))
    fake_tokenizer_module.hf_tokenizer(str(qwen_dir), fix_mistral_regex=True)

    assert calls["tokenizer"][0]["fix_mistral_regex"] is True
    assert calls["tokenizer"][1]["fix_mistral_regex"] is True
    assert calls["tokenizer"][2]["fix_mistral_regex"] is True
    assert fake_utils_pkg.hf_tokenizer is fake_tokenizer_module.hf_tokenizer


def test_sitecustomize_suppresses_text_only_processor_warning(monkeypatch) -> None:
    fake_verl_pkg = types.ModuleType("verl")
    fake_verl_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_utils_pkg = types.ModuleType("verl.utils")
    fake_utils_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_tokenizer_module = types.ModuleType("verl.utils.tokenizer")

    def _fake_hf_tokenizer(name_or_path, *args, **kwargs):
        _ = name_or_path, args, kwargs
        return {}

    def _fake_hf_processor(name_or_path, *args, **kwargs):
        _ = name_or_path, args, kwargs
        warnings.warn(
            "Failed to create processor: Unsupported processor type: Qwen2TokenizerFast. This may affect multimodal processing",
            UserWarning,
            stacklevel=1,
        )
        return None

    fake_tokenizer_module.hf_tokenizer = _fake_hf_tokenizer
    fake_tokenizer_module.hf_processor = _fake_hf_processor
    fake_utils_pkg.tokenizer = fake_tokenizer_module
    fake_utils_pkg.hf_tokenizer = _fake_hf_tokenizer
    fake_utils_pkg.hf_processor = _fake_hf_processor

    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_utils_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils.tokenizer", fake_tokenizer_module)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    sitecustomize.apply_small_swe_runtime_patches()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert fake_tokenizer_module.hf_processor("/tmp/model") is None

    assert not any("Unsupported processor type" in str(item.message) for item in captured)


def test_sitecustomize_marks_fast_tokenizer_pad_warning_as_handled(monkeypatch, tmp_path) -> None:
    class _FakeTokenizer:
        def __init__(self) -> None:
            self.deprecation_warnings: dict[str, bool] = {}

    fake_verl_pkg = types.ModuleType("verl")
    fake_verl_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_utils_pkg = types.ModuleType("verl.utils")
    fake_utils_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_tokenizer_module = types.ModuleType("verl.utils.tokenizer")

    calls: dict[str, list[dict[str, object]]] = {"kwargs": []}

    def _fake_hf_tokenizer(name_or_path, *args, **kwargs):
        _ = name_or_path, args
        calls["kwargs"].append(dict(kwargs))
        return _FakeTokenizer()

    def _fake_hf_processor(name_or_path, *args, **kwargs):
        _ = name_or_path, args, kwargs
        return None

    fake_tokenizer_module.hf_tokenizer = _fake_hf_tokenizer
    fake_tokenizer_module.hf_processor = _fake_hf_processor
    fake_utils_pkg.tokenizer = fake_tokenizer_module
    fake_utils_pkg.hf_tokenizer = _fake_hf_tokenizer
    fake_utils_pkg.hf_processor = _fake_hf_processor

    qwen_dir = tmp_path / "qwen"
    qwen_dir.mkdir()
    (qwen_dir / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")

    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_utils_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils.tokenizer", fake_tokenizer_module)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    sitecustomize.apply_small_swe_runtime_patches()

    tokenizer = fake_tokenizer_module.hf_tokenizer(str(qwen_dir))
    assert calls["kwargs"][-1]["fix_mistral_regex"] is True
    assert tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] is True


def test_sitecustomize_coerces_pad_outputs_to_tensors(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.deprecation_warnings: dict[str, bool] = {}

        def pad(self, *args, **kwargs):
            _ = args, kwargs
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    fake_verl_pkg = types.ModuleType("verl")
    fake_verl_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_utils_pkg = types.ModuleType("verl.utils")
    fake_utils_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_tokenizer_module = types.ModuleType("verl.utils.tokenizer")

    def _fake_hf_tokenizer(name_or_path, *args, **kwargs):
        _ = name_or_path, args, kwargs
        return _FakeTokenizer()

    def _fake_hf_processor(name_or_path, *args, **kwargs):
        _ = name_or_path, args, kwargs
        return None

    fake_tokenizer_module.hf_tokenizer = _fake_hf_tokenizer
    fake_tokenizer_module.hf_processor = _fake_hf_processor
    fake_utils_pkg.tokenizer = fake_tokenizer_module
    fake_utils_pkg.hf_tokenizer = _fake_hf_tokenizer
    fake_utils_pkg.hf_processor = _fake_hf_processor

    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_utils_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils.tokenizer", fake_tokenizer_module)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    sitecustomize.apply_small_swe_runtime_patches()
    tokenizer = fake_tokenizer_module.hf_tokenizer("/tmp/model")
    padded = tokenizer.pad({"input_ids": [1, 2, 3]}, return_tensors="pt")

    assert isinstance(padded["input_ids"], torch.Tensor)
    assert isinstance(padded["attention_mask"], torch.Tensor)


def test_sitecustomize_tokenizer_patch_falls_back_when_fix_flag_is_unsupported(monkeypatch) -> None:
    fake_verl_pkg = types.ModuleType("verl")
    fake_verl_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_utils_pkg = types.ModuleType("verl.utils")
    fake_utils_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_tokenizer_module = types.ModuleType("verl.utils.tokenizer")

    calls: list[dict[str, object]] = []

    def _fake_hf_tokenizer(name_or_path, *args, **kwargs):
        _ = name_or_path, args
        calls.append(dict(kwargs))
        if "fix_mistral_regex" in kwargs:
            raise TypeError("__init__() got an unexpected keyword argument 'fix_mistral_regex'")
        return {}

    def _fake_hf_processor(name_or_path, *args, **kwargs):
        _ = name_or_path, args, kwargs
        return None

    fake_tokenizer_module.hf_tokenizer = _fake_hf_tokenizer
    fake_tokenizer_module.hf_processor = _fake_hf_processor
    fake_utils_pkg.tokenizer = fake_tokenizer_module
    fake_utils_pkg.hf_tokenizer = _fake_hf_tokenizer
    fake_utils_pkg.hf_processor = _fake_hf_processor

    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils", fake_utils_pkg)
    monkeypatch.setitem(sys.modules, "verl.utils.tokenizer", fake_tokenizer_module)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    sitecustomize.apply_small_swe_runtime_patches()
    assert fake_tokenizer_module.hf_tokenizer("/tmp/model") == {}

    assert len(calls) == 2
    assert calls[0]["fix_mistral_regex"] is True
    assert "fix_mistral_regex" not in calls[1]


def test_sitecustomize_patches_transformers_torch_dtype_property(monkeypatch) -> None:
    transformers = pytest.importorskip("transformers")
    _ = transformers
    import transformers.configuration_utils as configuration_utils
    from transformers.configuration_utils import PretrainedConfig

    original_property = PretrainedConfig.torch_dtype
    had_marker = hasattr(PretrainedConfig, "_small_swe_torch_dtype_property_patch")
    original_marker = getattr(PretrainedConfig, "_small_swe_torch_dtype_property_patch", False)
    call_count = {"value": 0}

    def _fake_warning_once(*args, **kwargs):
        _ = args, kwargs
        call_count["value"] += 1

    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")
    monkeypatch.setattr(configuration_utils.logger, "warning_once", _fake_warning_once)

    try:
        sitecustomize.apply_small_swe_runtime_patches()
        cfg = PretrainedConfig()
        cfg.dtype = "float16"
        assert cfg.torch_dtype == "float16"
        cfg.torch_dtype = "float32"
        assert cfg.dtype == "float32"
        assert call_count["value"] == 0
    finally:
        PretrainedConfig.torch_dtype = original_property
        if had_marker:
            setattr(PretrainedConfig, "_small_swe_torch_dtype_property_patch", original_marker)
        else:
            delattr(PretrainedConfig, "_small_swe_torch_dtype_property_patch")


def test_sitecustomize_patches_reward_loop_valid_response_length_indexing(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class _FakeLoop:
        async def run_in_executor(self, executor, fn):
            _ = executor
            return fn()

    class _FakeNaiveRewardManager:
        def __init__(self) -> None:
            self.loop = _FakeLoop()
            self.is_async_reward_score = False
            self.reward_router_address = None
            self.reward_model_tokenizer = None
            self.compute_score = lambda **kwargs: {"score": 0.25, "solution": kwargs["solution_str"]}
            self.decoded_ids: list[int] = []

            def _decode(token_ids, skip_special_tokens: bool = True):
                _ = skip_special_tokens
                if hasattr(token_ids, "tolist"):
                    token_ids = token_ids.tolist()
                self.decoded_ids = [int(item) for item in token_ids]
                return "decoded"

            self.tokenizer = types.SimpleNamespace(decode=_decode)

        async def run_single(self, data):
            data_item = data[0]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            _ = response_ids[:valid_response_length]
            return {"reward_score": 0.0, "reward_extra_info": {"acc": 0.0}}

    fake_verl_pkg = types.ModuleType("verl")
    fake_verl_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_experimental_pkg = types.ModuleType("verl.experimental")
    fake_experimental_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_reward_loop_pkg = types.ModuleType("verl.experimental.reward_loop")
    fake_reward_loop_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_reward_manager_pkg = types.ModuleType("verl.experimental.reward_loop.reward_manager")
    fake_reward_manager_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_naive_module = types.ModuleType("verl.experimental.reward_loop.reward_manager.naive")
    fake_naive_module.NaiveRewardManager = _FakeNaiveRewardManager

    monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
    monkeypatch.setitem(sys.modules, "verl.experimental", fake_experimental_pkg)
    monkeypatch.setitem(sys.modules, "verl.experimental.reward_loop", fake_reward_loop_pkg)
    monkeypatch.setitem(
        sys.modules,
        "verl.experimental.reward_loop.reward_manager",
        fake_reward_manager_pkg,
    )
    monkeypatch.setitem(sys.modules, "verl.experimental.reward_loop.reward_manager.naive", fake_naive_module)
    monkeypatch.setenv("SMALL_SWE_ENABLE_SDPO_RUNTIME_PATCH", "1")

    sitecustomize.apply_small_swe_runtime_patches()

    manager = _FakeNaiveRewardManager()
    item = types.SimpleNamespace(
        batch={
            "responses": torch.tensor([11, 22, 33, 44], dtype=torch.long),
            "attention_mask": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        },
        non_tensor_batch={
            "data_source": "swe-smith",
            "reward_model": {"ground_truth": {}},
        },
    )
    result = asyncio.run(manager.run_single([item]))

    assert result["reward_score"] == 0.25
    assert result["reward_extra_info"]["score"] == 0.25
    assert result["reward_extra_info"]["solution"] == "decoded"
    assert manager.decoded_ids == [11, 22, 33, 44]
