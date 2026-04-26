from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import config


def _load_entry_module():
    pytest.importorskip("transformers")
    return importlib.import_module("verl_integration.fsdp_sft_trainer_entry")


def test_patched_from_pretrained_uses_sdpa_fallback_when_flash_attn_disabled(
    monkeypatch,
) -> None:
    entry = _load_entry_module()

    captured: dict[str, object] = {}

    def _fake_from_pretrained(*args, **kwargs):
        del args
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(entry, "_ORIGINAL_FROM_PRETRAINED", _fake_from_pretrained)
    monkeypatch.setattr(entry, "_FLASH_ATTN_DISABLED", True)
    monkeypatch.delenv("SMALL_SWE_RFT_ATTN_IMPL", raising=False)
    monkeypatch.delenv("SMALL_SWE_FALLBACK_ATTN_IMPL", raising=False)

    payload = entry._patched_from_pretrained(config.DEFAULT_TRAINING_MODEL_NAME)

    assert payload["attn_implementation"] == "sdpa"
    assert captured["attn_implementation"] == "sdpa"
    assert "use_flash_attention_2" not in payload
    assert "use_flash_attention_2" not in captured


def test_patched_from_pretrained_honors_explicit_attn_impl_override(monkeypatch) -> None:
    entry = _load_entry_module()

    captured: dict[str, object] = {}

    def _fake_from_pretrained(*args, **kwargs):
        del args
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(entry, "_ORIGINAL_FROM_PRETRAINED", _fake_from_pretrained)
    monkeypatch.setattr(entry, "_FLASH_ATTN_DISABLED", True)
    monkeypatch.setenv("SMALL_SWE_RFT_ATTN_IMPL", "flash_attention_2")

    payload = entry._patched_from_pretrained(config.DEFAULT_TRAINING_MODEL_NAME)

    assert payload["attn_implementation"] == "flash_attention_2"
    assert "use_flash_attention_2" not in payload
    assert captured["attn_implementation"] == "flash_attention_2"


def test_patched_from_pretrained_replaces_flash_attn_impl_when_disabled(monkeypatch) -> None:
    entry = _load_entry_module()

    captured: dict[str, object] = {}

    def _fake_from_pretrained(*args, **kwargs):
        del args
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(entry, "_ORIGINAL_FROM_PRETRAINED", _fake_from_pretrained)
    monkeypatch.setattr(entry, "_FLASH_ATTN_DISABLED", True)
    monkeypatch.delenv("SMALL_SWE_RFT_ATTN_IMPL", raising=False)
    monkeypatch.setenv("SMALL_SWE_FALLBACK_ATTN_IMPL", "sdpa")

    payload = entry._patched_from_pretrained(
        config.DEFAULT_TRAINING_MODEL_NAME,
        attn_implementation="flash_attention_2",
    )

    assert payload["attn_implementation"] == "sdpa"
    assert captured["attn_implementation"] == "sdpa"
    assert "use_flash_attention_2" not in payload
    assert "use_flash_attention_2" not in captured


def test_patched_from_pretrained_sets_model_dtype_for_flash_attn(monkeypatch) -> None:
    entry = _load_entry_module()
    torch = pytest.importorskip("torch")

    captured: dict[str, object] = {}

    def _fake_from_pretrained(*args, **kwargs):
        del args
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(entry, "_ORIGINAL_FROM_PRETRAINED", _fake_from_pretrained)
    monkeypatch.setattr(entry, "_FLASH_ATTN_DISABLED", False)
    monkeypatch.setenv("SMALL_SWE_RFT_MODEL_DTYPE", "bf16")

    payload = entry._patched_from_pretrained(
        config.DEFAULT_TRAINING_MODEL_NAME,
        attn_implementation="flash_attention_2",
    )

    assert payload["dtype"] == torch.bfloat16
    assert captured["dtype"] == torch.bfloat16


def test_patched_from_pretrained_does_not_override_explicit_dtype(monkeypatch) -> None:
    entry = _load_entry_module()

    captured: dict[str, object] = {}

    def _fake_from_pretrained(*args, **kwargs):
        del args
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(entry, "_ORIGINAL_FROM_PRETRAINED", _fake_from_pretrained)
    monkeypatch.setattr(entry, "_FLASH_ATTN_DISABLED", False)
    monkeypatch.setenv("SMALL_SWE_RFT_MODEL_DTYPE", "bf16")

    payload = entry._patched_from_pretrained(
        config.DEFAULT_TRAINING_MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype="auto",
    )

    assert payload["torch_dtype"] == "auto"
    assert captured["torch_dtype"] == "auto"


def test_patched_from_pretrained_falls_back_to_torch_dtype_for_legacy_transformers(
    monkeypatch,
) -> None:
    entry = _load_entry_module()
    torch = pytest.importorskip("torch")

    captured: dict[str, object] = {}

    def _fake_from_pretrained(*args, **kwargs):
        del args
        if "dtype" in kwargs:
            raise TypeError("got an unexpected keyword argument 'dtype'")
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(entry, "_ORIGINAL_FROM_PRETRAINED", _fake_from_pretrained)
    monkeypatch.setattr(entry, "_FLASH_ATTN_DISABLED", False)
    monkeypatch.setenv("SMALL_SWE_RFT_MODEL_DTYPE", "bf16")

    payload = entry._patched_from_pretrained(
        config.DEFAULT_TRAINING_MODEL_NAME,
        attn_implementation="flash_attention_2",
    )

    assert "dtype" not in payload
    assert payload["torch_dtype"] == torch.bfloat16
    assert captured["torch_dtype"] == torch.bfloat16


def test_inner_validation_disabled_when_test_freq_zero_or_val_files_empty() -> None:
    entry = _load_entry_module()

    enabled_config = SimpleNamespace(
        trainer={"test_freq": 10},
        data={"val_files": ["/tmp/val.parquet"]},
    )
    zero_freq_config = SimpleNamespace(
        trainer={"test_freq": 0},
        data={"val_files": ["/tmp/val.parquet"]},
    )
    empty_val_config = SimpleNamespace(
        trainer={"test_freq": 10},
        data={"val_files": []},
    )

    assert entry._inner_validation_disabled(enabled_config) is False
    assert entry._inner_validation_disabled(zero_freq_config) is True
    assert entry._inner_validation_disabled(empty_val_config) is True


def test_patched_run_sft_uses_empty_validation_dataset_when_disabled(monkeypatch) -> None:
    entry = _load_entry_module()
    omegaconf = pytest.importorskip("omegaconf")

    config_payload = {
        "ulysses_sequence_parallel_size": 1,
        "model": {
            "partial_pretrain": "/tmp/model",
            "trust_remote_code": False,
        },
        "data": {
            "train_files": "/tmp/train.parquet",
            "val_files": [],
            "train_max_samples": -1,
        },
        "trainer": {
            "test_freq": 0,
        },
    }
    trainer_config = omegaconf.OmegaConf.create(config_payload)

    class _Mesh:
        def get_rank(self):
            return 0

        def size(self, *_args):
            return 1

        def get_local_rank(self, *_args):
            return 0

    captured: dict[str, object] = {}

    class _Trainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self) -> None:
            captured["fit_called"] = True

    monkeypatch.setattr(entry._verl_sft_trainer, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(
        entry._verl_sft_trainer,
        "initialize_global_process_group",
        lambda: (0, 0, 1),
    )
    monkeypatch.setattr(entry._verl_sft_trainer, "init_device_mesh", lambda **_kwargs: _Mesh())
    monkeypatch.setattr(entry._verl_sft_trainer, "copy_to_local", lambda **_kwargs: "/tmp/model")
    monkeypatch.setattr(entry, "_load_hf_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        entry._verl_sft_trainer,
        "create_sft_dataset",
        lambda *_args, **_kwargs: ["train"],
    )
    monkeypatch.setattr(entry, "_SmallSWEFSDPSFTTrainer", _Trainer)
    monkeypatch.setattr(entry._verl_sft_trainer, "destroy_global_process_group", lambda: None)

    entry._patched_run_sft(trainer_config)

    assert captured["train_dataset"] == ["train"]
    assert isinstance(captured["val_dataset"], entry._EmptyValidationDataset)
    assert captured["fit_called"] is True


def test_fit_without_validation_never_calls_validation_with_zero_test_freq(monkeypatch) -> None:
    entry = _load_entry_module()
    omegaconf = pytest.importorskip("omegaconf")

    class _Mesh:
        def get_rank(self):
            return 0

    class _Sampler:
        def set_epoch(self, *, epoch):
            self.epoch = epoch

    class _TensorDict:
        def __init__(self, data, *, batch_size):
            self.data = data
            self.batch_size = batch_size

        def to(self, _device_name):
            return self

    class _Tracking:
        def __init__(self, **_kwargs):
            self.logged = []

        def log(self, *, data, step):
            self.logged.append((step, data))

    config_payload = {
        "data": {
            "train_batch_size": 1,
        },
        "trainer": {
            "project_name": "project",
            "experiment_name": "experiment",
            "logger": ["console"],
            "group_name": None,
            "total_epochs": 1,
            "total_training_steps": 1,
            "test_freq": 0,
            "save_freq": 0,
        },
    }
    fake_trainer = SimpleNamespace(
        device_mesh=_Mesh(),
        config=omegaconf.OmegaConf.create(config_payload),
        resume_global_step=0,
        train_dataloader=[{"tokens": [1]}],
        steps_per_epoch=1,
        train_sampler=_Sampler(),
        device_name="cpu",
        total_training_steps=None,
    )
    saved_steps: list[int] = []
    fake_trainer.training_step = lambda _data: {"train/time(s)": 0.0}
    fake_trainer.save_checkpoint = lambda *, step: saved_steps.append(step)
    fake_trainer.validation_step = lambda _data: pytest.fail("validation should not run")

    monkeypatch.setattr(entry._verl_sft_trainer, "Tracking", _Tracking)
    monkeypatch.setattr(entry._verl_sft_trainer, "TensorDict", _TensorDict)
    monkeypatch.setattr(entry._verl_sft_trainer, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(entry._verl_sft_trainer, "log_with_rank", lambda *_args, **_kwargs: None)

    entry._fit_without_validation(fake_trainer)

    assert saved_steps == [1]
