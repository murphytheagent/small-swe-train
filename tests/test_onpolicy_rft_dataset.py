from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from verl_integration import onpolicy_rft_dataset as dataset_module
from verl_integration.onpolicy_rft_dataset import _cache_key


class _TokenizerStub:
    def __init__(
        self,
        *,
        name_or_path: str | None = None,
        vocab_size: int | None = None,
        added_vocab: dict[str, int] | None = None,
    ) -> None:
        if name_or_path is not None:
            self.name_or_path = name_or_path
        if vocab_size is not None:
            self.vocab_size = vocab_size
        self._added_vocab = dict(added_vocab or {})

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added_vocab)


def _build_key(tokenizer: object) -> str:
    return _cache_key(
        data_config_name="swebench_lite",
        turn_generator_mode="proof_tool_chain",
        total_steps=1,
        runtime_overrides={"seed": 11},
        data_overrides={"task_batch_size": 8},
        handoff_overrides={"selection_policy": "terminal_valid"},
        verify_submissions=False,
        parquet_files=["/tmp/train.parquet"],
        tokenizer=tokenizer,
    )


def _runtime_result(selected_count: int) -> dict[str, object]:
    rows = [[index + 1, index + 2] for index in range(selected_count)]
    return {
        "sft_batch": {
            "tensors": {
                "input_ids": rows,
                "attention_mask": [[1] * len(row) for row in rows],
                "position_ids": [list(range(len(row))) for row in rows],
                "loss_mask": [[1] * len(row) for row in rows],
            },
            "grouping_metadata": {
                "group_id": [f"group-{index}" for index in range(selected_count)],
                "task_id": [f"task-{index}" for index in range(selected_count)],
                "attempt_index": [0 for _ in range(selected_count)],
            },
            "meta_info": {"selected_count": selected_count},
        },
        "rejected_rows": [],
    }


def _fake_torch_module() -> SimpleNamespace:
    return SimpleNamespace(
        tensor=lambda values, dtype=None: list(values),
        long="long",
        distributed=None,
    )


@pytest.fixture(autouse=True)
def _clear_onpolicy_cache() -> None:
    dataset_module._ONPOLICY_RFT_CACHE.clear()
    yield
    dataset_module._ONPOLICY_RFT_CACHE.clear()


def test_cache_key_changes_when_tokenizer_checkpoint_changes() -> None:
    key_a = _build_key(_TokenizerStub(name_or_path="checkpoint-a", vocab_size=50000))
    key_b = _build_key(_TokenizerStub(name_or_path="checkpoint-b", vocab_size=50000))

    assert key_a != key_b


def test_cache_key_changes_when_added_vocab_changes() -> None:
    key_a = _build_key(_TokenizerStub(name_or_path="checkpoint-a", vocab_size=50000, added_vocab={"<repo>": 50001}))
    key_b = _build_key(_TokenizerStub(name_or_path="checkpoint-a", vocab_size=50000, added_vocab={"<patch>": 50001}))

    assert key_a != key_b


def test_cache_key_falls_back_to_instance_identity_when_metadata_missing() -> None:
    class MinimalTokenizer:
        pass

    tokenizer_a = MinimalTokenizer()
    tokenizer_b = MinimalTokenizer()
    key_a = _build_key(tokenizer_a)
    key_b = _build_key(tokenizer_b)

    assert key_a != key_b


def test_cache_key_includes_parquet_split_fingerprint() -> None:
    tokenizer = _TokenizerStub(name_or_path="checkpoint-a", vocab_size=50000)
    train_key = _cache_key(
        data_config_name="swebench_lite",
        turn_generator_mode="proof_tool_chain",
        total_steps=1,
        runtime_overrides={"seed": 11},
        data_overrides={"task_batch_size": 8},
        handoff_overrides={"selection_policy": "terminal_valid"},
        verify_submissions=False,
        parquet_files=["/tmp/train.parquet"],
        tokenizer=tokenizer,
    )
    val_key = _cache_key(
        data_config_name="swebench_lite",
        turn_generator_mode="proof_tool_chain",
        total_steps=1,
        runtime_overrides={"seed": 11},
        data_overrides={"task_batch_size": 8},
        handoff_overrides={"selection_policy": "terminal_valid"},
        verify_submissions=False,
        parquet_files=["/tmp/val.parquet"],
        tokenizer=tokenizer,
    )

    assert train_key != val_key


def test_onpolicy_dataset_does_not_cache_empty_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}
    tokenizer = _TokenizerStub(name_or_path="checkpoint-a", vocab_size=50000)

    def _fake_collect(**kwargs) -> dict[str, object]:
        del kwargs
        calls["count"] += 1
        return _runtime_result(0)

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setattr(dataset_module, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    config = {"on_policy": {"enabled": True}}

    with pytest.raises(ValueError, match="zero selected rows"):
        dataset_module.OnPolicyRFTDataset(
            parquet_files=["/tmp/train.parquet"],
            tokenizer=tokenizer,
            config=config,
        )
    with pytest.raises(ValueError, match="zero selected rows"):
        dataset_module.OnPolicyRFTDataset(
            parquet_files=["/tmp/train.parquet"],
            tokenizer=tokenizer,
            config=config,
        )

    assert calls["count"] == 2
    assert not dataset_module._ONPOLICY_RFT_CACHE


def test_onpolicy_dataset_caches_non_empty_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}
    tokenizer = _TokenizerStub(name_or_path="checkpoint-a", vocab_size=50000)

    def _fake_collect(**kwargs) -> dict[str, object]:
        del kwargs
        calls["count"] += 1
        return _runtime_result(2)

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setattr(dataset_module, "collect_onpolicy_rft_runtime_batch", _fake_collect)
    config = {"on_policy": {"enabled": True}}

    first = dataset_module.OnPolicyRFTDataset(
        parquet_files=["/tmp/train.parquet"],
        tokenizer=tokenizer,
        config=config,
    )
    second = dataset_module.OnPolicyRFTDataset(
        parquet_files=["/tmp/train.parquet"],
        tokenizer=tokenizer,
        config=config,
    )

    assert len(first) == 2
    assert len(second) == 2
    assert calls["count"] == 1


def test_onpolicy_dataset_evicts_stale_empty_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _TokenizerStub(name_or_path="checkpoint-a", vocab_size=50000)
    key = _cache_key(
        data_config_name=dataset_module.DEFAULT_ON_POLICY_DATA_CONFIG_NAME,
        turn_generator_mode="default",
        total_steps=1,
        runtime_overrides={},
        data_overrides={},
        handoff_overrides={},
        verify_submissions=False,
        parquet_files=["/tmp/train.parquet"],
        tokenizer=tokenizer,
    )
    dataset_module._ONPOLICY_RFT_CACHE[key] = _runtime_result(0)

    calls = {"count": 0}

    def _fake_collect(**kwargs) -> dict[str, object]:
        del kwargs
        calls["count"] += 1
        return _runtime_result(2)

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setattr(dataset_module, "collect_onpolicy_rft_runtime_batch", _fake_collect)

    with pytest.raises(ValueError, match="zero selected rows"):
        dataset_module.OnPolicyRFTDataset(
            parquet_files=["/tmp/train.parquet"],
            tokenizer=tokenizer,
            config={"on_policy": {"enabled": True}},
        )

    assert key not in dataset_module._ONPOLICY_RFT_CACHE
    assert calls["count"] == 0


def test_onpolicy_dataset_forwards_verify_submissions_from_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests = []

    def _fake_collect(*, request, tokenizer):
        del tokenizer
        captured_requests.append(request)
        return _runtime_result(1)

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setattr(dataset_module, "collect_onpolicy_rft_runtime_batch", _fake_collect)

    dataset_module.OnPolicyRFTDataset(
        parquet_files=["/tmp/train.parquet"],
        tokenizer=_TokenizerStub(name_or_path="checkpoint-a", vocab_size=50000),
        config={
            "on_policy": {
                "enabled": True,
                "runtime_overrides": {"verify_submissions": True},
            }
        },
    )

    assert len(captured_requests) == 1
    assert captured_requests[0].verify_submissions is True
