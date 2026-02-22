from __future__ import annotations

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
        tokenizer=tokenizer,
    )


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
