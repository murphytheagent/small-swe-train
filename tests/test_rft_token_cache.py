from __future__ import annotations

from types import SimpleNamespace

import pytest

from prompts import build_onpolicy_initial_user_message
from trainer.rft_token_cache import (
    CachedRFTSFTDataset,
    LengthBucketDistributedSampler,
    build_rft_token_cache_fingerprint,
    build_token_cache_records,
    collate_token_cache_rows,
    write_selected_rows_to_token_cache_parquet,
)


class _Tokenizer:
    name_or_path = "Qwen/Qwen3-8B"
    vocab_size = 256
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    chat_template = "{{ messages }}"

    def __init__(self, added_vocab: dict[str, int] | None = None) -> None:
        self._added_vocab = dict(added_vocab or {})

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added_vocab)

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        enable_thinking=None,
        chat_template_kwargs=None,
        **_kwargs,
    ):
        template_kwargs = dict(chat_template_kwargs or {})
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        text = self.render(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=template_kwargs.get("enable_thinking"),
        )
        if not tokenize:
            return text
        input_ids = [ord(char) for char in text]
        if return_dict:
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        return input_ids

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
        **_kwargs,
    ):
        assert add_special_tokens is False
        input_ids = [ord(char) for char in text]
        payload = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload

    def render(self, messages, *, add_generation_prompt=False, enable_thinking=None) -> str:
        chunks = ["SYS"]
        last_query_index = len(messages) - 1
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            content = message.get("content", "")
            if (
                message["role"] == "user"
                and isinstance(content, str)
                and not (
                    content.startswith("<tool_response>")
                    and content.endswith("</tool_response>")
                )
            ):
                last_query_index = index
                break
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            prefix = f"<{role}>"
            if (
                role == "assistant"
                and enable_thinking is False
                and index > last_query_index
                and index == len(messages) - 1
            ):
                prefix += "<think>\n\n</think>\n\n"
            chunks.append(f"{prefix}{content}</{role}>")
        if add_generation_prompt:
            prompt = "<assistant>"
            if enable_thinking is False:
                prompt += "<think>\n\n</think>\n\n"
            chunks.append(prompt)
        return "".join(chunks)


def _selected_row(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    prompt: str = "Fix the bug.",
    assistant_response: str = '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
) -> dict[str, object]:
    return {
        "input_ids": input_ids,
        "action_mask_rft": loss_mask,
        "prompt": prompt,
        "assistant_response": assistant_response,
        "task_id": "task-1",
        "attempt_index": 0,
        "step_index": 0,
        "turn_index": 0,
    }


def test_token_cache_records_keep_one_row_per_prompt_and_required_columns() -> None:
    tokenizer = _Tokenizer()
    records = build_token_cache_records(
        [
            _selected_row(
                [1, 2, 3, 4],
                [0, 1, 1, 1],
                prompt="Task prompt",
                assistant_response="Action",
            ),
            _selected_row(
                [5, 6],
                [1, 1],
                prompt="Second task",
                assistant_response="Done",
            ),
        ],
        tokenizer=tokenizer,
        max_sequence_length=512,
        cache_fingerprint="abc123",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert len(records) == 2
    rendered = "".join(chr(item) for item in records[0]["input_ids"])
    generation_prefix = "<assistant><think>\n\n</think>\n\n"
    assistant_content_start = rendered.index(generation_prefix) + len(generation_prefix)
    assert "Task objective:" in rendered
    assert "Task prompt" in rendered
    assert "<think>\n\n</think>" in rendered
    assert "Action" in rendered
    assert [1, 2, 3, 4] != records[0]["input_ids"]
    assert records[0]["attention_mask"] == [1] * records[0]["sequence_length"]
    assert records[0]["position_ids"] == list(range(records[0]["sequence_length"]))
    assert records[0]["loss_mask"][:assistant_content_start] == [0] * assistant_content_start
    assert records[0]["loss_mask"][assistant_content_start:] == [
        1
    ] * (records[0]["sequence_length"] - assistant_content_start)
    assert records[0]["loss_token_count"] == records[0]["sequence_length"] - assistant_content_start
    assert records[0]["cache_schema_version"] == 1
    assert records[0]["cache_fingerprint"] == "abc123"


def test_token_cache_records_handle_qwen_multiturn_thinking_prefill() -> None:
    tokenizer = _Tokenizer()
    rows = [
        _selected_row(
            [1, 2, 3, 4],
            [0, 1, 1, 1],
            prompt="Task prompt",
            assistant_response="",
        )
    ]
    rows[0]["trajectory_history"] = [
        '<tool_call>{"tool":"bash","args":{"command":"echo hi"}}</tool_call>',
        "<tool_response>hi</tool_response>",
        '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
    ]

    records = build_token_cache_records(
        rows,
        tokenizer=tokenizer,
        max_sequence_length=2048,
        cache_fingerprint="abc123",
        chat_template_kwargs={"enable_thinking": False},
    )

    rendered = "".join(chr(item) for item in records[0]["input_ids"])
    first_action = '<tool_call>{"tool":"bash","args":{"command":"echo hi"}}</tool_call>'
    final_action = '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
    first_action_start = rendered.index(first_action)
    first_action_end = rendered.index("</assistant>", first_action_start) + len("</assistant>")
    final_prefix = "<assistant><think>\n\n</think>\n\n"
    final_action_start = rendered.index(final_prefix) + len(final_prefix)

    assert rendered.count("<think>\n\n</think>") == 1
    assert records[0]["loss_mask"][first_action_start:first_action_end] == [
        1
    ] * (first_action_end - first_action_start)
    assert records[0]["loss_mask"][first_action_end:final_action_start] == [
        0
    ] * (final_action_start - first_action_end)
    assert final_action in rendered
    assert records[0]["loss_token_count"] > len(final_action)


def test_cached_dataset_validates_fingerprint_and_collates_to_batch_max(tmp_path) -> None:
    pytest.importorskip("pandas")
    torch = pytest.importorskip("torch")
    parquet_path = tmp_path / "cache.parquet"
    tokenizer = _Tokenizer()
    write_selected_rows_to_token_cache_parquet(
        [
            _selected_row(
                [1, 2, 3, 4],
                [0, 1, 1, 1],
                prompt="Task prompt",
                assistant_response="Action",
            ),
            _selected_row(
                [5, 6],
                [1, 1],
                prompt="Second task",
                assistant_response="Done",
            ),
        ],
        parquet_path,
        tokenizer=tokenizer,
        cache_fingerprint="expected",
        chat_template_kwargs={"enable_thinking": False},
    )

    dataset = CachedRFTSFTDataset(
        str(parquet_path),
        tokenizer=SimpleNamespace(pad_token_id=0),
        config={
            "train_min_rows": 2,
            "token_cache": {
                "schema_version": 1,
                "expected_fingerprint": "expected",
            },
        },
    )

    batch = collate_token_cache_rows([dataset[0], dataset[1]], pad_token_id=0)
    expected_lengths = [
        len(
            tokenizer.render(
                [
                    {
                        "role": "user",
                        "content": build_onpolicy_initial_user_message(
                            problem_statement="Task prompt"
                        ),
                    },
                    {"role": "assistant", "content": "Action"},
                ],
                add_generation_prompt=False,
                enable_thinking=False,
            )
        ),
        len(
            tokenizer.render(
                [
                    {
                        "role": "user",
                        "content": build_onpolicy_initial_user_message(
                            problem_statement="Second task"
                        ),
                    },
                    {"role": "assistant", "content": "Done"},
                ],
                add_generation_prompt=False,
                enable_thinking=False,
            )
        ),
    ]
    assert len(dataset) == 2
    assert dataset.sequence_lengths == expected_lengths
    assert batch["input_ids"].shape == torch.Size([2, max(expected_lengths)])
    assert batch["attention_mask"][0, : expected_lengths[0]].tolist() == [1] * expected_lengths[0]
    assert batch["attention_mask"][1, : expected_lengths[1]].tolist() == [1] * expected_lengths[1]
    assert batch["attention_mask"][1, expected_lengths[1] :].tolist() == [0] * (
        max(expected_lengths) - expected_lengths[1]
    )
    assert batch["loss_mask"][0].sum().item() > 0

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CachedRFTSFTDataset(
            str(parquet_path),
            tokenizer=SimpleNamespace(pad_token_id=0),
            config={
                "token_cache": {
                    "schema_version": 1,
                    "expected_fingerprint": "different",
                },
            },
        )


def test_cache_fingerprint_changes_when_added_vocab_changes() -> None:
    base = build_rft_token_cache_fingerprint(
        tokenizer=_Tokenizer({"<extra_a>": 10}),
        max_model_len=128,
        data_max_length=128,
    )
    changed = build_rft_token_cache_fingerprint(
        tokenizer=_Tokenizer({"<extra_a>": 10, "<extra_b>": 11}),
        max_model_len=128,
        data_max_length=128,
    )

    assert base != changed


def test_cache_fingerprint_changes_when_chat_template_kwargs_change() -> None:
    base = build_rft_token_cache_fingerprint(
        tokenizer=_Tokenizer(),
        max_model_len=128,
        data_max_length=128,
        chat_template_kwargs={"enable_thinking": False},
    )
    changed = build_rft_token_cache_fingerprint(
        tokenizer=_Tokenizer(),
        max_model_len=128,
        data_max_length=128,
        chat_template_kwargs={"enable_thinking": True},
    )

    assert base != changed


def test_bucket_sampler_is_length_aware_stable_len_and_epoch_shuffle() -> None:
    dataset = SimpleNamespace(sequence_lengths=[2, 10, 3, 11, 4, 12, 5, 13])
    dataset.__len__ = lambda self=dataset: len(dataset.sequence_lengths)  # type: ignore[method-assign]

    class _Dataset:
        sequence_lengths = [2, 10, 3, 11, 4, 12, 5, 13]

        def __len__(self) -> int:
            return len(self.sequence_lengths)

    sampler = LengthBucketDistributedSampler(
        _Dataset(),
        num_replicas=2,
        rank=0,
        batch_size=2,
        seed=7,
        drop_last=True,
        bucket_size=4,
    )
    first_epoch = list(iter(sampler))
    sampler.set_epoch(1)
    second_epoch = list(iter(sampler))

    assert len(sampler) == 4
    assert len(first_epoch) == 4
    assert sorted(first_epoch) != []
    assert first_epoch != second_epoch
