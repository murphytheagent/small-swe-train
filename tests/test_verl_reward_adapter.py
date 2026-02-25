from __future__ import annotations

from dataclasses import dataclass

import pytest

from verl_integration import reward_adapter


@dataclass
class _FakeBatch:
    batch: dict
    non_tensor_batch: dict


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return " ".join(str(token) for token in token_ids)


def test_dataproto_to_rows_extracts_metadata_and_tool_outputs() -> None:
    batch = _FakeBatch(
        batch={
            "responses": [
                [11, 12, 13],
                [21, 22],
            ],
            "response_mask": [
                [1, 1, 1],
                [1, 1],
            ],
        },
        non_tensor_batch={
            "raw_prompt": [
                [{"role": "user", "content": "Fix task one"}],
                [{"role": "system", "content": "guide"}, {"role": "user", "content": "Fix task two"}],
            ],
            "task_id": ["task-1", "task-2"],
            "image_name": ["img-1", "img-2"],
            "step_index": [3, 4],
            "attempt_index": [0, 1],
            "turn_index": [2, 3],
            "trajectory_steps": [
                [
                    {
                        "tool": "bash",
                        "stdout": "done",
                        "stderr": "",
                        "exit_code": 0,
                        "metadata": {"k": "v"},
                    }
                ],
                [],
            ],
            "reward_model": [
                {"ground_truth": {"resolved": True}},
                {"ground_truth": {"resolved": False}},
            ],
            "tool_response_blocks": [["<tool_response>{}</tool_response>"], []],
        },
    )

    rows = reward_adapter.dataproto_to_rows(batch=batch, tokenizer=_FakeTokenizer())

    assert len(rows) == 2
    assert rows[0]["prompt"] == "Fix task one"
    assert rows[0]["task_id"] == "task-1"
    assert rows[0]["image_name"] == "img-1"
    assert rows[0]["response_text"] == "11 12 13"
    assert rows[0]["tool_output"]["exit_code"] == 0
    assert rows[0]["resolved"] is True
    assert rows[0]["_response_mask"] == [1, 1, 1]
    assert rows[1]["prompt"] == "Fix task two"
    assert rows[1]["tool_output"] == {}


def test_rows_to_reward_tensor_requires_torch_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reward_adapter, "torch", None)

    with pytest.raises(RuntimeError, match="requires torch"):
        reward_adapter.rows_to_reward_tensor(
            [
                {
                    "response_text": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
                    "assistant_response": '<tool_call>{"tool":"submit","args":{"final_response":"ok"}}</tool_call>',
                    "_response_mask": [1],
                    "resolved": True,
                }
            ]
        )


def test_rows_to_reward_tensor_marks_last_valid_response_token_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    rows = [
        {
            "response_text": '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
            "assistant_response": '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>',
            "_response_mask": [0, 1, 1, 0],
            "resolved": True,
        },
        {
            "response_text": '<tool_call>{"tool":"search","args":{"query":"needle"}}</tool_call>',
            "assistant_response": '<tool_call>{"tool":"search","args":{"query":"needle"}}</tool_call>',
            "_response_mask": [1, 1, 0, 0],
            "resolved": False,
        },
    ]

    reward_tensor, reward_extra_infos = reward_adapter.rows_to_reward_tensor(
        rows,
        response_width=4,
    )

    assert reward_tensor.shape == (2, 4)
    assert float(reward_tensor[0, 2].item()) == pytest.approx(1.0)
    assert float(reward_tensor[0].sum().item()) == pytest.approx(1.0)
    assert float(reward_tensor[1].sum().item()) == pytest.approx(0.0)
    assert "feedback" in reward_extra_infos
    assert len(reward_extra_infos["feedback"]) == 2
    _ = torch  # silence lint for optional import in skip environments
