from __future__ import annotations

import pytest

from verl_integration.swe_bridge_agent_loop import (
    append_response_tokens,
    build_bridge_task_context,
    build_tool_response_messages,
)


def test_build_bridge_task_context_requires_task_metadata() -> None:
    with pytest.raises(ValueError, match="task_id"):
        build_bridge_task_context(
            {
                "image_name": "sweb.eval.x86_64",
                "raw_prompt": [{"role": "user", "content": "Fix test."}],
            }
        )


def test_build_bridge_task_context_extracts_prompt_and_patch() -> None:
    context = build_bridge_task_context(
        {
            "task_id": "task-17",
            "image_name": "sweb.eval.x86_64",
            "patch": "diff --git ...",
            "raw_prompt": [
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": "Fix failing tests."},
            ],
        }
    )
    assert context.task_id == "task-17"
    assert context.image_name == "sweb.eval.x86_64"
    assert context.prompt_text == "Fix failing tests."
    assert context.patch == "diff --git ..."


def test_build_tool_response_messages_keeps_non_empty_blocks() -> None:
    messages = build_tool_response_messages(
        [
            "  ",
            "<tool_response>{\"stdout\":\"ok\"}</tool_response>",
        ]
    )
    assert messages == [{"role": "user", "content": "<tool_response>{\"stdout\":\"ok\"}</tool_response>"}]


def test_append_response_tokens_preserves_generated_vs_tool_masks() -> None:
    full_token_ids: list[int] = [100, 101]
    response_mask: list[int] = []
    response_logprobs: list[float] = []

    reached_limit = append_response_tokens(
        full_token_ids=full_token_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        token_ids=[11, 12],
        generated=True,
        max_response_length=4,
        token_logprobs=[-0.2, -0.3],
    )
    assert not reached_limit

    reached_limit = append_response_tokens(
        full_token_ids=full_token_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        token_ids=[21, 22],
        generated=False,
        max_response_length=4,
    )
    assert reached_limit
    assert full_token_ids == [100, 101, 11, 12, 21, 22]
    assert response_mask == [1, 1, 0, 0]
    assert response_logprobs == [-0.2, -0.3, 0.0, 0.0]


def test_append_response_tokens_clips_to_available_budget() -> None:
    full_token_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []

    reached_limit = append_response_tokens(
        full_token_ids=full_token_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        token_ids=[1, 2, 3],
        generated=True,
        max_response_length=2,
        token_logprobs=[-1.0, -2.0, -3.0],
    )

    assert reached_limit
    assert full_token_ids == [1, 2]
    assert response_mask == [1, 1]
    assert response_logprobs == [-1.0, -2.0]
