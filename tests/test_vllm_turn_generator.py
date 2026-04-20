from __future__ import annotations

import config
from env.task_dataset import TaskSample
import pytest
import rollout.vllm_turn_generator as vllm_turn_generator_module
from rollout.vllm_turn_generator import (
    VLLMTurnGeneratorConfig,
    _build_chat_completions_endpoint,
    _extract_assistant_content,
    build_vllm_turn_generator,
    load_vllm_turn_generator_config,
)


def _sample_task() -> TaskSample:
    return TaskSample(
        task_id="task-1",
        image_name="img:1",
        problem_statement="Fix failing test",
        fail_to_pass=["tests/test_bug.py::test_bug"],
        pass_to_pass=["tests/test_ok.py::test_ok"],
        raw={},
    )


def test_build_chat_completions_endpoint_normalizes_v1_paths() -> None:
    assert (
        _build_chat_completions_endpoint("http://localhost:8000/v1")
        == "http://localhost:8000/v1/chat/completions"
    )
    assert (
        _build_chat_completions_endpoint("http://localhost:8000")
        == "http://localhost:8000/v1/chat/completions"
    )
    assert (
        _build_chat_completions_endpoint("http://localhost:8000/v1/chat/completions")
        == "http://localhost:8000/v1/chat/completions"
    )


def test_extract_assistant_content_supports_tool_calls_payload() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit",
                                "arguments": '{"final_response":"done"}',
                            }
                        }
                    ]
                }
            }
        ]
    }
    content = _extract_assistant_content(payload)
    if config.ACTION_PAYLOAD_FORMAT == "xml":
        assert '<tool_call name="submit">' in content
        assert "<final_response><![CDATA[done]]></final_response>" in content
    else:
        assert '"tool":"submit"' in content
        assert '"final_response":"done"' in content


def test_build_vllm_turn_generator_calls_chat_completion(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post_chat_completion(*, base_url, payload, timeout_sec):
        captured["base_url"] = base_url
        captured["payload"] = payload
        captured["timeout_sec"] = timeout_sec
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(
        vllm_turn_generator_module,
        "_post_chat_completion",
        fake_post_chat_completion,
    )

    generator = build_vllm_turn_generator(
        VLLMTurnGeneratorConfig(
            base_url="http://localhost:8000/v1",
            model_name=config.DEFAULT_TRAINING_MODEL_NAME,
            request_timeout_sec=12,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            system_prompt="contract",
        )
    )
    turn = generator(
        task=_sample_task(),
        attempt_index=1,
        turn_index=0,
        step_index=2,
        history=['<tool_response>{"stdout":"ok","stderr":"","exit_code":0}</tool_response>'],
    )

    assert turn == '<tool_call>{"tool":"submit","args":{"final_response":"done"}}</tool_call>'
    assert captured["base_url"] == "http://localhost:8000/v1"
    assert captured["timeout_sec"] == 12
    payload = captured["payload"]
    assert isinstance(payload, dict)
    messages = payload.get("messages")
    assert isinstance(messages, list)
    assert len(messages) >= 3
    first_user = messages[1]
    assert isinstance(first_user, dict)
    first_user_content = str(first_user.get("content", ""))
    assert "Task objective" in first_user_content
    assert "Task ID:" not in first_user_content
    assert "Step Index:" not in first_user_content
    assert "FAIL_TO_PASS" not in first_user_content
    assert "PASS_TO_PASS" not in first_user_content


def test_build_messages_starts_with_single_user_message() -> None:
    messages = vllm_turn_generator_module._build_messages(
        config=VLLMTurnGeneratorConfig(
            base_url="http://localhost:8000/v1",
            model_name=config.DEFAULT_TRAINING_MODEL_NAME,
            request_timeout_sec=12,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            system_prompt="contract",
        ),
        task=_sample_task(),
        attempt_index=0,
        turn_index=0,
        step_index=0,
        history=[],
    )
    roles = [str(message.get("role", "")) for message in messages if isinstance(message, dict)]
    assert roles.count("user") == 1


def test_parse_tool_response_block_uses_model_delimiters(monkeypatch) -> None:
    class _FakeDelimiters:
        tool_response_start = "<resp>"
        tool_response_end = "</resp>"

    monkeypatch.setattr(
        vllm_turn_generator_module,
        "default_delimiters",
        lambda: _FakeDelimiters(),
    )
    parsed = vllm_turn_generator_module._parse_tool_response_block('<resp>{"stdout":"ok"}</resp>')
    assert parsed == {"stdout": "ok"}


def test_load_vllm_turn_generator_config_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SMALL_SWE_VLLM_BASE_URL", "http://localhost:9000/v1/")
    config = load_vllm_turn_generator_config()
    assert config.base_url == "http://localhost:9000/v1"


def test_load_vllm_turn_generator_config_requires_vllm_mapping(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_turn_generator_module,
        "rft_runtime_defaults",
        lambda: {},
    )
    with pytest.raises(ValueError, match="rft_runtime.vllm"):
        load_vllm_turn_generator_config()


def test_load_vllm_turn_generator_config_rejects_invalid_max_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_turn_generator_module,
        "rft_runtime_defaults",
        lambda: {
            "vllm": {
                "base_url": "http://127.0.0.1:8000/v1",
                "model_name": config.DEFAULT_TRAINING_MODEL_NAME,
                "request_timeout_sec": 90,
                "max_tokens": 0,
                "temperature": 0.0,
                "top_p": 1.0,
            }
        },
    )
    with pytest.raises(ValueError, match="max_tokens"):
        load_vllm_turn_generator_config()
