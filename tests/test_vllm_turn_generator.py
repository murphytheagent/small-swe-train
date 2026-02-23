from __future__ import annotations

from env.task_dataset import TaskSample
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
            model_name="Qwen/Qwen3-4B-Instruct-2507",
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

    assert '"tool":"submit"' in turn
    assert captured["base_url"] == "http://localhost:8000/v1"
    assert captured["timeout_sec"] == 12
    payload = captured["payload"]
    assert isinstance(payload, dict)
    messages = payload.get("messages")
    assert isinstance(messages, list)
    assert len(messages) >= 3


def test_load_vllm_turn_generator_config_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SMALL_SWE_VLLM_BASE_URL", "http://localhost:9000/v1/")
    config = load_vllm_turn_generator_config()
    assert config.base_url == "http://localhost:9000/v1"
