"""OpenAI-compatible vLLM assistant-turn generator for on-policy rollouts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from config import rft_runtime_defaults
from env.task_dataset import TaskSample
from prompts.model_delimiters import default_delimiters
from prompts.runtime_messages import (
    build_onpolicy_initial_user_message,
    build_onpolicy_system_prompt,
)
from rollout.action_format import render_tool_call_block

_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class VLLMTurnGeneratorConfig:
    base_url: str
    model_name: str
    request_timeout_sec: int
    max_tokens: int
    temperature: float
    top_p: float
    system_prompt: str


def build_vllm_turn_generator(
    config: VLLMTurnGeneratorConfig | None = None,
):
    """Build a callable turn generator that queries a vLLM OpenAI endpoint."""
    resolved = config or load_vllm_turn_generator_config()

    def _generate_turn(
        *,
        task: TaskSample,
        attempt_index: int,
        turn_index: int,
        step_index: int,
        history: Sequence[str],
    ) -> str:
        messages = _build_messages(
            config=resolved,
            task=task,
            attempt_index=attempt_index,
            turn_index=turn_index,
            step_index=step_index,
            history=history,
        )
        payload = {
            "model": resolved.model_name,
            "messages": messages,
            "temperature": resolved.temperature,
            "top_p": resolved.top_p,
            "max_tokens": resolved.max_tokens,
        }
        completion_payload = _post_chat_completion(
            base_url=resolved.base_url,
            payload=payload,
            timeout_sec=resolved.request_timeout_sec,
        )
        assistant_text = _extract_assistant_content(completion_payload)
        if not assistant_text.strip():
            raise RuntimeError("vLLM returned an empty assistant turn.")
        return assistant_text.strip()

    return _generate_turn


def load_vllm_turn_generator_config() -> VLLMTurnGeneratorConfig:
    """Resolve vLLM chat settings from centralized defaults + environment overrides."""
    runtime_defaults = rft_runtime_defaults()
    vllm_defaults = _require_mapping(
        runtime_defaults.get("vllm"),
        label="rft_runtime.vllm",
    )

    configured_base_url = _require_non_empty_str(
        vllm_defaults.get("base_url"),
        label="rft_runtime.vllm.base_url",
    )
    configured_model_name = _require_non_empty_str(
        vllm_defaults.get("model_name"),
        label="rft_runtime.vllm.model_name",
    )
    configured_timeout_sec = _require_positive_int(
        vllm_defaults.get("request_timeout_sec"),
        label="rft_runtime.vllm.request_timeout_sec",
    )
    configured_max_tokens = _require_positive_int(
        vllm_defaults.get("max_tokens"),
        label="rft_runtime.vllm.max_tokens",
    )
    configured_temperature = _require_float(
        vllm_defaults.get("temperature"),
        label="rft_runtime.vllm.temperature",
    )
    configured_top_p = _require_float(
        vllm_defaults.get("top_p"),
        label="rft_runtime.vllm.top_p",
    )

    base_url = _env_or_default(
        "SMALL_SWE_VLLM_BASE_URL",
        configured_base_url,
    )
    model_name = _env_or_default(
        "SMALL_SWE_VLLM_MODEL",
        configured_model_name,
    )
    request_timeout_sec = _coerce_positive_int(
        _env_or_default(
            "SMALL_SWE_VLLM_REQUEST_TIMEOUT_SEC",
            str(configured_timeout_sec),
        ),
        fallback=configured_timeout_sec,
    )
    max_tokens = _coerce_positive_int(
        _env_or_default(
            "SMALL_SWE_VLLM_MAX_TOKENS",
            str(configured_max_tokens),
        ),
        fallback=configured_max_tokens,
    )
    temperature = _coerce_float(
        _env_or_default(
            "SMALL_SWE_VLLM_TEMPERATURE",
            str(configured_temperature),
        ),
        fallback=configured_temperature,
    )
    top_p = _coerce_float(
        _env_or_default(
            "SMALL_SWE_VLLM_TOP_P",
            str(configured_top_p),
        ),
        fallback=configured_top_p,
    )
    system_prompt = build_onpolicy_system_prompt()

    return VLLMTurnGeneratorConfig(
        base_url=_normalize_base_url(base_url),
        model_name=model_name,
        request_timeout_sec=request_timeout_sec,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
    )


def _build_messages(
    *,
    config: VLLMTurnGeneratorConfig,
    task: TaskSample,
    attempt_index: int,
    turn_index: int,
    step_index: int,
    history: Sequence[str],
) -> list[dict[str, str]]:
    initial_user_message = build_onpolicy_initial_user_message(
        problem_statement=task.problem_statement,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": initial_user_message},
    ]

    for item in history:
        text = str(item)
        parsed_tool_response = _parse_tool_response_block(text)
        if parsed_tool_response is not None:
            messages.append(
                {
                    "role": "user",
                    "content": "Tool response:\n" + _stable_json(parsed_tool_response),
                }
            )
            continue
        messages.append({"role": "assistant", "content": text})

    return messages


def _parse_tool_response_block(value: str) -> Mapping[str, Any] | None:
    text = value.strip()
    delimiters = default_delimiters()
    start = delimiters.tool_response_start
    end = delimiters.tool_response_end
    if not text.startswith(start) or not text.endswith(end):
        return None
    payload = text[len(start) : -len(end)].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    return parsed


def _post_chat_completion(
    *,
    base_url: str,
    payload: Mapping[str, Any],
    timeout_sec: int,
) -> Mapping[str, Any]:
    endpoint = _build_chat_completions_endpoint(base_url)
    headers = {"Content-Type": "application/json"}
    api_key = _resolve_api_key()
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps(dict(payload), ensure_ascii=True).encode("utf-8")
    request = urllib_request.Request(endpoint, data=body, headers=headers, method="POST")

    try:
        with urllib_request.urlopen(request, timeout=timeout_sec) as response:
            response_body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"vLLM chat completion failed with HTTP {exc.code}: {detail}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Unable to reach vLLM endpoint {endpoint}: {exc.reason}") from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"vLLM returned non-JSON chat completion payload: {response_body[:200]!r}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise RuntimeError("vLLM chat completion payload must be a JSON object.")
    return decoded


def _extract_assistant_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise RuntimeError("vLLM response is missing non-empty `choices`.")

    first = choices[0]
    if not isinstance(first, Mapping):
        raise RuntimeError("vLLM response `choices[0]` must be a JSON object.")

    message = first.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("vLLM response is missing `choices[0].message` object.")

    content = message.get("content")
    if isinstance(content, str):
        if content.strip():
            return content
    elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    chunks.append(maybe_text)
        joined = "".join(chunks).strip()
        if joined:
            return joined

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)) and tool_calls:
        first_call = tool_calls[0]
        if isinstance(first_call, Mapping):
            function_payload = _as_mapping(first_call.get("function"))
            tool_name = str(function_payload.get("name", "")).strip()
            args_payload = function_payload.get("arguments", "{}")
            args_dict = _coerce_json_mapping(args_payload, fallback={})
            if tool_name:
                return render_tool_call_block(
                    {"tool": tool_name, "args": dict(args_dict)},
                    payload_format="json",
                    compact=True,
                )

    raise RuntimeError("vLLM response did not include assistant content.")


def _resolve_api_key() -> str | None:
    for name in ("SMALL_SWE_VLLM_API_KEY", "OPENAI_API_KEY"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return value
    return None


def _build_chat_completions_endpoint(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def _normalize_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"`{label}` must be configured as a mapping.")


def _require_non_empty_str(value: Any, *, label: str) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    raise ValueError(f"`{label}` must be a non-empty string.")


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"`{label}` must be an integer >= 1.")
    if isinstance(value, int) and value >= 1:
        return int(value)
    raise ValueError(f"`{label}` must be an integer >= 1.")


def _require_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"`{label}` must be numeric.")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"`{label}` must be numeric.")


def _coerce_json_mapping(value: Any, *, fallback: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return fallback
        if isinstance(parsed, Mapping):
            return parsed
    return fallback


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _coerce_positive_int(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        try:
            parsed = int(stripped)
        except ValueError:
            return fallback
        if parsed >= 1:
            return parsed
    return fallback


def _coerce_float(value: Any, *, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in _TRUE_STRINGS:
            return 1.0
        if not stripped:
            return fallback
        try:
            return float(stripped)
        except ValueError:
            return fallback
    return fallback
