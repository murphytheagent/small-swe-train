"""Custom SDPO agent loop that executes local SWE bridge tool turns."""

from __future__ import annotations

import asyncio
import logging
import numbers
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

from config import on_policy_runtime_defaults
from env.container_pool import BatchContainerPool, ContainerHandle
from env.docker_executor import DockerToolExecutor
from env.runtime_protocol import EnvironmentStep, ToolRequest
from env.task_dataset import TaskSample
from rollout.onpolicy_collector import _build_patch_apply_command
from verl_integration.env_bridge import run_env_bridge_step

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:  # pragma: no cover - exercised in train runtime
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        AsyncLLMServerManager,
        DictConfigWrap,
        register,
    )
except ModuleNotFoundError:  # pragma: no cover - unit-test shim without train extras
    class AgentLoopBase:  # type: ignore[override]
        def __init__(
            self,
            trainer_config: Any,
            server_manager: Any,
            tokenizer: Any,
            processor: Any,
            **kwargs: Any,
        ) -> None:
            self.config = trainer_config.config
            self.server_manager = server_manager
            self.tokenizer = tokenizer
            self.processor = processor
            self.dataset_cls = kwargs.get("dataset_cls")
            self.dataset_config = kwargs.get("dataset_config")
            self.apply_chat_template_kwargs = {}
            self.loop = asyncio.get_event_loop()

        async def process_vision_info(self, messages: list[dict]) -> dict[str, Any]:
            del messages
            return {}

        async def apply_chat_template(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            images: list[Any] | None = None,
            videos: list[Any] | None = None,
            remove_system_prompt: bool = False,
        ) -> list[int]:
            del tools, images, videos, remove_system_prompt
            return [len(str(messages))]

    @dataclass(frozen=True)
    class AgentLoopOutput:  # type: ignore[override]
        prompt_ids: list[int]
        response_ids: list[int]
        response_mask: list[int]
        response_logprobs: list[float] | None = None
        multi_modal_data: dict[str, Any] | None = None
        num_turns: int = 0
        metrics: dict[str, float] | None = None
        extra_fields: dict[str, Any] | None = None

    class AsyncLLMServerManager:  # type: ignore[override]
        async def generate(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    @dataclass(frozen=True)
    class DictConfigWrap:  # type: ignore[override]
        config: Any

    def register(_name: str):  # type: ignore[override]
        def _decorate(subclass: type) -> type:
            return subclass

        return _decorate


@dataclass(frozen=True)
class BridgeLoopTaskContext:
    task_id: str
    image_name: str
    prompt_text: str
    patch: str | None


def build_bridge_task_context(kwargs: Mapping[str, Any]) -> BridgeLoopTaskContext:
    task_id = _require_non_empty_text(kwargs.get("task_id"), "task_id")
    image_name = _require_non_empty_text(kwargs.get("image_name"), "image_name")
    prompt_text = _extract_prompt_text(kwargs)
    patch = _optional_patch(kwargs.get("patch"))
    return BridgeLoopTaskContext(
        task_id=task_id,
        image_name=image_name,
        prompt_text=prompt_text,
        patch=patch,
    )


def build_tool_response_messages(tool_response_blocks: Sequence[str]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for block in tool_response_blocks:
        text = str(block).strip()
        if not text:
            continue
        messages.append({"role": "user", "content": text})
    return messages


def append_response_tokens(
    *,
    full_token_ids: list[int],
    response_mask: list[int],
    response_logprobs: list[float],
    token_ids: Sequence[int],
    generated: bool,
    max_response_length: int,
    token_logprobs: Sequence[float] | None = None,
) -> bool:
    if max_response_length < 1:
        raise ValueError("max_response_length must be >= 1.")

    available = max_response_length - len(response_mask)
    if available <= 0:
        return True

    clipped_token_ids = [int(token) for token in token_ids[:available]]
    if not clipped_token_ids:
        return len(response_mask) >= max_response_length

    full_token_ids.extend(clipped_token_ids)
    response_mask.extend([1 if generated else 0] * len(clipped_token_ids))
    if response_logprobs:
        if token_logprobs is None:
            response_logprobs.extend([0.0] * len(clipped_token_ids))
        else:
            clipped_logprobs = [float(value) for value in token_logprobs[: len(clipped_token_ids)]]
            if len(clipped_logprobs) < len(clipped_token_ids):
                clipped_logprobs.extend([0.0] * (len(clipped_token_ids) - len(clipped_logprobs)))
            response_logprobs.extend(clipped_logprobs)
    elif token_logprobs:
        clipped_logprobs = [float(value) for value in token_logprobs[: len(clipped_token_ids)]]
        if len(clipped_logprobs) < len(clipped_token_ids):
            clipped_logprobs.extend([0.0] * (len(clipped_token_ids) - len(clipped_logprobs)))
        response_logprobs.extend(clipped_logprobs)

    return len(response_mask) >= max_response_length


@register("swe_bridge_agent")
class SWEBridgeAgentLoop(AgentLoopBase):
    """Run SDPO rollout turns against local docker-backed SWE bridge execution."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: Any,
        processor: Any,
        *,
        tool_timeout_sec: int = 60,
        container_start_timeout_sec: int = 120,
        cleanup_timeout_sec: int = 30,
        attempt_timeout_sec: int = 300,
        max_tool_calls_per_turn: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config
        defaults = on_policy_runtime_defaults()

        self.max_user_turns = int(config.actor_rollout_ref.rollout.multi_turn.max_user_turns)
        self.max_assistant_turns = int(config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns)
        self.prompt_length = int(config.actor_rollout_ref.rollout.prompt_length)
        self.response_length = int(config.actor_rollout_ref.rollout.response_length)
        self.tool_timeout_sec = _coerce_positive_int(
            tool_timeout_sec,
            fallback=_coerce_positive_int(defaults.get("tool_timeout_sec"), fallback=60),
        )
        self.container_start_timeout_sec = _coerce_positive_int(
            container_start_timeout_sec,
            fallback=_coerce_positive_int(defaults.get("container_start_timeout_sec"), fallback=120),
        )
        self.cleanup_timeout_sec = _coerce_positive_int(cleanup_timeout_sec, fallback=30)
        self.attempt_timeout_sec = _coerce_positive_int(
            attempt_timeout_sec,
            fallback=_coerce_positive_int(defaults.get("attempt_timeout_sec"), fallback=300),
        )
        self.max_tool_calls_per_turn = _coerce_positive_int(
            max_tool_calls_per_turn,
            fallback=_coerce_positive_int(defaults.get("max_tool_calls_per_turn"), fallback=3),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        full_token_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0}

        task_context = build_bridge_task_context(kwargs)
        logger.info(
            "swe_bridge_agent start task_id=%s image_name=%s",
            task_context.task_id,
            task_context.image_name,
        )
        task_sample = _build_task_sample(task_context=task_context, raw_kwargs=kwargs)
        request_id = uuid4().hex
        assistant_turns = 0
        user_turns = 0
        bridge_step_index = 0
        bridge_error = ""
        timeout_error = ""
        trajectory_steps: list[dict[str, Any]] = []
        tool_response_blocks: list[str] = []
        validation_errors: list[str] = []
        final_turn_has_submit = False
        final_submit_format_valid = False

        pool = BatchContainerPool(
            env_pool_size=1,
            container_start_timeout_sec=self.container_start_timeout_sec,
            cleanup_timeout_sec=self.cleanup_timeout_sec,
            name_prefix="sdpo-swe-bridge",
        )
        handle: ContainerHandle | None = None
        try:
            handles = await asyncio.to_thread(pool.acquire, [task_sample])
            if len(handles) != 1:
                raise RuntimeError("swe_bridge_agent requires exactly one container handle per sample.")
            handle = handles[0]
            executor = DockerToolExecutor(
                container_id=handle.container_id,
                tool_timeout_sec=self.tool_timeout_sec,
            )
            await asyncio.to_thread(_maybe_apply_patch, executor, task_context.patch, self.tool_timeout_sec)

            started_at = time.monotonic()
            while len(response_mask) < self.response_length:
                if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                    break
                if self.max_user_turns and user_turns >= self.max_user_turns:
                    break
                if (time.monotonic() - started_at) > self.attempt_timeout_sec:
                    timeout_error = (
                        f"swe_bridge_agent timed out after {self.attempt_timeout_sec}s "
                        f"for task {task_context.task_id!r}."
                    )
                    break

                generate_started = time.monotonic()
                generation_output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=full_token_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                )
                metrics["generate_sequences"] += time.monotonic() - generate_started

                generated_ids = _coerce_token_ids(getattr(generation_output, "token_ids", []))
                generated_logprobs = _coerce_logprobs(getattr(generation_output, "log_probs", None))
                if not generated_ids:
                    bridge_error = "Model returned empty token_ids in swe_bridge_agent generation."
                    break

                available_tokens = max(0, self.response_length - len(response_mask))
                clipped_generated_ids = generated_ids[:available_tokens]
                clipped_generated_logprobs = (
                    generated_logprobs[:available_tokens] if generated_logprobs is not None else None
                )
                reached_limit = append_response_tokens(
                    full_token_ids=full_token_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    token_ids=clipped_generated_ids,
                    generated=True,
                    max_response_length=self.response_length,
                    token_logprobs=clipped_generated_logprobs,
                )
                assistant_turn_ids = clipped_generated_ids
                assistant_text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(assistant_turn_ids, skip_special_tokens=True),
                )
                assistant_turns += 1
                if reached_limit:
                    break

                bridge_started = time.monotonic()
                try:
                    bridge_result = await asyncio.to_thread(
                        run_env_bridge_step,
                        assistant_text,
                        executor=executor,
                        max_tool_calls=self.max_tool_calls_per_turn,
                        step_index_start=bridge_step_index,
                    )
                except Exception as exc:
                    bridge_error = f"swe_bridge_agent bridge failure: {exc}"
                    break
                metrics["tool_calls"] += time.monotonic() - bridge_started

                bridge_step_index += len(bridge_result.steps)
                trajectory_steps.extend(_serialize_environment_steps(bridge_result.steps))
                turn_validation_errors = _collect_validation_errors(bridge_result.steps)
                if turn_validation_errors:
                    validation_errors.extend(turn_validation_errors)
                if bridge_result.is_terminal:
                    final_turn_has_submit = True
                    final_submit_format_valid = not bool(turn_validation_errors)

                if bridge_result.tool_response_blocks:
                    tool_response_blocks.extend(str(block) for block in bridge_result.tool_response_blocks)
                    tool_messages = build_tool_response_messages(bridge_result.tool_response_blocks)
                    if tool_messages:
                        tool_ids = await self.apply_chat_template(
                            tool_messages,
                            remove_system_prompt=True,
                        )
                        append_response_tokens(
                            full_token_ids=full_token_ids,
                            response_mask=response_mask,
                            response_logprobs=response_logprobs,
                            token_ids=tool_ids,
                            generated=False,
                            max_response_length=self.response_length,
                        )
                        user_turns += 1

                if bridge_result.is_terminal:
                    logger.info(
                        "swe_bridge_agent terminal task_id=%s assistant_turns=%d user_turns=%d",
                        task_context.task_id,
                        assistant_turns,
                        user_turns,
                    )
                    break
        finally:
            await asyncio.to_thread(pool.release_all)

        response_len = len(response_mask)
        if response_len > 0:
            response_ids = full_token_ids[-response_len:]
            prompt_ids = full_token_ids[:-response_len]
        else:
            response_ids = []
            prompt_ids = full_token_ids

        output_multi_modal_data = {}
        if images is not None:
            output_multi_modal_data["images"] = images
        if videos is not None:
            output_multi_modal_data["videos"] = videos

        extra_fields = {
            "task_id": task_context.task_id,
            "image_name": task_context.image_name,
            "container_id": handle.container_id if handle is not None else "",
            "trajectory_steps": trajectory_steps,
            "tool_response_blocks": tool_response_blocks,
            "trajectory_tool_validation_errors": _stable_unique_strings(validation_errors),
            "final_turn_has_submit": final_turn_has_submit,
            "final_submit_format_valid": final_submit_format_valid,
        }
        if bridge_error:
            extra_fields["bridge_error"] = bridge_error
        if timeout_error:
            extra_fields["timeout_error"] = timeout_error

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            multi_modal_data=output_multi_modal_data,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )


def _build_task_sample(*, task_context: BridgeLoopTaskContext, raw_kwargs: Mapping[str, Any]) -> TaskSample:
    return TaskSample(
        task_id=task_context.task_id,
        image_name=task_context.image_name,
        problem_statement=task_context.prompt_text,
        fail_to_pass=raw_kwargs.get("fail_to_pass"),
        pass_to_pass=raw_kwargs.get("pass_to_pass"),
        raw=dict(raw_kwargs),
    )


def _extract_prompt_text(kwargs: Mapping[str, Any]) -> str:
    raw_prompt = kwargs.get("raw_prompt")
    if isinstance(raw_prompt, Sequence) and not isinstance(raw_prompt, (str, bytes)):
        for item in raw_prompt:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role", "")).strip().lower()
            if role != "user":
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content
    fallback = kwargs.get("prompt")
    if isinstance(fallback, str) and fallback.strip():
        return fallback
    return "SWE task prompt unavailable."


def _optional_patch(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    patch = value.strip()
    if not patch:
        return None
    return value


def _require_non_empty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"swe_bridge_agent requires non-empty `{label}` metadata.")
    return value.strip()


def _coerce_positive_int(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, numbers.Integral) and int(value) >= 1:
        return int(value)
    if isinstance(value, float) and value.is_integer() and int(value) >= 1:
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            if parsed >= 1:
                return parsed
    return fallback


def _coerce_token_ids(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    token_ids: list[int] = []
    for token in value:
        if isinstance(token, bool):
            continue
        if isinstance(token, numbers.Integral):
            token_ids.append(int(token))
    return token_ids


def _coerce_logprobs(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    output: list[float] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, numbers.Real):
            output.append(float(item))
    return output if output else None


def _serialize_environment_steps(steps: Sequence[EnvironmentStep]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for step in steps:
        payload.append(
            {
                "step_index": int(step.step_index),
                "tool": step.request.tool,
                "args": dict(step.request.args),
                "stdout": step.response.stdout,
                "stderr": step.response.stderr,
                "exit_code": int(step.response.exit_code),
                "metadata": dict(step.response.metadata),
            }
        )
    return payload


def _collect_validation_errors(steps: Sequence[EnvironmentStep]) -> list[str]:
    errors: list[str] = []
    for step in steps:
        raw_errors = step.response.metadata.get("validation_errors")
        if not isinstance(raw_errors, Sequence) or isinstance(raw_errors, (str, bytes)):
            continue
        for raw_error in raw_errors:
            text = str(raw_error).strip()
            if text:
                errors.append(text)
    return errors


def _stable_unique_strings(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _maybe_apply_patch(executor: DockerToolExecutor, patch: str | None, timeout_sec: int) -> None:
    if patch is None:
        return
    response = executor.run(
        ToolRequest(
            tool="bash",
            args={
                "command": _build_patch_apply_command(),
                "stdin": patch,
                "timeout_sec": timeout_sec,
            },
        )
    )
    if response.exit_code == 0:
        return
    detail = response.stderr.strip() or f"exit_code={response.exit_code}"
    raise RuntimeError(f"swe_bridge_agent failed to apply task patch: {detail}")
