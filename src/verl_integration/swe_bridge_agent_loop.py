"""Custom SDPO agent loop that executes local SWE bridge tool turns."""

from __future__ import annotations

import asyncio
import logging
import numbers
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence, TypeVar
from uuid import uuid4

from env.container_pool import BatchContainerPool, ContainerHandle
from env.docker_executor import DockerToolExecutor
from env.runtime_protocol import EnvironmentStep, ToolRequest
from env.task_dataset import TaskSample
from prompts import build_onpolicy_system_prompt, build_sdpo_rollout_followup_user_message
from rollout.onpolicy_collector import _build_patch_apply_command
from verl_integration.env_bridge import run_env_bridge_step
from verl_integration.submission_verifier import run_submission_verifier

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - runtime env should include train deps
    yaml = None  # type: ignore[assignment]

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


@dataclass(frozen=True)
class BridgeLoopRuntimeSettings:
    env_pool_size: int
    tool_timeout_sec: int
    container_start_timeout_sec: int
    cleanup_timeout_sec: int
    attempt_timeout_sec: int
    max_tool_calls_per_turn: int
    verifier_timeout_sec: int


_FALLBACK_RUNTIME_DEFAULTS: dict[str, int] = {
    "env_pool_size": 8,
    "tool_timeout_sec": 60,
    "container_start_timeout_sec": 120,
    "cleanup_timeout_sec": 30,
    "attempt_timeout_sec": 300,
    "max_tool_calls_per_turn": 3,
    "verifier_timeout_sec": 600,
}
_DEFAULT_LOOP_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/verl/agent_loops/swe_bridge_agent.yaml"
_CONTAINER_SLOT_LOCK = threading.Lock()
_CONTAINER_SLOT_GATES: dict[int, threading.BoundedSemaphore] = {}


def _load_runtime_defaults_from_yaml() -> dict[str, int]:
    defaults = dict(_FALLBACK_RUNTIME_DEFAULTS)

    def _parse_positive_int(raw_value: Any, *, fallback: int) -> int:
        if isinstance(raw_value, bool):
            return fallback
        if isinstance(raw_value, numbers.Number):
            candidate = int(raw_value)
            return candidate if candidate > 0 else fallback
        if isinstance(raw_value, str):
            stripped = raw_value.strip()
            if not stripped:
                return fallback
            try:
                candidate = int(stripped)
            except ValueError:
                return fallback
            return candidate if candidate > 0 else fallback
        return fallback

    if yaml is None:
        logger.warning("PyYAML unavailable; using fallback SWE bridge runtime defaults.")
        return defaults
    try:
        parsed = yaml.safe_load(_DEFAULT_LOOP_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - fallback for non-standard runtime packaging
        logger.warning(
            "Failed to read SWE bridge runtime defaults from %s; using fallback defaults: %s",
            _DEFAULT_LOOP_CONFIG_PATH,
            exc,
        )
        return defaults

    config_entry: Mapping[str, Any] | None = None
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)) and parsed:
        first_item = parsed[0]
        if isinstance(first_item, Mapping):
            config_entry = first_item
    elif isinstance(parsed, Mapping):
        config_entry = parsed

    if config_entry is None:
        logger.warning(
            "Unexpected swe_bridge_agent config shape in %s; using fallback defaults.",
            _DEFAULT_LOOP_CONFIG_PATH,
        )
        return defaults

    for key, fallback in _FALLBACK_RUNTIME_DEFAULTS.items():
        value = config_entry.get(key)
        defaults[key] = _parse_positive_int(value, fallback=fallback)
    return defaults


_RUNTIME_DEFAULTS = _load_runtime_defaults_from_yaml()
_DEFAULT_ENV_POOL_SIZE = _RUNTIME_DEFAULTS["env_pool_size"]
_DEFAULT_TOOL_TIMEOUT_SEC = _RUNTIME_DEFAULTS["tool_timeout_sec"]
_DEFAULT_CONTAINER_START_TIMEOUT_SEC = _RUNTIME_DEFAULTS["container_start_timeout_sec"]
_DEFAULT_CLEANUP_TIMEOUT_SEC = _RUNTIME_DEFAULTS["cleanup_timeout_sec"]
_DEFAULT_ATTEMPT_TIMEOUT_SEC = _RUNTIME_DEFAULTS["attempt_timeout_sec"]
_DEFAULT_MAX_TOOL_CALLS_PER_TURN = _RUNTIME_DEFAULTS["max_tool_calls_per_turn"]
_DEFAULT_VERIFIER_TIMEOUT_SEC = _RUNTIME_DEFAULTS["verifier_timeout_sec"]
_DEFAULT_STAGE_HEARTBEAT_SEC = 60

_T = TypeVar("_T")


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


def resolve_bridge_loop_runtime_config(
    *,
    env_pool_size: int | None = None,
    tool_timeout_sec: int | None = None,
    container_start_timeout_sec: int | None = None,
    cleanup_timeout_sec: int | None = None,
    attempt_timeout_sec: int | None = None,
    max_tool_calls_per_turn: int | None = None,
    verifier_timeout_sec: int | None = None,
) -> BridgeLoopRuntimeSettings:
    return BridgeLoopRuntimeSettings(
        env_pool_size=_coerce_positive_int(env_pool_size, fallback=_DEFAULT_ENV_POOL_SIZE),
        tool_timeout_sec=_coerce_positive_int(
            tool_timeout_sec,
            fallback=_DEFAULT_TOOL_TIMEOUT_SEC,
        ),
        container_start_timeout_sec=_coerce_positive_int(
            container_start_timeout_sec,
            fallback=_DEFAULT_CONTAINER_START_TIMEOUT_SEC,
        ),
        cleanup_timeout_sec=_coerce_positive_int(
            cleanup_timeout_sec,
            fallback=_DEFAULT_CLEANUP_TIMEOUT_SEC,
        ),
        attempt_timeout_sec=_coerce_positive_int(
            attempt_timeout_sec,
            fallback=_DEFAULT_ATTEMPT_TIMEOUT_SEC,
        ),
        max_tool_calls_per_turn=_coerce_positive_int(
            max_tool_calls_per_turn,
            fallback=_DEFAULT_MAX_TOOL_CALLS_PER_TURN,
        ),
        verifier_timeout_sec=_coerce_positive_int(
            verifier_timeout_sec,
            fallback=_DEFAULT_VERIFIER_TIMEOUT_SEC,
        ),
    )


def _remaining_attempt_timeout_sec(*, started_at: float, attempt_timeout_sec: int) -> float:
    return attempt_timeout_sec - (time.monotonic() - started_at)


async def _await_with_attempt_timeout(
    awaitable: Awaitable[_T],
    *,
    task_id: str,
    stage: str,
    started_at: float,
    attempt_timeout_sec: int,
) -> _T:
    remaining_sec = _remaining_attempt_timeout_sec(started_at=started_at, attempt_timeout_sec=attempt_timeout_sec)
    if remaining_sec <= 0:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise TimeoutError(
            f"swe_bridge_agent attempt timed out after {attempt_timeout_sec}s for task {task_id!r} before stage {stage!r}."
        )
    try:
        return await asyncio.wait_for(awaitable, timeout=remaining_sec)
    except asyncio.TimeoutError as exc:
        elapsed_sec = time.monotonic() - started_at
        raise TimeoutError(
            f"swe_bridge_agent attempt timed out after {elapsed_sec:.1f}s/{attempt_timeout_sec}s "
            f"for task {task_id!r} in stage {stage!r}."
        ) from exc


async def _emit_stage_heartbeats(
    *,
    task_id: str,
    image_name: str,
    started_at: float,
    stage_getter: Callable[[], str],
    stop_event: asyncio.Event,
    interval_sec: int = _DEFAULT_STAGE_HEARTBEAT_SEC,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            continue
        except asyncio.TimeoutError:
            logger.warning(
                "swe_bridge_agent heartbeat task_id=%s image_name=%s stage=%s elapsed=%.1fs",
                task_id,
                image_name,
                stage_getter(),
                time.monotonic() - started_at,
            )


async def _acquire_container_slot(
    gate: threading.BoundedSemaphore,
    *,
    task_id: str,
    stage: str,
    started_at: float,
    attempt_timeout_sec: int,
    poll_interval_sec: float = 0.05,
) -> None:
    while True:
        if gate.acquire(blocking=False):
            return
        remaining_sec = _remaining_attempt_timeout_sec(started_at=started_at, attempt_timeout_sec=attempt_timeout_sec)
        if remaining_sec <= 0:
            raise TimeoutError(
                f"swe_bridge_agent attempt timed out after {attempt_timeout_sec}s for task {task_id!r} in stage {stage!r}."
            )
        await asyncio.sleep(min(poll_interval_sec, max(0.01, remaining_sec)))


def build_agent_loop_messages(kwargs: Mapping[str, Any]) -> list[dict[str, str]]:
    """Normalize prompt messages and enforce a tool-contract-guided generation state."""
    parsed_messages: list[dict[str, str]] = []
    raw_prompt = kwargs.get("raw_prompt")
    if isinstance(raw_prompt, Sequence) and not isinstance(raw_prompt, (str, bytes)):
        for item in raw_prompt:
            if not isinstance(item, Mapping):
                continue
            role = _as_role_text(item.get("role"))
            content = _as_content_text(item.get("content"))
            if role and content:
                parsed_messages.append({"role": role, "content": content})

    if not parsed_messages:
        fallback_prompt = _extract_prompt_text(kwargs).strip()
        if fallback_prompt:
            parsed_messages.append({"role": "user", "content": fallback_prompt})

    if not parsed_messages:
        raise ValueError("swe_bridge_agent requires at least one prompt message.")

    system_contract = build_onpolicy_system_prompt()
    if parsed_messages[0]["role"] == "system":
        parsed_messages[0] = {
            "role": "system",
            "content": f"{system_contract}\n\n{parsed_messages[0]['content']}",
        }
    else:
        parsed_messages.insert(0, {"role": "system", "content": system_contract})

    if parsed_messages[-1]["role"] != "user":
        parsed_messages.append(
            {
                "role": "user",
                "content": build_sdpo_rollout_followup_user_message(),
            }
        )

    return parsed_messages


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
        tool_timeout_sec: int | None = None,
        container_start_timeout_sec: int | None = None,
        cleanup_timeout_sec: int | None = None,
        attempt_timeout_sec: int | None = None,
        max_tool_calls_per_turn: int | None = None,
        env_pool_size: int | None = None,
        verifier_timeout_sec: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config
        runtime_config = resolve_bridge_loop_runtime_config(
            env_pool_size=env_pool_size,
            tool_timeout_sec=tool_timeout_sec,
            container_start_timeout_sec=container_start_timeout_sec,
            cleanup_timeout_sec=cleanup_timeout_sec,
            attempt_timeout_sec=attempt_timeout_sec,
            max_tool_calls_per_turn=max_tool_calls_per_turn,
            verifier_timeout_sec=verifier_timeout_sec,
        )

        self.max_user_turns = int(config.actor_rollout_ref.rollout.multi_turn.max_user_turns)
        self.max_assistant_turns = int(config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns)
        self.prompt_length = int(config.actor_rollout_ref.rollout.prompt_length)
        self.response_length = int(config.actor_rollout_ref.rollout.response_length)
        self.env_pool_size = runtime_config.env_pool_size
        self.tool_timeout_sec = runtime_config.tool_timeout_sec
        self.container_start_timeout_sec = runtime_config.container_start_timeout_sec
        self.cleanup_timeout_sec = runtime_config.cleanup_timeout_sec
        self.attempt_timeout_sec = runtime_config.attempt_timeout_sec
        self.max_tool_calls_per_turn = runtime_config.max_tool_calls_per_turn
        self.verifier_timeout_sec = runtime_config.verifier_timeout_sec

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        task_context = build_bridge_task_context(kwargs)
        started_at = time.monotonic()
        current_stage = "init"
        stage_stop_event = asyncio.Event()
        heartbeat_task: asyncio.Task[None] | None = None

        def _set_stage(stage: str) -> None:
            nonlocal current_stage
            if stage == current_stage:
                return
            current_stage = stage
            logger.warning(
                "swe_bridge_agent stage task_id=%s image_name=%s stage=%s elapsed=%.1fs",
                task_context.task_id,
                task_context.image_name,
                current_stage,
                time.monotonic() - started_at,
            )

        logger.warning(
            "swe_bridge_agent start task_id=%s image_name=%s",
            task_context.task_id,
            task_context.image_name,
        )

        _set_stage("build_messages")
        messages = build_agent_loop_messages(kwargs)
        _set_stage("process_vision_info")
        multi_modal_data = await _await_with_attempt_timeout(
            self.process_vision_info(messages),
            task_id=task_context.task_id,
            stage=current_stage,
            started_at=started_at,
            attempt_timeout_sec=self.attempt_timeout_sec,
        )
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        _set_stage("tokenize_prompt")
        raw_prompt_ids = await _await_with_attempt_timeout(
            self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
            ),
            task_id=task_context.task_id,
            stage=current_stage,
            started_at=started_at,
            attempt_timeout_sec=self.attempt_timeout_sec,
        )
        raw_prompt_token_ids = _coerce_token_ids(raw_prompt_ids)
        canonical_prompt_ids = _clip_prompt_for_rollout_context(
            raw_prompt_token_ids,
            prompt_length=self.prompt_length,
        )
        if not canonical_prompt_ids:
            raise ValueError("swe_bridge_agent produced an empty prompt token sequence.")
        if len(canonical_prompt_ids) < len(raw_prompt_token_ids):
            logger.info(
                "swe_bridge_agent clipped prompt context from %d to %d tokens (prompt_length=%d).",
                len(raw_prompt_token_ids),
                len(canonical_prompt_ids),
                self.prompt_length,
            )
        full_token_ids = list(canonical_prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0}

        task_sample = _build_task_sample(task_context=task_context, raw_kwargs=kwargs)
        request_id = uuid4().hex
        assistant_turns = 0
        user_turns = 0
        bridge_step_index = 0
        bridge_error = ""
        timeout_error = ""
        loop_exit_reason = "response_length_budget_exhausted"
        trajectory_steps: list[dict[str, Any]] = []
        tool_response_blocks: list[str] = []
        trajectory_assistant_turns: list[str] = []
        trajectory_assistant_turn_token_lengths: list[int] = []
        trajectory_turn_tool_response_blocks: list[list[str]] = []
        validation_errors: list[str] = []
        final_turn_has_submit = False
        final_submit_format_valid = False
        verification_metadata: dict[str, Any] = {}

        container_slot_gate = _get_container_slot_gate(self.env_pool_size)
        pool = BatchContainerPool(
            # One loop invocation operates on one task sample; shared slot gating
            # enforces global env_pool_size concurrency across concurrent loop runs.
            env_pool_size=1,
            container_start_timeout_sec=self.container_start_timeout_sec,
            cleanup_timeout_sec=self.cleanup_timeout_sec,
            name_prefix="sdpo-swe-bridge",
        )
        handle: ContainerHandle | None = None
        container_slot_acquired = False
        heartbeat_task = asyncio.create_task(
            _emit_stage_heartbeats(
                task_id=task_context.task_id,
                image_name=task_context.image_name,
                started_at=started_at,
                stage_getter=lambda: current_stage,
                stop_event=stage_stop_event,
            )
        )
        try:
            _set_stage("wait_container_slot")
            await _acquire_container_slot(
                container_slot_gate,
                task_id=task_context.task_id,
                stage=current_stage,
                started_at=started_at,
                attempt_timeout_sec=self.attempt_timeout_sec,
            )
            container_slot_acquired = True

            _set_stage("acquire_container")
            handles = await _await_with_attempt_timeout(
                asyncio.to_thread(pool.acquire, [task_sample]),
                task_id=task_context.task_id,
                stage=current_stage,
                started_at=started_at,
                attempt_timeout_sec=self.attempt_timeout_sec,
            )
            if len(handles) != 1:
                raise RuntimeError("swe_bridge_agent requires exactly one container handle per sample.")
            handle = handles[0]
            logger.warning(
                "swe_bridge_agent container_ready task_id=%s container_id=%s elapsed=%.1fs",
                task_context.task_id,
                handle.container_id,
                time.monotonic() - started_at,
            )
            executor = DockerToolExecutor(
                container_id=handle.container_id,
                tool_timeout_sec=self.tool_timeout_sec,
            )
            _set_stage("maybe_apply_patch")
            await _await_with_attempt_timeout(
                asyncio.to_thread(_maybe_apply_patch, executor, task_context.patch, self.tool_timeout_sec),
                task_id=task_context.task_id,
                stage=current_stage,
                started_at=started_at,
                attempt_timeout_sec=self.attempt_timeout_sec,
            )

            while len(response_mask) < self.response_length:
                if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                    loop_exit_reason = "max_assistant_turns_reached"
                    break
                if self.max_user_turns and user_turns >= self.max_user_turns:
                    loop_exit_reason = "max_user_turns_reached"
                    break
                if _remaining_attempt_timeout_sec(
                    started_at=started_at,
                    attempt_timeout_sec=self.attempt_timeout_sec,
                ) <= 0:
                    timeout_error = (
                        f"swe_bridge_agent timed out after {self.attempt_timeout_sec}s "
                        f"for task {task_context.task_id!r} in stage {current_stage!r}."
                    )
                    loop_exit_reason = "attempt_timeout"
                    break

                available_tokens = self.response_length - len(response_mask)
                if available_tokens <= 0:
                    loop_exit_reason = "response_length_budget_exhausted"
                    break

                _validate_rollout_context_alignment(
                    canonical_prompt_ids=canonical_prompt_ids,
                    full_token_ids=full_token_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs if response_logprobs else None,
                )

                turn_sampling_params = dict(sampling_params)
                requested_tokens_raw = turn_sampling_params.get(
                    "max_tokens",
                    turn_sampling_params.get("max_new_tokens", available_tokens),
                )
                try:
                    requested_tokens = int(requested_tokens_raw)
                except (TypeError, ValueError):
                    requested_tokens = available_tokens
                if requested_tokens < 1:
                    requested_tokens = available_tokens
                turn_sampling_params["max_tokens"] = max(1, min(available_tokens, requested_tokens))

                generate_started = time.monotonic()
                turn_idx = assistant_turns + 1
                _set_stage(f"generate_turn_{turn_idx}")
                generation_output = await _await_with_attempt_timeout(
                    self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=full_token_ids,
                        sampling_params=turn_sampling_params,
                        image_data=images,
                        video_data=videos,
                    ),
                    task_id=task_context.task_id,
                    stage=current_stage,
                    started_at=started_at,
                    attempt_timeout_sec=self.attempt_timeout_sec,
                )
                metrics["generate_sequences"] += time.monotonic() - generate_started

                generated_ids = _coerce_token_ids(getattr(generation_output, "token_ids", []))
                generated_logprobs = _coerce_logprobs(getattr(generation_output, "log_probs", None))
                if not generated_ids:
                    bridge_error = "Model returned empty token_ids in swe_bridge_agent generation."
                    loop_exit_reason = "empty_generation"
                    break

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
                _set_stage(f"decode_turn_{turn_idx}")
                assistant_text = await _await_with_attempt_timeout(
                    self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.decode(assistant_turn_ids, skip_special_tokens=True),
                    ),
                    task_id=task_context.task_id,
                    stage=current_stage,
                    started_at=started_at,
                    attempt_timeout_sec=self.attempt_timeout_sec,
                )
                assistant_turns += 1
                trajectory_assistant_turns.append(assistant_text)
                trajectory_assistant_turn_token_lengths.append(len(assistant_turn_ids))
                trajectory_turn_tool_response_blocks.append([])
                if reached_limit:
                    loop_exit_reason = "response_length_budget_exhausted"
                    break

                bridge_started = time.monotonic()
                try:
                    _set_stage(f"bridge_turn_{turn_idx}")
                    bridge_result = await _await_with_attempt_timeout(
                        asyncio.to_thread(
                            run_env_bridge_step,
                            assistant_text,
                            executor=executor,
                            max_tool_calls=self.max_tool_calls_per_turn,
                            step_index_start=bridge_step_index,
                        ),
                        task_id=task_context.task_id,
                        stage=current_stage,
                        started_at=started_at,
                        attempt_timeout_sec=self.attempt_timeout_sec,
                    )
                except TimeoutError as exc:
                    timeout_error = str(exc)
                    loop_exit_reason = "attempt_timeout"
                    logger.warning(
                        "swe_bridge_agent bridge timeout task_id=%s assistant_turn=%d stage=%s error=%s",
                        task_context.task_id,
                        assistant_turns,
                        current_stage,
                        exc,
                    )
                    break
                except Exception as exc:
                    bridge_error = f"swe_bridge_agent bridge failure: {exc}"
                    loop_exit_reason = "bridge_failure"
                    logger.warning(
                        "swe_bridge_agent bridge failure task_id=%s assistant_turn=%d error=%s assistant_excerpt=%s",
                        task_context.task_id,
                        assistant_turns,
                        exc,
                        _truncate_text(assistant_text, limit=240),
                    )
                    break
                metrics["tool_calls"] += time.monotonic() - bridge_started

                bridge_step_index += len(bridge_result.steps)
                serialized_steps = _serialize_environment_steps(bridge_result.steps)
                trajectory_steps.extend(serialized_steps)
                turn_validation_errors = _collect_validation_errors(bridge_result.steps)
                if turn_validation_errors:
                    validation_errors.extend(turn_validation_errors)
                if bridge_result.is_terminal:
                    final_turn_has_submit = True
                    final_submit_format_valid = not bool(turn_validation_errors)
                    final_response_text = _extract_final_submit_text(
                        bridge_result.envelope.tool_calls,
                        bridge_result.steps,
                    )
                    _set_stage("verify_submission")
                    verification_metadata = await _await_with_attempt_timeout(
                        asyncio.to_thread(
                            _verify_terminal_submission,
                            executor,
                            task_sample,
                            self.verifier_timeout_sec,
                            final_submit_format_valid,
                            final_response_text,
                        ),
                        task_id=task_context.task_id,
                        stage=current_stage,
                        started_at=started_at,
                        attempt_timeout_sec=self.attempt_timeout_sec,
                    )
                    if serialized_steps:
                        metadata = serialized_steps[-1].setdefault("metadata", {})
                        if isinstance(metadata, dict):
                            metadata.update(verification_metadata)

                if bridge_result.tool_response_blocks:
                    current_turn_tool_blocks = [str(block) for block in bridge_result.tool_response_blocks]
                    tool_response_blocks.extend(current_turn_tool_blocks)
                    trajectory_turn_tool_response_blocks[-1] = current_turn_tool_blocks
                    tool_messages = build_tool_response_messages(bridge_result.tool_response_blocks)
                    if tool_messages:
                        _set_stage(f"tokenize_tool_feedback_turn_{assistant_turns}")
                        tool_ids = await _await_with_attempt_timeout(
                            self.apply_chat_template(
                                tool_messages,
                                remove_system_prompt=True,
                            ),
                            task_id=task_context.task_id,
                            stage=current_stage,
                            started_at=started_at,
                            attempt_timeout_sec=self.attempt_timeout_sec,
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
                    loop_exit_reason = "terminal"
                    break
        except TimeoutError as exc:
            timeout_error = str(exc)
            loop_exit_reason = "attempt_timeout"
            logger.warning(
                "swe_bridge_agent timeout task_id=%s stage=%s elapsed=%.1fs error=%s",
                task_context.task_id,
                current_stage,
                time.monotonic() - started_at,
                exc,
            )
        except Exception as exc:
            bridge_error = f"swe_bridge_agent setup failure: {exc}"
            loop_exit_reason = "setup_failure"
            logger.exception(
                "swe_bridge_agent setup failure task_id=%s stage=%s error=%s",
                task_context.task_id,
                current_stage,
                exc,
            )
        finally:
            _set_stage("cleanup")
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(pool.release_all),
                    timeout=max(5, self.cleanup_timeout_sec + 5),
                )
            except Exception as exc:
                logger.warning(
                    "swe_bridge_agent cleanup issue task_id=%s stage=%s error=%s",
                    task_context.task_id,
                    current_stage,
                    exc,
                )
            if container_slot_acquired:
                try:
                    container_slot_gate.release()
                except ValueError:
                    logger.warning(
                        "swe_bridge_agent container slot release mismatch task_id=%s",
                        task_context.task_id,
                    )
            stage_stop_event.set()
            if heartbeat_task is not None:
                await heartbeat_task

        logger.warning(
            "swe_bridge_agent stop task_id=%s reason=%s assistant_turns=%d user_turns=%d tool_response_blocks=%d elapsed=%.1fs",
            task_context.task_id,
            loop_exit_reason,
            assistant_turns,
            user_turns,
            len(tool_response_blocks),
            time.monotonic() - started_at,
        )

        _validate_rollout_context_alignment(
            canonical_prompt_ids=canonical_prompt_ids,
            full_token_ids=full_token_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
        )
        prompt_ids = list(canonical_prompt_ids)
        response_ids = full_token_ids[len(canonical_prompt_ids) :]

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
            "trajectory_assistant_turns": trajectory_assistant_turns,
            "trajectory_assistant_turn_token_lengths": trajectory_assistant_turn_token_lengths,
            "trajectory_turn_tool_response_blocks": trajectory_turn_tool_response_blocks,
            "assistant_turns": assistant_turns,
            "user_turns": user_turns,
            "tool_response_block_count": len(tool_response_blocks),
            "loop_exit_reason": loop_exit_reason,
            "trajectory_tool_validation_errors": _stable_unique_strings(validation_errors),
            "final_turn_has_submit": final_turn_has_submit,
            "final_submit_format_valid": final_submit_format_valid,
            "bridge_error": bridge_error,
            "timeout_error": timeout_error,
            "executor_error": "",
        }
        extra_fields.update(
            _initial_verification_extra_fields(
                fail_to_pass=task_sample.fail_to_pass,
                pass_to_pass=task_sample.pass_to_pass,
            )
        )
        _apply_verification_metadata(extra_fields, verification_metadata)

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
    reward_ground_truth = _extract_reward_ground_truth_from_kwargs(raw_kwargs)
    return TaskSample(
        task_id=task_context.task_id,
        image_name=task_context.image_name,
        problem_statement=task_context.prompt_text,
        fail_to_pass=_resolve_task_sample_test_targets(
            raw_kwargs,
            reward_ground_truth=reward_ground_truth,
            key="fail_to_pass",
        ),
        pass_to_pass=_resolve_task_sample_test_targets(
            raw_kwargs,
            reward_ground_truth=reward_ground_truth,
            key="pass_to_pass",
        ),
        raw=dict(raw_kwargs),
    )


def _initial_verification_extra_fields(
    *,
    fail_to_pass: Sequence[Any] | None,
    pass_to_pass: Sequence[Any] | None,
) -> dict[str, Any]:
    return {
        "fail_to_pass": _coerce_test_targets(fail_to_pass),
        "pass_to_pass": _coerce_test_targets(pass_to_pass),
        "verification_feedback": "",
        "fail_to_pass_results": {},
        "pass_to_pass_results": {},
        "fail_to_pass_all_passed": None,
        "pass_to_pass_all_passed": None,
        "fail_to_pass_verified": None,
        "pass_to_pass_verified": None,
        "verification_missing": None,
        "verification_error": "",
        "submission_final_response": "",
        "resolved": False,
    }


def _apply_verification_metadata(
    extra_fields: dict[str, Any],
    verification_metadata: Mapping[str, Any] | None,
) -> None:
    if not verification_metadata:
        return
    extra_fields.update(
        {
            "fail_to_pass": _coerce_test_targets(
                verification_metadata.get("fail_to_pass", extra_fields.get("fail_to_pass", []))
            ),
            "pass_to_pass": _coerce_test_targets(
                verification_metadata.get("pass_to_pass", extra_fields.get("pass_to_pass", []))
            ),
            "verification_feedback": verification_metadata.get(
                "verification_feedback",
                extra_fields.get("verification_feedback", ""),
            ),
            "fail_to_pass_results": verification_metadata.get(
                "fail_to_pass_results",
                extra_fields.get("fail_to_pass_results", {}),
            ),
            "pass_to_pass_results": verification_metadata.get(
                "pass_to_pass_results",
                extra_fields.get("pass_to_pass_results", {}),
            ),
            "fail_to_pass_all_passed": verification_metadata.get(
                "fail_to_pass_all_passed",
                extra_fields.get("fail_to_pass_all_passed"),
            ),
            "pass_to_pass_all_passed": verification_metadata.get(
                "pass_to_pass_all_passed",
                extra_fields.get("pass_to_pass_all_passed"),
            ),
            "fail_to_pass_verified": verification_metadata.get(
                "fail_to_pass_verified",
                extra_fields.get("fail_to_pass_verified"),
            ),
            "pass_to_pass_verified": verification_metadata.get(
                "pass_to_pass_verified",
                extra_fields.get("pass_to_pass_verified"),
            ),
            "verification_missing": verification_metadata.get(
                "verification_missing",
                extra_fields.get("verification_missing"),
            ),
            "verification_error": verification_metadata.get(
                "verification_error",
                extra_fields.get("verification_error", ""),
            ),
            "submission_final_response": verification_metadata.get(
                "submission_final_response",
                extra_fields.get("submission_final_response", ""),
            ),
            "resolved": verification_metadata.get(
                "resolved",
                extra_fields.get("resolved", False),
            ),
        }
    )


def _verify_terminal_submission(
    executor: DockerToolExecutor,
    task_sample: TaskSample,
    verifier_timeout_sec: int,
    final_submit_format_valid: bool,
    final_response: str,
) -> dict[str, Any]:
    if not final_submit_format_valid:
        return {
            "submission_final_response": final_response,
            "fail_to_pass": _coerce_test_targets(task_sample.fail_to_pass),
            "pass_to_pass": _coerce_test_targets(task_sample.pass_to_pass),
            "fail_to_pass_results": {},
            "pass_to_pass_results": {},
            "fail_to_pass_verified": False,
            "pass_to_pass_verified": False,
            "verification_missing": False,
            "verification_error": "terminal submit failed tool-argument validation",
            "verification_feedback": "Verifier skipped: terminal submit format was invalid.",
            "resolved": False,
        }
    try:
        return run_submission_verifier(
            executor=executor,
            fail_to_pass=task_sample.fail_to_pass,
            pass_to_pass=task_sample.pass_to_pass,
            verifier_timeout_sec=verifier_timeout_sec,
            final_response=final_response,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        return {
            "submission_final_response": final_response,
            "fail_to_pass": _coerce_test_targets(task_sample.fail_to_pass),
            "pass_to_pass": _coerce_test_targets(task_sample.pass_to_pass),
            "fail_to_pass_results": {},
            "pass_to_pass_results": {},
            "fail_to_pass_verified": False,
            "pass_to_pass_verified": False,
            "verification_missing": False,
            "verification_error": f"terminal verifier execution failed: {exc}",
            "verification_feedback": "",
            "resolved": False,
        }


def _extract_final_submit_text(
    tool_calls: Sequence[Any],
    steps: Sequence[EnvironmentStep],
) -> str:
    if tool_calls:
        first_call = tool_calls[0]
        if getattr(first_call, "tool", "") == "submit":
            args = getattr(first_call, "args", {})
            if isinstance(args, Mapping):
                value = args.get("final_response")
                if isinstance(value, str):
                    return value
                if value is not None:
                    return str(value)
    for step in reversed(steps):
        if step.request.tool != "submit":
            continue
        value = step.request.args.get("final_response")
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return str(value)
    return ""


def _coerce_test_targets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        targets: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                targets.append(text)
        return targets
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _extract_reward_ground_truth_from_kwargs(raw_kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    reward_model = raw_kwargs.get("reward_model")
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, Mapping):
            return ground_truth
    return {}


def _resolve_task_sample_test_targets(
    raw_kwargs: Mapping[str, Any],
    *,
    reward_ground_truth: Mapping[str, Any],
    key: str,
) -> list[str]:
    for source in (raw_kwargs, reward_ground_truth):
        for candidate_key in (key, key.upper()):
            if candidate_key in source:
                return _coerce_test_targets(source.get(candidate_key))
    return []


def _get_container_slot_gate(env_pool_size: int) -> threading.BoundedSemaphore:
    with _CONTAINER_SLOT_LOCK:
        gate = _CONTAINER_SLOT_GATES.get(env_pool_size)
        if gate is None:
            gate = threading.BoundedSemaphore(value=env_pool_size)
            _CONTAINER_SLOT_GATES[env_pool_size] = gate
        return gate


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


def _as_role_text(value: Any) -> str:
    role = _as_content_text(value).strip().lower()
    if role not in {"system", "user", "assistant"}:
        return ""
    return role


def _as_content_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text_piece = item.get("text")
                if isinstance(text_piece, str) and text_piece.strip():
                    chunks.append(text_piece.strip())
            elif isinstance(item, str) and item.strip():
                chunks.append(item.strip())
        return "\n".join(chunks).strip()
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _truncate_text(value: str, *, limit: int) -> str:
    if limit < 4:
        return value[:limit]
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


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


def _clip_prompt_for_rollout_context(prompt_ids: Any, *, prompt_length: int) -> list[int]:
    if prompt_length < 1:
        raise ValueError("prompt_length must be >= 1.")
    normalized_prompt_ids = _coerce_token_ids(prompt_ids)
    if len(normalized_prompt_ids) <= prompt_length:
        return normalized_prompt_ids
    return normalized_prompt_ids[-prompt_length:]


def _validate_rollout_context_alignment(
    *,
    canonical_prompt_ids: Sequence[int],
    full_token_ids: Sequence[int],
    response_mask: Sequence[int],
    response_logprobs: Sequence[float] | None = None,
) -> None:
    prompt_len = len(canonical_prompt_ids)
    full_len = len(full_token_ids)
    if full_len < prompt_len:
        raise RuntimeError(
            "swe_bridge_agent context mismatch: full_token_ids shorter than canonical prompt "
            f"(full={full_len}, prompt={prompt_len})."
        )

    if list(full_token_ids[:prompt_len]) != list(canonical_prompt_ids):
        raise RuntimeError("swe_bridge_agent context mismatch: prompt prefix diverged from rollout context.")

    response_len = full_len - prompt_len
    if response_len != len(response_mask):
        raise RuntimeError(
            "swe_bridge_agent context mismatch: response length does not match response_mask "
            f"(response={response_len}, mask={len(response_mask)})."
        )

    if response_logprobs is not None and len(response_logprobs) != len(response_mask):
        raise RuntimeError(
            "swe_bridge_agent context mismatch: response_logprobs length does not match response_mask "
            f"(logprobs={len(response_logprobs)}, mask={len(response_mask)})."
        )


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
