#!/usr/bin/env python3
"""Run paired student-vs-teacher rollout pilot using turn-level teacher reprompts."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

try:
    from config import DEFAULT_ON_POLICY_DATA_CONFIG_NAME, resolve_on_policy_settings
    from rollout.onpolicy_collector import OnPolicyRolloutCollector
    from rollout.turn_parser import parse_assistant_turn_payload
    from rollout.vllm_turn_generator import (
        VLLMTurnGeneratorConfig,
        _extract_assistant_content,
        _post_chat_completion,
        build_vllm_turn_generator,
        load_vllm_turn_generator_config,
    )
    from verl_integration.reprompt_adapter import (
        DEFAULT_MAX_REPROMPT_LEN,
        DEFAULT_NUM_RECENT_RAW_BLOCKS,
        build_self_distillation_batch,
    )
    from verl_integration.reward_function import reward_fn
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from config import DEFAULT_ON_POLICY_DATA_CONFIG_NAME, resolve_on_policy_settings
    from rollout.onpolicy_collector import OnPolicyRolloutCollector
    from rollout.turn_parser import parse_assistant_turn_payload
    from rollout.vllm_turn_generator import (
        VLLMTurnGeneratorConfig,
        _extract_assistant_content,
        _post_chat_completion,
        build_vllm_turn_generator,
        load_vllm_turn_generator_config,
    )
    from verl_integration.reprompt_adapter import (
        DEFAULT_MAX_REPROMPT_LEN,
        DEFAULT_NUM_RECENT_RAW_BLOCKS,
        build_self_distillation_batch,
    )
    from verl_integration.reward_function import reward_fn

_TOOL_RESPONSE_START = "<tool_response>"
_TOOL_RESPONSE_END = "</tool_response>"
_TURN_SUPERVISION_CURRENT = "current_turn"
_TURN_INDEX_MODE_FIXED = "fixed"
_TURN_INDEX_MODE_DYNAMIC_MIDDLE = "dynamic_middle"
_TEACHER_REPROMPT_TURN_INDEX_MODES = (
    _TURN_INDEX_MODE_FIXED,
    _TURN_INDEX_MODE_DYNAMIC_MIDDLE,
)
_RFT_MANIFEST_NAME = "rft_runtime_loop_manifest.json"
_RFT_MANIFEST_GLOBS = (
    "outputs/rft_runtime/*/rft_runtime_loop_manifest.json",
    "outputs/slurm/rft_runtime/*/rft_runtime_loop_manifest.json",
)


@dataclass(frozen=True)
class BaselineTrace:
    task_id: str
    attempt_index: int
    problem_statement: str
    raw_prompt_messages: tuple[dict[str, str], ...]
    assistant_turns: tuple[str, ...]
    turn_tool_response_blocks: tuple[tuple[str, ...], ...]
    verification_feedback: str
    verification_error: str
    resolved: bool
    trajectory_steps: tuple[Mapping[str, Any], ...] = ()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=1024)
    parser.add_argument("--attempts-per-task", type=int, default=8)
    parser.add_argument("--max-turns-per-attempt", type=int, default=16)
    parser.add_argument("--max-in-flight-tasks", type=int, default=32)
    parser.add_argument("--data-config-name", default=DEFAULT_ON_POLICY_DATA_CONFIG_NAME)
    parser.add_argument("--teacher-reprompt-turn-index", type=int, default=-1)
    parser.add_argument(
        "--teacher-reprompt-turn-index-mode",
        default=_TURN_INDEX_MODE_DYNAMIC_MIDDLE,
        choices=sorted(_TEACHER_REPROMPT_TURN_INDEX_MODES),
    )
    parser.add_argument(
        "--max-reprompt-len",
        type=int,
        default=DEFAULT_MAX_REPROMPT_LEN,
    )
    parser.add_argument(
        "--num-recent-raw-blocks",
        type=int,
        default=DEFAULT_NUM_RECENT_RAW_BLOCKS,
    )
    parser.add_argument("--turn-supervision-mode", default=_TURN_SUPERVISION_CURRENT)
    parser.add_argument("--verifier-feedback-mode", default="all_turns")
    parser.add_argument(
        "--no-verify-submissions",
        dest="verify_submissions",
        action="store_false",
        help="Disable submission verification in pilot rollouts.",
    )
    parser.add_argument(
        "--verifier-timeout-sec",
        type=int,
        default=600,
        help="Timeout (sec) for submission verification.",
    )
    parser.add_argument(
        "--rft-checkpoint",
        default="",
        help="Optional model/checkpoint override for vLLM requests.",
    )
    parser.add_argument(
        "--rft-manifest",
        type=Path,
        default=None,
        help=f"Explicit path to an {_RFT_MANIFEST_NAME} file.",
    )
    parser.add_argument(
        "--load-latest-rft-checkpoint",
        action="store_true",
        help=(
            "Resolve model/checkpoint from the newest rft manifest under "
            "outputs/slurm/rft_runtime (or outputs/rft_runtime)."
        ),
    )
    parser.add_argument(
        "--print-resolved-rft-checkpoint",
        action="store_true",
        help=(
            "Print the resolved RFT checkpoint path/model and exit. "
            "Returns an empty line when no RFT override flag is set."
        ),
    )
    return parser


def _is_tool_response_block(value: str) -> bool:
    text = value.strip()
    return text.startswith(_TOOL_RESPONSE_START) and text.endswith(_TOOL_RESPONSE_END)


def derive_turn_tool_response_blocks(
    *,
    history: Sequence[Any],
    assistant_turns: Sequence[str],
) -> list[list[str]]:
    # Collector history is [assistant_turn, tool_response*, assistant_turn, ...].
    tool_blocks: list[list[str]] = [[] for _ in assistant_turns]
    if not history or not assistant_turns:
        return tool_blocks

    turn_index = -1
    for raw_item in history:
        text = str(raw_item)
        if _is_tool_response_block(text):
            if 0 <= turn_index < len(tool_blocks):
                tool_blocks[turn_index].append(text)
            continue
        turn_index += 1
        if turn_index >= len(tool_blocks):
            break
    return tool_blocks


def derive_assistant_turns_from_history(*, history: Sequence[Any]) -> list[str]:
    assistant_turns: list[str] = []
    for raw_item in history:
        text = str(raw_item)
        if _is_tool_response_block(text):
            continue
        if text.strip():
            assistant_turns.append(text)
    return assistant_turns


def _coerce_text_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _coerce_message_list(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        rows.append({"role": role, "content": content})
    return rows


def _coerce_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        rows.append(dict(item))
    return rows


def _assistant_turn_has_terminal_submit(turn_text: str) -> bool:
    text = str(turn_text).strip()
    if not text:
        return False
    try:
        envelope = parse_assistant_turn_payload(text)
    except Exception:
        return False
    return any(getattr(tool_call, "tool", "") == "submit" for tool_call in envelope.tool_calls)


def _resolve_temperature_override(*, env_var_name: str, fallback: float) -> float:
    raw_value = os.environ.get(env_var_name)
    if raw_value is None:
        return float(fallback)
    normalized = str(raw_value).strip()
    if not normalized:
        return float(fallback)
    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"{env_var_name} must be a float if set; received {raw_value!r}.") from exc


def _resolve_pilot_vllm_configs(
    *,
    base_config: VLLMTurnGeneratorConfig,
) -> tuple[VLLMTurnGeneratorConfig, VLLMTurnGeneratorConfig]:
    student_temperature = _resolve_temperature_override(
        env_var_name="PILOT_STUDENT_TEMPERATURE",
        fallback=base_config.temperature,
    )
    teacher_temperature = _resolve_temperature_override(
        env_var_name="PILOT_TEACHER_TEMPERATURE",
        fallback=base_config.temperature,
    )
    return (
        replace(base_config, temperature=student_temperature),
        replace(base_config, temperature=teacher_temperature),
    )


def _build_teacher_request_messages(
    *,
    trace: BaselineTrace,
    teacher_prompt: str,
    vllm_config: VLLMTurnGeneratorConfig,
) -> list[dict[str, str]]:
    # Match runtime distillation chat formatting: prior context messages plus
    # reprompt as the final user message.
    prefix_messages = list(trace.raw_prompt_messages[:-1]) if trace.raw_prompt_messages else []
    messages: list[dict[str, str]] = []
    for message in prefix_messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    if not messages:
        system_prompt = str(getattr(vllm_config, "system_prompt", "")).strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": teacher_prompt})
    return messages


def _build_baseline_trace_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], BaselineTrace]:
    trace_map: dict[tuple[str, int], BaselineTrace] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        attempt_index = int(row.get("attempt_index", 0) or 0)
        assistant_turns = _coerce_text_list(row.get("trajectory_assistant_turns"))
        history = _coerce_text_list(row.get("trajectory_history"))
        tool_blocks = derive_turn_tool_response_blocks(history=history, assistant_turns=assistant_turns)
        trace_map[(task_id, attempt_index)] = BaselineTrace(
            task_id=task_id,
            attempt_index=attempt_index,
            problem_statement=str(row.get("prompt", "")),
            raw_prompt_messages=tuple(_coerce_message_list(row.get("_raw_prompt_messages"))),
            assistant_turns=tuple(assistant_turns),
            turn_tool_response_blocks=tuple(tuple(items) for items in tool_blocks),
            verification_feedback=str(row.get("verification_feedback", "")),
            verification_error=str(row.get("verification_error", "")),
            resolved=bool(row.get("resolved", False)),
            trajectory_steps=tuple(_coerce_mapping_list(row.get("trajectory_steps"))),
        )
    return trace_map


def _resolve_teacher_reprompt_turn_index(
    *,
    trace: BaselineTrace,
    teacher_reprompt_turn_index: int,
    teacher_reprompt_turn_index_mode: str,
    resolved_turn_index_cache: dict[tuple[str, int], int] | None = None,
) -> int:
    mode = str(teacher_reprompt_turn_index_mode).strip().lower()
    if mode == _TURN_INDEX_MODE_DYNAMIC_MIDDLE:
        cache_key = (trace.task_id, trace.attempt_index)
        if resolved_turn_index_cache is not None and cache_key in resolved_turn_index_cache:
            return resolved_turn_index_cache[cache_key]
        eligible_turn_indices = [
            index
            for index, tool_blocks in enumerate(trace.turn_tool_response_blocks)
            if index < len(trace.assistant_turns)
            and trace.assistant_turns[index].strip()
            and tool_blocks
            and not _assistant_turn_has_terminal_submit(trace.assistant_turns[index])
        ]
        injectable_turn_indices = [
            index
            for index, assistant_turn in enumerate(trace.assistant_turns)
            if assistant_turn.strip() and not _assistant_turn_has_terminal_submit(assistant_turn)
        ]
        turn_count = len(trace.assistant_turns)
        if eligible_turn_indices:
            resolved_turn_index = random.choice(eligible_turn_indices)
        elif injectable_turn_indices:
            resolved_turn_index = injectable_turn_indices[(len(injectable_turn_indices) - 1) // 2]
        elif turn_count <= 0:
            resolved_turn_index = max(0, int(teacher_reprompt_turn_index))
        else:
            resolved_turn_index = (turn_count - 1) // 2
        if resolved_turn_index_cache is not None:
            resolved_turn_index_cache[cache_key] = resolved_turn_index
        return resolved_turn_index
    return max(0, int(teacher_reprompt_turn_index))


def _build_teacher_turn_generator(
    *,
    baseline_trace_map: Mapping[tuple[str, int], BaselineTrace],
    fallback_turn_generator: Any,
    vllm_config: VLLMTurnGeneratorConfig,
    teacher_reprompt_turn_index: int,
    teacher_reprompt_turn_index_mode: str,
    max_reprompt_len: int,
    num_recent_raw_blocks: int,
    turn_supervision_mode: str,
    verifier_feedback_mode: str,
):
    normalized_turn_supervision_mode = str(turn_supervision_mode).strip().lower()
    if normalized_turn_supervision_mode != _TURN_SUPERVISION_CURRENT:
        raise ValueError(
            "Teacher-reprompt pilot only supports --turn-supervision-mode=current_turn."
        )
    resolved_turn_index_cache: dict[tuple[str, int], int] = {}

    def _generate(
        *,
        task: Any,
        attempt_index: int,
        turn_index: int,
        step_index: int,
        history: Sequence[str],
    ) -> str:
        trace = baseline_trace_map.get((str(task.task_id), int(attempt_index)))
        if trace is None:
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )

        resolved_turn_index = _resolve_teacher_reprompt_turn_index(
            trace=trace,
            teacher_reprompt_turn_index=teacher_reprompt_turn_index,
            teacher_reprompt_turn_index_mode=teacher_reprompt_turn_index_mode,
            resolved_turn_index_cache=resolved_turn_index_cache,
        )
        turn_count = len(trace.assistant_turns)
        if turn_count <= 0:
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )
        if resolved_turn_index < 0 or resolved_turn_index >= turn_count:
            resolved_turn_index = 0
        injection_turn_index = resolved_turn_index + 1

        if turn_index <= resolved_turn_index:
            if turn_index < len(trace.assistant_turns):
                return trace.assistant_turns[turn_index]
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )

        if turn_index != injection_turn_index:
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )

        history_assistant_turns = derive_assistant_turns_from_history(history=history)
        expected_replayed_turns = resolved_turn_index + 1
        if len(history_assistant_turns) != expected_replayed_turns:
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )
        history_turn_tool_blocks = derive_turn_tool_response_blocks(
            history=history,
            assistant_turns=history_assistant_turns,
        )

        sample = {
            "prompt": trace.problem_statement,
            "_raw_prompt_messages": [dict(message) for message in trace.raw_prompt_messages],
            "trajectory_assistant_turns": list(history_assistant_turns),
            "trajectory_turn_tool_response_blocks": [list(items) for items in history_turn_tool_blocks],
            "trajectory_steps": [dict(step) for step in trace.trajectory_steps],
            "verification_feedback": trace.verification_feedback,
            "verification_error": trace.verification_error,
            "resolved": trace.resolved,
        }
        reprompt_batch = build_self_distillation_batch(
            [sample],
            include_student_attempt_for_teacher=True,
            max_reprompt_len=max_reprompt_len,
            num_recent_raw_blocks=num_recent_raw_blocks,
            turn_supervision_mode=turn_supervision_mode,
            verifier_feedback_mode=verifier_feedback_mode,
            legacy_distillation_gating_policy="feedback_present",
        )
        prompts = reprompt_batch.get("turn_teacher_prompts", [[]])
        prompts_for_row = prompts[0] if prompts else []
        if not prompts_for_row:
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )

        prompt_index = int(resolved_turn_index)
        if prompt_index < 0 or prompt_index >= len(prompts_for_row):
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )
        teacher_prompt = str(prompts_for_row[prompt_index])
        try:
            completion_payload = _post_chat_completion(
                base_url=vllm_config.base_url,
                payload={
                    "model": vllm_config.model_name,
                    "messages": _build_teacher_request_messages(
                        trace=trace,
                        teacher_prompt=teacher_prompt,
                        vllm_config=vllm_config,
                    ),
                    "temperature": vllm_config.temperature,
                    "top_p": vllm_config.top_p,
                    "max_tokens": vllm_config.max_tokens,
                },
                timeout_sec=vllm_config.request_timeout_sec,
            )
            teacher_turn = _extract_assistant_content(completion_payload).strip()
        except Exception:
            teacher_turn = ""
        if teacher_turn:
            return teacher_turn
        return fallback_turn_generator(
            task=task,
            attempt_index=attempt_index,
            turn_index=turn_index,
            step_index=step_index,
            history=history,
        )

    return _generate


def summarize_pair_rewards(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(item["reward_delta"]) for item in pairs]
    student_rewards = [float(item["student_reward"]) for item in pairs]
    teacher_rewards = [float(item["teacher_reward"]) for item in pairs]
    improved = sum(1 for delta in deltas if delta > 0)
    worsened = sum(1 for delta in deltas if delta < 0)
    tied = sum(1 for delta in deltas if delta == 0)

    def _safe_mean(values: Sequence[float]) -> float:
        return float(mean(values)) if values else 0.0

    return {
        "pair_count": len(pairs),
        "student_mean_reward": _safe_mean(student_rewards),
        "teacher_mean_reward": _safe_mean(teacher_rewards),
        "mean_reward_delta": _safe_mean(deltas),
        "improved_count": improved,
        "worsened_count": worsened,
        "tied_count": tied,
    }


def extract_format_metrics(*, reward_info: Mapping[str, Any]) -> dict[str, float]:
    raw_blocks = reward_info.get("format_metrics")
    if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, (str, bytes)) or not raw_blocks:
        raise ValueError("reward_fn info must include format_metrics[0].")
    metrics_block = raw_blocks[0]
    if not isinstance(metrics_block, Mapping):
        raise ValueError("reward_fn info format_metrics[0] must be a mapping.")

    metrics: dict[str, float] = {}
    for key, value in metrics_block.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[str(key)] = float(value)
    return metrics


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _discover_latest_rft_manifest(*, project_root: Path) -> Path | None:
    latest_manifest: Path | None = None
    for pattern in _RFT_MANIFEST_GLOBS:
        for candidate in project_root.glob(pattern):
            if not candidate.is_file():
                continue
            if latest_manifest is None:
                latest_manifest = candidate
                continue
            try:
                if candidate.stat().st_mtime > latest_manifest.stat().st_mtime:
                    latest_manifest = candidate
            except OSError:
                continue
    return latest_manifest


def _manifest_checkpoint_candidates(payload: Mapping[str, Any]) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []

    def _append_candidate(value: Any) -> None:
        if not isinstance(value, str):
            return
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

        path = Path(normalized)
        sibling_name: str | None = None
        if path.name == "huggingface_vllm_merged":
            sibling_name = "huggingface"
        elif path.name == "huggingface":
            sibling_name = "huggingface_vllm_merged"
        if sibling_name is None:
            return
        sibling = str(path.with_name(sibling_name))
        if sibling not in seen:
            seen.add(sibling)
            candidates.append(sibling)

    for key in ("final_model_path", "latest_vllm_checkpoint", "latest_hf_checkpoint"):
        _append_candidate(payload.get(key))

    steps = payload.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        for raw_step in reversed(steps):
            if not isinstance(raw_step, Mapping):
                continue
            for key in ("latest_vllm_checkpoint", "latest_hf_checkpoint"):
                _append_candidate(raw_step.get(key))
    return candidates


def _resolve_checkpoint_from_manifest(*, manifest_path: Path) -> str:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid RFT manifest payload (expected mapping): {manifest_path}")
    candidates = _manifest_checkpoint_candidates(payload)
    if not candidates:
        raise ValueError(f"No checkpoint candidates found in RFT manifest: {manifest_path}")
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def _resolve_rft_checkpoint_override(
    *,
    explicit_checkpoint: str,
    explicit_manifest: Path | None,
    load_latest_rft_checkpoint: bool,
    project_root: Path,
) -> tuple[str | None, Path | None]:
    checkpoint_override = str(explicit_checkpoint).strip()
    if checkpoint_override:
        return checkpoint_override, None

    manifest_path: Path | None = None
    if explicit_manifest is not None:
        manifest_path = explicit_manifest.expanduser()
    elif load_latest_rft_checkpoint:
        manifest_path = _discover_latest_rft_manifest(project_root=project_root)
        if manifest_path is None:
            raise FileNotFoundError(
                "Unable to discover latest RFT manifest under outputs/slurm/rft_runtime "
                "or outputs/rft_runtime."
            )

    if manifest_path is None:
        return None, None
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RFT manifest does not exist: {manifest_path}")
    checkpoint = _resolve_checkpoint_from_manifest(manifest_path=manifest_path)
    return checkpoint, manifest_path


def main() -> None:
    args = _build_parser().parse_args()
    resolved_rft_checkpoint, resolved_rft_manifest = _resolve_rft_checkpoint_override(
        explicit_checkpoint=str(args.rft_checkpoint),
        explicit_manifest=args.rft_manifest,
        load_latest_rft_checkpoint=bool(args.load_latest_rft_checkpoint),
        project_root=Path(__file__).resolve().parents[1],
    )
    if bool(args.print_resolved_rft_checkpoint):
        print(str(resolved_rft_checkpoint or ""))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --print-resolved-rft-checkpoint is set.")

    if args.step_index < 0:
        raise ValueError("--step-index must be >= 0.")
    turn_index_mode = str(args.teacher_reprompt_turn_index_mode).strip().lower()
    turn_supervision_mode = str(args.turn_supervision_mode).strip().lower()
    if turn_supervision_mode != _TURN_SUPERVISION_CURRENT:
        raise ValueError("Teacher-reprompt pilot only supports --turn-supervision-mode=current_turn.")
    if turn_index_mode == _TURN_INDEX_MODE_DYNAMIC_MIDDLE:
        if int(args.teacher_reprompt_turn_index) != -1:
            raise ValueError("--teacher-reprompt-turn-index must be -1 when using dynamic_middle mode.")
    elif args.teacher_reprompt_turn_index < 0:
        raise ValueError("--teacher-reprompt-turn-index must be >= 0 when using fixed mode.")

    runtime_overrides = {
        "task_batch_size": int(args.task_batch_size),
        "attempts_per_task": int(args.attempts_per_task),
        "max_turns_per_attempt": int(args.max_turns_per_attempt),
        "max_in_flight_tasks": int(args.max_in_flight_tasks),
        "verify_submissions": bool(args.verify_submissions),
        "verifier_timeout_sec": int(args.verifier_timeout_sec),
    }
    settings = resolve_on_policy_settings(
        data_config_name=str(args.data_config_name),
        runtime_overrides=runtime_overrides,
    )

    vllm_config = load_vllm_turn_generator_config()
    if resolved_rft_checkpoint is not None:
        vllm_config = replace(vllm_config, model_name=resolved_rft_checkpoint)
    student_vllm_config, teacher_vllm_config = _resolve_pilot_vllm_configs(base_config=vllm_config)
    student_turn_generator = build_vllm_turn_generator(student_vllm_config)

    baseline_collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=student_turn_generator,
    )
    baseline_rows = [dict(row) for row in baseline_collector.collect_step(int(args.step_index))]
    baseline_trace_map = _build_baseline_trace_map(baseline_rows)

    teacher_turn_generator = _build_teacher_turn_generator(
        baseline_trace_map=baseline_trace_map,
        fallback_turn_generator=student_turn_generator,
        vllm_config=teacher_vllm_config,
        teacher_reprompt_turn_index=int(args.teacher_reprompt_turn_index),
        teacher_reprompt_turn_index_mode=turn_index_mode,
        max_reprompt_len=int(args.max_reprompt_len),
        num_recent_raw_blocks=int(args.num_recent_raw_blocks),
        turn_supervision_mode=turn_supervision_mode,
        verifier_feedback_mode=str(args.verifier_feedback_mode),
    )
    teacher_collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=teacher_turn_generator,
    )
    teacher_rows = [dict(row) for row in teacher_collector.collect_step(int(args.step_index))]

    teacher_map: dict[tuple[str, int], dict[str, Any]] = {}
    for row in teacher_rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        attempt_index = int(row.get("attempt_index", 0) or 0)
        teacher_map[(task_id, attempt_index)] = row

    baseline_rewards, _baseline_info = reward_fn(
        baseline_rows,
        max_tool_calls=settings.runtime.max_tool_calls_per_turn,
    )
    teacher_rewards, _teacher_info = reward_fn(
        teacher_rows,
        max_tool_calls=settings.runtime.max_tool_calls_per_turn,
    )
    baseline_format_metrics = extract_format_metrics(reward_info=_baseline_info)
    teacher_format_metrics = extract_format_metrics(reward_info=_teacher_info)
    baseline_reward_map = {
        (str(row.get("task_id", "")).strip(), int(row.get("attempt_index", 0) or 0)): float(
            baseline_rewards[index]
        )
        for index, row in enumerate(baseline_rows)
    }
    teacher_reward_map = {
        (str(row.get("task_id", "")).strip(), int(row.get("attempt_index", 0) or 0)): float(
            teacher_rewards[index]
        )
        for index, row in enumerate(teacher_rows)
    }

    pairs: list[dict[str, Any]] = []
    for baseline_row in baseline_rows:
        task_id = str(baseline_row.get("task_id", "")).strip()
        if not task_id:
            continue
        attempt_index = int(baseline_row.get("attempt_index", 0) or 0)
        key = (task_id, attempt_index)
        teacher_row = teacher_map.get(key)
        student_reward = float(baseline_reward_map.get(key, 0.0))
        teacher_reward = float(teacher_reward_map.get(key, 0.0))
        pairs.append(
            {
                "task_id": task_id,
                "attempt_index": attempt_index,
                "student_reward": student_reward,
                "teacher_reward": teacher_reward,
                "reward_delta": teacher_reward - student_reward,
                "teacher_row_missing": teacher_row is None,
            }
        )

    summary = {
        "data_config_name": str(args.data_config_name),
        "step_index": int(args.step_index),
        "task_batch_size": int(args.task_batch_size),
        "attempts_per_task": int(args.attempts_per_task),
        "teacher_reprompt_turn_index": int(args.teacher_reprompt_turn_index),
        "teacher_reprompt_turn_index_mode": turn_index_mode,
        "turn_supervision_mode": str(args.turn_supervision_mode),
        "verifier_feedback_mode": str(args.verifier_feedback_mode),
        "verify_submissions": bool(args.verify_submissions),
        "verifier_timeout_sec": int(args.verifier_timeout_sec),
        "max_reprompt_len": int(args.max_reprompt_len),
        "num_recent_raw_blocks": int(args.num_recent_raw_blocks),
        "vllm_model_name": str(vllm_config.model_name),
        "student_temperature": float(student_vllm_config.temperature),
        "teacher_temperature": float(teacher_vllm_config.temperature),
        "rft_checkpoint_override": resolved_rft_checkpoint,
        "rft_manifest_path": str(resolved_rft_manifest) if resolved_rft_manifest is not None else None,
        "baseline_row_count": len(baseline_rows),
        "teacher_row_count": len(teacher_rows),
        "missing_teacher_rows": int(sum(1 for item in pairs if item["teacher_row_missing"])),
        "baseline_format_metrics": baseline_format_metrics,
        "teacher_format_metrics": teacher_format_metrics,
        "reward_summary": summarize_pair_rewards(pairs),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "baseline_rollout_rows.jsonl", baseline_rows)
    _write_jsonl(output_dir / "teacher_rollout_rows.jsonl", teacher_rows)
    _write_jsonl(output_dir / "pair_rewards.jsonl", pairs)
    _write_json(output_dir / "pilot_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
