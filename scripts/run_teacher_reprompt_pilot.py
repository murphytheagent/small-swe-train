#!/usr/bin/env python3
"""Run paired student-vs-teacher rollout pilot using turn-level teacher reprompts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

try:
    from config import DEFAULT_ON_POLICY_DATA_CONFIG_NAME, resolve_on_policy_settings
    from rollout.onpolicy_collector import OnPolicyRolloutCollector
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
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from config import DEFAULT_ON_POLICY_DATA_CONFIG_NAME, resolve_on_policy_settings
    from rollout.onpolicy_collector import OnPolicyRolloutCollector
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

_TOOL_RESPONSE_START = "<tool_response>"
_TOOL_RESPONSE_END = "</tool_response>"
_TURN_SUPERVISION_CURRENT = "current_turn"
_TURN_INDEX_MODE_FIXED = "fixed"
_TURN_INDEX_MODE_DYNAMIC_MIDDLE = "dynamic_middle"
_TEACHER_REPROMPT_TURN_INDEX_MODES = (
    _TURN_INDEX_MODE_FIXED,
    _TURN_INDEX_MODE_DYNAMIC_MIDDLE,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=128)
    parser.add_argument("--attempts-per-task", type=int, default=8)
    parser.add_argument("--max-turns-per-attempt", type=int, default=16)
    parser.add_argument("--max-in-flight-tasks", type=int, default=32)
    parser.add_argument("--data-config-name", default=DEFAULT_ON_POLICY_DATA_CONFIG_NAME)
    parser.add_argument("--teacher-reprompt-turn-index", type=int, default=1)
    parser.add_argument(
        "--teacher-reprompt-turn-index-mode",
        default=_TURN_INDEX_MODE_FIXED,
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
        )
    return trace_map


def _resolve_teacher_reprompt_turn_index(
    *,
    trace: BaselineTrace,
    teacher_reprompt_turn_index: int,
    teacher_reprompt_turn_index_mode: str,
) -> int:
    mode = str(teacher_reprompt_turn_index_mode).strip().lower()
    if mode == _TURN_INDEX_MODE_DYNAMIC_MIDDLE:
        turn_count = len(trace.assistant_turns)
        if turn_count <= 0:
            return max(0, int(teacher_reprompt_turn_index))
        return turn_count // 2
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
        )
        if turn_index < resolved_turn_index:
            if turn_index < len(trace.assistant_turns):
                return trace.assistant_turns[turn_index]
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )

        # Inject the teacher reprompt once at the configured turn, then let the
        # rollout proceed with the normal turn generator on the updated history.
        if turn_index > resolved_turn_index:
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )

        if turn_index >= len(trace.assistant_turns):
            return fallback_turn_generator(
                task=task,
                attempt_index=attempt_index,
                turn_index=turn_index,
                step_index=step_index,
                history=history,
            )

        sample = {
            "prompt": trace.problem_statement,
            "_raw_prompt_messages": [dict(message) for message in trace.raw_prompt_messages],
            "trajectory_assistant_turns": list(trace.assistant_turns[: turn_index + 1]),
            "trajectory_turn_tool_response_blocks": [
                list(items) for items in trace.turn_tool_response_blocks[: turn_index + 1]
            ],
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

        prompt_index = int(turn_index)
        if str(turn_supervision_mode).strip().lower() == "next_turn":
            prompt_index = prompt_index - 1
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def main() -> None:
    args = _build_parser().parse_args()
    if args.step_index < 0:
        raise ValueError("--step-index must be >= 0.")
    turn_index_mode = str(args.teacher_reprompt_turn_index_mode).strip().lower()
    if args.teacher_reprompt_turn_index < 0 and turn_index_mode != _TURN_INDEX_MODE_DYNAMIC_MIDDLE:
        raise ValueError("--teacher-reprompt-turn-index must be >= 0.")
    if args.teacher_reprompt_turn_index < 0 and int(args.teacher_reprompt_turn_index) != -1:
        raise ValueError("--teacher-reprompt-turn-index must be -1 when using dynamic_middle mode.")

    runtime_overrides = {
        "task_batch_size": int(args.task_batch_size),
        "attempts_per_task": int(args.attempts_per_task),
        "max_turns_per_attempt": int(args.max_turns_per_attempt),
        "max_in_flight_tasks": int(args.max_in_flight_tasks),
    }
    settings = resolve_on_policy_settings(
        data_config_name=str(args.data_config_name),
        runtime_overrides=runtime_overrides,
    )

    vllm_config = load_vllm_turn_generator_config()
    student_turn_generator = build_vllm_turn_generator(vllm_config)

    baseline_collector = OnPolicyRolloutCollector(
        settings=settings,
        turn_generator=student_turn_generator,
    )
    baseline_rows = [dict(row) for row in baseline_collector.collect_step(int(args.step_index))]
    baseline_trace_map = _build_baseline_trace_map(baseline_rows)

    teacher_turn_generator = _build_teacher_turn_generator(
        baseline_trace_map=baseline_trace_map,
        fallback_turn_generator=student_turn_generator,
        vllm_config=vllm_config,
        teacher_reprompt_turn_index=int(args.teacher_reprompt_turn_index),
        teacher_reprompt_turn_index_mode=turn_index_mode,
        max_reprompt_len=int(args.max_reprompt_len),
        num_recent_raw_blocks=int(args.num_recent_raw_blocks),
        turn_supervision_mode=str(args.turn_supervision_mode),
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

    pairs: list[dict[str, Any]] = []
    for baseline_row in baseline_rows:
        task_id = str(baseline_row.get("task_id", "")).strip()
        if not task_id:
            continue
        attempt_index = int(baseline_row.get("attempt_index", 0) or 0)
        key = (task_id, attempt_index)
        teacher_row = teacher_map.get(key)
        student_reward = 1.0 if bool(baseline_row.get("resolved", False)) else 0.0
        teacher_reward = 1.0 if bool(teacher_row and teacher_row.get("resolved", False)) else 0.0
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
        "max_reprompt_len": int(args.max_reprompt_len),
        "num_recent_raw_blocks": int(args.num_recent_raw_blocks),
        "baseline_row_count": len(baseline_rows),
        "teacher_row_count": len(teacher_rows),
        "missing_teacher_rows": int(sum(1 for item in pairs if item["teacher_row_missing"])),
        "reward_summary": summarize_pair_rewards(pairs),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "baseline_rollout_rows.jsonl", baseline_rows)
    _write_jsonl(args.output_dir / "teacher_rollout_rows.jsonl", teacher_rows)
    _write_jsonl(args.output_dir / "pair_rewards.jsonl", pairs)
    _write_json(args.output_dir / "pilot_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
