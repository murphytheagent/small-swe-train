"""End-to-end ingestion pipeline for SWE-style tool trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, cast

from data.feedback_canonicalizer import build_feedback_packet
from data.tool_schema_adapter import adapt_external_tool_call
from env import EnvironmentStep, ToolRequest, ToolResponse
from losses.action_masking import TokenLabel, build_action_token_mask
from schemas import FeedbackPacket

_TRAJECTORY_KEYS = ("trajectory", "steps")
_TOOL_NAME_KEYS = ("tool", "tool_name", "name")
_TOOL_ARGS_KEYS = ("args", "arguments", "tool_args", "tool_input", "input", "kwargs")
_TOOL_OUTPUT_KEYS = (
    "tool_output",
    "output",
    "observation",
    "response",
    "environment_output",
    "result",
)
_TOOL_OUTPUT_LIST_KEYS = ("tool_outputs", "outputs", "observations", "responses")
_THINKING_KEYS = ("thinking", "thought", "analysis", "reasoning")


@dataclass(frozen=True)
class Episode:
    episode_id: str
    source_format: str
    environment_steps: tuple[EnvironmentStep, ...]
    feedback_packets: tuple[FeedbackPacket, ...]


@dataclass(frozen=True)
class _StepEntry:
    tool_name: str
    args: dict[str, Any]
    tool_output: dict[str, Any]
    thinking: str | None


class SupportsOffsetsTokenizer(Protocol):
    """Minimal tokenizer protocol used by ingestion helpers."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Mapping[str, Any]:
        ...


def load_raw_records(input_path: Path) -> list[dict[str, Any]]:
    """Load trajectory records from JSON/JSONL file or directory."""
    if input_path.is_dir():
        records: list[dict[str, Any]] = []
        for path in sorted(input_path.rglob("*.json")):
            records.extend(_load_json_file(path))
        for path in sorted(input_path.rglob("*.jsonl")):
            records.extend(_load_jsonl_file(path))
        if not records:
            raise ValueError(f"No .json/.jsonl records found under {input_path}")
        return records

    suffix = input_path.suffix.lower()
    if suffix == ".json":
        return _load_json_file(input_path)
    if suffix == ".jsonl":
        return _load_jsonl_file(input_path)
    raise ValueError(f"Unsupported input file type: {input_path}")


def build_episodes(raw_records: Sequence[Mapping[str, Any]]) -> list[Episode]:
    """Convert raw records into canonical episodes."""
    return [
        build_episode_from_record(record, fallback_index=index)
        for index, record in enumerate(raw_records)
    ]


def build_episode_from_record(record: Mapping[str, Any], *, fallback_index: int) -> Episode:
    """Build one canonical episode from a raw trajectory record."""
    source_format = _infer_source_format(record)
    episode_id = _extract_episode_id(record, fallback_index=fallback_index)
    entries = list(_iter_step_entries(record))
    if not entries:
        raise ValueError(f"Record {episode_id!r} has no tool-call steps to ingest.")

    environment_steps: list[EnvironmentStep] = []
    feedback_packets: list[FeedbackPacket] = []
    for step_index, entry in enumerate(entries):
        canonical_call = adapt_external_tool_call(entry.tool_name, entry.args)
        response = _tool_response_from_output(entry.tool_output)
        step = EnvironmentStep(
            step_index=step_index,
            request=ToolRequest(tool=canonical_call.tool, args=dict(canonical_call.args)),
            response=response,
            thinking=entry.thinking,
        )
        packet = build_feedback_packet(
            step_index=step_index,
            tool=canonical_call.tool,
            tool_input=canonical_call.args,
            tool_output=entry.tool_output,
        )
        environment_steps.append(step)
        feedback_packets.append(packet)

    return Episode(
        episode_id=episode_id,
        source_format=source_format,
        environment_steps=tuple(environment_steps),
        feedback_packets=tuple(feedback_packets),
    )


def render_episode_chatml(episode: Episode) -> tuple[str, list[TokenLabel]]:
    """Render episode to ChatML-like text and token-aligned labels."""
    text, labeled_spans = _build_episode_text_and_spans(episode)
    return text, labeled_spans


def tokenize_episode(
    episode: Episode,
    tokenizer: SupportsOffsetsTokenizer,
) -> tuple[str, list[int], list[TokenLabel], list[bool], list[bool]]:
    """Tokenize episode text and derive stage-specific masks."""
    text, labeled_spans = render_episode_chatml(episode)
    input_ids, token_labels = _tokenize_with_labels(text, labeled_spans, tokenizer)
    return (
        text,
        input_ids,
        token_labels,
        build_action_token_mask(token_labels, stage="rft"),
        build_action_token_mask(token_labels, stage="step_sdpo"),
    )


def build_training_record(
    episode: Episode,
    *,
    tokenizer: SupportsOffsetsTokenizer,
) -> dict[str, Any]:
    """Build one training-ready record from a canonical episode."""
    (
        sequence_text,
        input_ids,
        token_labels,
        action_mask_rft,
        action_mask_step_sdpo,
    ) = tokenize_episode(episode, tokenizer)

    return {
        "episode_id": episode.episode_id,
        "source_format": episode.source_format,
        "num_steps": len(episode.environment_steps),
        "sequence_text": sequence_text,
        "input_ids": input_ids,
        "token_labels": token_labels,
        "action_mask_rft": action_mask_rft,
        "action_mask_step_sdpo": action_mask_step_sdpo,
        "environment_steps": [_environment_step_to_dict(step) for step in episode.environment_steps],
        "feedback_packets": [packet.to_dict() for packet in episode.feedback_packets],
    }


def build_training_records(
    episodes: Sequence[Episode],
    *,
    tokenizer: SupportsOffsetsTokenizer,
) -> list[dict[str, Any]]:
    """Build all training records for *episodes*."""
    return [build_training_record(episode, tokenizer=tokenizer) for episode in episodes]


def write_training_records(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write records to JSONL, Arrow IPC file, or Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=True))
                handle.write("\n")
        return

    if suffix in {".parquet", ".arrow"}:
        _write_arrow_family(records, output_path=output_path, suffix=suffix)
        return

    raise ValueError(f"Unsupported output extension for {output_path}. Use .jsonl/.parquet/.arrow")


def load_qwen_tokenizer(model_name: str = "Qwen/Qwen3-4B") -> SupportsOffsetsTokenizer:
    """Load a fast tokenizer compatible with Qwen3-4B chat format."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency is runtime optional for tests
        raise RuntimeError(
            "transformers is required for tokenizer loading. Install with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return cast(SupportsOffsetsTokenizer, tokenizer)


def run_ingestion(
    *,
    input_path: Path,
    output_path: Path,
    tokenizer_model: str = "Qwen/Qwen3-4B",
    max_episodes: int | None = None,
) -> dict[str, int]:
    """Run end-to-end ingestion and return summary stats."""
    raw_records = load_raw_records(input_path)
    episodes = build_episodes(raw_records)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    tokenizer = load_qwen_tokenizer(tokenizer_model)
    training_records = build_training_records(episodes, tokenizer=tokenizer)
    write_training_records(training_records, output_path)
    return {
        "raw_records": len(raw_records),
        "episodes_ingested": len(episodes),
        "records_written": len(training_records),
    }


def _load_json_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [_expect_mapping(item, context=str(path)) for item in payload]
    if isinstance(payload, dict):
        trajectories = payload.get("trajectories")
        if isinstance(trajectories, list):
            return [_expect_mapping(item, context=str(path)) for item in trajectories]
        return [dict(payload)]
    raise ValueError(f"JSON payload in {path} must be an object or list of objects.")


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            decoded = json.loads(stripped)
            records.append(_expect_mapping(decoded, context=f"{path}:{line_number}"))
    return records


def _expect_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must contain JSON objects only.")
    return dict(value)


def _infer_source_format(record: Mapping[str, Any]) -> str:
    explicit = record.get("source_format") or record.get("dataset")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if "history" in record:
        return "swe-bench"
    if any(key in record for key in _TRAJECTORY_KEYS):
        return "swe-smith"
    return "unknown"


def _extract_episode_id(record: Mapping[str, Any], *, fallback_index: int) -> str:
    for key in ("instance_id", "episode_id", "id", "task_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"episode-{fallback_index:06d}"


def _iter_step_entries(record: Mapping[str, Any]) -> Iterable[_StepEntry]:
    for key in _TRAJECTORY_KEYS:
        candidate = record.get(key)
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            for raw_step in candidate:
                if not isinstance(raw_step, Mapping):
                    continue
                yield from _entries_from_step(raw_step)
            return

    history = record.get("history")
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        yield from _entries_from_history(history)
        return


def _entries_from_history(history: Sequence[Any]) -> Iterable[_StepEntry]:
    pending_calls: list[tuple[str, dict[str, Any], str | None]] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue

        role = str(item.get("role", "")).strip().lower()
        if role == "assistant":
            pending_calls = _extract_tool_calls_from_payload(item)
            if not pending_calls:
                continue
            direct_output = _extract_tool_output(item, call=None, call_index=0)
            if direct_output:
                for tool_name, args, thinking in pending_calls:
                    yield _StepEntry(
                        tool_name=tool_name,
                        args=args,
                        tool_output=direct_output,
                        thinking=thinking,
                    )
                pending_calls = []
            continue

        if role in {"tool", "environment", "observation"} and pending_calls:
            output = _extract_tool_output(item, call=None, call_index=0)
            tool_name, args, thinking = pending_calls.pop(0)
            yield _StepEntry(
                tool_name=tool_name,
                args=args,
                tool_output=output,
                thinking=thinking,
            )

    for tool_name, args, thinking in pending_calls:
        yield _StepEntry(
            tool_name=tool_name,
            args=args,
            tool_output={},
            thinking=thinking,
        )


def _entries_from_step(step: Mapping[str, Any]) -> Iterable[_StepEntry]:
    calls = _extract_tool_calls_from_payload(step)
    if not calls:
        return

    for call_index, (tool_name, args, thinking) in enumerate(calls):
        output = _extract_tool_output(step, call=step, call_index=call_index)
        yield _StepEntry(
            tool_name=tool_name,
            args=args,
            tool_output=output,
            thinking=thinking,
        )


def _extract_tool_calls_from_payload(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any], str | None]]:
    extracted: list[tuple[str, dict[str, Any], str | None]] = []
    top_level_thinking = _extract_thinking(payload)

    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)):
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            tool_name = _extract_tool_name(call)
            if not tool_name:
                continue
            args = _extract_tool_args(call, default_source=payload, tool_name=tool_name)
            thinking = _extract_thinking(call) or top_level_thinking
            extracted.append((tool_name, args, thinking))
        if extracted:
            return extracted

    for key in ("tool_call", "action"):
        call = payload.get(key)
        if isinstance(call, Mapping):
            tool_name = _extract_tool_name(call)
            if tool_name:
                args = _extract_tool_args(call, default_source=payload, tool_name=tool_name)
                extracted.append((tool_name, args, _extract_thinking(call) or top_level_thinking))
                return extracted

    direct_tool_name = _extract_tool_name(payload)
    if direct_tool_name:
        args = _extract_tool_args(payload, default_source=payload, tool_name=direct_tool_name)
        extracted.append((direct_tool_name, args, top_level_thinking))

    return extracted


def _extract_tool_name(payload: Mapping[str, Any]) -> str | None:
    for key in _TOOL_NAME_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    function_payload = payload.get("function")
    if isinstance(function_payload, Mapping):
        value = function_payload.get("name")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _extract_tool_args(
    payload: Mapping[str, Any],
    *,
    default_source: Mapping[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    for key in _TOOL_ARGS_KEYS:
        value = payload.get(key)
        parsed = _coerce_args(value)
        if parsed is not None:
            return _normalize_submit_args(tool_name=tool_name, args=parsed, context=default_source)

    function_payload = payload.get("function")
    if isinstance(function_payload, Mapping):
        parsed = _coerce_args(function_payload.get("arguments"))
        if parsed is not None:
            return _normalize_submit_args(tool_name=tool_name, args=parsed, context=default_source)

    fallback = _normalize_submit_args(tool_name=tool_name, args={}, context=default_source)
    return fallback


def _coerce_args(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        if stripped.startswith("{") and stripped.endswith("}"):
            decoded = json.loads(stripped)
            if not isinstance(decoded, Mapping):
                raise ValueError("Decoded arguments JSON must be an object.")
            return dict(decoded)
        return {"command": value}
    return {"value": value}


def _normalize_submit_args(
    *,
    tool_name: str,
    args: dict[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_tool = tool_name.strip().lower()
    if normalized_tool not in {"submit", "answer"}:
        return dict(args)
    if any(key in args for key in ("final_response", "answer")):
        return dict(args)
    for key in ("content", "text", "final_response", "answer"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            copied = dict(args)
            copied["answer"] = value
            return copied
    return dict(args)


def _extract_tool_output(
    step: Mapping[str, Any],
    *,
    call: Mapping[str, Any] | None,
    call_index: int,
) -> dict[str, Any]:
    if call is not None:
        for key in _TOOL_OUTPUT_KEYS:
            if key in call:
                return _coerce_tool_output(call.get(key))

    for key in _TOOL_OUTPUT_LIST_KEYS:
        value = step.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if call_index < len(value):
                return _coerce_tool_output(value[call_index])

    for key in _TOOL_OUTPUT_KEYS:
        if key in step:
            return _coerce_tool_output(step.get(key))

    if "content" in step:
        return _coerce_tool_output(step.get("content"))
    return {}


def _coerce_tool_output(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return {"stdout": value}
    return {"stdout": json.dumps(value, sort_keys=True, ensure_ascii=True)}


def _extract_thinking(payload: Mapping[str, Any]) -> str | None:
    for key in _THINKING_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _tool_response_from_output(tool_output: Mapping[str, Any]) -> ToolResponse:
    stdout = str(tool_output.get("stdout", ""))
    stderr = str(tool_output.get("stderr", ""))
    raw_exit_code = tool_output.get("exit_code", 0)
    try:
        exit_code = int(raw_exit_code)
    except (TypeError, ValueError):
        exit_code = 0
    metadata = {
        key: value
        for key, value in tool_output.items()
        if key not in {"stdout", "stderr", "exit_code"}
    }
    return ToolResponse(stdout=stdout, stderr=stderr, exit_code=exit_code, metadata=metadata)


def _build_episode_text_and_spans(episode: Episode) -> tuple[str, list[tuple[int, int, TokenLabel]]]:
    chunks: list[str] = []
    labeled_spans: list[tuple[int, int, TokenLabel]] = []
    cursor = 0

    for step, packet in zip(episode.environment_steps, episode.feedback_packets):
        cursor = _append_segment(chunks, text="<|im_start|>assistant\n", cursor=cursor)
        if step.thinking:
            thinking_block = f"<think>{step.thinking}</think>\n"
            cursor = _append_segment(
                chunks,
                text=thinking_block,
                cursor=cursor,
                labeled_spans=labeled_spans,
                label="think",
            )
        tool_payload = json.dumps(
            {"tool": step.request.tool, "args": dict(step.request.args)},
            sort_keys=True,
            ensure_ascii=True,
        )
        tool_call_block = f"<tool_call>{tool_payload}</tool_call>\n"
        cursor = _append_segment(
            chunks,
            text=tool_call_block,
            cursor=cursor,
            labeled_spans=labeled_spans,
            label="tool_call",
        )
        cursor = _append_segment(chunks, text="<|im_end|>\n", cursor=cursor)

        feedback_text = packet.canonical_feedback.normalized_text
        cursor = _append_segment(
            chunks,
            text="<|im_start|>tool\n<tool_response>",
            cursor=cursor,
        )
        cursor = _append_segment(chunks, text=feedback_text, cursor=cursor)
        cursor = _append_segment(
            chunks,
            text="</tool_response>\n<|im_end|>\n",
            cursor=cursor,
        )

    return "".join(chunks), labeled_spans


def _append_segment(
    chunks: list[str],
    *,
    text: str,
    cursor: int,
    labeled_spans: list[tuple[int, int, TokenLabel]] | None = None,
    label: TokenLabel | None = None,
) -> int:
    chunks.append(text)
    end = cursor + len(text)
    if labeled_spans is not None and label is not None and text:
        labeled_spans.append((cursor, end, label))
    return end


def _tokenize_with_labels(
    text: str,
    labeled_spans: Sequence[tuple[int, int, TokenLabel]],
    tokenizer: SupportsOffsetsTokenizer,
) -> tuple[list[int], list[TokenLabel]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    raw_ids = encoded.get("input_ids")
    raw_offsets = encoded.get("offset_mapping")
    if raw_ids is None or raw_offsets is None:
        raise ValueError("Tokenizer must return `input_ids` and `offset_mapping`.")

    input_ids = _normalize_ints(raw_ids)
    offsets = _normalize_offsets(raw_offsets)
    if len(input_ids) != len(offsets):
        raise ValueError("Tokenizer returned inconsistent input_ids and offset_mapping lengths.")

    token_labels = [_label_for_offset(start, end, labeled_spans) for start, end in offsets]
    return input_ids, token_labels


def _normalize_ints(raw_ids: Any) -> list[int]:
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise ValueError("Tokenizer `input_ids` must be a sequence of ints.")
    return [int(item) for item in raw_ids]


def _normalize_offsets(raw_offsets: Any) -> list[tuple[int, int]]:
    if not isinstance(raw_offsets, Sequence) or isinstance(raw_offsets, (str, bytes)):
        raise ValueError("Tokenizer `offset_mapping` must be a sequence of pairs.")
    normalized: list[tuple[int, int]] = []
    for offset in raw_offsets:
        if not isinstance(offset, Sequence) or isinstance(offset, (str, bytes)) or len(offset) != 2:
            raise ValueError("Each offset mapping entry must be a (start, end) pair.")
        normalized.append((int(offset[0]), int(offset[1])))
    return normalized


def _label_for_offset(
    start: int,
    end: int,
    labeled_spans: Sequence[tuple[int, int, TokenLabel]],
) -> TokenLabel:
    if end <= start:
        return "other"
    label: TokenLabel = "other"
    for span_start, span_end, span_label in labeled_spans:
        if end <= span_start or start >= span_end:
            continue
        if span_label == "tool_call":
            return "tool_call"
        if span_label == "think":
            label = "think"
    return label


def _environment_step_to_dict(step: EnvironmentStep) -> dict[str, Any]:
    return {
        "step_index": step.step_index,
        "request": {
            "tool": step.request.tool,
            "args": dict(step.request.args),
        },
        "response": {
            "stdout": step.response.stdout,
            "stderr": step.response.stderr,
            "exit_code": step.response.exit_code,
            "metadata": dict(step.response.metadata),
        },
        "thinking": step.thinking,
    }


def _write_arrow_family(
    records: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    suffix: str,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "pyarrow is required for .parquet/.arrow outputs. Install with `pip install pyarrow`."
        ) from exc

    table = pa.Table.from_pylist([dict(record) for record in records])
    if suffix == ".parquet":
        pq.write_table(table, output_path)
        return

    with pa.OSFile(output_path, "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
