"""Pre-tokenized RFT cache dataset and length-bucketed sampling."""

from __future__ import annotations

import hashlib
import json
import numbers
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from trainer.rft_multiturn_dataset import build_multiturn_messages

RFT_TOKEN_CACHE_SCHEMA_VERSION = 1

REQUIRED_TOKEN_CACHE_COLUMNS: tuple[str, ...] = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "loss_mask",
    "sequence_length",
    "loss_token_count",
    "cache_schema_version",
    "cache_fingerprint",
)

FINGERPRINT_SOURCE_PATHS: tuple[str, ...] = (
    "src/prompts/runtime_messages.py",
    "src/data/tokenization.py",
    "src/data/feedback_canonicalizer.py",
    "src/rollout/action_format.py",
    "src/trainer/rft_multiturn_dataset.py",
    "src/trainer/rft_token_cache.py",
    "src/trainer/rft_runtime_loop.py",
)


def build_rft_token_cache_fingerprint(
    *,
    tokenizer: Any,
    max_model_len: int,
    data_max_length: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> str:
    """Hash tokenizer/template/config/code inputs that define cached RFT tokens."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    payload = {
        "schema_version": RFT_TOKEN_CACHE_SCHEMA_VERSION,
        "tokenizer": _tokenizer_fingerprint_payload(tokenizer),
        "chat_template_kwargs": _json_safe_mapping(chat_template_kwargs or {}),
        "max_model_len": int(max_model_len),
        "data_max_length": int(data_max_length),
        "max_tool_calls_per_turn": _max_tool_calls_per_turn(),
        "source_hashes": _source_hashes(root),
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_selected_rows_to_token_cache_parquet(
    selected_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    tokenizer: Any | None = None,
    max_sequence_length: int | None = None,
    cache_fingerprint: str,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> int:
    """Write selected rows as one pre-tokenized SFT sample per parquet row."""
    if tokenizer is None:
        raise ValueError("tokenizer is required to build full multiturn RFT token cache rows.")
    if not selected_rows:
        raise ValueError("Cannot write RFT token cache parquet with zero rows.")
    records = build_token_cache_records(
        selected_rows,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        cache_fingerprint=cache_fingerprint,
        chat_template_kwargs=chat_template_kwargs,
    )
    _write_records_to_parquet(records, output_path)
    return len(records)


def build_token_cache_records(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_sequence_length: int | None,
    cache_fingerprint: str,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not cache_fingerprint or not str(cache_fingerprint).strip():
        raise ValueError("cache_fingerprint must be a non-empty string.")
    limit = int(max_sequence_length) if max_sequence_length is not None else None
    if limit is not None and limit < 2:
        raise ValueError("max_sequence_length must be >= 2 when provided.")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        messages = build_multiturn_messages(row, row_index=index)
        input_ids, loss_mask, attention_mask = _tokenize_multiturn_messages(
            messages=messages,
            tokenizer=tokenizer,
            chat_template_kwargs=chat_template_kwargs,
        )
        if not input_ids:
            raise ValueError(f"rows[{index}] produced an empty multiturn token sequence.")
        length = min(len(input_ids), len(loss_mask), len(attention_mask))
        if limit is not None and length > limit:
            raise ValueError(
                f"rows[{index}] multiturn token length {length} exceeds max_sequence_length={limit}."
            )
        if length < 2:
            raise ValueError(f"rows[{index}] has fewer than 2 usable tokens.")
        input_ids = input_ids[:length]
        loss_mask = loss_mask[:length]
        attention_mask = attention_mask[:length]
        records.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": list(range(length)),
                "loss_mask": loss_mask,
                "sequence_length": length,
                "loss_token_count": int(sum(loss_mask)),
                "cache_schema_version": RFT_TOKEN_CACHE_SCHEMA_VERSION,
                "cache_fingerprint": str(cache_fingerprint),
                "task_id": str(row.get("task_id", "")).strip(),
                "attempt_index": _coerce_int(row.get("attempt_index"), fallback=0),
                "step_index": _coerce_int(row.get("step_index"), fallback=0),
                "turn_index": _coerce_int(row.get("turn_index"), fallback=0),
            }
        )
    return records


def _tokenize_multiturn_messages(
    *,
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> tuple[list[int], list[int], list[int]]:
    """Tokenize messages with the same per-turn mask semantics as verl MultiTurnSFTDataset."""
    rendered_text = _render_chat_template_text(
        tokenizer,
        [dict(message) for message in messages],
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    input_ids, attention_mask, offsets = _tokenize_rendered_text_with_offsets(
        tokenizer,
        rendered_text,
    )
    template_input_ids, _ = _apply_chat_template_ids_and_attention(
        tokenizer,
        [dict(message) for message in messages],
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if input_ids != template_input_ids:
        raise ValueError(
            "Rendered RFT token cache text does not match chat template tokenization. "
            "Check tokenizer chat-template settings."
        )

    loss_mask = [0] * len(input_ids)
    for start_char, end_char in _assistant_loss_char_spans(rendered_text, messages):
        for token_index, (token_start, token_end) in enumerate(offsets):
            if token_end <= token_start:
                continue
            if token_start < end_char and token_end > start_char:
                loss_mask[token_index] = 1
    return input_ids, loss_mask, attention_mask


def _render_chat_template_text(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("tokenizer must define callable apply_chat_template for RFT token cache.")
    rendered = apply_chat_template(
        list(messages),
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **dict(chat_template_kwargs or {}),
    )
    if not isinstance(rendered, str):
        raise ValueError("chat template text rendering did not return a string.")
    return rendered


def _tokenize_rendered_text_with_offsets(
    tokenizer: Any,
    rendered_text: str,
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    if not callable(tokenizer):
        raise ValueError("tokenizer must be callable with return_offsets_mapping=True.")
    payload = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = _extract_token_ids(payload)
    attention_mask = _extract_attention_mask(payload, length=len(input_ids))
    offsets = _extract_offset_mapping(payload, length=len(input_ids))
    return input_ids, attention_mask, offsets


def _assistant_loss_char_spans(
    rendered_text: str,
    messages: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip()
        candidate = _rendered_message_content_candidate(
            rendered_text,
            message,
            cursor=cursor,
        )
        if candidate is None:
            if role == "assistant":
                raise ValueError(
                    f"Assistant message {index} content does not align with the full "
                    "chat-template rendering."
                )
            continue
        content_start, content_text = candidate
        content_end = content_start + len(content_text)
        cursor = max(cursor, content_end)
        if role != "assistant":
            continue
        turn_end = _assistant_turn_end(rendered_text, content_end)
        if turn_end < content_start:
            raise ValueError("Assistant-turn RFT token cache span is inverted.")
        spans.append((content_start, turn_end))
    return spans


def _rendered_message_content_candidate(
    rendered_text: str,
    message: Mapping[str, Any],
    *,
    cursor: int,
) -> tuple[int, str] | None:
    candidates = _rendered_message_content_candidates(message)
    if not candidates:
        return None

    best: tuple[int, str] | None = None
    for candidate in candidates:
        start = rendered_text.find(candidate, cursor)
        if start < 0:
            continue
        if best is None or start < best[0] or (start == best[0] and len(candidate) > len(best[1])):
            best = (start, candidate)
    return best


def _rendered_message_content_candidates(message: Mapping[str, Any]) -> tuple[str, ...]:
    content = message.get("content", "")
    if not isinstance(content, str):
        return ()
    role = str(message.get("role", "")).strip()
    candidates = [content]
    if role == "assistant":
        if "</think>" in content:
            candidates.append(content.split("</think>")[-1].lstrip("\n"))
        candidates.append(content.lstrip("\n"))
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _assistant_turn_end(rendered_text: str, content_end: int) -> int:
    candidates: list[tuple[int, int]] = []
    for marker in ("<|im_end|>\n", "<|im_end|>", "</assistant>"):
        marker_start = rendered_text.find(marker, content_end)
        if marker_start < 0:
            continue
        marker_end = marker_start + len(marker)
        if marker == "<|im_end|>" and marker_end < len(rendered_text) and rendered_text[marker_end] == "\n":
            marker_end += 1
        candidates.append((marker_start, marker_end))
    if not candidates:
        return len(rendered_text)
    _, marker_end = min(candidates, key=lambda item: item[0])
    return marker_end


def _apply_chat_template_ids_and_attention(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> tuple[list[int], list[int]]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("tokenizer must define callable apply_chat_template for RFT token cache.")
    try:
        payload = apply_chat_template(
            list(messages),
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            **dict(chat_template_kwargs or {}),
        )
    except TypeError:
        payload = apply_chat_template(
            list(messages),
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **dict(chat_template_kwargs or {}),
        )
    input_ids = _extract_token_ids(payload)
    attention_mask = _extract_attention_mask(payload, length=len(input_ids))
    return input_ids, attention_mask


def _extract_token_ids(payload: Any) -> list[int]:
    if isinstance(payload, Mapping):
        payload = payload.get("input_ids")
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if isinstance(payload, Sequence) and payload and hasattr(payload[0], "tolist"):
        payload = [item.tolist() for item in payload]
    if (
        isinstance(payload, Sequence)
        and payload
        and isinstance(payload[0], Sequence)
        and not isinstance(payload[0], (str, bytes))
    ):
        payload = payload[0]
    return _coerce_int_list(payload, label="chat_template.input_ids")


def _extract_attention_mask(payload: Any, *, length: int) -> list[int]:
    if isinstance(payload, Mapping):
        raw_attention = payload.get("attention_mask")
        if raw_attention is not None:
            if hasattr(raw_attention, "tolist"):
                raw_attention = raw_attention.tolist()
            if (
                isinstance(raw_attention, Sequence)
                and raw_attention
                and isinstance(raw_attention[0], Sequence)
                and not isinstance(raw_attention[0], (str, bytes))
            ):
                raw_attention = raw_attention[0]
            attention = _coerce_int_list(raw_attention, label="chat_template.attention_mask")
            if len(attention) == length:
                return [1 if item else 0 for item in attention]
    return [1] * length


def _extract_offset_mapping(payload: Any, *, length: int) -> list[tuple[int, int]]:
    raw_offsets: Any = None
    if isinstance(payload, Mapping):
        raw_offsets = payload.get("offset_mapping")
    elif hasattr(payload, "offset_mapping"):
        raw_offsets = getattr(payload, "offset_mapping")
    if raw_offsets is None:
        raise ValueError("tokenizer did not return offset_mapping for rendered chat template text.")
    if hasattr(raw_offsets, "tolist"):
        raw_offsets = raw_offsets.tolist()
    if (
        isinstance(raw_offsets, Sequence)
        and raw_offsets
        and isinstance(raw_offsets[0], Sequence)
        and raw_offsets[0]
        and isinstance(raw_offsets[0][0], Sequence)
        and not isinstance(raw_offsets[0][0], (str, bytes))
    ):
        raw_offsets = raw_offsets[0]
    if not isinstance(raw_offsets, Sequence) or isinstance(raw_offsets, (str, bytes)):
        raise ValueError("offset_mapping must be a sequence of token offset pairs.")

    offsets: list[tuple[int, int]] = []
    for item in raw_offsets:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError("offset_mapping must contain pairs of token offsets.")
        offsets.append((int(item[0]), int(item[1])))
    if len(offsets) != length:
        raise ValueError(
            f"offset_mapping length {len(offsets)} does not match input_ids length {length}."
        )
    return offsets


class CachedRFTSFTDataset:
    """verl SFT dataset that consumes pre-tokenized cache parquet rows."""

    def __init__(
        self,
        parquet_files: str | Sequence[str],
        tokenizer: Any,
        config: Mapping[str, Any],
        processor: Any | None = None,
        max_samples: int = -1,
    ) -> None:
        del processor
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - train-only dependency
            raise RuntimeError(
                "CachedRFTSFTDataset requires torch. Install training extras (`pip install -e \".[train]\"`)."
            ) from exc

        self._torch = torch
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        records = _read_parquet_records(parquet_files)
        if max_samples > 0:
            records = records[: int(max_samples)]
        _validate_required_columns(records)

        expected_schema = _config_get_int(config, "token_cache.schema_version", RFT_TOKEN_CACHE_SCHEMA_VERSION)
        expected_fingerprint = str(_config_get(config, "token_cache.expected_fingerprint", "") or "").strip()
        train_min_rows = _config_get_int(config, "train_min_rows", 1)
        if len(records) < train_min_rows:
            raise ValueError(
                f"Cached RFT dataset has {len(records)} rows, below data.train_min_rows={train_min_rows}."
            )

        self._samples: list[dict[str, Any]] = []
        self.sequence_lengths: list[int] = []
        for index, record in enumerate(records):
            schema_version = int(record["cache_schema_version"])
            if schema_version != expected_schema:
                raise ValueError(
                    f"RFT token cache schema mismatch at row {index}: "
                    f"{schema_version} != expected {expected_schema}."
                )
            fingerprint = str(record["cache_fingerprint"])
            if expected_fingerprint and fingerprint != expected_fingerprint:
                raise ValueError(
                    f"RFT token cache fingerprint mismatch at row {index}: "
                    f"{fingerprint} != expected {expected_fingerprint}."
                )
            sample = {
                "input_ids": torch.tensor(_coerce_int_list(record["input_ids"], label="input_ids"), dtype=torch.long),
                "attention_mask": torch.tensor(_coerce_int_list(record["attention_mask"], label="attention_mask"), dtype=torch.long),
                "position_ids": torch.tensor(_coerce_int_list(record["position_ids"], label="position_ids"), dtype=torch.long),
                "loss_mask": torch.tensor(_coerce_loss_mask(record["loss_mask"], label="loss_mask"), dtype=torch.long),
            }
            length = int(record.get("sequence_length", sample["input_ids"].numel()))
            self.sequence_lengths.append(length)
            self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index]

    def collate_fn(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collate_token_cache_rows(rows, pad_token_id=self.pad_token_id)


def collate_token_cache_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot collate an empty RFT token-cache batch.")
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - train-only dependency
        raise RuntimeError("collate_token_cache_rows requires torch.") from exc

    max_len = max(int(row["input_ids"].numel()) for row in rows)

    def _pad_tensor(key: str, pad_value: int) -> Any:
        padded = []
        for row in rows:
            value = row[key]
            pad_len = max_len - int(value.numel())
            if pad_len > 0:
                pad = torch.full((pad_len,), int(pad_value), dtype=value.dtype)
                value = torch.cat([value, pad], dim=0)
            padded.append(value)
        return torch.stack(padded, dim=0)

    return {
        "input_ids": _pad_tensor("input_ids", pad_token_id),
        "attention_mask": _pad_tensor("attention_mask", 0),
        "position_ids": _pad_tensor("position_ids", 0),
        "loss_mask": _pad_tensor("loss_mask", 0),
    }


class LengthBucketDistributedSampler:
    """Distributed sampler that groups similar sequence lengths without packing."""

    def __init__(
        self,
        dataset: Any,
        *,
        num_replicas: int,
        rank: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        bucket_size: int | None = None,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.lengths = _dataset_lengths(dataset)
        global_batch_size = self.batch_size * self.num_replicas
        self.bucket_size = int(bucket_size or max(global_batch_size * 8, global_batch_size))
        if self.drop_last:
            self.num_samples = len(self.lengths) // self.num_replicas
        else:
            self.num_samples = (len(self.lengths) + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda index: self.lengths[index])
        buckets = [indices[offset : offset + self.bucket_size] for offset in range(0, len(indices), self.bucket_size)]

        if self.shuffle:
            import torch

            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            bucket_order = torch.randperm(len(buckets), generator=generator).tolist()
            shuffled: list[int] = []
            for bucket_index in bucket_order:
                bucket = list(buckets[bucket_index])
                order = torch.randperm(len(bucket), generator=generator).tolist()
                shuffled.extend(bucket[item] for item in order)
            indices = shuffled
        else:
            indices = [index for bucket in buckets for index in bucket]

        if self.drop_last:
            indices = indices[: self.total_size]
        elif indices:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                repeats = (indices * ((padding_size // len(indices)) + 1))[:padding_size]
                indices.extend(repeats)
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _dataset_lengths(dataset: Any) -> list[int]:
    lengths = getattr(dataset, "sequence_lengths", None)
    if isinstance(lengths, Sequence) and not isinstance(lengths, (str, bytes)):
        return [int(item) for item in lengths]
    return [1 for _ in range(len(dataset))]


def _read_parquet_records(parquet_files: str | Sequence[str]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on train extras
        raise RuntimeError(
            "Reading RFT token-cache parquet requires pandas/pyarrow. "
            "Install training extras (`pip install -e \".[train]\"`)."
        ) from exc

    paths = _normalize_parquet_files(parquet_files)
    if not paths:
        raise ValueError("CachedRFTSFTDataset requires at least one parquet file.")
    records: list[dict[str, Any]] = []
    for path in paths:
        dataframe = pd.read_parquet(path)
        records.extend(dataframe.to_dict(orient="records"))
    return records


def _write_records_to_parquet(records: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on train extras
        raise RuntimeError(
            "Writing RFT token-cache parquet requires pandas/pyarrow. "
            "Install training extras (`pip install -e \".[train]\"`)."
        ) from exc

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(target, index=False)


def _validate_required_columns(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("Cached RFT token-cache parquet has zero rows.")
    missing = [key for key in REQUIRED_TOKEN_CACHE_COLUMNS if key not in records[0]]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Cached RFT token-cache parquet is missing required columns: {joined}.")


def _normalize_parquet_files(parquet_files: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(parquet_files, str):
        raw_items = [parquet_files]
    elif isinstance(parquet_files, Sequence):
        raw_items = list(parquet_files)
    else:
        raw_items = [str(parquet_files)]
    return tuple(str(item).strip() for item in raw_items if str(item).strip())


def _config_get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if isinstance(current, Mapping):
            current = current.get(part, default)
        elif hasattr(current, "get"):
            try:
                current = current.get(part, default)
            except TypeError:
                current = getattr(current, part, default)
        else:
            current = getattr(current, part, default)
        if current is default:
            break
    return current


def _config_get_int(config: Mapping[str, Any], dotted_key: str, default: int) -> int:
    value = _config_get(config, dotted_key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _tokenizer_fingerprint_payload(tokenizer: Any) -> dict[str, Any]:
    added_vocab: Mapping[str, Any] = {}
    getter = getattr(tokenizer, "get_added_vocab", None)
    if callable(getter):
        try:
            added_vocab = dict(getter())
        except Exception:
            added_vocab = {}
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "chat_template": str(getattr(tokenizer, "chat_template", "")),
        "added_vocab": {str(key): int(value) for key, value in sorted(added_vocab.items())},
    }


def _source_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in FINGERPRINT_SOURCE_PATHS:
        path = root / relative_path
        if not path.is_file():
            hashes[relative_path] = "missing"
            continue
        hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _json_safe_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in sorted(payload.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        else:
            safe[str(key)] = str(value)
    return safe


def _max_tool_calls_per_turn() -> int:
    try:
        import config

        return int(config.MAX_TOOL_CALLS_PER_TURN)
    except Exception:
        return -1


def _coerce_int(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return fallback
    return fallback


def _coerce_int_list(value: Any, *, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(f"{label} must be a sequence of ints.")
    return [int(item) for item in value]


def _coerce_loss_mask(value: Any, *, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(f"{label} must be a sequence of bool/int values.")
    return [1 if _coerce_bool(item) else 0 for item in value]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Number):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return False
