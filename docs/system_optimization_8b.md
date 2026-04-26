# Qwen3-8B System Optimization Tracker

Last updated: 2026-04-26.

## Goal

Make the default RFT and turn-SDPO path viable for `Qwen/Qwen3-8B` on the existing single-node GPU target while preserving exact one-row-per-prompt training semantics.

## Current Snapshot

- Default model: `Qwen/Qwen3-8B`.
- Thinking policy: disabled by default. RFT and teacher-reprompt pilot vLLM launches use `--default-chat-template-kwargs '{"enable_thinking": false}'`, launchers still pass `data.apply_chat_template_kwargs.enable_thinking=false`, and tokenizers created through `verl.utils.hf_tokenizer()` default `apply_chat_template()` to `enable_thinking=False` unless a caller overrides it.
- RFT memory defaults: microbatch per GPU `1`; vLLM GPU memory utilization `0.75`; `max_num_seqs=32`; `max_num_batched_tokens=131072`.
- SDPO memory defaults: actor microbatch per GPU `1`; rollout/ref log-prob microbatch per GPU `1`; `ppo_max_token_len_per_gpu=16384`; rollout GPU memory utilization `0.80`; `max_num_seqs=32`; `max_num_batched_tokens=131072`.
- Existing optimizations kept: FSDP2, activation/gradient checkpointing, bf16 model dtype, remove-padding, SDPO dynamic token batching, and checkpoint retention `checkpoint_keep_last=1`.
- `enforce_eager=true` remains enabled for compatibility. This is a known throughput cost and should be measured against non-eager/cuda-graph rollout before flipping.

## RFT Token Cache

RFT runtime now writes a per-outer-step pre-tokenized parquet cache and launches the inner SFT trainer against `trainer.rft_token_cache.CachedRFTSFTDataset`. Cache rows render the same full multiturn transcript shape produced by `build_multiturn_messages(...)`, then store token tensors and the assistant-turn loss mask for the inner trainer.

- One parquet row remains one prompt/trajectory; no concatenation or attention-semantics change.
- Required columns: `input_ids`, `attention_mask`, `position_ids`, `loss_mask`, `sequence_length`, `loss_token_count`, `cache_schema_version`, `cache_fingerprint`.
- Cache fingerprint covers tokenizer metadata, chat template, `data.apply_chat_template_kwargs`, added vocab, max lengths, selected prompt/tokenization/action-format/feedback/RFT source files, and `MAX_TOOL_CALLS_PER_TURN`.
- The inner trainer disables verl `MultiTurnSFTDataset` fallback and inner validation. Fixed held-out convergence eval remains outer-loop only.
- `data.train_min_rows` defaults to the global train batch size; too-few selected rows fail before trainer launch.

## Length Bucketing

RFT uses length-bucketed distributed sampling only.

- No sequence packing.
- No row concatenation.
- The sampler groups similar `sequence_length` rows, reports stable `len()`, supports `set_epoch()`, and uses a rank-independent epoch seed.
- With verl remove-padding enabled, the expected win is lower peak per-step token count and better per-rank token balance, not simply padding removal.

## Profiler Keys

Profiler metrics use the `profiler/*` namespace and are safe for file logging and W&B filtering.

Minimum RFT outer keys:

- `profiler/rft_outer_rollout_collect_sec`
- `profiler/rft_outer_eval_collect_sec`
- `profiler/rft_token_cache_write_sec`
- `profiler/rft_trainer_wall_sec`
- `profiler/rft_vllm_restart_sec`
- `profiler/rft_outer_step_sec`

Minimum RFT/SDPO token and memory keys:

- `profiler/total_tokens`
- `profiler/non_padding_tokens`
- `profiler/loss_tokens`
- `profiler/non_padding_ratio`
- `profiler/loss_tokens_per_total_tokens`
- `profiler/global_tokens_per_sec`
- `profiler/tokens_per_sec_per_gpu`
- `profiler/gpu_memory_allocated_bytes`
- `profiler/gpu_memory_reserved_bytes`
- `profiler/gpu_memory_max_reserved_bytes`
- `profiler/oom_margin`

JSONL locations:

- RFT outer: `${RFT_OUTPUT_DIR}/profiler.jsonl`.
- RFT inner: `rft_step_*/profiler/rank_*.jsonl`.
- SDPO: file logger output under `VERL_FILE_LOGGER_ROOT`; actor-update profiler keys are returned with normal verl metrics.

RFT inner token-throughput counts are reduced across ranks before global and per-GPU rates are computed. `nvidia-smi` utilization is sampled only at coarse boundaries and is allowed to be missing. Hot paths use torch CUDA memory counters.

## Follow-Up Measurements

- Compare `enforce_eager=true` against non-eager/cuda-graph rollout at 8B.
- Use `profiler/rft_vllm_restart_sec` to decide whether warm vLLM weight reload or a persistent trainer/server process is worth building.
- Track SDPO rollout KV pressure with `max_num_seqs`, `n`, context length, and `max_num_batched_tokens` before increasing rollout parallelism.

## Rollback To 4B

Single-PR rollback checklist:

- Set `configs/verl/model_defaults.yaml` and `configs/runtime/training_policy_defaults.v1.json` back to the 4B model id.
- Restore RFT vLLM memory utilization and batch-token limits if 4B throughput is preferred.
- Restore SDPO `ppo_max_token_len_per_gpu`, rollout `gpu_memory_utilization`, `max_num_seqs`, and `max_num_batched_tokens`.
- Decide whether to keep RFT token caching and length bucketing; they are model-size independent and should normally remain.
- Keep thinking suppression unless the chosen model is a non-thinking-only fork and the suppression has been explicitly retired.
