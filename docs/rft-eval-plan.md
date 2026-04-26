# Fixed Held-Out RFT Eval

## Status

Implemented as an outer-loop RFT telemetry path.

The current behavior is:

- `rft_runtime.loop.eval_task_count` configures a fixed number of valid held-out tasks. The default is `50`.
- Before a non-dry-run RFT loop starts step 0, the runtime validates that exactly this many held-out eval tasks can be reserved from the configured valid task pool.
- The same deterministic held-out eval partition is used at every outer RFT step.
- Step 0 evals the initial model. Later outer steps eval the current checkpoint before that step's SFT update.
- Format RFT and positive RFT both use the outer-step eval path, with stage-specific selection and verifier semantics.
- The eval path is telemetry only. It does not affect row selection, checkpoint acceptance, stopping, or the SFT trainer's validation loop.

## Config Surface

Primary config:

```json
{
  "rft_runtime": {
    "loop": {
      "eval_task_count": 50
    }
  }
}
```

Launcher/runtime overrides:

- `RFT_EVAL_TASK_COUNT`
- `--eval-task-count`
- `data.on_policy.task_eval_task_count` for direct mode

Legacy fraction settings remain available for compatibility:

- `eval_split_fraction`
- `eval_min_rows`
- `task_eval_split_fraction`
- `task_eval_min_rows`

When `eval_task_count > 0`, the fixed count is authoritative.

Setting `eval_task_count=0` disables fixed-count mode. If legacy
`eval_split_fraction` remains positive, the fraction-based holdout path is used;
set both to zero to disable held-out eval entirely.

## Runtime Contract

The RFT runtime validates the fixed held-out task pool before loading the tokenizer or starting managed vLLM. The launcher also validates the fixed pool before direct-mode RFT starts.

For each outer step:

1. Collect training candidates from the train partition.
2. Collect one eval attempt for each task in the fixed eval partition.
3. Record eval selected/rejected counts and task-family/difficulty telemetry in the step summary.
4. Train inner SFT only on the train parquet.

The inner SFT trainer is deliberately not used for held-out eval:

- `trainer.test_freq=0`
- `data.val_files=[]`

The local `verl_integration.fsdp_sft_trainer_entry` handles this disabled state by
skipping validation dataset construction and verl's otherwise-unconditional
last-step validation.

This keeps the signal at the outer RFT step boundary only.

## Fail-Closed Behavior

The runtime does not fall back from held-out eval to train rows.

It fails before the run if the requested fixed eval task count cannot be materialized. It also fails before the inner SFT trainer if held-out eval collection produces zero selected rows after filtering, because that would make the outer-step eval signal invalid.

## Non-Goals

- No committed 100-task manifest is used by the current implementation.
- No separate benchmark runner is introduced here.
- No inner-SFT validation curve is treated as RFT convergence signal.
