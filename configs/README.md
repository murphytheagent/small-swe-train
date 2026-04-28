# Config Layout

This repo intentionally keeps local configs minimal.
To align launch-time behavior, config edits should stay near the scripts that consume them.

## Active Entrypoints

- `configs/verl/sdpo_swe.yaml` is the SDPO training entry config.
- `configs/verl/rft_swe.yaml` is the RFT training entry config.

Both are launched by scripts in `scripts/` with:
- `--config-dir configs/verl`
- `--config-name sdpo_swe` or `rft_swe`

These scripts also pass runtime-specific overrides through `configs/runtime/` and `configs/data/` (for example `RFT_TURN_GENERATOR_MODE`, `RFT_TASK_BATCH_SIZE`, etc.).

The centralized runtime policy in `configs/runtime/training_policy_defaults.v1.json`
tracks the canonical staged pipeline as:
- `format_rft`
- optional `positive_rft`
- `turn_sdpo`

Set `SMALL_SWE_TRAINING_POLICY_CONFIG` to select a different checked-in policy
JSON under `configs/`, for example the JSON/XML preflight variants in
`configs/runtime/training_policy_preflight_*.v1.json`. This selector is for
experiment policy files; launch-only values such as output directories,
ports, stage name, and checkpoint handoff remain script arguments or
environment values.

## Hydra Composition (Current)

- `sdpo_swe.yaml` defaults: `model_defaults`, `ppo_trainer`, `_self_`
- `rft_swe.yaml` defaults: `model_defaults`, `sft_trainer`, `_self_`

`ppo_trainer` / `sft_trainer` come from installed `verl` package configs.

## Length Fields (`turn_sdpo`)

In `configs/verl/sdpo_swe.yaml`:

- `max_model_len`:
  upper bound on total sequence context for rollout engine
  (`prompt_context + generated_response`).
- `data.max_prompt_length`:
  prompt-context budget before rollout (left-clip if longer).
- `data.max_response_length`:
  generated-token budget during rollout.
- `actor_rollout_ref.rollout.prompt_length` and `response_length`:
  wired from local `data.max_*` fields.
- `actor_rollout_ref.rollout.max_model_len`:
  wired from local `max_model_len`.
- `actor_rollout_ref.model.override_config.max_position_embeddings`:
  pinned to `max_model_len` so vLLM inherits the same sequence cap
  (verl overrides rollout max length with the HF config value).

Practical rule:
- keep `max_model_len >= max_prompt_length + max_response_length`.

## Config Directories

- `configs/verl/`: primary training configs used by launch scripts.
- `configs/data/`: on-policy dataset/runtime source settings.
- `configs/runtime/`: centralized runtime policy JSON defaults.
- `configs/model/`: optional delimiter override files by model family.
  Used by `src/config.py::resolve_model_config_path()` before bundled defaults.
  In this repo, bundled families are mirrored here so runtime uses local copies.
- `configs/experiments/`: not used by current runtime and intentionally omitted.
