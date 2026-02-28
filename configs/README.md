# Config Layout

This repo intentionally keeps local configs minimal.

## Active Entrypoints

- `configs/verl/sdpo_swe.yaml` is the SDPO training entry config.
- `configs/verl/rft_swe.yaml` is the RFT training entry config.

Both are launched by scripts in `scripts/` with:
- `--config-dir configs/verl`
- `--config-name sdpo_swe` or `rft_swe`

## Hydra Composition (Current)

- `sdpo_swe.yaml` defaults: `model_defaults`, `ppo_trainer`, `_self_`
- `rft_swe.yaml` defaults: `model_defaults`, `sft_trainer`, `_self_`

`ppo_trainer` / `sft_trainer` come from installed `verl` package configs.

## Length Fields (SDPO)

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
