# small-swe-train (Draft)

Initial implementation draft for training a local SWE agent from a 4B base model.

Scope of this draft:
- Stage 0: format bootstrapping (tool-call schema correctness)
- Stage 1: SDFT-style demo-conditioned on-policy distillation
- Stage 2: multi-turn Step-SDPO (feedback-conditioned self-distillation)
- Stage 3: optional terminal hindsight distillation for delayed errors

This repository is intentionally lightweight. It provides interfaces, data types, and
training-loop scaffolding so the next step can wire in a real model/runtime stack.

## Repository layout

```text
small_swe_train/
  config.py              # stage-level config dataclasses
  types.py               # trajectory/tool schema
  training/
    metrics.py           # metric sink helper
    stages.py            # stage runners + model/runtime protocols
configs/
  default.json           # runnable draft config
train.py                 # entrypoint for staged training run
```

## Run

```bash
python3 train.py --config configs/default.json
```

Expected behavior right now:
- config loading and stage orchestration run
- placeholder stage implementations log TODO markers for model/runtime hooks

## Core metrics to track

- `format.valid_action_rate`: parsed tool actions / total actions
- `format.invalid_schema_rate`: invalid JSON/XML schema outputs
- `sdft.reverse_kl`: reverse KL on student rollouts vs demo-conditioned teacher
- `sdft.demo_uplift`: delta in step success with demo-conditioning
- `sdpo.teacher_student_kl`: teacher vs student KL on action tokens
- `sdpo.step_fix_rate`: fraction of failed steps improved after feedback
- `sdpo.terminal_hindsight_gain`: success delta with terminal-feedback hindsight
- `env.episode_success_rate`: pass-rate on held-out SWE episodes

## Next milestone

1. Bind `PolicyModel` to your training stack (Transformers/vLLM/etc.).
2. Bind `SWEEnvironment` to local dockerized execution traces.
3. Replace placeholder losses with real tensor ops and optimizer steps.
