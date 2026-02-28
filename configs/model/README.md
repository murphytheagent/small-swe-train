# Model Delimiter Overrides

This directory is intentionally checked in as the runtime source of truth for
model-family delimiter configs used by this repo.

- path pattern: `configs/model/<family>.yaml`
- example family key: `qwen3`

Resolution order at runtime:

1. `configs/model/<family>.yaml` (repo override, preferred)
2. bundled `src/prompts/model_configs/<family>.yaml` (fallback only)

Policy in this repo:

- every bundled model delimiter config should have a mirrored file here.
- mirrored files should stay identical unless we intentionally diverge.
