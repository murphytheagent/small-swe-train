# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Python package. Key areas include `src/schemas/` (JSON schema contracts), `src/rollout/` (ChatML parsing and rollout collection), `src/data/` (canonicalization and adapters), `src/teacher/` (prompt builders), and `src/trainer/` (RFT/SDPO scaffolds and runtime loop). 
- `tests/` contains pytest suites; most tests follow `tests/test_*.py` naming.
- `configs/` stores runtime defaults and training configs; see `configs/README.md` and `configs/runtime/training_policy_defaults.v1.json`.
- `scripts/` provides launch helpers for RFT/SDPO and utilities (e.g., `scripts/run_rft.sh`, `scripts/run_sdpo.sh`).
- `data/`, `assets/`, `outputs/`, and `benchmarks/` contain fixtures, media, run artifacts, and benchmark assets.
- Root docs are limited to `AGENTS.md`, `README.md`, and `STATUS.md`; keep all other Markdown documentation under `docs/`.
- `STATUS.md` is the current status and todo tracker. Design history and implementation notes live under `docs/` (starting with `docs/design.md`).

## Build, Test, and Development Commands
- `make build-train CORES=2` — syncs training deps with `uv` and verifies `flash-attn`.
- `make build-dev` — installs dev-only deps (pytest).
- `uv sync --python 3.13 --extra train` — direct env setup (Python 3.11+ required, 3.13 is the default in `Makefile`).
- `uv run python -m pytest -q` — run the test suite.
- All launcher scripts assume execution under Slurm (submit or `srun` on a compute node). Use them only in a Slurm session.
- `bash scripts/run_rft.sh --dry-run trainer.total_training_steps=1` — validate the RFT loop config.
- `bash scripts/run_rft.sh` — full on-policy RFT loop (collector → rejection → parquet handoff → trainer → checkpoint → vLLM restart).
- `bash scripts/run_sdpo.sh trainer.total_training_steps=2` — SDPO runtime (expects Slurm and a writable `RAY_TMPDIR`).

## Coding Style & Naming Conventions
- Python style only; no formatter is enforced. Use 4-space indentation and follow existing file patterns.
- Prefer explicit type hints for public interfaces and configuration payloads.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Keep configuration keys aligned with existing JSON/YAML schemas in `configs/` and `src/schemas/`.

## Testing Guidelines
- Framework: `pytest` (see `pyproject.toml` for options; `tests/` is the default root).
- Name tests `test_*.py` and keep fixtures close to the tests that use them.
- Run focused tests with `uv run python -m pytest tests/test_onpolicy_rollout_adapter.py -k format_valid`.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short, imperative, sentence case (e.g., “Fix turn-span fallback”). Avoid noisy prefixes unless needed.
- PRs should include a concise summary, testing notes (`uv run python -m pytest ...`), and links to related issues or plans when applicable.
- For runtime changes, call out config keys touched and any new environment variables (for example, `SMALL_SWE_VLLM_*`).

## Documentation Guidelines
- Keep `README.md` focused on orientation, setup, and launch commands.
- Keep `STATUS.md` focused on active status, next actions, and known blockers.
- Start feature work by recording the active item in `STATUS.md`, and end feature work by updating `STATUS.md` with the completed outcome and any remaining blockers.
- Put research notes, migration plans, evaluation plans, and implementation design packets in `docs/`.
- When moving or adding docs, update relative links in `README.md`, `STATUS.md`, `AGENTS.md`, and neighboring docs.
