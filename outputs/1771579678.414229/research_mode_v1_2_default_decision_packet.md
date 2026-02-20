# small-swe-train: Research Mode v1.2 (Default Decision Packet)

Generated: 2026-02-20 23:47 UTC
Thread: 1771579678.414229

Purpose:
- Provide one recommended default for each unresolved v1.1 question.
- Reduce decision friction so research-mode can be frozen with a single approval/override pass.

## Proposed defaults (approve or override by number)

1. JSON schema finalization
- Default: keep one unified envelope with strict per-tool `args` schemas.
- Envelope:
  - `{"tool": "bash|search|edit|submit", "args": {...}}`
- Validation:
  - first validate envelope,
  - then validate `args` against the selected tool schema.

2. Invalid-output recovery policy
- Default: one deterministic parser-guided repair pass.
- If repair fails: mark step invalid and continue episode with explicit environment error feedback.
- No retry loops.

3. Format-stability gate (to enter main SDPO)
- Default thresholds over rolling last `N = 200` episodes:
  - `parse_valid_rate >= 0.985`
  - `allowed_tool_rate >= 0.995`
  - `required_arg_presence >= 0.985`
  - `single_object_rate >= 0.98` (exactly one JSON action object per step)

4. Self-contained feedback criterion (for include-attempt toggle)
- Default rule: feedback is self-contained iff all three are present:
  - failing artifact identity (test/cmd/file/function),
  - actionable error text,
  - localization hint (where to inspect/fix).
- If any is missing, include student attempt in teacher context.

5. Deterministic truncation constants
- Default: `H = 768`, `T = 768` tokens.
- Apply truncation after canonicalizing tool/env payload text and before adding teacher prompt wrappers.

6. Pre-main trainer mix
- Default: `RFT -> short SDFT -> SDPO`, with SDFT behind a config flag (`enabled=true/false`).
- Suggested short SDFT budget: 5-10% of pre-main tokens.

7. Teacher regularization in SDPO
- Default: EMA teacher.
- Update: `phi <- (1 - beta) * phi + beta * theta` with `beta = 0.005` initially.
- Tuning range for sweeps: `beta in [0.001, 0.01]`.

8. Top-K distillation
- Default: fixed `K = 100` in v1 (no schedule).
- Keep schedule support in config but disabled initially.

9. LoRA scope and precision
- Default LoRA targets: attention projections only (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- Precision default: bf16 compute + LoRA adapters; optional 4-bit base loading as a runtime switch.

10. Sandbox execution policy
- Default: Docker sandbox + command allowlist for v1.
- Start allowlist with read/write safe dev commands (`ls`, `cat`, `grep/rg`, `git diff`, test runners, formatters).
- Keep explicit denylist for destructive/network-sensitive commands.

11. Research-mode exit checklist
- Default required sign-off set before coding restart:
  - schema locked,
  - format gate numbers locked,
  - teacher-context inclusion rule locked,
  - truncation constants locked,
  - SDPO regularization and top-K locked,
  - sandbox policy locked,
  - benchmark/eval protocol locked,
  - experiment matrix approved (at least one minimal run + one ablation).

## Math/implementation implications

- Main stage objective remains unchanged:
- `L_step_sdpo = E[ sum_t sum_{n in M_t} KL( pi_theta(.|h_t,a_{t,<n}) || stopgrad(q_phi(.|h_t,f_{t+1},a_{t,<n})) ) ]`
- `M_t` continues to mask only action tokens.
- This packet only fixes unresolved policy constants/contracts, not algorithm direction.

## Proposed human reply format
- "Approve defaults 1-11"
- or "Approve all except 3=..., 5=..., 10=..."

After approval, next step is contract/metrics scaffolding only (no full training loop yet).
