# Minimal Experiment Matrix (v3)

Generated: 2026-02-21 06:24 UTC
Thread: 1771579678.414229

Research-mode exit checklist requires at least one minimal run plus one ablation run.

## Run A: minimal reference
- Pipeline: `RFT -> short SDFT -> SDPO`
- Output contract: optional `<think>` + `1..M` `<tool_call>` JSON blocks (`M=3` default)
- Terminal action: `submit`
- Teacher: EMA (`beta=0.005`)
- Top-K: `K=100`
- Truncation: `H=768`, `T=768`
- Adaptation: LoRA attention projections only
- Masking policy:
  - RFT masks out think tokens
  - step-SDPO trains think + tool-call tokens

Acceptance checks:
- Entry gate metrics satisfy thresholds over last `N=200` episodes.
- `Delta_rand(k) > 0` on selected checkpoints.
- No hard block from format regression.

## Run B: ablation
- Same as Run A except `SDFT disabled`.

Comparison target:
- Quantify difference in gate attainment speed and early SDPO stability.

## Optional sweep queue (after sign-off)
- EMA beta sweep: `0.001`, `0.005`, `0.01`
- Top-K schedule enabled vs fixed `K=100`
- Multi-tool cap sweep: `M=2` vs `M=3`
