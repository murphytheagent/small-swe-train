# Minimal Experiment Matrix (v2)

Generated: 2026-02-21 21:37 UTC
Thread: 1771579678.414229

Research-mode exit checklist requires at least one minimal run plus one ablation run.

## Run A: minimal reference
- Pipeline: `RFT -> short SDFT -> SDPO`
- Output contract: optional `<think>` + one `<tool_call>` JSON block
- Terminal action: `answer`
- Teacher: EMA (`beta=0.005`)
- Top-K: `K=100`
- Truncation: `H=768`, `T=768`
- Adaptation: LoRA attention projections only

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
