# H1 — Distribution shift (auto)

## Claim being tested
Sink mass distribution differs between positive class (hallucinated/incorrect) and negative (non-hallucinated/correct).

## What was run
- runs: 8
- label: `hallucinated`

## Results (Cohen's d)
- finite d: 6
- sign breakdown: 1 positive, 5 negative

### Strongest effects (by d)
Most negative (pos < neg):
- mistralai/Mistral-7B-Instruct-v0.3: d=-0.205 (n=415, pos_rate=0.429)
- microsoft/phi-2: d=-0.125 (n=415, pos_rate=0.542)
- mistralai/Mistral-Nemo-Instruct-2407: d=-0.105 (n=415, pos_rate=0.405)

Most positive (pos > neg):
- Qwen/Qwen2.5-0.5B-Instruct: d=+0.169 (n=415, pos_rate=0.737)
- EleutherAI/pythia-2.8b-deduped: d=-0.010 (n=415, pos_rate=0.766)
- TinyLlama/TinyLlama-1.1B-Chat-v1.0: d=-0.051 (n=415, pos_rate=0.742)

## Artifacts
- per-run table: `plots/metrics.json`
- manifest: `plots/manifest.json`

## Plots
- baseline: `plots/h1/` and `plots/h1_heatmap/`
- per-run diagnostics: `plots/runs/`
- aggregate analysis: `plots/agg/` (table: `runs_summary.csv`)

## Status
- outcome: **preliminary** (sign can flip across models; treat as a heterogeneity finding, not a single universal monotone law)
