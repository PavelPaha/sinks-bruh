# H7 — Sink mass ↔ Entropy (auto)
## Claim being tested
`entropy` (next-token uncertainty) is statistically linked to `sink_mass`.
## What was run
- run files: 32
- analyzed runs (min_rows>=50): 26
## Summary across models (median correlations)
- **truthfulqa_mc** (K1, q=last, chat=auto): median Spearman ρ=-0.027, Pearson r=+0.026, partial(ρ|seq_len)=-0.042 over n_models=7
- **truthfulqa_mc** (K1, q=range@32, chat=auto): median Spearman ρ=+0.025, Pearson r=+0.011, partial(ρ|seq_len)=+0.001 over n_models=6
- **truthfulqa_mc** (K4, q=last, chat=auto): median Spearman ρ=-0.065, Pearson r=-0.065, partial(ρ|seq_len)=-0.130 over n_models=7
- **truthfulqa_mc** (K4, q=range@32, chat=auto): median Spearman ρ=+0.024, Pearson r=+0.020, partial(ρ|seq_len)=-0.011 over n_models=6

## Stability under seq_len control
- median( partial - raw ): -0.029
- median(|partial - raw|): 0.029
Interpretation: if these are near 0, controlling for `seq_len` doesn't materially change the sink↔entropy conclusion for these runs.

## Artifacts
- metrics: `plots/metrics.json` (and `plots/metrics.csv`)
- manifest: `plots/manifest.json`
- shared diagnostics: `plots/runs/` and `plots/agg/` (notably `plots/agg/scatter_partial_vs_spearman.png`)

## Status
- outcome: **preliminary** (replicate across tasks, compare instruct vs base, sweep K/Q)
