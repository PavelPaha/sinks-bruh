# H3 — Added value over entropy (train/test)

## Claim being tested
`sink_mass` adds predictive value beyond next-token `entropy`.

## Protocol
- eval_mode: `holdout` (test_frac=0.3, repeats=5)
- bootstrap: 300

## What we fit
- A: logistic regression `y ~ zscore(entropy)` trained on train, evaluated on test
- B: logistic regression `y ~ zscore(entropy) + zscore(sink_mass)` trained on train, evaluated on test

## Results
We report mean Δ over runs (positive => sink helps):
- mean Δlogloss: -0.0029
- mean ΔAUROC: -0.0124

Interpretation: **does not support (no added value or unstable)** (check per-run + per-repeat in `plots/metrics.json`).

## Artifacts
- report: `plots/metrics.json`
- aggregated plots: `plots/agg/`

## Extra diagnostics
- per-run: `plots/runs/`
- aggregate: `plots/agg/` (table: `runs_summary.csv`)
