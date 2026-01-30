# H4 — Localization (auto)

## Claim being tested
The sink/label effect is localized to specific layers/heads (not uniform).

## What was run
- runs: 8
- inputs: 8 file(s)

## Results
Top heads by |Δ sink| per run are in `plots/metrics.json`.

Largest |Cohen's d| across runs (overall sink_mass shift):
- truthfulqa_mc / mistralai/Mistral-7B-Instruct-v0.3: d=-0.205 (n=415)
- truthfulqa_mc / Qwen/Qwen2.5-0.5B-Instruct: d=+0.169 (n=415)
- truthfulqa_mc / microsoft/phi-2: d=-0.125 (n=415)

## Plots
- heatmaps: `plots/h1_heatmap/`
- layer profiles: `plots/layers/`

## Extra diagnostics
- per-run: `plots/runs/`
- aggregate: `plots/agg/` (table: `runs_summary.csv`)

## Status
- outcome: **preliminary** (depends on stability across models/tasks)
