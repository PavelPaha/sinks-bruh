# H5 — Chat sensitivity (auto)

## Claim being tested
Changing chat formatting (chat template on/off) changes sink/label conclusions.

## What was run
- inputs: 16 run file(s)
- paired comparisons: 8

## Results
- metrics: `plots/metrics.json`
- Δd summary (off - auto): mean=+0.010, median=+0.010

Largest |Δd| pairs:
- truthfulqa_mc / Qwen/Qwen2.5-0.5B-Instruct: Δd=-0.113 (auto d=+0.169, off d=+0.055)
- truthfulqa_mc / mistralai/Mistral-7B-Instruct-v0.3: Δd=+0.104 (auto d=-0.205, off d=-0.101)
- truthfulqa_mc / mistralai/Mistral-Nemo-Instruct-2407: Δd=+0.048 (auto d=-0.105, off d=-0.057)

## Plots
- `plots/delta_cohens_d.png` (if generated)
- `plots/h1/h1_grid_*.png` (if generated)
- `plots/h1_heatmap/heatmap_grid_*.png` (if generated)

## Extra diagnostics
- per-run: `plots/runs/`
- aggregate: `plots/agg/` (table: `runs_summary.csv`)

## Status
- outcome: **preliminary** (needs more models/tasks)
