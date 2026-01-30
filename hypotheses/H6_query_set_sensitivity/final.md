# H6 — Query-set sensitivity (auto)

## Claim being tested
The sink signal depends on the query set Q (e.g., last-token vs tail-range) and on `query_start`.

## What was run
- inputs: 32 run file(s)
- runs analyzed: 32

## Results
- metrics table: `plots/metrics.json`
- best seen:
- best AUROC=0.666 at query_mode=last, query_start=0 (task=truthfulqa_mc, model=Qwen/Qwen2.5-7B-Instruct)

Key curves:
- `plots/auroc_vs_query.png`
- `plots/mean_sink_vs_query.png`

## Extra diagnostics
- per-run: `plots/runs/`
- aggregate: `plots/agg/` (table: `runs_summary.csv`)

## Status
- outcome: **preliminary** (needs more query_start sweep + other models/tasks)
