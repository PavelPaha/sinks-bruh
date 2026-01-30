# H0 — Sanity (auto)

## What was run
- inputs: 1 run file(s)
- rows (pooled): 100
- label: `incorrect` (positive rate=0.660)

## Basic checks
- accuracy (if present): 0.340
- sink_mass: mean=0.3711, std=0.0124, range=[0.3430, 0.4021]

## Strongest signals (sanity)
Largest |Cohen's d| across runs (sink vs label):
- mmlu / Qwen/Qwen2.5-0.5B-Instruct: d=+0.121 (n=100)

Largest |Spearman ρ| across runs (sink vs entropy):
- mmlu / Qwen/Qwen2.5-0.5B-Instruct: ρ=-0.458 (n=100)

## Local analysis artifacts
- per-run diagnostics: `plots/runs/`
- aggregated summaries: `plots/agg/`
- per-run table: `runs_summary.csv`

## Status
- outcome: **ready** (data format ok; plots build)
