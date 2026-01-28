# Hypothesis: <ID> — <short title>

## Status
- status: `pending`
- owner: <name/initials>
- last updated: 2026-01-25

## One-liner
<1 sentence: what claim are we testing?>

## What we test (operationalization)
- **label**: `hallucinated` / `incorrect` / `correct` (define exactly)
- **signal(s)**: `sink_mass`, `sink_by_layer`, `sink_by_layer_head`, `entropy`, …
- **dataset(s)**: which tasks/splits (truthfulqa_mc/halueval/freshqa_false_premise/mmlu)

## Protocol (frozen for this hypothesis)
- **K (sink_tokens)**: <int>
- **Q (query)**: `last` OR `range` + `query_start=<int>`
- **chat**: `auto|off|on`
- **samples**: <int> (and seed)
- **quantization**: `none|4bit|8bit`

## Methodology (reasonable + limitations)
- <what statistical test/metric we use and why>
- **Known simplifications**: <what we intentionally ignore and why it’s acceptable for MVP>

## Acceptance / rejection criteria
- Accept if: <clear quantitative + qualitative criteria>
- Reject if: <clear criteria>

## Artifacts
- Data: `data/` (expected file patterns)
- Plots: `plots/` + `plots/manifest.json`

## Commands
See `scripts/`.

