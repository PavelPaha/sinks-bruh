# H1 — Distribution shift: sink_mass отличается на hallucinations

## Status

- status: `pending`
- last updated: 2026-01-25

## One-liner

На hallucination-тасках распределение `sink_mass` статистически отличается между `hallucinated=1` и `hallucinated=0` (или хотя бы показывает устойчивую “scaling inversion”).

## What we test

- **label (primary)**: `hallucinated`
  - `halueval`, `freshqa_false_premise`: из датасет-лейбла
  - `truthfulqa_mc`: operationally `hallucinated = not correct`
- **signal**: `sink_mass` (и опционально `sink_by_layer`, чтобы понимать “где” эффект)
- **datasets**: `truthfulqa_mc`, `halueval`, `freshqa_false_premise` (MMLU — контроль)

## Protocol (frozen baseline)

- `K=4` (`--sink_tokens 4`)
- `Q=last` (`--query_mode last`)
- `chat=auto` (и отдельная абляция `chat=off`)
- `seed=42`

## Methodology (reasonable + limitations)

- Основная метрика эффекта: **Cohen’s d** (как в `scripts/plot_h1.py`).
- Визуализация: violin/box (разность распределений).
- **Упрощение**: мы трактуем `truthfulqa_mc` как hallucination через `hallucinated = not correct` (это proxy, не “free-form hallucination”).

## Acceptance criteria

- На ≥2 hallucination-тасках эффект |d| ≥ 0.2 на нескольких моделях, **или**
- наблюдается структурный паттерн “scaling inversion” (знак меняется с размером) и он воспроизводим при фиксированном протоколе.

## Artifacts

- `data/`: `.jsonl.gz` для выбранных моделей/тасков
- `plots/`: `h1_grid_<task>.png`, `h1_summary.csv`
- `plots/manifest.json`: индекс

## Commands

См. `scripts/`.
