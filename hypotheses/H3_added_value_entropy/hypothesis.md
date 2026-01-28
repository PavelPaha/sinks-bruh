# H3 — Added value: sink даёт сигнал сверх entropy

## Status
- status: `pending`
- last updated: 2026-01-25

## One-liner
Даже контролируя неопределённость (`entropy`), `sink_mass` даёт дополнительную информацию о `hallucinated`.

## What we test
- **label**: `hallucinated` (primary)
- **features**: `entropy`, `sink_mass`
- **datasets**: hallucination-таски

## Protocol
- фиксируем \(K,Q,chat\) для честного сравнения

## Methodology (paper-style)
- сравниваем две модели:
  - A: `label ~ entropy`
  - B: `label ~ entropy + sink_mass`
- метрики: log-loss / AUROC (bootstrap CI для Δ)

## Note
В репо уже есть scatter/corr `sink↔entropy` (`scripts/plot_sink_entropy.py`), но для H3 нужен отдельный анализатор (или ноутбук).

