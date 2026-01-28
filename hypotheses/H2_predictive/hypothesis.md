# H2 — Predictive: sink_mass предсказывает hallucinations

## Status
- status: `pending`
- last updated: 2026-01-25

## One-liner
`sink_mass` обладает предсказательной силой для `hallucinated` (или `incorrect` как fallback), измеряемой AUROC/AUPRC.

## What we test
- **label**: `hallucinated` (primary), fallback `incorrect=~correct`
- **signal**: `sink_mass` (scalar)
- **datasets**: hallucination-таски (TruthfulQA/HaluEval/FreshQA)

## Protocol
- фиксируем \(K,Q,chat\) и сравниваем модели

## Methodology
- считаем AUROC/AUPRC на одном split’е (без подгона порога)
- сравниваем с random baseline

## Note
В текущем репо есть “accuracy vs sink” curves, но **нет отдельного AUROC‑скрипта**. Для H2 нужно будет добавить простой анализатор (или сделать в ноутбуке).

