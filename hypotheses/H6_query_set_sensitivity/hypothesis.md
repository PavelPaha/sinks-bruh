# H6 — Q sensitivity: “не смотреть с первых токенов” меняет эффект

## Status
- status: `pending`
- last updated: 2026-01-25

## One-liner
Если первые токены промпта “ещё без контекста”, то измерять sink лучше из хвоста (tail). Проверяем, что смена \(Q\) (query set) меняет силу/стабильность эффекта.

## What we test
- **label**: `hallucinated` (primary), fallback `incorrect`
- **signals**: `sink_mass`, `sink_by_layer_head`
- **manipulation**: `query_mode=last` vs `query_mode=range --query_start p`

## Protocol
- фиксируем `K` и `chat`, меняем только \(Q\)
- рекомендуемые p: 16/32/64 (в зависимости от длины промпта)

## Methodology
- сравниваем H1 (Cohen’s d) и H4 (heatmap) между режимами
- критерий “лучше”: эффект сильнее и/или стабильнее (меньше разброс между моделями/тасками)

