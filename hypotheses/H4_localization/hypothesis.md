# H4 — Localization: эффект локализован по слоям/головам

## Status

- status: `pending`
- last updated: 2026-01-25

## One-liner

Разница sink-attention между классами (hallucinated vs non-hallucinated / incorrect vs correct) **концентрируется** в subset голов/слоёв, а не “размазана” равномерно.

## What we test

- **label**: `hallucinated` (если доступен) иначе `incorrect = ~correct`
- **signal**: `sink_by_layer_head` (L×H), опционально `sink_by_layer` (L)
- **datasets**: hallucination-таски (`truthfulqa_mc`, `halueval`, `freshqa_false_premise`)

## Protocol (baseline)

- `K=4`
- `Q=last`
- `chat=auto` (+ ablation `chat=off`)

## Methodology

- Строим \(\Delta_{l,h} = \mathbb{E}[\text{sink}_{l,h} | pos] - \mathbb{E}[\text{sink}_{l,h} | neg]\)
- Визуализируем heatmap (Layer × Head).
- Дополнительно: “layer profile” — как средняя sink масса меняется по глубине.

**Упрощение:** пока без формального статистического теста по каждому head (иначе проблема множественных проверок). Для MVP: описательная + reproducibility across runs/models.

## Acceptance criteria

- В heatmap’ах видны стабильные “спайки” (top-|Δ| heads), повторяющиеся между:
  - близкими моделями (одно семейство), и/или
  - разными hallucination-тасками.

## Commands

См. `scripts/`.
