# H0 — Sanity: метрики/лейблы корректны

## Status

- status: `running`
- last updated: 2026-01-25

## One-liner

Прежде чем интерпретировать sink-графики, проверяем, что **лейблы и метрики корректны**, результаты не вырождены и plot-пайплайн воспроизводим.

## What we test (operationalization)

- **label**
  - `correct`: A/B/C/D по логитам (`_predict_choice_from_logits` в `scripts/measure_sink_text.py`)
  - `hallucinated`:
    - `halueval`, `freshqa_false_premise`: из датасет-лейбла
    - `truthfulqa_mc`: operationally `hallucinated = not correct` (в этом репо так и сделано)
- **signals**: `sink_mass`, `entropy`, `sink_by_layer` (L), `sink_by_layer_head` (L×H)
- **datasets**: берём 1–2 существующих run-файла из `data/` (или из общего `artifacts/sink_runs/`)

## Protocol (frozen)

Используем **как записано в run-файлах** (не меняем), потому что цель — sanity.

## Methodology

Проверяем:

- наличие ключей в jsonl-строках
- диапазоны/NaN:
  - `sink_mass` должен быть finite в большинстве строк
  - `entropy` finite
  - `sink_by_layer_head` имеет одинаковую форму (L×H) во всех строках run
- sanity по метке:
  - доля `correct` не должна быть ~0% или ~100% (для нормального датасета и модели)
  - если есть `hallucinated`, оба класса присутствуют

## Acceptance criteria

- plot-скрипты успешно строят фигуры по run-файлам
- нет явных артефактов типа “всё NaN”/“все метки одинаковые”

## Artifacts

- `data/`: положить 1–2 `.jsonl.gz` (симлинки не используем; копируем файл)
- `plots/`: сюда пишем графики sanity (можно просто дублировать общие plot’ы для одного run)
- `plots/manifest.json`: индекс графиков

## Commands

См. `scripts/`.
