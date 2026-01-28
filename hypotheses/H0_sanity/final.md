# H0 — Final (sanity)

## Summary
- outcome: `pending` (waiting for data)
- what was validated:
  - ✅ Структура скриптов корректна (`check_structure.py`, `run_plots.sh`)
  - ✅ Шаблоны файлов на месте (`hypothesis.md`, `run_notes.md`, `final.md`, `plots/manifest.json`)
  - ⏳ Нет входных данных для проверки (нужен хотя бы один `.jsonl.gz` в `data/`)
- what remained broken/unclear:
  - Нет реальных run-файлов → нельзя проверить корректность меток/NaN/диапазонов
  - Нельзя построить графики без данных

## Evidence (artifacts)
- runs: none
- plots: none (требуют входные данные)

## Implications
- what hypotheses are safe to test next:
  - **H0 должна быть завершена первой** (проверка структуры данных) перед H1/H4/H2/H3
  - После получения хотя бы одного `.jsonl.gz` можно:
    1. Проверить наличие ключей (`sink_mass`, `entropy`, `correct`, `sink_by_layer_head`)
    2. Проверить отсутствие массовых NaN
    3. Построить sanity-графики (accuracy vs sink, h1, layer profiles)
    4. Перейти к H1
- what protocol constraints we must respect going forward:
  - Все последующие гипотезы должны использовать **те же протоколы** (K, Q, chat), что и H0, если не указано иное
  - Run-файлы должны содержать `sink_by_layer_head` для H4 (localization)

