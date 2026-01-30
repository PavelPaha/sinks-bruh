# H0 — Run notes

## Run 1 (2026-01-25)
- date: 2026-01-25
- run file(s): none (no data files found)
- protocol extracted from run meta: N/A
- quick checks:
  - keys present: N/A (no files to check)
  - NaN rate: N/A
  - label balance: N/A

## Observations
- Проверили структуру: скрипт `check_structure.py` работает корректно
- В `hypotheses/H0_sanity/data/` нет `.jsonl.gz` файлов
- В `artifacts/sink_runs/` тоже нет файлов
- Скрипты для plotting (`run_plots.sh`) готовы, но требуют входные данные

## Next step
**Нужно запустить минимальный measurement:**
```bash
python scripts/measure_sink_text.py \
  --task mmlu --split test \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --samples 100 \
  --sink_tokens 4 --query_mode last --chat auto \
  --out_dir hypotheses/H0_sanity/data
```

После этого:
1. Перезапустить `python hypotheses/H0_sanity/scripts/check_structure.py` для валидации структуры
2. Запустить `bash hypotheses/H0_sanity/scripts/run_plots.sh` для построения графиков
3. Обновить этот файл с реальными цифрами

