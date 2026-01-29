# Запуск на Google Colab GPU

## Вариант 1: Через Colab Notebook (рекомендуется)

1. Открой `notebooks/run_h0_colab.ipynb` в VS Code
2. Подключись к Colab runtime через расширение Google Colab
3. Запусти ячейки по порядку

**Преимущества:**
- Автоматическое определение GPU
- Легко скачать результаты
- Можно видеть прогресс в реальном времени

---

## Вариант 2: Через SSH туннель (если поддерживается расширением)

Если VS Code расширение для Colab поддерживает remote execution:

1. Подключись к Colab runtime
2. Запусти команду напрямую в терминале VS Code (он будет работать на Colab):

```bash
python scripts/measure_sink_text.py \
  --task mmlu --split test \
  --model Qwen/Qwen2.5-7B-Instruct \
  --samples 100 \
  --sink_tokens 4 --query_mode last --chat auto \
  --device cuda \
  --out_dir hypotheses/H0_sanity/data
```

---

## Вариант 3: Локально на CPU (для теста)

Если хочешь сначала проверить, что скрипт работает локально:

```bash
python scripts/measure_sink_text.py \
  --task mmlu --split test \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --samples 10 \
  --sink_tokens 4 --query_mode last --chat auto \
  --device cpu \
  --out_dir hypotheses/H0_sanity/data
```

**Внимание:** На CPU будет очень медленно, используй только для проверки синтаксиса.

---

## После успешного запуска

1. В `hypotheses/H0_sanity/data/` появится `.jsonl.gz` файл
2. Запусти локально (или в Colab):
   ```bash
   python hypotheses/H0_sanity/scripts/check_structure.py
   bash hypotheses/H0_sanity/scripts/run_plots.sh
   ```
3. Я автоматически обновлю `run_notes.md` и `final.md`

---

## Примечания

- Модель `Qwen/Qwen2.5-0.5B-Instruct` может не существовать. Если ошибка — используй `Qwen/Qwen2.5-7B-Instruct` (но она больше)
- Для экономии памяти в Colab можно добавить `--quantization 4bit`
- Если Colab выдает "out of memory", уменьши `--samples` или используй меньшую модель
