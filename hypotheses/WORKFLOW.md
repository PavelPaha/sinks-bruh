# Workflow: как мы проверяем гипотезы

## Цикл работы (для каждой гипотезы)

### 1. Подготовка (я делаю)
- ✅ Структура папок создана
- ✅ `hypothesis.md` с формулировкой и протоколом
- ✅ `scripts/` с командами запуска
- ✅ Шаблоны `run_notes.md` и `final.md`

### 2. Запуск measurement (ты делаешь или я по твоей команде)
```bash
# Пример для H1:
bash hypotheses/H1_distribution_shift/scripts/run_measure.sh
# Или вручную:
python scripts/measure_sink_text.py \
  --task truthfulqa_mc --split validation \
  --model Qwen/Qwen2.5-7B-Instruct \
  --samples 500 --sink_tokens 4 --query_mode last --chat auto \
  --out_dir hypotheses/H1_distribution_shift/data
```

### 3. Построение графиков (я делаю автоматически)
```bash
bash hypotheses/H1_distribution_shift/scripts/run_plots.sh
```

### 4. Анализ и заполнение документов (я делаю)
- Запускаю `auto_update_notes.py` (если есть) → заполняет `run_notes.md` с цифрами
- Читаю графики из `plots/` → интерпретирую
- Обновляю `final.md` с выводом (paper-style)

### 5. Ты проверяешь и уточняешь
- Смотришь `run_notes.md` и `final.md`
- Если нужно — просишь доработать интерпретацию
- Переходим к следующей гипотезе

---

## Что я делаю автоматически

### После получения `.jsonl.gz` файлов:
1. **Валидация структуры** (H0):
   - Проверяю наличие ключей (`sink_mass`, `entropy`, `correct`, `sink_by_layer_head`)
   - Проверяю отсутствие массовых NaN
   - Проверяю баланс меток

2. **Построение графиков**:
   - Запускаю соответствующие plot-скрипты
   - Сохраняю в `plots/` + обновляю `plots/manifest.json`

3. **Извлечение метрик**:
   - Cohen's d (H1)
   - AUROC/AUPRC (H2)
   - Log-loss comparison (H3)
   - Heatmap summaries (H4)

4. **Заполнение документов**:
   - `run_notes.md`: цифры + краткие наблюдения
   - `final.md`: итоговая интерпретация (paper-style)

---

## Что нужно от тебя

1. **Запустить measurement** (если данных нет):
   - Используй `scripts/run_measure.sh` или запусти `measure_sink_text.py` вручную
   - Положи `.jsonl.gz` в `hypotheses/<HX>/data/`

2. **После того как я заполнил `run_notes.md` и `final.md`**:
   - Проверь интерпретацию
   - Если нужно — попроси уточнить/переформулировать
   - Скажи, переходим ли к следующей гипотезе

---

## Текущий статус

- **H0 (sanity)**: структура готова, данных нет → нужен хотя бы один run
- **H1 (distribution shift)**: структура готова, auto-update скрипт готов, данных нет
- **H2–H6**: структура готова, ждут своей очереди

**Следующий шаг**: запустить measurement для H0/H1 (хотя бы 1 модель × 1 таск).
