# Hypotheses: Attention Sinks ↔ Hallucinations (2026-01-25)

## Контекст
Хочется сфокусироваться на основном тейке: **паттерны sink-attention связаны с галлюцинациями**.
Мы проверяем это на широком спектре моделей (0.5B – 72B) и датасетов (hallucination detection vs control tasks).

---

## Операциональные определения (чтобы не спорить словами)

### Что такое “галлюцинация” в экспериментах
Нужно привязать “hallucination” к измеряемой метке.

#### Вариант A (рекомендуется для старта): TruthfulQA
- Берём TruthfulQA в формате multiple-choice (если используем MC) или scoring-метку “truthful”.
- **hallucinated = 1**, если ответ **неправильный/неправдивый** по метке.
- **hallucinated = 0**, если ответ **правильный/правдивый**.

#### Вариант B: HaluEval
- Специализированный датасет на детекцию.
- **hallucinated = 1**, если лейбл "FAIL" (not supported).
- **hallucinated = 0**, если лейбл "PASS" (supported).

#### Вариант C: FreshQA (False Premise)
- **hallucinated = 1**, если вопрос содержит ложную предпосылку (False Premise), а модель пытается на него ответить.
- **hallucinated = 0**, если предпосылки нет (и модель отвечает верно).

### Что такое sink mass (в общем виде)
Определяем как агрегат внимания на первые \(K\) позиций (sink window) при выборе набора query-позиций \(Q\):

\[
s(Q,K) = \frac{1}{|Q|}\sum_{i\in Q}\sum_{j=0}^{K-1}\bar{A}[i,j]
\]

Где \(\bar{A}\) — attention, усреднённый по слоям/головам (либо анализируется по слоям/головам отдельно).

**Базовый протокол:** \(Q=\{S-1\}\) (last-token), \(K=4\) (и абляции по \(K\)).

---

## Preliminary Findings (на 25.01.2026)

На основе первых запусков (Qwen 0.5B-32B, TinyLlama):

1. **Scaling Inversion (Эффект меняет знак с масштабом):**
   - На FreshQA (False Premise):
     - **Qwen-7B/14B:** Sink Mass **падает** при галлюцинации (`d ~ -0.5..-0.9`). Модель "теряет фокус" на контексте?
     - **Qwen-32B:** Sink Mass **растет** при галлюцинации (`d ~ +0.8`). Модель жестко фокусируется на sink (safety refusal pattern?).
   - Это ключевой инсайт для статьи: нельзя просто сказать "sink растет", всё зависит от capability модели.

2. **Task Dependency:**
   - **TruthfulQA:** Сигнал преимущественно отрицательный (меньше sink mass = больше вранья).
   - **MMLU (Control):** Сигнал слабый позитивный (больше sink mass = больше ошибок). Это напоминает "Sink-Confidence Paradox" (неуверенность -> sink).

3. **Локализация:**
   - Эффект не равномерный. Heatmap показывает, что разница (Delta) сосредоточена в специфических головах (induction heads?), а не размазана по всей модели.

---

## Гипотезы (sink ↔ hallucination)

### H1 — Distribution shift: sink mass отличается на галлюцинациях
**Гипотеза:** при фиксированной модели/протоколе распределение `sink_mass` на `hallucinated=1` отличается от `hallucinated=0`.

**Тесты:**
- разница средних/медиан + bootstrap CI
- effect size (например Cohen’s d)
- визуализация: KDE/violin + доверительные интервалы

**Комментарий:** это “безопасная” гипотеза: не требует утверждать причинность.

### H2 — Predictive: sink mass предсказывает галлюцинации как классификатор
**Гипотеза:** `sink_mass` даёт ненулевую предсказательную силу для `hallucinated`.

**Метрики:**
- AUROC / AUPRC
- калибровка: Accuracy vs Sink Mass curves

### H4 — Localization: эффект локализован по слоям/головам
**Гипотеза:** разница sink-attention между hallucinated и non-hallucinated концентрируется в определённых слоях или головах (induction heads?).

---

## Визуализация и Анализ (Pipeline 2.0)

Мы построили модульный пайплайн: `measure` -> `jsonl.gz` -> `plot`.

### 1. Сбор данных
Скрипт: `scripts/measure_sink_text.py`
- Прогоняет одну модель на одном датасете.
- Сохраняет сжатый JSONL (`.jsonl.gz`) с полями: `sink_mass`, `entropy`, `correct`, `hallucinated`, и матрицы `sink_by_layer`, `sink_by_layer_head`.
- Поддерживает квантизацию (4bit/8bit) для больших моделей.

### 2. Графики (Paper-Grade Plots)

| Скрипт | Выходной график | Описание |
|--------|-----------------|----------|
| `scripts/plot_h1.py` | `*_h1.png` | **Distribution Shift:** KDE + Violin plot распределений Sink Mass для hallucinated vs correct. Считает Cohen's d и 95% CI. |
| `scripts/plot_h1_heatmap.py` | `*_h1_heatmap.png` | **Localization:** Хитмапа (Layer × Head) разницы `Δ = Sink(Hallucinated) - Sink(Correct)`. Показывает, в каких головах возникает аномалия. |
| `scripts/plot_accuracy_vs_sink_runs.py` | `*_accuracy_vs_sink.png` | **Calibration:** Кривая зависимости точности от Sink Mass (бинирование по квантилям). Показывает "Sink-Confidence Paradox". |
| `scripts/plot_layer_profiles.py` | `*_layer_profile.png` | **Depth Profile:** Как меняется Sink Mass по глубине сети (от 1-го до последнего слоя). Сравнение разных моделей на одной оси (0..1 relative depth). |

---

## Текущий набор моделей и датасетов

### Датасеты
1. **TruthfulQA (MC)**: основной бенчмарк на правдивость. (hallucinated = incorrect)
2. **HaluEval**: детекция галлюцинаций в RAG-контексте. (hallucinated = FAIL)
3. **FreshQA (False Premise)**: устойчивость к ложным предпосылкам.
4. **MMLU**: контроль (General Knowledge).

### Модели (Text-Only)
Мы используем широкий спектр открытых моделей разных размеров (0.5B – 72B):

- **Qwen2.5:** 0.5B, 1.5B, 7B, 14B, 32B, **72B** (Instruct)
- **LLaMA-3.2:** 1B, 3B (Instruct)
- **Gemma-2:** 2B, 9B, 27B (Instruct)
- **Mistral:** 7B (v0.3), Nemo (12B), Mixtral 8x7B (MoE)
- **Другие:** TinyLlama-1.1B, OpenLLaMA, Phi-2, Pythia.

---

## Обязательные абляции (Robustness)

- **Prompting / chat-template:** `chat=off` vs `chat=on` (проверка, не является ли sink артефактом системного промпта).
- **Sink window \(K\):** \(K \in \{1,4\}\). Мы используем **K=4** как дефолт, так как "sink" часто размазан по первым 3-4 токенам (Xiao et al., 2024).
- **Quantization:** Сравнение fp16 vs 4bit (для больших моделей используется 4bit NF4).

---

## Дополнительные наблюдения

### Sink токены — не только первый токен!
Эмпирическое наблюдение: attention может скапливаться не только на токеене 0 (`<bos>`), но на **первых 3-4 токенах**. Это согласуется с литературой ("Mirage in the Eyes", Wang et al., 2025).

### Связь sink → hallucination
Статья "Mirage in the Eyes" (arXiv:2501.15269) показывает, что Sink появляется на "переломе" релевантности — после sink токена качество ответа падает. Наша гипотеза H4 (heatmap) должна это подтвердить локализацией в средних/поздних слоях.
