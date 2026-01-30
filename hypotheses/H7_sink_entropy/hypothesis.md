## H7 — Sink mass ↔ Entropy (uncertainty) linkage

### One-liner
Проверяем, что **`sink_mass` статистически связан с `entropy`** (неопределённостью next-token), в разрезе (model × task × K × Q × chat).

### Claim (what we test)
Для фиксированных настроек измерения внимания:
- \(K\): `sink_tokens` (размер sink-окна в токенах),
- \(Q\): `query_mode` + `query_start` (какие query-токены агрегируем),

мы проверяем, что на одном и том же наборе примеров существует связь:
\[
\rho(\text{sink\_mass}, \text{entropy}) \neq 0
\]
и/или остаётся после контроля длины контекста:
\[
\rho(\text{sink\_mass}, \text{entropy}\;|\;\text{seq\_len}) \neq 0.
\]

### Why this matters (paper framing)
`entropy` — прокси неопределённости; если `sink_mass` систематически связан с `entropy`, то sink может служить косвенным маркером режимов “уверенности/неуверенности” и потенциально быть прокси качества/галлюцинаций (дальше связывается с H1–H3).

### Protocol (minimal, reproducible)
- **Within-run**: на уровне одного `*.jsonl.gz` (одна модель, один таск, фиксированные \(K,Q,chat\)):
  - Pearson \(r\) и Spearman \(\rho\) между `sink_mass` и `entropy`.
  - Partial correlation по `seq_len` (через регрессию-residuals).
- **Across models**: агрегируем распределение корреляций по моделям, сравниваем группы (семейства/квантизация/чат-режим) при одинаковых \(K,Q\).

### Outputs
В `plots/` сохраняются:
- `metrics.json` + `metrics.csv` (таблица per-run метрик)
- `corr_grid_<task>__K<...>__q<...>.png` (сеточные scatter-плоты по моделям)
- `corr_hist_<task>__K<...>__q<...>.png` (распределение корреляций по моделям)
- `manifest.json`

