# H5 — Chat/system sensitivity: chat-template меняет выводы

## Status
- status: `pending`
- last updated: 2026-01-25

## One-liner
В chat‑моделях “первые токены” включают системный префикс, поэтому эффекты sink могут быть **артефактом протокола** (chat on/off) или качественно меняться.

## What we test
- сравниваем одни и те же гипотезы H1/H4 (и позже H2/H3) между:
  - `chat=auto` (или `on`)
  - `chat=off` (raw prompts)
- datasets: hallucination‑таски

## Protocol
- фиксируем K и Q
- меняем только `chat`

## Methodology
- запускаем кампанию/подмножество из `configs/sink_campaign.json` и `configs/sink_campaign_nochat.json`
- сравниваем:
  - знак/величину Cohen’s d (H1)
  - структуру heatmap (H4)

