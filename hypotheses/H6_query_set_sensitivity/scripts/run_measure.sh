#!/usr/bin/env bash
set -euo pipefail

# H6: compare query modes (last vs range) with fixed K/chat.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HYP_DIR="$ROOT/hypotheses/H6_query_set_sensitivity"
OUT_DIR="$HYP_DIR/data"

mkdir -p "$OUT_DIR"

run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  elif command -v python >/dev/null 2>&1; then
    python "$@"
  else
    python3 "$@"
  fi
}

MODEL="Qwen/Qwen2.5-7B-Instruct"
TASK="truthfulqa_mc"
SPLIT="validation"
SAMPLES=500
SEED=42
K=4
CHAT=auto

echo "== last =="
run_py "$ROOT/scripts/measure_sink_text.py" \
  --task "$TASK" --split "$SPLIT" --model "$MODEL" \
  --samples "$SAMPLES" --seed "$SEED" \
  --sink_tokens "$K" --query_mode last --chat "$CHAT" \
  --out_dir "$OUT_DIR"

for p in 16 32 64; do
  echo "== range start=$p =="
  run_py "$ROOT/scripts/measure_sink_text.py" \
    --task "$TASK" --split "$SPLIT" --model "$MODEL" \
    --samples "$SAMPLES" --seed "$SEED" \
    --sink_tokens "$K" --query_mode range --query_start "$p" --chat "$CHAT" \
    --out_dir "$OUT_DIR"
done

