#!/usr/bin/env bash
set -euo pipefail

# H5: run the same (model×task) twice: chat=auto vs chat=off.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HYP_DIR="$ROOT/hypotheses/H5_chat_sensitivity"
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
QUERY_MODE=last
QUANT=none

for CHAT in auto off; do
  echo "== chat=$CHAT =="
  run_py "$ROOT/scripts/measure_sink_text.py" \
    --task "$TASK" --split "$SPLIT" --model "$MODEL" \
    --samples "$SAMPLES" --seed "$SEED" \
    --sink_tokens "$K" --query_mode "$QUERY_MODE" \
    --chat "$CHAT" --quantization "$QUANT" \
    --out_dir "$OUT_DIR"
done

