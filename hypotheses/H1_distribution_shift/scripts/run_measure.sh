#!/usr/bin/env bash
set -euo pipefail

# H1: run a small set of (task × model) jobs into hypotheses/H1_distribution_shift/data
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HYP_DIR="$ROOT/hypotheses/H1_distribution_shift"
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

# Edit this list as you iterate.
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)

TASKS=(
  "truthfulqa_mc:validation"
  "halueval:test"
  "freshqa_false_premise:test"
)

K=4
CHAT=auto
QUERY_MODE=last
SAMPLES=500
SEED=42
QUANT=none

for model in "${MODELS[@]}"; do
  for ts in "${TASKS[@]}"; do
    task="${ts%%:*}"
    split="${ts##*:}"
    echo "== model=$model task=$task split=$split =="
    run_py "$ROOT/scripts/measure_sink_text.py" \
      --task "$task" \
      --split "$split" \
      --model "$model" \
      --samples "$SAMPLES" \
      --seed "$SEED" \
      --sink_tokens "$K" \
      --query_mode "$QUERY_MODE" \
      --chat "$CHAT" \
      --quantization "$QUANT" \
      --out_dir "$OUT_DIR"
  done
done

