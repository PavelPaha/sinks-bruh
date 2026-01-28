#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HYP_DIR="$ROOT/hypotheses/H6_query_set_sensitivity"

INPUTS=("$HYP_DIR"/data/*.jsonl.gz)
OUT_PLOTS="$HYP_DIR/plots"

if [ ! -e "${INPUTS[0]}" ]; then
  echo "No inputs in $HYP_DIR/data/*.jsonl.gz" >&2
  exit 1
fi

mkdir -p "$OUT_PLOTS"

run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  elif command -v python >/dev/null 2>&1; then
    python "$@"
  else
    python3 "$@"
  fi
}

run_py "$ROOT/scripts/plot_h1.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/h1" --label auto
run_py "$ROOT/scripts/plot_h1_heatmap.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/h1_heatmap" --label auto

echo "Done. See $OUT_PLOTS"

