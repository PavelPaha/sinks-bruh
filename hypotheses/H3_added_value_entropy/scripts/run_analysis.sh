#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HYP_DIR="$ROOT/hypotheses/H3_added_value_entropy"

INPUTS=("$HYP_DIR"/data/*.jsonl.gz)
OUT_PLOTS="$HYP_DIR/plots"

if [ ! -e "${INPUTS[0]}" ]; then
  echo "No inputs in $HYP_DIR/data/*.jsonl.gz" >&2
  exit 1
fi

mkdir -p "$OUT_PLOTS"

# prefer uv if present; otherwise use python/python3
run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  elif command -v python >/dev/null 2>&1; then
    python "$@"
  else
    python3 "$@"
  fi
}

# Context plot: sink↔entropy scatter
run_py "$ROOT/scripts/plot_sink_entropy.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/correlation"

# Main H3 test: entropy-only vs entropy+sink
run_py "$HYP_DIR/scripts/fit_logreg.py" --inputs "${INPUTS[@]}" --label auto --bootstrap 200 --seed 0 --out "$OUT_PLOTS/metrics.json"

echo "Done. See $OUT_PLOTS"

