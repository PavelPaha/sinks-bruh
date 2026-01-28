#!/usr/bin/env bash
set -euo pipefail

# H0: sanity plots from local data/*.jsonl.gz
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HYP_DIR="$ROOT/hypotheses/H0_sanity"

INPUTS=("$HYP_DIR"/data/*.jsonl.gz)
OUT_PLOTS="$HYP_DIR/plots"

if [ ! -e "${INPUTS[0]}" ]; then
  echo "No inputs in $HYP_DIR/data/*.jsonl.gz" >&2
  echo "Copy a run file into hypotheses/H0_sanity/data first." >&2
  exit 1
fi

mkdir -p "$OUT_PLOTS"

run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  else
    python "$@"
  fi
}

run_py "$ROOT/scripts/plot_accuracy_vs_sink_runs.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/accuracy_vs_sink"
run_py "$ROOT/scripts/plot_h1.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/h1" --label auto
run_py "$ROOT/scripts/plot_h1_heatmap.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/h1_heatmap" --label auto
run_py "$ROOT/scripts/plot_layer_profiles.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/layers"
run_py "$ROOT/scripts/plot_sink_entropy.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/correlation"

echo "Done. See $OUT_PLOTS"

