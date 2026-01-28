#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HYP_DIR="$ROOT/hypotheses/H4_localization"

INPUTS=("$HYP_DIR"/data/*.jsonl.gz)
OUT_PLOTS="$HYP_DIR/plots"

if [ ! -e "${INPUTS[0]}" ]; then
  echo "No inputs in $HYP_DIR/data/*.jsonl.gz" >&2
  echo "Copy or generate run files into hypotheses/H4_localization/data first." >&2
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

run_py "$ROOT/scripts/plot_h1_heatmap.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/h1_heatmap" --label auto
run_py "$ROOT/scripts/plot_layer_profiles.py" --inputs "${INPUTS[@]}" --out_dir "$OUT_PLOTS/layers"

echo "Done. See $OUT_PLOTS"

