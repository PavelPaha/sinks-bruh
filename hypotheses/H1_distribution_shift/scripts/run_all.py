from __future__ import annotations

# Allow running directly without package install.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import time
from typing import Any, Dict, List

import numpy as np

from hypotheses._lib.io import ensure_dir, list_run_files, load_runs, read_json, require_keys, write_json
from hypotheses._lib.metrics import choose_label, cohens_d
from hypotheses._lib.repo import find_repo_root
from hypotheses._lib.runner import run_measure_sink_text, run_plot_script


def _update_plots_manifest(plots_dir: Path, *, inputs: List[Path], items: List[Dict[str, Any]]) -> None:
    write_json(
        plots_dir / "manifest.json",
        {"hypothesis_id": "H1_distribution_shift", "generated_at": time.time(), "inputs": [str(p) for p in inputs], "items": items},
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure", action="store_true", help="Run measurements from config.json into data/")
    p.add_argument("--label", default="auto", choices=["auto", "hallucinated", "incorrect"])
    args = p.parse_args()

    hyp_dir = Path(__file__).resolve().parents[1]
    repo_root = find_repo_root(hyp_dir)

    data_dir = ensure_dir(hyp_dir / "data")
    plots_dir = ensure_dir(hyp_dir / "plots")
    cfg = read_json(hyp_dir / "config.json")

    if args.measure:
        for m in cfg.get("measurements", []):
            run_measure_sink_text(
                repo_root,
                data_dir,
                task=m["task"],
                split=m.get("split", "test"),
                model=m["model"],
                samples=int(m.get("samples", 500)),
                seed=int(m.get("seed", 42)),
                sink_tokens=int(m.get("sink_tokens", 4)),
                query_mode=str(m.get("query_mode", "last")),
                query_start=int(m.get("query_start", 0)),
                chat=str(m.get("chat", "auto")),
                device=str(m.get("device", "cuda")),
                quantization=str(m.get("quantization", "none")),
                revision=m.get("revision"),
            )

    inputs = list_run_files(data_dir)
    if not inputs:
        raise SystemExit(f"No run files in {data_dir}. Put *.jsonl.gz there or run with --measure")

    runs = load_runs(data_dir, min_rows=50)
    if not runs:
        raise SystemExit("Found run files but none had >=50 rows (possibly empty/corrupted).")

    # validate required keys
    required_ok, missing = require_keys([r for run in runs for r in run.rows], ["sink_mass", "correct"])
    if not required_ok:
        raise SystemExit(f"Missing required keys across runs: {missing}")

    # plots via repo scripts
    plot_items: List[Dict[str, Any]] = []
    for spec in cfg.get("plots", []):
        out = ensure_dir(plots_dir / str(spec["out_subdir"]))
        run_plot_script(repo_root, str(spec["script"]), inputs=inputs, out_dir=out, extra_args=list(spec.get("extra_args", [])))
        plot_items.append({"script": spec["script"], "out_dir": str(out), "extra_args": spec.get("extra_args", [])})
    _update_plots_manifest(plots_dir, inputs=inputs, items=plot_items)

    # metrics per run (Cohen's d)
    metrics = []
    for run in runs:
        rows = [r for r in run.rows if r.get("sink_mass") is not None and r.get("correct") is not None]
        label_name, y = choose_label(rows, args.label)
        sink = np.array([float(r["sink_mass"]) for r in rows], dtype=float)
        x1 = sink[y == 1]
        x0 = sink[y == 0]
        d = cohens_d(x1, x0)
        metrics.append(
            {
                "run": run.path.name,
                "task": run.task,
                "model": run.model,
                "chat_mode": run.chat_mode,
                "quantization": run.quantization,
                "label": label_name,
                "n": int(len(rows)),
                "pos_rate": float(np.mean(y)),
                "cohens_d": float(d),
                "mean_pos": float(np.mean(x1)) if len(x1) else float("nan"),
                "mean_neg": float(np.mean(x0)) if len(x0) else float("nan"),
            }
        )

    write_json(plots_dir / "metrics.json", {"metrics": metrics})

    # aggregated text
    ds = [m["cohens_d"] for m in metrics if np.isfinite(m["cohens_d"])]
    n_pos = sum(1 for d in ds if d > 0)
    n_neg = sum(1 for d in ds if d < 0)
    md = f"""# H1 — Distribution shift (auto)\n\n## Claim being tested\nSink mass distribution differs between positive class (hallucinated/incorrect) and negative (non-hallucinated/correct).\n\n## What was run\n- runs: {len(metrics)}\n- label: `{metrics[0]['label'] if metrics else args.label}`\n\n## Results (Cohen's d)\n- finite d: {len(ds)}\n- sign breakdown: {n_pos} positive, {n_neg} negative\n\nPer-run table is in `{plots_dir / 'metrics.json'}`.\n\n## Plots\nSee `{plots_dir}` (manifest: `{plots_dir / 'manifest.json'}`)\n\n## Status\n- outcome: **preliminary** (depends on sign stability across models/tasks)\n"""
    (hyp_dir / "final.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

