from __future__ import annotations

# Allow running this file directly without installing the repo as a package.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import time
from typing import Any, Dict, List

import numpy as np

from hypotheses._lib.io import ensure_dir, list_run_files, load_runs, read_json, require_keys, write_json
from hypotheses._lib.metrics import choose_label
from hypotheses._lib.repo import find_repo_root
from hypotheses._lib.runner import run_measure_sink_text, run_plot_script


def _update_plots_manifest(plots_dir: Path, *, inputs: List[Path], items: List[Dict[str, Any]]) -> None:
    path = plots_dir / "manifest.json"
    obj = {"hypothesis_id": "H0_sanity", "generated_at": time.time(), "inputs": [str(p) for p in inputs], "items": items}
    write_json(path, obj)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure", action="store_true", help="Run measurements defined in config.json into data/")
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
                samples=int(m.get("samples", 100)),
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

    runs = load_runs(data_dir, min_rows=10)
    if not runs:
        raise SystemExit("Found run files but none had >=10 rows (possibly empty/corrupted).")

    # validate keys
    required_ok, missing = require_keys([r for run in runs for r in run.rows], ["sink_mass", "correct"])
    if not required_ok:
        raise SystemExit(f"Missing required keys across runs: {missing}")

    # run plots via repo scripts (consistent visuals)
    plot_items: List[Dict[str, Any]] = []
    for spec in cfg.get("plots", []):
        out_sub = str(spec["out_subdir"])
        out = ensure_dir(plots_dir / out_sub)
        run_plot_script(repo_root, str(spec["script"]), inputs=inputs, out_dir=out, extra_args=list(spec.get("extra_args", [])))
        plot_items.append({"script": spec["script"], "out_dir": str(out), "extra_args": spec.get("extra_args", [])})

    _update_plots_manifest(plots_dir, inputs=inputs, items=plot_items)

    # write a short auto-summary (for paper drafting)
    all_rows = [row for run in runs for row in run.rows if row.get("sink_mass") is not None and row.get("correct") is not None]
    label_name, y = choose_label(all_rows, args.label)
    sink = np.array([float(r["sink_mass"]) for r in all_rows], dtype=float)
    acc = float(np.mean([1.0 if r.get("correct") else 0.0 for r in all_rows]))
    pos_rate = float(np.mean(y))

    summary = {
        "hypothesis": "H0_sanity",
        "n_rows": int(len(all_rows)),
        "label": label_name,
        "pos_rate": pos_rate,
        "accuracy": acc,
        "sink_mass": {"mean": float(np.mean(sink)), "std": float(np.std(sink)), "min": float(np.min(sink)), "max": float(np.max(sink))},
        "inputs": [str(p) for p in inputs],
    }
    write_json(plots_dir / "summary.json", summary)

    md = f"""# H0 — Sanity (auto)\n\n## What was run\n- inputs: {len(inputs)} run file(s)\n- rows: {summary['n_rows']}\n- label: `{label_name}` (positive rate={pos_rate:.3f})\n\n## Basic checks\n- accuracy (if present): {acc:.3f}\n- sink_mass: mean={summary['sink_mass']['mean']:.4f}, std={summary['sink_mass']['std']:.4f}, range=[{summary['sink_mass']['min']:.4f}, {summary['sink_mass']['max']:.4f}]\n\n## Plots\nSee `{plots_dir}` (manifest: `{plots_dir / 'manifest.json'}`)\n\n## Status\n- outcome: **ready** (data format ok; plots build)\n"""
    (hyp_dir / "final.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

