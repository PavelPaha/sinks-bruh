from __future__ import annotations

# Allow running directly without package install.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from hypotheses._lib.io import ensure_dir, list_run_files, load_runs, read_json, require_keys, write_json
from hypotheses._lib.metrics import choose_label
from hypotheses._lib.repo import find_repo_root
from hypotheses._lib.runner import run_measure_sink_text, run_plot_script


def compute_delta_map(rows: List[Dict[str, Any]], label_mode: str) -> Tuple[np.ndarray, str]:
    # Determine label
    label_name, y = choose_label(rows, label_mode)
    # Determine shape
    L = H = 0
    for r in rows:
        m = r.get("sink_by_layer_head")
        if isinstance(m, list) and m:
            a = np.array(m)
            if a.ndim == 2:
                L, H = a.shape
                break
    if L == 0:
        raise ValueError("No sink_by_layer_head found")

    sum_pos = np.zeros((L, H), dtype=float)
    sum_neg = np.zeros((L, H), dtype=float)
    n_pos = n_neg = 0
    for r, yi in zip(rows, y.tolist()):
        m = r.get("sink_by_layer_head")
        if not isinstance(m, list):
            continue
        a = np.array(m, dtype=float)
        if a.shape != (L, H):
            continue
        if yi == 1:
            sum_pos += a
            n_pos += 1
        else:
            sum_neg += a
            n_neg += 1
    if n_pos < 5 or n_neg < 5:
        raise ValueError("Too few examples per class to compute delta map")
    delta = (sum_pos / n_pos) - (sum_neg / n_neg)
    return delta, label_name


def topk(delta: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
    flat = np.abs(delta).ravel()
    idx = np.argsort(-flat)[:k]
    out = []
    L, H = delta.shape
    for j in idx:
        l = int(j // H)
        h = int(j % H)
        out.append({"layer": l, "head": h, "delta": float(delta[l, h]), "abs_delta": float(abs(delta[l, h]))})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure", action="store_true")
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

    runs = load_runs(data_dir, min_rows=100)
    all_rows = [row for run in runs for row in run.rows]
    required_ok, missing = require_keys(all_rows, ["sink_by_layer_head", "correct"])
    if not required_ok:
        raise SystemExit(f"Missing required keys across runs: {missing}")

    # Build plots (heatmaps + layer profiles) using shared plot scripts
    plot_items: List[Dict[str, Any]] = []
    for spec in cfg.get("plots", []):
        out = ensure_dir(plots_dir / str(spec["out_subdir"]))
        run_plot_script(repo_root, str(spec["script"]), inputs=inputs, out_dir=out, extra_args=list(spec.get("extra_args", [])))
        plot_items.append({"script": spec["script"], "out_dir": str(out), "extra_args": spec.get("extra_args", [])})
    write_json(plots_dir / "manifest.json", {"hypothesis_id": "H4_localization", "generated_at": time.time(), "inputs": [str(p) for p in inputs], "items": plot_items})

    # Compute delta maps + top heads per run
    metrics = []
    for run in runs:
        delta, label_name = compute_delta_map(run.rows, args.label)
        metrics.append(
            {
                "run": run.path.name,
                "task": run.task,
                "model": run.model,
                "chat_mode": run.chat_mode,
                "quantization": run.quantization,
                "label": label_name,
                "shape": [int(delta.shape[0]), int(delta.shape[1])],
                "top10": topk(delta, 10),
            }
        )
    write_json(plots_dir / "metrics.json", {"metrics": metrics})

    md = f"""# H4 — Localization (auto)\n\n## Claim being tested\nThe sink/label effect is localized to specific layers/heads (not uniform).\n\n## What was run\n- runs: {len(metrics)}\n- inputs: {len(inputs)} file(s)\n\n## Results\nTop heads by |Δ sink| per run are in `{plots_dir / 'metrics.json'}`.\n\n## Plots\n- heatmaps: `{plots_dir / 'h1_heatmap'}`\n- layer profiles: `{plots_dir / 'layers'}`\n\n## Status\n- outcome: **preliminary** (depends on stability across models/tasks)\n"""
    (hyp_dir / "final.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

