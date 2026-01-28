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
import matplotlib.pyplot as plt

from hypotheses._lib.io import ensure_dir, list_run_files, load_runs, read_json, require_keys, write_json
from hypotheses._lib.metrics import choose_label, pr_auc, roc_auc
from hypotheses._lib.repo import find_repo_root
from hypotheses._lib.runner import run_measure_sink_text


def query_key(run) -> Tuple[str, int]:
    # (query_mode, query_start)
    meta = run.meta or {}
    qm = str(meta.get("query_mode", run.rows[0].get("query_mode", "last")))
    qs = int(meta.get("query_start", run.rows[0].get("query_start", 0)))
    return qm, qs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure", action="store_true")
    p.add_argument("--label", default="auto", choices=["auto", "hallucinated", "incorrect"])
    args = p.parse_args()

    hyp_dir = Path(__file__).resolve().parents[1]
    repo_root = find_repo_root(hyp_dir)
    data_dir = ensure_dir(hyp_dir / "data")
    plots_dir = ensure_dir(hyp_dir / "plots")

    if args.measure:
        cfg = read_json(hyp_dir / "config.json")
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

    runs = load_runs(data_dir, min_rows=200)
    all_rows = [row for run in runs for row in run.rows]
    required_ok, missing = require_keys(all_rows, ["sink_mass", "correct"])
    if not required_ok:
        raise SystemExit(f"Missing required keys across runs: {missing}")

    rows_out = []
    for run in runs:
        qm, qs = query_key(run)
        rows = [r for r in run.rows if r.get("sink_mass") is not None and r.get("correct") is not None]
        label_name, y = choose_label(rows, args.label)
        sink = np.array([float(r["sink_mass"]) for r in rows], dtype=float)
        rows_out.append(
            {
                "run": run.path.name,
                "task": run.task,
                "model": run.model,
                "query_mode": qm,
                "query_start": int(qs),
                "label": label_name,
                "n": int(len(y)),
                "pos_rate": float(np.mean(y)),
                "sink_mean": float(np.mean(sink)),
                "auroc": float(roc_auc(y, sink)),
                "auprc": float(pr_auc(y, sink)),
            }
        )

    write_json(plots_dir / "metrics.json", {"generated_at": time.time(), "inputs": [str(p) for p in inputs], "metrics": rows_out})

    # Plot: AUROC vs query_start (range); include last as special point at -1
    # Map x:
    xs = []
    ys = []
    labels = []
    for r in rows_out:
        x = -1 if r["query_mode"] == "last" else int(r["query_start"])
        xs.append(x)
        ys.append(r["auroc"])
        labels.append("last" if x == -1 else f"range@{x}")

    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    labels = np.array(labels)[order]

    plt.figure(figsize=(5.5, 3.2))
    plt.plot(xs, ys, marker="o")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylabel("AUROC (sink_mass)")
    plt.title("H6: Query-set sensitivity")
    plt.tight_layout()
    plt.savefig(plots_dir / "auroc_vs_query.png", dpi=150)
    plt.close()

    # Plot: mean sink_mass vs query
    plt.figure(figsize=(5.5, 3.2))
    plt.plot(xs, [rows_out[i]["sink_mean"] for i in order], marker="o")
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylabel("mean sink_mass")
    plt.title("H6: mean sink_mass vs query mode/start")
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_sink_vs_query.png", dpi=150)
    plt.close()

    md = f"""# H6 — Query-set sensitivity (auto)\n\n## Claim being tested\nThe sink signal depends on the query set Q (e.g., last-token vs tail-range) and on `query_start`.\n\n## What was run\n- inputs: {len(inputs)} run file(s)\n- runs analyzed: {len(rows_out)}\n\n## Results\nMetrics table: `{plots_dir / 'metrics.json'}`\n\nKey curves:\n- `{plots_dir / 'auroc_vs_query.png'}`\n- `{plots_dir / 'mean_sink_vs_query.png'}`\n\n## Status\n- outcome: **preliminary** (needs more query_start sweep + other models/tasks)\n"""
    (hyp_dir / "final.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

