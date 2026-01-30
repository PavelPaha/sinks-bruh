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
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from hypotheses._lib.analysis_ext import (
    plot_aggregate_summary_table,
    plot_aggregate_layer_profiles,
    plot_basic_run_diagnostics,
    plot_paired_chat_deltas,
    plot_signflip_and_effects,
    summarize_run,
    write_manifest,
)
from hypotheses._lib.io import ensure_dir, list_run_files, load_runs, read_json, require_keys, write_json
from hypotheses._lib.metrics import choose_label, cohens_d, pr_auc, roc_auc
from hypotheses._lib.repo import find_repo_root
from hypotheses._lib.runner import run_measure_sink_text, run_plot_script


def run_key(run) -> Tuple[str, str, str, int, str, int, int]:
    # (task, model, quant, K, query_mode, query_start, seed)
    meta = run.meta or {}
    k = int(meta.get("sink_tokens", run.rows[0].get("sink_tokens", 0)))
    qs = int(meta.get("query_start", run.rows[0].get("query_start", 0)))
    seed = int(meta.get("seed", 0))
    return (run.task, run.model, run.quantization, k, str(run.rows[0].get("query_mode", "last")), qs, seed)


def metrics_for_rows(rows: List[Dict[str, Any]], label_mode: str) -> Dict[str, float]:
    rows = [r for r in rows if r.get("sink_mass") is not None and r.get("correct") is not None]
    label_name, y = choose_label(rows, label_mode)
    sink = np.array([float(r["sink_mass"]) for r in rows], dtype=float)
    x1 = sink[y == 1]
    x0 = sink[y == 0]
    d = cohens_d(x1, x0)
    return {"label": label_name, "n": float(len(rows)), "pos_rate": float(np.mean(y)), "cohens_d": float(d), "mean_pos": float(np.mean(x1)) if len(x1) else float("nan"), "mean_neg": float(np.mean(x0)) if len(x0) else float("nan")}


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

    # Group by everything except chat_mode
    groups: Dict[Tuple, Dict[str, Any]] = {}
    for run in runs:
        key = run_key(run)
        groups.setdefault(key, {})[run.chat_mode] = run

    rows_out = []
    for key, dct in groups.items():
        if "auto" not in dct or "off" not in dct:
            continue
        ra = dct["auto"]
        ro = dct["off"]
        ma = metrics_for_rows(ra.rows, args.label)
        mo = metrics_for_rows(ro.rows, args.label)
        rows_out.append(
            {
                "task": key[0],
                "model": key[1],
                "quantization": key[2],
                "sink_tokens": key[3],
                "query_mode": key[4],
                "query_start": key[5],
                "seed": key[6],
                "auto": ma,
                "off": mo,
                "delta_cohens_d": float(mo["cohens_d"] - ma["cohens_d"]),
                "delta_mean_pos": float(mo["mean_pos"] - ma["mean_pos"]),
                "delta_mean_neg": float(mo["mean_neg"] - ma["mean_neg"]),
            }
        )

    out = {"generated_at": time.time(), "inputs": [str(p) for p in inputs], "pairs": rows_out}
    write_json(plots_dir / "metrics.json", out)

    # Extra local-only analysis: per-run diagnostics + aggregate summaries.
    per_run_summaries = []
    extra_items: List[Dict[str, Any]] = []
    runs_out_dir = ensure_dir(plots_dir / "runs")
    for run in runs:
        rid = run.path.name.replace(".jsonl.gz", "")
        per_run_summaries.append(summarize_run(run.rows, run_id=rid, label_mode=args.label))
        extra_items.extend(plot_basic_run_diagnostics(run.rows, runs_out_dir / rid, title=rid, label_mode=args.label))

    agg_dir = ensure_dir(plots_dir / "agg")
    summary_csv = plot_aggregate_summary_table(per_run_summaries, agg_dir)
    extra_items.append({"kind": "table", "path": str(summary_csv), "desc": "Per-run summary table (csv)."})
    # Intentionally: no generic cross-run diagnostics in agg/ (only hypothesis-specific aggregations).
    # H5-specific aggregates: paired chat deltas + stacked layer profiles (if available).
    extra_items.extend(plot_paired_chat_deltas(per_run_summaries, agg_dir, title="H5 aggregate — chat deltas"))
    extra_items.extend(
        plot_aggregate_layer_profiles([(s.run_id, run.rows) for s, run in zip(per_run_summaries, runs)], agg_dir, title="H5 aggregate — layer profiles", label_mode=args.label)
    )

    # Sanity plots on the combined inputs (helps visually see differences)
    try:
        run_plot_script(repo_root, "plot_h1.py", inputs=inputs, out_dir=ensure_dir(plots_dir / "h1"), extra_args=["--label", "auto"])
        run_plot_script(
            repo_root, "plot_h1_heatmap.py", inputs=inputs, out_dir=ensure_dir(plots_dir / "h1_heatmap"), extra_args=["--label", "auto"]
        )
    except Exception:
        # Plots are optional; do not block metrics/final if plotting deps are missing.
        pass

    # Simple plot: delta d histogram
    deltas = [r["delta_cohens_d"] for r in rows_out if np.isfinite(r["delta_cohens_d"])]
    if plt is not None and deltas:
        plt.figure(figsize=(5, 3))
        plt.hist(deltas, bins=20)
        plt.axvline(0, color="black", linewidth=1)
        plt.title("H5: Δ Cohen's d (chat=off - chat=auto)")
        plt.tight_layout()
        plt.savefig(plots_dir / "delta_cohens_d.png", dpi=150)
        plt.close()

    write_manifest(
        plots_dir,
        hypothesis_id="H5_chat_sensitivity",
        inputs=[Path(p) for p in inputs],
        items=[
            {"kind": "metrics", "path": str(plots_dir / "metrics.json"), "desc": "Paired chat(auto/off) comparisons."},
            *(
                [{"kind": "plot", "path": str(plots_dir / "delta_cohens_d.png"), "desc": "Histogram of Δ Cohen's d (off - auto)."}]
                if plt is not None
                else []
            ),
            *extra_items,
        ],
    )

    deltas_arr = np.array([float(r["delta_cohens_d"]) for r in rows_out], dtype=float) if rows_out else np.array([], dtype=float)
    deltas_arr = deltas_arr[np.isfinite(deltas_arr)]
    top_abs = sorted(
        rows_out,
        key=lambda r: abs(float(r.get("delta_cohens_d", float("nan")))) if np.isfinite(float(r.get("delta_cohens_d", float("nan")))) else -1.0,
        reverse=True,
    )[:3] if rows_out else []
    top_abs_str = "\n".join(
        [
            f"- {r['task']} / {r['model']}: Δd={float(r['delta_cohens_d']):+.3f} (auto d={float(r['auto']['cohens_d']):+.3f}, off d={float(r['off']['cohens_d']):+.3f})"
            for r in top_abs
        ]
    ) if top_abs else "- (none)"

    if not rows_out:
        md = f"""# H5 — Chat sensitivity (auto)\n\n## Claim being tested\nChanging chat formatting (chat template on/off) changes sink/label conclusions.\n\n## Status\n- outcome: **blocked** (no paired runs)\n\n## Why\nNo paired runs found. H5 requires **both** `chat=auto` and `chat=off` runs for the same (task, model, K, Q, seed).\n\n## What we still produced\n- per-run diagnostics: `plots/runs/`\n- aggregate analysis: `plots/agg/` (table: `{summary_csv.name}`)\n\n## How to fix\nRun the missing chat mode into `{data_dir}`.\n"""
    else:
        md = f"""# H5 — Chat sensitivity (auto)\n\n## Claim being tested\nChanging chat formatting (chat template on/off) changes sink/label conclusions.\n\n## What was run\n- inputs: {len(inputs)} run file(s)\n- paired comparisons: {len(rows_out)}\n\n## Results\n- metrics: `plots/metrics.json`\n- Δd summary (off - auto): mean={float(np.mean(deltas_arr)) if deltas_arr.size else float('nan'):+.3f}, median={float(np.median(deltas_arr)) if deltas_arr.size else float('nan'):+.3f}\n\nLargest |Δd| pairs:\n{top_abs_str}\n\n## Plots\n- `plots/delta_cohens_d.png` (if generated)\n- `plots/h1/h1_grid_*.png` (if generated)\n- `plots/h1_heatmap/heatmap_grid_*.png` (if generated)\n\n## Extra diagnostics\n- per-run: `plots/runs/`\n- aggregate: `plots/agg/` (table: `{summary_csv.name}`)\n\n## Status\n- outcome: **preliminary** (needs more models/tasks)\n"""
    (hyp_dir / "final.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

