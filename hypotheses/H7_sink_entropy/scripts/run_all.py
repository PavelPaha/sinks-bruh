from __future__ import annotations

# Allow running directly without package install.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore
try:
    from scipy.stats import pearsonr, spearmanr  # type: ignore
except Exception:  # pragma: no cover
    def pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            return float("nan"), float("nan")
        return float(np.corrcoef(x, y)[0, 1]), float("nan")

    def spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            return float("nan"), float("nan")
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1]), float("nan")

from hypotheses._lib.analysis_ext import (
    plot_aggregate_summary_table,
    plot_basic_run_diagnostics,
    plot_signflip_and_effects,
    save_table_csv,
    summarize_run,
    write_manifest,
)
from hypotheses._lib.io import ensure_dir, list_run_files, load_runs, read_json, write_json
from hypotheses._lib.repo import find_repo_root
from hypotheses._lib.runner import run_measure_sink_text


def _short_name(model: str, quant: str, chat: str) -> str:
    name = model.split("/")[-1] if model else "?"
    if quant not in (None, "none", ""):
        name += " (4b)"
    if chat == "off":
        name += " [raw]"
    return name


def _nanfloat(x: Any) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def _partial_corr_by_residuals(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Partial correlation corr(x,y | z) via linear residualization against z.
    Works as: corr(resid(x~z), resid(y~z)).
    """
    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(m.sum()) < 20:
        return float("nan")
    x = x[m]
    y = y[m]
    z = z[m]
    Z = np.stack([np.ones_like(z), _zscore(z)], axis=1)  # [n,2]
    # least squares
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    if np.nanstd(rx) < 1e-12 or np.nanstd(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass(frozen=True)
class RunKey:
    task: str
    sink_tokens: int
    query_mode: str
    query_start: int
    chat_mode: str


def _key_from_run(rows0: Dict[str, Any]) -> RunKey:
    return RunKey(
        task=str(rows0.get("task", "unknown")),
        sink_tokens=int(rows0.get("sink_tokens", 0) or 0),
        query_mode=str(rows0.get("query_mode", "last")),
        query_start=int(rows0.get("query_start", 0) or 0),
        chat_mode=str(rows0.get("chat_mode", "auto")),
    )


def _expand_measurements(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    defaults = dict(cfg.get("defaults", {}))
    tasks = list(cfg.get("tasks", []))
    models = list(cfg.get("models", []))
    sink_tokens_list = list(defaults.get("sink_tokens_list", [4]))
    query_specs = list(defaults.get("query_specs", [{"query_mode": "last", "query_start": 0}]))

    out: List[Dict[str, Any]] = []
    for t in tasks:
        for m in models:
            for k in sink_tokens_list:
                for q in query_specs:
                    out.append(
                        {
                            "task": t["task"],
                            "split": t.get("split", "test"),
                            "model": m["model"],
                            "samples": int(t.get("samples", defaults.get("samples", 500))),
                            "seed": int(defaults.get("seed", 42)),
                            "sink_tokens": int(k),
                            "query_mode": str(q.get("query_mode", "last")),
                            "query_start": int(q.get("query_start", 0)),
                            "chat": str(defaults.get("chat", "auto")),
                            "device": str(defaults.get("device", "cuda")),
                            "quantization": str(m.get("quantization", defaults.get("quantization", "none"))),
                            "revision": m.get("revision"),
                        }
                    )
    return out


def _bin_curve(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 50:
        return np.array([]), np.array([])
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    # avoid empty bins due to ties
    xs = []
    ys = []
    for i in range(n_bins):
        lo = qs[i]
        hi = qs[i + 1]
        if i == n_bins - 1:
            sel = (x >= lo) & (x <= hi)
        else:
            sel = (x >= lo) & (x < hi)
        if int(sel.sum()) < 5:
            continue
        xs.append(float(np.mean(x[sel])))
        ys.append(float(np.mean(y[sel])))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _plot_grid(
    out_path: Path,
    *,
    task: str,
    items: List[Dict[str, Any]],
    title_suffix: str,
) -> None:
    import math

    if plt is None:
        return
    items = list(items)
    items.sort(key=lambda d: d.get("spearman", 0.0))
    N = len(items)
    if N == 0:
        return
    cols = min(4, N)
    rows_n = math.ceil(N / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.5 * rows_n), constrained_layout=True)
    fig.suptitle(f"H7 Sink↔Entropy: {task} {title_suffix}".strip(), fontsize=14, fontweight="bold")
    flat = axes.flatten() if N > 1 else [axes]

    for i, ax in enumerate(flat):
        if i >= N:
            ax.axis("off")
            continue
        it = items[i]
        x = np.asarray(it["sink"], dtype=float)
        y = np.asarray(it["entropy"], dtype=float)
        ax.scatter(x, y, s=6, alpha=0.25, color="steelblue")
        # regression line (for visualization only)
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) >= 20:
            try:
                z = np.polyfit(x[m], y[m], 1)
                p = np.poly1d(z)
                xx = np.linspace(float(np.min(x[m])), float(np.max(x[m])), 50)
                ax.plot(xx, p(xx), color="red", linewidth=1)
            except Exception:
                pass

        r = it.get("pearson")
        rho = it.get("spearman")
        pr = it.get("partial_seq_len")
        ax.set_title(f"{it['name']}\nr={r:+.2f}  ρ={rho:+.2f}\nρ|len={pr:+.2f}" if np.isfinite(pr) else f"{it['name']}\nr={r:+.2f}  ρ={rho:+.2f}", fontsize=9)
        ax.set_xlabel("sink_mass" if i >= (rows_n - 1) * cols else "")
        ax.set_ylabel("entropy" if i % cols == 0 else "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_hist(out_path: Path, *, task: str, rows: List[Dict[str, Any]], title_suffix: str) -> None:
    if plt is None:
        return
    rs = np.array([_nanfloat(r.get("pearson")) for r in rows], dtype=float)
    rhos = np.array([_nanfloat(r.get("spearman")) for r in rows], dtype=float)
    rs = rs[np.isfinite(rs)]
    rhos = rhos[np.isfinite(rhos)]
    if rs.size == 0 and rhos.size == 0:
        return
    plt.figure(figsize=(6, 3.5))
    if rs.size:
        plt.hist(rs, bins=20, alpha=0.6, label="Pearson r")
    if rhos.size:
        plt.hist(rhos, bins=20, alpha=0.6, label="Spearman ρ")
    plt.axvline(0.0, color="black", linewidth=1)
    plt.title(f"H7 correlation distribution: {task} {title_suffix}".strip())
    plt.xlabel("correlation")
    plt.ylabel("count (models)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure", action="store_true", help="Run measurements into data/")
    p.add_argument("--max_runs", type=int, default=None, help="Debug: cap number of runs to analyze")
    # Kept for consistency with hypotheses/analyze_all.py (ignored by core H7 metric computation).
    p.add_argument("--label", default="auto", choices=["auto", "hallucinated", "incorrect"])
    args = p.parse_args()

    hyp_dir = Path(__file__).resolve().parents[1]
    repo_root = find_repo_root(hyp_dir)
    data_dir = ensure_dir(hyp_dir / "data")
    plots_dir = ensure_dir(hyp_dir / "plots")
    cfg = read_json(hyp_dir / "config.json")

    if args.measure:
        measurements = cfg.get("measurements")
        if not measurements:
            measurements = _expand_measurements(cfg)
        for m in measurements:
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
        raise SystemExit(f"No run files in {data_dir}. Run with --measure or put *.jsonl.gz there.")

    runs = load_runs(data_dir, min_rows=50)
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]

    # per-run metrics
    metrics: List[Dict[str, Any]] = []
    grid_payload: Dict[RunKey, List[Dict[str, Any]]] = {}

    for run in runs:
        rows = run.rows
        # require columns
        sink = np.array([_nanfloat(r.get("sink_mass")) for r in rows], dtype=float)
        ent = np.array([_nanfloat(r.get("entropy")) for r in rows], dtype=float)
        seq = np.array([_nanfloat(r.get("seq_len")) for r in rows], dtype=float)
        m = np.isfinite(sink) & np.isfinite(ent)
        if int(m.sum()) < 50:
            continue
        sink = sink[m]
        ent = ent[m]
        seq = seq[m] if np.isfinite(seq).any() else np.full_like(sink, np.nan)

        try:
            r, _ = pearsonr(sink, ent)
        except Exception:
            r = float("nan")
        try:
            rho, _ = spearmanr(sink, ent)
        except Exception:
            rho = float("nan")
        pr = _partial_corr_by_residuals(sink, ent, seq) if np.isfinite(seq).any() else float("nan")

        key = _key_from_run(rows[0] if rows else {})
        name = _short_name(run.model, run.quantization, run.chat_mode)

        bx, by = _bin_curve(sink, ent, n_bins=10)

        rec = {
            "run": run.path.name,
            "task": key.task,
            "model": run.model,
            "chat_mode": key.chat_mode,
            "quantization": run.quantization,
            "sink_tokens": key.sink_tokens,
            "query_mode": key.query_mode,
            "query_start": key.query_start,
            "n": int(len(sink)),
            "sink_mean": float(np.mean(sink)),
            "entropy_mean": float(np.mean(ent)),
            "pearson": float(r),
            "spearman": float(rho),
            "partial_seq_len": float(pr),
            "bin_curve": {"x": bx.tolist(), "y": by.tolist()},
        }
        metrics.append(rec)
        grid_payload.setdefault(key, []).append({"name": name, "sink": sink, "entropy": ent, "pearson": float(r), "spearman": float(rho), "partial_seq_len": float(pr)})

    # Write metrics
    out = {
        "hypothesis_id": "H7_sink_entropy",
        "generated_at": time.time(),
        "n_runs": int(len(metrics)),
        "inputs": [str(p) for p in inputs],
        "metrics": metrics,
    }
    write_json(plots_dir / "metrics.json", out)

    # CSV (minimal, no pandas dependency)
    try:
        save_table_csv(plots_dir / "metrics.csv", [{k: v for k, v in m.items() if k not in ("bin_curve",)} for m in metrics])
    except Exception:
        # keep going even if filesystem is read-only or rows are empty
        pass

    # Plots per (task,K,Q,chat)
    plot_items: List[Dict[str, Any]] = []
    if plt is not None:
        for key, items in grid_payload.items():
            suffix = f"__K{key.sink_tokens}__q{key.query_mode}{'' if key.query_mode=='last' else f'_start{key.query_start}'}__chat{key.chat_mode}"
            grid_path = plots_dir / f"corr_grid_{key.task}{suffix}.png"
            hist_path = plots_dir / f"corr_hist_{key.task}{suffix}.png"
            _plot_grid(grid_path, task=key.task, items=items, title_suffix=suffix)
            _plot_hist(hist_path, task=key.task, rows=[{"pearson": it["pearson"], "spearman": it["spearman"]} for it in items], title_suffix=suffix)
            plot_items.append({"type": "grid", "path": str(grid_path), "key": key.__dict__})
            plot_items.append({"type": "hist", "path": str(hist_path), "key": key.__dict__})

    # Extra local-only analysis: per-run diagnostics + aggregate summaries (reuses shared tooling).
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

    # Manifest (union of H7-specific plots + shared diagnostics)
    write_manifest(
        plots_dir,
        hypothesis_id="H7_sink_entropy",
        inputs=inputs,
        items=[
            *plot_items,
            {"kind": "metrics", "path": str(plots_dir / "metrics.json"), "desc": "Per-run sink↔entropy correlations."},
            {"kind": "table", "path": str(plots_dir / "metrics.csv"), "desc": "Flat CSV (if generated)."},
            *extra_items,
        ],
    )

    # Auto final.md
    # summarize by median correlation per key
    summaries = []
    for key, items in grid_payload.items():
        rhos = np.array([_nanfloat(it.get("spearman")) for it in items], dtype=float)
        rs = np.array([_nanfloat(it.get("pearson")) for it in items], dtype=float)
        prs = np.array([_nanfloat(it.get("partial_seq_len")) for it in items], dtype=float)
        rhos = rhos[np.isfinite(rhos)]
        rs = rs[np.isfinite(rs)]
        prs = prs[np.isfinite(prs)]
        summaries.append(
            {
                "key": key,
                "n_models": len(items),
                "pearson_median": float(np.median(rs)) if rs.size else float("nan"),
                "spearman_median": float(np.median(rhos)) if rhos.size else float("nan"),
                "partial_median": float(np.median(prs)) if prs.size else float("nan"),
            }
        )

    lines = []
    lines.append("# H7 — Sink mass ↔ Entropy (auto)\n")
    lines.append("## Claim being tested\n`entropy` (next-token uncertainty) is statistically linked to `sink_mass`.\n")
    lines.append("## What was run\n")
    lines.append(f"- run files: {len(inputs)}\n- analyzed runs (min_rows>=50): {len(metrics)}\n")
    lines.append("## Summary across models (median correlations)\n")
    for s in sorted(summaries, key=lambda x: (x["key"].task, x["key"].sink_tokens, x["key"].query_mode, x["key"].query_start, x["key"].chat_mode)):
        key = s["key"]
        suffix = f"K{key.sink_tokens}, q={key.query_mode}{'' if key.query_mode=='last' else f'@{key.query_start}'}, chat={key.chat_mode}"
        lines.append(f"- **{key.task}** ({suffix}): median Spearman ρ={s['spearman_median']:+.3f}, Pearson r={s['pearson_median']:+.3f}, partial(ρ|seq_len)={s['partial_median']:+.3f} over n_models={s['n_models']}\n")
    # quick stability summary for the "seq_len doesn't matter" claim
    diffs = []
    for s in summaries:
        if np.isfinite(s["spearman_median"]) and np.isfinite(s["partial_median"]):
            diffs.append(float(s["partial_median"] - s["spearman_median"]))
    diffs_arr = np.asarray(diffs, dtype=float)

    lines.append("\n## Stability under seq_len control\n")
    if diffs_arr.size:
        lines.append(f"- median( partial - raw ): {float(np.median(diffs_arr)):+.3f}\n")
        lines.append(f"- median(|partial - raw|): {float(np.median(np.abs(diffs_arr))):.3f}\n")
        lines.append("Interpretation: if these are near 0, controlling for `seq_len` doesn't materially change the sink↔entropy conclusion for these runs.\n")
    else:
        lines.append("- (not enough data to summarize partial-vs-raw deltas)\n")

    lines.append("\n## Artifacts\n")
    lines.append("- metrics: `plots/metrics.json` (and `plots/metrics.csv`)\n")
    lines.append("- manifest: `plots/manifest.json`\n")
    lines.append("- shared diagnostics: `plots/runs/` and `plots/agg/` (notably `plots/agg/scatter_partial_vs_spearman.png`)\n")
    lines.append("\n## Status\n- outcome: **preliminary** (replicate across tasks, compare instruct vs base, sweep K/Q)\n")

    (hyp_dir / "final.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

