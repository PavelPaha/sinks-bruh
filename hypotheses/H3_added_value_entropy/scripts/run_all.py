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
    plot_basic_run_diagnostics,
    plot_signflip_and_effects,
    summarize_run,
    write_manifest,
)
from hypotheses._lib.io import ensure_dir, list_run_files, load_runs, read_json, require_keys, write_json
from hypotheses._lib.metrics import choose_label, roc_auc
from hypotheses._lib.logreg import eval_probs, fit_logistic, logloss, predict_proba, zscore_apply, zscore_fit
from hypotheses._lib.repo import find_repo_root
from hypotheses._lib.runner import run_measure_sink_text


def _bootstrap_delta_test(
    rng: np.random.Generator, y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int
) -> Dict[str, Dict[str, float]]:
    n = len(y)
    d_ll = []
    d_auc = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        # skip degenerate
        if int(yb.sum()) == 0 or int(yb.sum()) == len(yb):
            continue
        pa = p_a[idx]
        pb = p_b[idx]
        d_ll.append(float(logloss(yb, pa) - logloss(yb, pb)))
        d_auc.append(float(roc_auc(yb, pb) - roc_auc(yb, pa)))
    d_ll = np.array(d_ll, dtype=float)
    d_auc = np.array(d_auc, dtype=float)
    if d_ll.size == 0:
        return {
            "delta_logloss": {"mean": float("nan"), "p05": float("nan"), "p95": float("nan")},
            "delta_auroc": {"mean": float("nan"), "p05": float("nan"), "p95": float("nan")},
        }
    return {
        "delta_logloss": {"mean": float(np.mean(d_ll)), "p05": float(np.quantile(d_ll, 0.05)), "p95": float(np.quantile(d_ll, 0.95))},
        "delta_auroc": {"mean": float(np.mean(d_auc)), "p05": float(np.quantile(d_auc, 0.05)), "p95": float(np.quantile(d_auc, 0.95))},
    }


def bootstrap_delta(
    rng: np.random.Generator, X_a: np.ndarray, X_b: np.ndarray, y: np.ndarray, n_boot: int
) -> Dict[str, Any]:
    n = len(y)
    d_ll: List[float] = []
    d_auc: List[float] = []
    skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        # Skip degenerate resamples (all-one class -> AUROC undefined)
        if int(yb.sum()) == 0 or int(yb.sum()) == len(yb):
            skipped += 1
            continue
        try:
            Xa = X_a[idx]
            Xb = X_b[idx]
            fa = fit_logistic(Xa, yb)
            fb = fit_logistic(Xb, yb)
            pa = predict_proba(Xa, fa)
            pb = predict_proba(Xb, fb)
            ea = eval_probs(yb, pa)
            eb = eval_probs(yb, pb)
            d_ll.append(float(ea["logloss"] - eb["logloss"]))  # positive => B better
            d_auc.append(float(eb["auroc"] - ea["auroc"]))     # positive => B better
        except Exception:
            skipped += 1
            continue

    d_ll_arr = np.array(d_ll, dtype=float)
    d_auc_arr = np.array(d_auc, dtype=float)

    def _summ(x: np.ndarray) -> Dict[str, float]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"mean": float("nan"), "p05": float("nan"), "p95": float("nan")}
        return {
            "mean": float(np.mean(x)),
            "p05": float(np.quantile(x, 0.05)),
            "p95": float(np.quantile(x, 0.95)),
        }

    return {
        "delta_logloss": _summ(d_ll_arr),
        "delta_auroc": _summ(d_auc_arr),
        "n_boot_requested": int(n_boot),
        "n_boot_used": int(len(d_ll_arr)),
        "n_boot_skipped": int(skipped),
        "samples": {"delta_logloss": d_ll_arr.tolist(), "delta_auroc": d_auc_arr.tolist()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure", action="store_true")
    p.add_argument("--label", default="auto", choices=["auto", "hallucinated", "incorrect"])
    p.add_argument("--eval_mode", choices=["holdout", "in_sample"], default="holdout")
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=None, help="bootstrap reps (holdout: on test deltas; in_sample: refit bootstrap)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    hyp_dir = Path(__file__).resolve().parents[1]
    repo_root = find_repo_root(hyp_dir)
    data_dir = ensure_dir(hyp_dir / "data")
    plots_dir = ensure_dir(hyp_dir / "plots")

    cfg = read_json(hyp_dir / "config.json")
    n_boot = int(args.bootstrap if args.bootstrap is not None else cfg.get("bootstrap", 300))

    if args.measure:
        for m in cfg.get("measurements", []):
            run_measure_sink_text(
                repo_root,
                data_dir,
                task=m["task"],
                split=m.get("split", "test"),
                model=m["model"],
                samples=int(m.get("samples", 800)),
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
    required_ok, missing = require_keys(all_rows, ["sink_mass", "entropy", "correct"])
    if not required_ok:
        raise SystemExit(f"Missing required keys across runs: {missing}")

    rng = np.random.default_rng(args.seed)

    def stratified_split(y_arr: np.ndarray, test_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        y_arr = y_arr.astype(int)
        idx_pos = np.where(y_arr == 1)[0]
        idx_neg = np.where(y_arr == 0)[0]
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)
        n_pos_test = max(1, int(round(len(idx_pos) * test_frac))) if len(idx_pos) else 0
        n_neg_test = max(1, int(round(len(idx_neg) * test_frac))) if len(idx_neg) else 0
        test_idx = np.concatenate([idx_pos[:n_pos_test], idx_neg[:n_neg_test]])
        train_idx = np.setdiff1d(np.arange(len(y_arr)), test_idx, assume_unique=False)
        return train_idx, test_idx

    # Evaluate PER RUN (per model), not pooled across all models.
    per_run_results: List[Dict[str, Any]] = []

    for run in runs:
        run_rows = [r for r in run.rows if r.get("sink_mass") is not None and r.get("entropy") is not None and r.get("correct") is not None]
        if not run_rows:
            continue
        label_name, y = choose_label(run_rows, args.label)
        y = y.astype(int)
        sink = np.array([float(r["sink_mass"]) for r in run_rows], dtype=float)
        ent = np.array([float(r["entropy"]) for r in run_rows], dtype=float)
        m = np.isfinite(sink) & np.isfinite(ent) & np.isfinite(y.astype(float))
        sink = sink[m]
        ent = ent[m]
        y = y[m].astype(int)
        if sink.size < 80 or len(set(int(v) for v in y)) < 2:
            continue

        per_rep = []
        for rep in range(int(args.repeats)):
            if args.eval_mode == "in_sample":
                train_idx = np.arange(len(y))
                test_idx = np.arange(len(y))
            else:
                train_idx, test_idx = stratified_split(y, float(args.test_frac), rng)

            y_tr, y_te = y[train_idx], y[test_idx]
            e_tr, e_te = ent[train_idx], ent[test_idx]
            s_tr, s_te = sink[train_idx], sink[test_idx]

            e_stats = zscore_fit(e_tr[np.isfinite(e_tr)])
            s_stats = zscore_fit(s_tr[np.isfinite(s_tr)])
            e_tr_z = zscore_apply(e_tr, e_stats)[:, None]
            e_te_z = zscore_apply(e_te, e_stats)[:, None]
            both_tr = np.stack([zscore_apply(e_tr, e_stats), zscore_apply(s_tr, s_stats)], axis=1)
            both_te = np.stack([zscore_apply(e_te, e_stats), zscore_apply(s_te, s_stats)], axis=1)

            fit_a = fit_logistic(e_tr_z, y_tr)
            fit_b = fit_logistic(both_tr, y_tr)
            p_a = predict_proba(e_te_z, fit_a)
            p_b = predict_proba(both_te, fit_b)

            mp = np.isfinite(p_a) & np.isfinite(p_b)
            y_te2 = y_te[mp]
            p_a2 = p_a[mp]
            p_b2 = p_b[mp]

            eval_a = eval_probs(y_te2, p_a2)
            eval_b = eval_probs(y_te2, p_b2)
            boot = _bootstrap_delta_test(rng, y_te2, p_a2, p_b2, n_boot=n_boot) if args.eval_mode == "holdout" else bootstrap_delta(rng, e_tr_z, both_tr, y_tr, n_boot=n_boot)

            per_rep.append(
                {
                    "rep": rep,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "pos_rate_train": float(np.mean(y_tr)),
                    "pos_rate_test": float(np.mean(y_te2)),
                    "entropy_only": eval_a,
                    "entropy_plus_sink": eval_b,
                    "bootstrap": boot,
                }
            )

        d_ll = np.array([r["bootstrap"]["delta_logloss"]["mean"] for r in per_rep], dtype=float)
        d_auc = np.array([r["bootstrap"]["delta_auroc"]["mean"] for r in per_rep], dtype=float)
        d_ll = d_ll[np.isfinite(d_ll)]
        d_auc = d_auc[np.isfinite(d_auc)]

        per_run_results.append(
            {
                "run_id": run.path.name.replace(".jsonl.gz", ""),
                "task": run.task,
                "model": run.model,
                "chat_mode": run.chat_mode,
                "quantization": run.quantization,
                "label": label_name,
                "n": int(len(y)),
                "pos_rate": float(y.mean()),
                "eval_mode": args.eval_mode,
                "test_frac": float(args.test_frac),
                "repeats": int(args.repeats),
                "bootstrap": int(n_boot),
                "per_repeat": per_rep,
                "summary": {
                    "delta_logloss_mean_over_repeats": float(np.mean(d_ll)) if d_ll.size else float("nan"),
                    "delta_auroc_mean_over_repeats": float(np.mean(d_auc)) if d_auc.size else float("nan"),
                },
            }
        )

    out = {
        "eval_mode": args.eval_mode,
        "test_frac": float(args.test_frac),
        "repeats": int(args.repeats),
        "bootstrap": int(n_boot),
        "n_runs": int(len(per_run_results)),
        "per_run": per_run_results,
        "inputs": [str(p) for p in inputs],
        "note": "Per-run evaluation (per model). Holdout: fit on train, evaluate on test. Bootstrap deltas computed on test predictions.",
        "generated_at": time.time(),
    }
    # H3 summary must aggregate across runs (not the last processed run).
    run_dll = (
        np.array([float(r["summary"]["delta_logloss_mean_over_repeats"]) for r in per_run_results], dtype=float)
        if per_run_results
        else np.array([], dtype=float)
    )
    run_dauc = (
        np.array([float(r["summary"]["delta_auroc_mean_over_repeats"]) for r in per_run_results], dtype=float)
        if per_run_results
        else np.array([], dtype=float)
    )
    run_dll = run_dll[np.isfinite(run_dll)]
    run_dauc = run_dauc[np.isfinite(run_dauc)]
    out["summary"] = {
        "delta_logloss_mean_over_runs": float(np.mean(run_dll)) if run_dll.size else float("nan"),
        "delta_auroc_mean_over_runs": float(np.mean(run_dauc)) if run_dauc.size else float("nan"),
        "n_runs_used": int(run_dll.size) if run_dll.size else 0,
    }
    write_json(plots_dir / "metrics.json", out)

    agg_dir = ensure_dir(plots_dir / "agg")

    # H3-specific aggregates -> plots/agg/
    if plt is not None and per_run_results:
        labels = np.array([str(r["model"]) for r in per_run_results], dtype=object)
        d_ll_run = np.array([float(r["summary"]["delta_logloss_mean_over_repeats"]) for r in per_run_results], dtype=float)
        d_auc_run = np.array([float(r["summary"]["delta_auroc_mean_over_repeats"]) for r in per_run_results], dtype=float)

        # per-model ΔAUROC
        m = np.isfinite(d_auc_run)
        if int(m.sum()) >= 1:
            order = np.argsort(d_auc_run[m])
            plt.figure(figsize=(10.8, 4.2))
            plt.plot(d_auc_run[m][order], marker="o")
            plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
            plt.xticks(range(int(m.sum())), labels[m][order], rotation=60, ha="right", fontsize=7)
            plt.ylabel("ΔAUROC (entropy+sink - entropy)")
            plt.title("H3: per-model added value (ΔAUROC over repeats)")
            plt.tight_layout()
            plt.savefig(agg_dir / "per_model_delta_auroc.png", dpi=150)
            plt.close()

        # per-model Δlogloss
        m = np.isfinite(d_ll_run)
        if int(m.sum()) >= 1:
            order = np.argsort(d_ll_run[m])
            plt.figure(figsize=(10.8, 4.2))
            plt.plot(d_ll_run[m][order], marker="o")
            plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
            plt.xticks(range(int(m.sum())), labels[m][order], rotation=60, ha="right", fontsize=7)
            plt.ylabel("Δlogloss (entropy+sink - entropy)")
            plt.title("H3: per-model added value (Δlogloss over repeats)")
            plt.tight_layout()
            plt.savefig(agg_dir / "per_model_delta_logloss.png", dpi=150)
            plt.close()

        # scatter ΔAUROC vs Δlogloss (runs)
        m = np.isfinite(d_auc_run) & np.isfinite(d_ll_run)
        if int(m.sum()) >= 3:
            plt.figure(figsize=(4.8, 4.2))
            plt.scatter(d_ll_run[m], d_auc_run[m], s=35, alpha=0.8)
            plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
            plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
            plt.xlabel("Δlogloss")
            plt.ylabel("ΔAUROC")
            plt.title("H3: added value across models")
            plt.tight_layout()
            plt.savefig(agg_dir / "scatter_delta_auroc_vs_delta_logloss.png", dpi=150)
            plt.close()

        # histograms across models
        if int(np.isfinite(d_auc_run).sum()) >= 3:
            x = d_auc_run[np.isfinite(d_auc_run)]
            plt.figure(figsize=(5.0, 3.0))
            plt.hist(x, bins=20)
            plt.axvline(0.0, color="black", linewidth=1)
            plt.title("H3: ΔAUROC distribution across models")
            plt.xlabel("ΔAUROC")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(agg_dir / "hist_delta_auroc_across_models.png", dpi=150)
            plt.close()

        if int(np.isfinite(d_ll_run).sum()) >= 3:
            x = d_ll_run[np.isfinite(d_ll_run)]
            plt.figure(figsize=(5.0, 3.0))
            plt.hist(x, bins=20)
            plt.axvline(0.0, color="black", linewidth=1)
            plt.title("H3: Δlogloss distribution across models")
            plt.xlabel("Δlogloss")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(agg_dir / "hist_delta_logloss_across_models.png", dpi=150)
            plt.close()

        # per-repeat deltas pooled across all models (stability check)
        all_rep_dll = np.array(
            [float(rr["bootstrap"]["delta_logloss"]["mean"]) for r in per_run_results for rr in r.get("per_repeat", [])],
            dtype=float,
        )
        all_rep_dauc = np.array(
            [float(rr["bootstrap"]["delta_auroc"]["mean"]) for r in per_run_results for rr in r.get("per_repeat", [])],
            dtype=float,
        )
        all_rep_dll = all_rep_dll[np.isfinite(all_rep_dll)]
        all_rep_dauc = all_rep_dauc[np.isfinite(all_rep_dauc)]

        if all_rep_dll.size:
            plt.figure(figsize=(5.0, 3.0))
            plt.hist(all_rep_dll, bins=min(25, max(8, int(all_rep_dll.size // 2))))
            plt.axvline(float(np.mean(all_rep_dll)), color="red", linewidth=1)
            plt.axvline(0.0, color="black", linewidth=1)
            plt.title("H3: per-repeat Δlogloss (pooled across models)")
            plt.xlabel("Δlogloss")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(agg_dir / "hist_delta_logloss_over_repeats_pooled.png", dpi=150)
            plt.close()

        if all_rep_dauc.size:
            plt.figure(figsize=(5.0, 3.0))
            plt.hist(all_rep_dauc, bins=min(25, max(8, int(all_rep_dauc.size // 2))))
            plt.axvline(float(np.mean(all_rep_dauc)), color="red", linewidth=1)
            plt.axvline(0.0, color="black", linewidth=1)
            plt.title("H3: per-repeat ΔAUROC (pooled across models)")
            plt.xlabel("ΔAUROC")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(agg_dir / "hist_delta_auroc_over_repeats_pooled.png", dpi=150)
            plt.close()

    # Extra local-only analysis: per-run diagnostics + aggregate summaries.
    per_run_summaries = []
    extra_items: List[Dict[str, Any]] = []
    runs_out_dir = ensure_dir(plots_dir / "runs")
    for run in runs:
        rid = run.path.name.replace(".jsonl.gz", "")
        per_run_summaries.append(summarize_run(run.rows, run_id=rid, label_mode=args.label))
        extra_items.extend(plot_basic_run_diagnostics(run.rows, runs_out_dir / rid, title=rid, label_mode=args.label))

    summary_csv = plot_aggregate_summary_table(per_run_summaries, agg_dir)
    extra_items.append({"kind": "table", "path": str(summary_csv), "desc": "Per-run summary table (csv)."})
    # Intentionally: no generic cross-run diagnostics in agg/ (only hypothesis-specific aggregations).

    write_manifest(
        plots_dir,
        hypothesis_id="H3_added_value_entropy",
        inputs=[Path(p) for p in inputs],
        items=[
            {"kind": "metrics", "path": str(plots_dir / "metrics.json"), "desc": "Pooled train/test delta metrics."},
            *(
                [
                    {"kind": "plot", "path": str(agg_dir / "per_model_delta_auroc.png"), "desc": "H3 aggregate: per-model ΔAUROC."},
                    {"kind": "plot", "path": str(agg_dir / "per_model_delta_logloss.png"), "desc": "H3 aggregate: per-model Δlogloss."},
                    {"kind": "plot", "path": str(agg_dir / "scatter_delta_auroc_vs_delta_logloss.png"), "desc": "H3 aggregate: ΔAUROC vs Δlogloss scatter."},
                    {"kind": "plot", "path": str(agg_dir / "hist_delta_auroc_across_models.png"), "desc": "H3 aggregate: ΔAUROC histogram across models."},
                    {"kind": "plot", "path": str(agg_dir / "hist_delta_logloss_across_models.png"), "desc": "H3 aggregate: Δlogloss histogram across models."},
                    {"kind": "plot", "path": str(agg_dir / "hist_delta_logloss_over_repeats_pooled.png"), "desc": "H3 aggregate: per-repeat Δlogloss pooled across models."},
                    {"kind": "plot", "path": str(agg_dir / "hist_delta_auroc_over_repeats_pooled.png"), "desc": "H3 aggregate: per-repeat ΔAUROC pooled across models."},
                ]
                if plt is not None
                else []
            ),
            *extra_items,
        ],
    )

    d_ll = float(out["summary"]["delta_logloss_mean_over_runs"])
    d_auc = float(out["summary"]["delta_auroc_mean_over_runs"])
    verdict = "supports (sink adds value)" if (np.isfinite(d_ll) and d_ll > 0) or (np.isfinite(d_auc) and d_auc > 0) else "does not support (no added value or unstable)"

    md = f"""# H3 — Added value over entropy (train/test)\n\n## Claim being tested\n`sink_mass` adds predictive value beyond next-token `entropy`.\n\n## Protocol\n- eval_mode: `{args.eval_mode}` (test_frac={args.test_frac}, repeats={args.repeats})\n- bootstrap: {n_boot}\n\n## What we fit\n- A: logistic regression `y ~ zscore(entropy)` trained on train, evaluated on test\n- B: logistic regression `y ~ zscore(entropy) + zscore(sink_mass)` trained on train, evaluated on test\n\n## Results\nWe report mean Δ over runs (positive => sink helps):\n- mean Δlogloss: {d_ll:.4f}\n- mean ΔAUROC: {d_auc:.4f}\n\nInterpretation: **{verdict}** (check per-run + per-repeat in `plots/metrics.json`).\n\n## Artifacts\n- report: `plots/metrics.json`\n- aggregated plots: `plots/agg/`\n\n## Extra diagnostics\n- per-run: `plots/runs/`\n- aggregate: `plots/agg/` (table: `{summary_csv.name}`)\n"""
    (hyp_dir / "final.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

