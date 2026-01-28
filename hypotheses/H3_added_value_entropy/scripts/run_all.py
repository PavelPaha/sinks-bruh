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

    rows = [r for r in all_rows if r.get("sink_mass") is not None and r.get("entropy") is not None and r.get("correct") is not None]
    label_name, y = choose_label(rows, args.label)
    y = y.astype(int)
    sink = np.array([float(r["sink_mass"]) for r in rows], dtype=float)
    ent = np.array([float(r["entropy"]) for r in rows], dtype=float)

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

        e_stats = zscore_fit(e_tr)
        s_stats = zscore_fit(s_tr)
        e_tr_z = zscore_apply(e_tr, e_stats)[:, None]
        e_te_z = zscore_apply(e_te, e_stats)[:, None]
        both_tr = np.stack([zscore_apply(e_tr, e_stats), zscore_apply(s_tr, s_stats)], axis=1)
        both_te = np.stack([zscore_apply(e_te, e_stats), zscore_apply(s_te, s_stats)], axis=1)

        fit_a = fit_logistic(e_tr_z, y_tr)
        fit_b = fit_logistic(both_tr, y_tr)
        p_a = predict_proba(e_te_z, fit_a)
        p_b = predict_proba(both_te, fit_b)

        eval_a = {**eval_probs(y_te, p_a), "auprc": float("nan")}
        eval_b = {**eval_probs(y_te, p_b), "auprc": float("nan")}

        boot = _bootstrap_delta_test(rng, y_te, p_a, p_b, n_boot=n_boot) if args.eval_mode == "holdout" else bootstrap_delta(rng, e_tr_z, both_tr, y_tr, n_boot=n_boot)

        per_rep.append(
            {
                "rep": rep,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "pos_rate_train": float(np.mean(y_tr)),
                "pos_rate_test": float(np.mean(y_te)),
                "entropy_only": eval_a,
                "entropy_plus_sink": eval_b,
                "bootstrap": boot,
            }
        )

    # aggregate deltas across repeats (mean)
    d_ll = np.array([r["bootstrap"]["delta_logloss"]["mean"] for r in per_rep], dtype=float)
    d_auc = np.array([r["bootstrap"]["delta_auroc"]["mean"] for r in per_rep], dtype=float)
    d_ll = d_ll[np.isfinite(d_ll)]
    d_auc = d_auc[np.isfinite(d_auc)]

    out = {
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
        "inputs": [str(p) for p in inputs],
        "note": "Holdout mode: fit on train, evaluate on test. Bootstrap deltas computed on test predictions.",
        "generated_at": time.time(),
    }
    write_json(plots_dir / "metrics.json", out)

    # plots: histogram of per-repeat delta means
    if d_ll.size:
        plt.figure(figsize=(5, 3))
        plt.hist(d_ll, bins=min(20, max(5, len(d_ll))))
        plt.axvline(float(np.mean(d_ll)), color="red", linewidth=1)
        plt.title("Holdout repeats: mean Δlogloss (positive => sink adds value)")
        plt.tight_layout()
        plt.savefig(plots_dir / "delta_logloss_over_repeats.png", dpi=150)
        plt.close()

    if d_auc.size:
        plt.figure(figsize=(5, 3))
        plt.hist(d_auc, bins=min(20, max(5, len(d_auc))))
        plt.axvline(float(np.mean(d_auc)), color="red", linewidth=1)
        plt.title("Holdout repeats: mean ΔAUROC (positive => sink adds value)")
        plt.tight_layout()
        plt.savefig(plots_dir / "delta_auroc_over_repeats.png", dpi=150)
        plt.close()

    md = f"""# H3 — Added value over entropy (train/test)\n\n## Claim being tested\n`sink_mass` adds predictive value beyond next-token `entropy`.\n\n## Protocol\n- eval_mode: `{args.eval_mode}` (test_frac={args.test_frac}, repeats={args.repeats})\n- bootstrap: {n_boot}\n\n## What we fit\n- A: logistic regression `y ~ zscore(entropy)` trained on train, evaluated on test\n- B: logistic regression `y ~ zscore(entropy) + zscore(sink_mass)` trained on train, evaluated on test\n\n## Results\nWe report mean Δ over repeats (positive => sink helps):\n- mean Δlogloss: {out['summary']['delta_logloss_mean_over_repeats']:.4f}\n- mean ΔAUROC: {out['summary']['delta_auroc_mean_over_repeats']:.4f}\n\nFull per-repeat report: `{plots_dir / 'metrics.json'}`\nPlots: `{plots_dir / 'delta_logloss_over_repeats.png'}`, `{plots_dir / 'delta_auroc_over_repeats.png'}`\n"""
    (hyp_dir / "final.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

