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
from hypotheses._lib.logreg import eval_probs, fit_logistic, predict_proba, zscore_apply, zscore_fit
from hypotheses._lib.repo import find_repo_root
from hypotheses._lib.runner import run_measure_sink_text


def roc_curve(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) points for ROC (no sklearn)."""
    order = np.argsort(-s, kind="mergesort")
    y = y[order].astype(int)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tp_total = tp[-1] if len(tp) else 0
    fp_total = fp[-1] if len(fp) else 0
    tpr = tp / max(1, tp_total)
    fpr = fp / max(1, fp_total)
    # include origin
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def pr_curve(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (recall, precision) points for PR (no sklearn)."""
    order = np.argsort(-s, kind="mergesort")
    y = y[order].astype(int)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n_pos = int(y.sum())
    recall = tp / max(1, n_pos)
    precision = tp / np.maximum(1, tp + fp)
    return np.concatenate([[0.0], recall]), np.concatenate([[1.0], precision])


def save_curves(out_dir: Path, *, y: np.ndarray, features: Dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    plt.figure(figsize=(4.5, 4.5))
    for name, s in features.items():
        fpr, tpr = roc_curve(y, s)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=150)
    plt.close()

    # PR
    plt.figure(figsize=(4.5, 4.5))
    for name, s in features.items():
        r, p = pr_curve(y, s)
        plt.plot(r, p, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pr.png", dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure", action="store_true", help="Run measurements from config.json into data/")
    p.add_argument("--label", default="auto", choices=["auto", "hallucinated", "incorrect"])
    p.add_argument("--eval_mode", choices=["holdout", "in_sample"], default="holdout")
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
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

    runs = load_runs(data_dir, min_rows=100)
    all_rows = [row for run in runs for row in run.rows]
    required_ok, missing = require_keys(all_rows, ["sink_mass", "entropy", "correct"])
    if not required_ok:
        raise SystemExit(f"Missing required keys across runs: {missing}")

    rows = [
        r
        for r in all_rows
        if r.get("sink_mass") is not None and r.get("entropy") is not None and r.get("correct") is not None
    ]
    label_name, y = choose_label(rows, args.label)
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

    # Evaluate: raw score (no training) AND trained logistic classifier on sink_mass.
    per_rep = []
    last_eval = None
    for rep in range(int(args.repeats)):
        if args.eval_mode == "in_sample":
            train_idx = np.arange(len(y))
            test_idx = np.arange(len(y))
        else:
            train_idx, test_idx = stratified_split(y, float(args.test_frac), rng)

        y_tr, y_te = y[train_idx], y[test_idx]
        s_tr, s_te = sink[train_idx], sink[test_idx]
        e_tr, e_te = ent[train_idx], ent[test_idx]

        # z-score on train only
        s_stats = zscore_fit(s_tr)
        e_stats = zscore_fit(e_tr)
        s_tr_z = zscore_apply(s_tr, s_stats)[:, None]
        s_te_z = zscore_apply(s_te, s_stats)[:, None]

        # logistic on sink
        fit = fit_logistic(s_tr_z, y_tr.astype(int))
        p_te = predict_proba(s_te_z, fit)

        eval_raw = {
            "auroc": float(roc_auc(y_te, s_te)),
            "auprc": float(pr_auc(y_te, s_te)),
        }
        eval_clf = {
            **eval_probs(y_te.astype(int), p_te),
            "auprc": float(pr_auc(y_te, p_te)),
        }

        per_rep.append(
            {
                "rep": rep,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "pos_rate_train": float(np.mean(y_tr)),
                "pos_rate_test": float(np.mean(y_te)),
                "raw_sink_mass": eval_raw,
                "logreg_sink_mass": eval_clf,
            }
        )
        last_eval = (y_te.astype(int), {"sink_mass": s_te, "sink_logreg": p_te})

    def agg(path: List[Dict[str, Any]], key_chain: List[str]) -> Dict[str, float]:
        vals = []
        for r in path:
            cur: Any = r
            for k in key_chain:
                cur = cur[k]
            vals.append(float(cur))
        arr = np.array(vals, dtype=float)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}

    out = {
        "label": label_name,
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
        "eval_mode": args.eval_mode,
        "test_frac": float(args.test_frac),
        "repeats": int(args.repeats),
        "per_repeat": per_rep,
        "summary": {
            "raw_sink_mass": {
                "auroc": agg(per_rep, ["raw_sink_mass", "auroc"]),
                "auprc": agg(per_rep, ["raw_sink_mass", "auprc"]),
            },
            "logreg_sink_mass": {
                "auroc": agg(per_rep, ["logreg_sink_mass", "auroc"]),
                "auprc": agg(per_rep, ["logreg_sink_mass", "auprc"]),
                "logloss": agg(per_rep, ["logreg_sink_mass", "logloss"]),
            },
        },
        "inputs": [str(p) for p in inputs],
        "generated_at": time.time(),
    }
    write_json(plots_dir / "metrics.json", out)

    # Curves: last split
    if last_eval is not None:
        y_plot, feats = last_eval
        save_curves(plots_dir, y=y_plot, features=feats)

    md = f"""# H2 — Predictive (train/test)\n\n## Claim being tested\n`sink_mass` can be used to predict the positive class (hallucinated/incorrect).\n\n## Protocol\n- label: `{label_name}`\n- eval_mode: `{args.eval_mode}` (test_frac={args.test_frac}, repeats={args.repeats})\n\n## What we fit\n- logistic regression: `y ~ zscore(sink_mass)` (trained on train, evaluated on test)\n- plus a no-training baseline: AUROC/AUPRC of raw `sink_mass`\n\n## Results (mean±std across repeats)\n- raw sink_mass AUROC: {out['summary']['raw_sink_mass']['auroc']['mean']:.3f} ± {out['summary']['raw_sink_mass']['auroc']['std']:.3f}\n- logreg(sink_mass) AUROC: {out['summary']['logreg_sink_mass']['auroc']['mean']:.3f} ± {out['summary']['logreg_sink_mass']['auroc']['std']:.3f}\n\nFull details: `{plots_dir / 'metrics.json'}`\nPlots (last split): `{plots_dir / 'roc.png'}`, `{plots_dir / 'pr.png'}`\n"""
    (hyp_dir / "final.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

