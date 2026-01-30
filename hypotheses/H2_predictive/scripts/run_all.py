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
    if plt is None:
        return

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

    # Evaluate PER RUN (per-model consistency), not pooled across all models.
    per_run_results: List[Dict[str, Any]] = []
    last_eval = None

    def agg(per_rep: List[Dict[str, Any]], key_chain: List[str]) -> Dict[str, float]:
        vals = []
        for r in per_rep:
            cur: Any = r
            for k in key_chain:
                cur = cur[k]
            vals.append(float(cur))
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}

    for run in runs:
        run_rows = [
            r
            for r in run.rows
            if (r.get("sink_mass") is not None)
            and (r.get("entropy") is not None)
            and (r.get("correct") is not None)
        ]
        if not run_rows:
            continue
        label_name, y = choose_label(run_rows, args.label)
        sink = np.array([float(r["sink_mass"]) for r in run_rows], dtype=float)
        ent = np.array([float(r["entropy"]) for r in run_rows], dtype=float)
        m = np.isfinite(sink) & np.isfinite(ent) & np.isfinite(y.astype(float))
        sink = sink[m]
        ent = ent[m]
        y = y[m].astype(int)
        if sink.size < 50 or len(set(int(v) for v in y)) < 2:
            continue

        per_rep = []
        for rep in range(int(args.repeats)):
            if args.eval_mode == "in_sample":
                train_idx = np.arange(len(y))
                test_idx = np.arange(len(y))
            else:
                train_idx, test_idx = stratified_split(y, float(args.test_frac), rng)

            y_tr, y_te = y[train_idx], y[test_idx]
            s_tr, s_te = sink[train_idx], sink[test_idx]

            # z-score on train only
            s_stats = zscore_fit(s_tr[np.isfinite(s_tr)])
            s_tr_z = zscore_apply(s_tr, s_stats)[:, None]
            s_te_z = zscore_apply(s_te, s_stats)[:, None]

            fit = fit_logistic(s_tr_z, y_tr.astype(int))
            p_te = predict_proba(s_te_z, fit)

            # Filter non-finite predictions just in case
            mp = np.isfinite(p_te)
            y_te2 = y_te[mp]
            s_te2 = s_te[mp]
            p_te2 = p_te[mp]

            eval_raw = {
                "auroc": float(roc_auc(y_te2, s_te2)),
                "auprc": float(pr_auc(y_te2, s_te2)),
            }
            eval_clf = {
                **eval_probs(y_te2.astype(int), p_te2),
                "auprc": float(pr_auc(y_te2, p_te2)),
            }

            per_rep.append(
                {
                    "rep": rep,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "pos_rate_train": float(np.mean(y_tr)),
                    "pos_rate_test": float(np.mean(y_te2)),
                    "raw_sink_mass": eval_raw,
                    "logreg_sink_mass": eval_clf,
                }
            )

            # Keep curves for the last processed run+rep
            last_eval = (y_te2.astype(int), {"sink_mass": s_te2, "sink_logreg": p_te2})

        per_run_results.append(
            {
                "run_id": run.path.name.replace(".jsonl.gz", ""),
                "task": run.task,
                "model": run.model,
                "chat_mode": run.chat_mode,
                "quantization": run.quantization,
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
            }
        )

    out = {
        "eval_mode": args.eval_mode,
        "test_frac": float(args.test_frac),
        "repeats": int(args.repeats),
        "n_runs": int(len(per_run_results)),
        "per_run": per_run_results,
        "inputs": [str(p) for p in inputs],
        "generated_at": time.time(),
    }
    write_json(plots_dir / "metrics.json", out)

    agg_dir = ensure_dir(plots_dir / "agg")

    # Curves: last split (from last processed run)
    if last_eval is not None and plt is not None:
        y_plot, feats = last_eval
        save_curves(plots_dir, y=y_plot, features=feats)

    # H2-specific aggregates (across models) -> plots/agg/
    if plt is not None and per_run_results:
        labels = np.array([str(r["model"]) for r in per_run_results], dtype=object)
        raw_auroc = np.array([float(r["summary"]["raw_sink_mass"]["auroc"]["mean"]) for r in per_run_results], dtype=float)
        clf_auroc = np.array([float(r["summary"]["logreg_sink_mass"]["auroc"]["mean"]) for r in per_run_results], dtype=float)
        raw_auprc = np.array([float(r["summary"]["raw_sink_mass"]["auprc"]["mean"]) for r in per_run_results], dtype=float)
        clf_auprc = np.array([float(r["summary"]["logreg_sink_mass"]["auprc"]["mean"]) for r in per_run_results], dtype=float)
        clf_ll = np.array([float(r["summary"]["logreg_sink_mass"]["logloss"]["mean"]) for r in per_run_results], dtype=float)

        # 1) AUROC canvas
        order = np.argsort(raw_auroc)
        plt.figure(figsize=(10.8, 4.2))
        plt.plot(raw_auroc[order], marker="o", label="raw sink_mass (AUROC)")
        plt.plot(clf_auroc[order], marker="o", label="logreg(sink_mass) (AUROC)")
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        plt.xticks(range(len(labels)), labels[order], rotation=60, ha="right", fontsize=7)
        plt.ylabel("AUROC")
        plt.title("H2: per-model predictive power (AUROC)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        p = agg_dir / "per_model_auroc.png"
        plt.savefig(p, dpi=150)
        plt.close()

        # 2) AUPRC canvas
        order = np.argsort(raw_auprc)
        plt.figure(figsize=(10.8, 4.2))
        plt.plot(raw_auprc[order], marker="o", label="raw sink_mass (AUPRC)")
        plt.plot(clf_auprc[order], marker="o", label="logreg(sink_mass) (AUPRC)")
        plt.xticks(range(len(labels)), labels[order], rotation=60, ha="right", fontsize=7)
        plt.ylabel("AUPRC")
        plt.title("H2: per-model predictive power (AUPRC)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        p = agg_dir / "per_model_auprc.png"
        plt.savefig(p, dpi=150)
        plt.close()

        # 3) Logloss by model (classifier)
        order = np.argsort(clf_ll)
        plt.figure(figsize=(10.8, 4.2))
        plt.plot(clf_ll[order], marker="o", label="logreg(sink_mass) logloss")
        plt.xticks(range(len(labels)), labels[order], rotation=60, ha="right", fontsize=7)
        plt.ylabel("logloss")
        plt.title("H2: per-model classifier calibration (logloss)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        p = agg_dir / "per_model_logloss.png"
        plt.savefig(p, dpi=150)
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
        hypothesis_id="H2_predictive",
        inputs=[Path(p) for p in inputs],
        items=[
            {"kind": "metrics", "path": str(plots_dir / "metrics.json"), "desc": "Pooled train/test evaluation metrics."},
            *(
                [
                    {"kind": "plot", "path": str(plots_dir / "roc.png"), "desc": "ROC curves on last split (pooled)."},
                    {"kind": "plot", "path": str(plots_dir / "pr.png"), "desc": "PR curves on last split (pooled)."},
                ]
                if (plt is not None and last_eval is not None)
                else []
            ),
            *(
                [
                    {"kind": "plot", "path": str(agg_dir / "per_model_auroc.png"), "desc": "H2 aggregate: per-model AUROC (raw vs logreg)."},
                    {"kind": "plot", "path": str(agg_dir / "per_model_auprc.png"), "desc": "H2 aggregate: per-model AUPRC (raw vs logreg)."},
                    {"kind": "plot", "path": str(agg_dir / "per_model_logloss.png"), "desc": "H2 aggregate: per-model logloss (logreg)."},
                ]
                if (plt is not None and (agg_dir / "per_model_auroc.png").exists())
                else []
            ),
            *extra_items,
        ],
    )

    # Summaries from per-run raw AUROC (from diagnostics) still useful, but H2's classifier results live in plots/metrics.json now.
    finite_auroc = [s for s in per_run_summaries if np.isfinite(s.auroc_sink_vs_label)]
    best = sorted(finite_auroc, key=lambda s: float(s.auroc_sink_vs_label), reverse=True)[:3]
    worst = sorted(finite_auroc, key=lambda s: float(s.auroc_sink_vs_label))[:3]
    best_str = "\n".join([f"- {s.task} / {s.model}: AUROC={float(s.auroc_sink_vs_label):.3f} (n={s.n})" for s in best]) if best else "- (none)"
    worst_str = "\n".join([f"- {s.task} / {s.model}: AUROC={float(s.auroc_sink_vs_label):.3f} (n={s.n})" for s in worst]) if worst else "- (none)"

    md = f"""# H2 — Predictive (train/test)\n\n## Claim being tested\n`sink_mass` can be used to predict the positive class (hallucinated/incorrect).\n\n## Protocol\n- eval_mode: `{args.eval_mode}` (test_frac={args.test_frac}, repeats={args.repeats})\n\n## What we fit (per model/run)\n- baseline: AUROC/AUPRC of raw `sink_mass`\n- classifier: logistic regression `y ~ zscore(sink_mass)` trained on train, evaluated on test\n\n## Where the *actual* results are\n- per-model/per-run results: `plots/metrics.json`\n- per-model AUROC canvas: `plots/per_model_auroc.png` (if generated)\n- example curves (last processed split): `plots/roc.png`, `plots/pr.png`\n\n## Heterogeneity across runs (raw sink AUROC from run summaries)\nBest:\n{best_str}\n\nWorst:\n{worst_str}\n\n## Extra diagnostics\n- per-run: `plots/runs/`\n- aggregate: `plots/agg/` (table: `{summary_csv.name}`)\n"""
    (hyp_dir / "final.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

