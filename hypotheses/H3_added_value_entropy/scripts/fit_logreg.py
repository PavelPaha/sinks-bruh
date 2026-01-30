from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize


def read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    s = s.astype(float)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    ranks = np.empty_like(s_sorted, dtype=float)
    i = 0
    r = 1.0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[i : j + 1] = avg_rank
        r += (j - i) + 1
        i = j + 1
    sum_r_pos = float(ranks[y_sorted == 1].sum())
    return (sum_r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def choose_label(rows: List[Dict[str, Any]], label_mode: str) -> Tuple[str, np.ndarray]:
    if label_mode == "hallucinated":
        y = np.array([1 if r.get("hallucinated") else 0 for r in rows], dtype=int)
        return "hallucinated", y
    if label_mode == "incorrect":
        y = np.array([0 if r.get("correct") else 1 for r in rows], dtype=int)
        return "incorrect", y
    # auto
    if any(r.get("hallucinated") is not None for r in rows):
        y = np.array([1 if r.get("hallucinated") else 0 for r in rows], dtype=int)
        return "hallucinated", y
    y = np.array([0 if r.get("correct") else 1 for r in rows], dtype=int)
    return "incorrect", y


@dataclass
class FitResult:
    w: np.ndarray
    bias: float


def fit_logistic(X: np.ndarray, y: np.ndarray) -> FitResult:
    # X: [n, d]
    n, d = X.shape

    def obj(theta: np.ndarray) -> float:
        w = theta[:d]
        b = theta[d]
        p = sigmoid(X @ w + b)
        return logloss(y, p)

    theta0 = np.zeros(d + 1, dtype=float)
    res = minimize(obj, theta0, method="BFGS")
    theta = res.x
    return FitResult(w=theta[:d], bias=float(theta[d]))


def eval_model(X: np.ndarray, y: np.ndarray, fit: FitResult) -> Dict[str, float]:
    p = sigmoid(X @ fit.w + fit.bias)
    return {
        "logloss": logloss(y, p),
        "auroc": roc_auc(y, p),
    }


def bootstrap_delta(
    rng: np.random.Generator,
    X_a: np.ndarray,
    X_b: np.ndarray,
    y: np.ndarray,
    n_boot: int,
) -> Dict[str, Dict[str, float]]:
    deltas_ll = []
    deltas_auc = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        Xa = X_a[idx]
        Xb = X_b[idx]
        fa = fit_logistic(Xa, yb)
        fb = fit_logistic(Xb, yb)
        ea = eval_model(Xa, yb, fa)
        eb = eval_model(Xb, yb, fb)
        deltas_ll.append(ea["logloss"] - eb["logloss"])  # positive means B better
        deltas_auc.append(eb["auroc"] - ea["auroc"])     # positive means B better
    deltas_ll = np.array(deltas_ll, dtype=float)
    deltas_auc = np.array(deltas_auc, dtype=float)
    return {
        "delta_logloss": {
            "mean": float(deltas_ll.mean()),
            "p05": float(np.quantile(deltas_ll, 0.05)),
            "p95": float(np.quantile(deltas_ll, 0.95)),
        },
        "delta_auroc": {
            "mean": float(deltas_auc.mean()),
            "p05": float(np.quantile(deltas_auc, 0.05)),
            "p95": float(np.quantile(deltas_auc, 0.95)),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=None)
    p.add_argument("--label", choices=["auto", "hallucinated", "incorrect"], default="auto")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    hyp_dir = repo_root / "hypotheses" / "H3_added_value_entropy"
    inputs = [Path(x) for x in args.inputs] if args.inputs else sorted((hyp_dir / "data").glob("*.jsonl.gz"))
    if not inputs:
        raise SystemExit("No inputs found. Put run files into hypotheses/H3_added_value_entropy/data/")

    rows_all: List[Dict[str, Any]] = []
    for path in inputs:
        rows_all.extend(read_jsonl_gz(path))

    rows = [
        r
        for r in rows_all
        if r.get("sink_mass") is not None
        and r.get("entropy") is not None
        and r.get("correct") is not None
    ]
    rows = [r for r in rows if np.isfinite(float(r["sink_mass"])) and np.isfinite(float(r["entropy"]))]  # type: ignore[index]
    if len(rows) < 200:
        raise SystemExit(f"Too few usable rows for stable fit: {len(rows)} (need ~200+)")

    label_name, y = choose_label(rows, args.label)
    y = y.astype(int)
    sink = np.array([float(r["sink_mass"]) for r in rows], dtype=float)
    ent = np.array([float(r["entropy"]) for r in rows], dtype=float)

    # standardize features
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    X_entropy = z(ent)[:, None]
    X_both = np.stack([z(ent), z(sink)], axis=1)

    fit_a = fit_logistic(X_entropy, y)
    fit_b = fit_logistic(X_both, y)
    eval_a = eval_model(X_entropy, y, fit_a)
    eval_b = eval_model(X_both, y, fit_b)

    rng = np.random.default_rng(args.seed)
    boot = bootstrap_delta(rng, X_entropy, X_both, y, n_boot=args.bootstrap)

    out = {
        "label": label_name,
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "entropy_only": eval_a,
        "entropy_plus_sink": eval_b,
        "bootstrap": boot,
        "inputs": [str(p) for p in inputs],
        "note": "Δlogloss = logloss(entropy-only) - logloss(entropy+sink). Positive => sink adds value.",
    }

    out_path = Path(args.out) if args.out else (hyp_dir / "plots" / "metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

