from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


def choose_label(rows: List[Dict], mode: str = "auto") -> Tuple[str, np.ndarray]:
    """
    Returns (label_name, y) where y=1 means "hallucinated/incorrect" (positive class).
    """
    mode = mode.lower()
    if mode == "hallucinated":
        y = np.array([1 if r.get("hallucinated") else 0 for r in rows], dtype=int)
        return "hallucinated", y
    if mode == "incorrect":
        y = np.array([0 if r.get("correct") else 1 for r in rows], dtype=int)
        return "incorrect", y
    # auto
    if any(r.get("hallucinated") is not None for r in rows):
        y = np.array([1 if r.get("hallucinated") else 0 for r in rows], dtype=int)
        return "hallucinated", y
    y = np.array([0 if r.get("correct") else 1 for r in rows], dtype=int)
    return "incorrect", y


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC via rank statistic; handles ties with average ranks."""
    y = y.astype(int)
    s = s.astype(float)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    # average ranks for ties
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


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUPRC via sorting by score desc and trapezoid on (recall, precision)."""
    y = y.astype(int)
    s = s.astype(float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]

    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for yi in y_sorted:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(1, tp + fp))
        recalls.append(tp / n_pos)

    recalls = np.array([0.0] + recalls, dtype=float)
    precisions = np.array([1.0] + precisions, dtype=float)
    return float(np.trapz(precisions, recalls))


def cohens_d(x1: np.ndarray, x0: np.ndarray) -> float:
    n1, n0 = len(x1), len(x0)
    if n1 < 2 or n0 < 2:
        return float("nan")
    s1, s0 = np.std(x1, ddof=1), np.std(x0, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    return float((np.mean(x1) - np.mean(x0)) / sp) if sp > 0 else float("nan")


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    p05: float
    p95: float


def bootstrap_ci(
    rng: np.random.Generator,
    y: np.ndarray,
    s: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 500,
) -> BootstrapCI:
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y[idx], s[idx]))
    v = np.array(vals, dtype=float)
    return BootstrapCI(mean=float(np.nanmean(v)), p05=float(np.nanquantile(v, 0.05)), p95=float(np.nanquantile(v, 0.95)))

