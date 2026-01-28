from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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

    # ensure start at recall=0 with precision=1 for area stability
    recalls = np.array([0.0] + recalls, dtype=float)
    precisions = np.array([1.0] + precisions, dtype=float)
    return float(np.trapz(precisions, recalls))


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=None, help="jsonl.gz run files; default: hypotheses/H2_predictive/data/*.jsonl.gz")
    p.add_argument("--label", choices=["auto", "hallucinated", "incorrect"], default="auto")
    p.add_argument("--out", default=None, help="Where to write metrics json")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    hyp_dir = repo_root / "hypotheses" / "H2_predictive"
    inputs = [Path(x) for x in args.inputs] if args.inputs else sorted((hyp_dir / "data").glob("*.jsonl.gz"))
    if not inputs:
        raise SystemExit("No inputs found. Put run files into hypotheses/H2_predictive/data/")

    rows_all: List[Dict[str, Any]] = []
    for path in inputs:
        rows_all.extend(read_jsonl_gz(path))

    # filter
    rows = [r for r in rows_all if r.get("sink_mass") is not None and r.get("correct") is not None]
    rows = [r for r in rows if np.isfinite(float(r["sink_mass"]))]  # type: ignore[index]
    if len(rows) < 50:
        raise SystemExit(f"Too few usable rows: {len(rows)}")

    label_name, y = choose_label(rows, args.label)
    s = np.array([float(r["sink_mass"]) for r in rows], dtype=float)

    auc = roc_auc(y, s)
    ap = pr_auc(y, s)

    out = {
        "label": label_name,
        "n": int(len(rows)),
        "pos_rate": float(y.mean()),
        "auroc": float(auc),
        "auprc": float(ap),
        "inputs": [str(p) for p in inputs],
    }

    out_path = Path(args.out) if args.out else (hyp_dir / "plots" / "metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

