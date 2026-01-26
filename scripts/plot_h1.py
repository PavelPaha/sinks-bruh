"""
H1 visualization: Distribution shift of sink_mass between hallucinated vs non-hallucinated.

Outputs (ONE file per task):
  - artifacts/plots/h1/h1_grid_<task>.png
  - artifacts/plots/h1/h1_summary.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


LabelMode = Literal["auto", "hallucinated", "incorrect"]


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cohens_d(x1: np.ndarray, x0: np.ndarray) -> float:
    n1, n0 = len(x1), len(x0)
    if n1 < 2 or n0 < 2:
        return float("nan")
    s1, s0 = np.std(x1, ddof=1), np.std(x0, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    return float((np.mean(x1) - np.mean(x0)) / sp) if sp > 0 else float("nan")


def _choose_label_mode(df: pd.DataFrame, mode: LabelMode) -> Tuple[str, pd.Series]:
    if mode == "hallucinated":
        return "hallucinated", df["hallucinated"].astype(bool)
    if mode == "incorrect":
        return "incorrect", (~df["correct"].astype(bool))
    # auto
    if "hallucinated" in df.columns and df["hallucinated"].nunique(dropna=True) >= 2:
        return "hallucinated", df["hallucinated"].astype(bool)
    if "correct" in df.columns and df["correct"].nunique(dropna=True) >= 2:
        return "incorrect", (~df["correct"].astype(bool))
    raise ValueError("No usable binary label")


def _short_name(model: str, quant: str, chat: str) -> str:
    name = model.split("/")[-1] if model else "?"
    if quant not in (None, "none", ""): name += " (4b)"
    if chat == "off": name += " [raw]"
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--label", type=str, default="auto", choices=["auto", "hallucinated", "incorrect"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "artifacts" / "plots" / "h1")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in args.inputs] if args.inputs else sorted((repo_root / "artifacts" / "sink_runs").glob("*.jsonl.gz"))
    if not paths:
        print("No inputs found.")
        return

    # Group by task
    task_data: Dict[str, List[Dict]] = {}
    
    for p in paths:
        try:
            rows = _read_jsonl_gz(p)
            if not rows: continue
            df = pd.DataFrame(rows)
            if "sink_mass" not in df.columns: continue
            df = df.dropna(subset=["sink_mass"]).copy()
            
            try:
                label_name, y = _choose_label_mode(df, args.label)
            except ValueError:
                continue
            
            df["label"] = y.astype(int)
            x1 = df.loc[df["label"] == 1, "sink_mass"].to_numpy()
            x0 = df.loc[df["label"] == 0, "sink_mass"].to_numpy()
            if len(x1) < 5 or len(x0) < 5: continue
            
            task = df["task"].iloc[0] if "task" in df.columns else "unknown"
            model = df["model"].iloc[0] if "model" in df.columns else "?"
            quant = df.get("quantization", pd.Series(["none"])).iloc[0]
            chat = df.get("chat_mode", pd.Series(["auto"])).iloc[0]
            
            d = _cohens_d(x1, x0)
            
            if task not in task_data: task_data[task] = []
            task_data[task].append({
                "name": _short_name(model, quant, chat),
                "df": df[["label", "sink_mass"]],
                "d": d,
                "label_name": label_name,
                "model": model,
                "quant": quant,
                "chat": chat,
                "n_pos": len(x1),
                "n_neg": len(x0),
            })
        except Exception as e:
            print(f"Skipped {p}: {e}")

    # Summary CSV
    summary_rows = []
    for task, items in task_data.items():
        for item in items:
            summary_rows.append({
                "task": task,
                "model": item["model"],
                "quantization": item["quant"],
                "chat_mode": item["chat"],
                "cohens_d": item["d"],
                "n_pos": item["n_pos"],
                "n_neg": item["n_neg"],
            })
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "h1_summary.csv", index=False)
        print(f"Saved {out_dir / 'h1_summary.csv'}")

    # Grid plot per task
    for task, items in task_data.items():
        if len(items) < 1: continue
        
        items.sort(key=lambda x: x["d"] if not np.isnan(x["d"]) else 0)
        
        N = len(items)
        cols = min(4, N)
        rows_n = math.ceil(N / cols)
        
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.5 * rows_n), constrained_layout=True)
        fig.suptitle(f"H1 Distribution Shift: {task}", fontsize=14, fontweight="bold")
        
        flat_axes = axes.flatten() if N > 1 else [axes]
        
        for i, ax in enumerate(flat_axes):
            if i < N:
                item = items[i]
                sns.violinplot(data=item["df"], x="label", y="sink_mass", ax=ax, inner="box", cut=0)
                d_val = item["d"]
                ax.set_title(f"{item['name']}\nd={d_val:+.2f}" if not np.isnan(d_val) else item["name"], fontsize=9)
                ax.set_xlabel("")
                ax.set_ylabel("sink" if i % cols == 0 else "")
            else:
                ax.axis("off")
        
        grid_path = out_dir / f"h1_grid_{task}.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
