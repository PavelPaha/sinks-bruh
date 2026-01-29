"""
Accuracy vs Sink Mass curves.

Outputs (ONE file per task):
  - artifacts/plots/accuracy_vs_sink/accuracy_grid_<task>.png
  - artifacts/plots/accuracy_vs_sink/accuracy_summary.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _short_name(model: str, quant: str, chat: str) -> str:
    name = model.split("/")[-1] if model else "?"
    if quant not in (None, "none", ""): name += " (4b)"
    if chat == "off": name += " [raw]"
    return name


def _bin_accuracy(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    df = df.dropna(subset=["sink_mass", "correct"]).copy()
    df["sink_mass"] = df["sink_mass"].astype(float)
    df["correct"] = df["correct"].astype(bool)
    if len(df) < 50: return pd.DataFrame()
    
    try:
        df["bin"] = pd.qcut(df["sink_mass"], q=bins, duplicates="drop")
    except:
        df["bin"] = pd.cut(df["sink_mass"], bins=bins)
    
    agg = df.groupby("bin", observed=True).agg(
        acc=("correct", "mean"),
        n=("correct", "count"),
        sink_center=("sink_mass", "mean"),
    ).reset_index(drop=True)
    
    return agg[agg["n"] >= 10].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "artifacts" / "plots" / "accuracy_vs_sink")
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
            if "sink_mass" not in df.columns or "correct" not in df.columns: continue
            
            agg = _bin_accuracy(df, args.bins)
            if agg.empty: continue
            
            task = df["task"].iloc[0] if "task" in df.columns else "unknown"
            model = df["model"].iloc[0] if "model" in df.columns else "?"
            quant = df.get("quantization", pd.Series(["none"])).iloc[0]
            chat = df.get("chat_mode", pd.Series(["auto"])).iloc[0]
            
            agg["name"] = _short_name(model, quant, chat)
            agg["task"] = task
            agg["model"] = model
            
            if task not in task_data: task_data[task] = []
            task_data[task].append({"name": agg["name"].iloc[0], "agg": agg, "model": model, "quant": quant, "chat": chat})
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
            })
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "accuracy_summary.csv", index=False)
        print(f"Saved {out_dir / 'accuracy_summary.csv'}")

    # Grid plot per task
    for task, items in task_data.items():
        if len(items) < 1: continue
        
        items.sort(key=lambda x: x["name"])
        
        N = len(items)
        cols = min(4, N)
        rows_n = math.ceil(N / cols)
        
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), constrained_layout=True)
        fig.suptitle(f"Accuracy vs Sink Mass: {task}", fontsize=14, fontweight="bold")
        
        flat_axes = axes.flatten() if N > 1 else [axes]
        
        for i, ax in enumerate(flat_axes):
            if i < N:
                item = items[i]
                sub = item["agg"].sort_values("sink_center")
                
                ax.plot(sub["sink_center"], sub["acc"], marker="o", linewidth=2, markersize=5)
                ax.axhline(sub["acc"].mean(), color="gray", linestyle="--", alpha=0.5)
                
                ax.set_title(item["name"], fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_xlabel("sink" if i >= (rows_n - 1) * cols else "")
                ax.set_ylabel("acc" if i % cols == 0 else "")
                ax.grid(True, alpha=0.3)
            else:
                ax.axis("off")
        
        grid_path = out_dir / f"accuracy_grid_{task}.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
