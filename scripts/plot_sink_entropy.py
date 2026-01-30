"""
Sink Mass vs Entropy correlation analysis.

Outputs (ONE file per task):
  - artifacts/plots/correlation/correlation_grid_<task>.png
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "artifacts" / "plots" / "correlation")
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
            if "sink_mass" not in df.columns or "entropy" not in df.columns: continue
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["sink_mass", "entropy"])
            if len(df) < 50: continue

            task = df["task"].iloc[0] if "task" in df.columns else "unknown"
            model = df["model"].iloc[0] if "model" in df.columns else "?"
            quant = df.get("quantization", pd.Series(["none"])).iloc[0]
            chat = df.get("chat_mode", pd.Series(["auto"])).iloc[0]
            
            r, _ = pearsonr(df["sink_mass"], df["entropy"])
            
            if task not in task_data: task_data[task] = []
            task_data[task].append({
                "name": _short_name(model, quant, chat),
                "df": df[["sink_mass", "entropy"]],
                "r": r,
            })
        except Exception as e:
            print(f"Skipped {p}: {e}")

    # Grid plot per task
    for task, items in task_data.items():
        if len(items) < 1: continue
        
        items.sort(key=lambda x: x["r"] if not np.isnan(x["r"]) else 0)
        
        N = len(items)
        cols = min(4, N)
        rows_n = math.ceil(N / cols)
        
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.5 * rows_n), constrained_layout=True)
        fig.suptitle(f"Sink-Entropy Correlation: {task}", fontsize=14, fontweight="bold")
        
        flat_axes = axes.flatten() if N > 1 else [axes]
        
        for i, ax in enumerate(flat_axes):
            if i < N:
                item = items[i]
                sub = item["df"]
                
                ax.scatter(sub["sink_mass"], sub["entropy"], alpha=0.3, s=8, c="steelblue")
                
                # Regression line
                z = np.polyfit(sub["sink_mass"], sub["entropy"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(sub["sink_mass"].min(), sub["sink_mass"].max(), 50)
                ax.plot(x_line, p(x_line), "r-", linewidth=1.5)
                
                r_val = item["r"]
                ax.set_title(f"{item['name']}\nr={r_val:.2f}", fontsize=9)
                ax.set_xlabel("sink" if i >= (rows_n - 1) * cols else "")
                ax.set_ylabel("entropy" if i % cols == 0 else "")
            else:
                ax.axis("off")
        
        grid_path = out_dir / f"correlation_grid_{task}.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
