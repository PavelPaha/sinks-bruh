"""
H1 heatmap: Layer×Head delta sink mass between classes.

Outputs (ONE file per task):
  - artifacts/plots/h1_heatmap/heatmap_grid_<task>.png
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


def _choose_label_mode(df: pd.DataFrame, mode: LabelMode) -> Tuple[str, pd.Series]:
    if mode == "hallucinated":
        return "hallucinated", df["hallucinated"].astype(bool)
    if mode == "incorrect":
        return "incorrect", (~df["correct"].astype(bool))
    if "hallucinated" in df.columns and df["hallucinated"].nunique(dropna=True) >= 2:
        return "hallucinated", df["hallucinated"].astype(bool)
    if "correct" in df.columns and df["correct"].nunique(dropna=True) >= 2:
        return "incorrect", (~df["correct"].astype(bool))
    raise ValueError("No usable binary label")


def _compute_delta(rows: List[Dict[str, Any]], label_mode: LabelMode) -> Tuple[Optional[np.ndarray], str]:
    if not rows: return None, ""
    df = pd.DataFrame(rows)
    if "sink_by_layer_head" not in df.columns: return None, ""
    
    try:
        label_name, y = _choose_label_mode(df, label_mode)
    except ValueError:
        return None, ""
    
    # Find shape
    L, H = 0, 0
    for r in rows:
        m = r.get("sink_by_layer_head")
        if isinstance(m, list) and m:
            try:
                arr = np.array(m)
                if arr.ndim == 2:
                    L, H = arr.shape
                    break
            except: pass
    if L == 0: return None, ""
    
    sum_pos = np.zeros((L, H))
    sum_neg = np.zeros((L, H))
    n_pos = n_neg = 0
    
    for r in rows:
        m = r.get("sink_by_layer_head")
        if not isinstance(m, list): continue
        try:
            a = np.array(m)
            if a.shape != (L, H): continue
        except: continue
        
        is_pos = bool(r.get(label_name)) if label_name == "hallucinated" else not bool(r.get("correct"))
        if is_pos:
            sum_pos += a; n_pos += 1
        else:
            sum_neg += a; n_neg += 1
    
    if n_pos < 5 or n_neg < 5: return None, ""
    return (sum_pos / n_pos) - (sum_neg / n_neg), label_name


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
    parser.add_argument("--vmax", type=float, default=0.08)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "artifacts" / "plots" / "h1_heatmap")
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
            
            delta, lbl = _compute_delta(rows, args.label)
            if delta is None: continue
            
            task = rows[0].get("task", "unknown")
            model = rows[0].get("model", "?")
            quant = rows[0].get("quantization")
            chat = rows[0].get("chat_mode")
            
            if task not in task_data: task_data[task] = []
            task_data[task].append({
                "name": _short_name(model, quant, chat),
                "data": delta,
            })
        except Exception as e:
            print(f"Skipped {p}: {e}")

    # Grid plot per task
    for task, items in task_data.items():
        if len(items) < 1: continue
        
        items.sort(key=lambda x: x["name"])
        
        N = len(items)
        cols = min(4, N)
        rows_n = math.ceil(N / cols)
        
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), constrained_layout=True)
        fig.suptitle(f"Heatmap Grid: {task}\nΔ Sink Mass (Hallucinated - Correct)", fontsize=14)
        
        flat_axes = axes.flatten() if N > 1 else [axes]
        
        for i, ax in enumerate(flat_axes):
            if i < N:
                h = items[i]
                sns.heatmap(h["data"], ax=ax, cmap="RdBu_r", center=0, vmin=-args.vmax, vmax=args.vmax, cbar=False)
                ax.set_title(h["name"], fontsize=9)
                ax.set_xlabel("Head", fontsize=8)
                ax.set_ylabel("Layer", fontsize=8)
                ax.tick_params(axis='both', labelsize=6)
            else:
                ax.axis("off")
        
        grid_path = out_dir / f"heatmap_grid_{task}.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
