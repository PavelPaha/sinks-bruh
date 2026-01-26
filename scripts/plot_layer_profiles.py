"""
Layer-wise Sink Mass profiles.

Outputs (ONE file per task):
  - artifacts/plots/layers/layer_grid_<task>.png
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


def _get_layer_profile(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate sink_by_layer into a profile: [layer_rel, mean_sink]."""
    all_profiles = []
    for r in rows:
        sbl = r.get("sink_by_layer")
        if not isinstance(sbl, list) or not sbl: continue
        L = len(sbl)
        for i, val in enumerate(sbl):
            if val is None: continue
            all_profiles.append({
                "layer_rel": i / max(1, L - 1),
                "sink_mass": float(val),
            })
    
    if not all_profiles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_profiles)
    # Aggregate by layer_rel (bin to 20 points)
    df["layer_bin"] = (df["layer_rel"] * 20).round() / 20
    agg = df.groupby("layer_bin")["sink_mass"].mean().reset_index()
    agg.columns = ["layer_rel", "sink_mass"]
    return agg.sort_values("layer_rel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "artifacts" / "plots" / "layers")
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
            
            profile = _get_layer_profile(rows)
            if profile.empty: continue
            
            task = rows[0].get("task", "unknown")
            model = rows[0].get("model", "?")
            quant = rows[0].get("quantization")
            chat = rows[0].get("chat_mode")
            
            if task not in task_data: task_data[task] = []
            task_data[task].append({
                "name": _short_name(model, quant, chat),
                "profile": profile,
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
        fig.suptitle(f"Layer Profile Grid: {task}", fontsize=14, fontweight="bold")
        
        flat_axes = axes.flatten() if N > 1 else [axes]
        
        for i, ax in enumerate(flat_axes):
            if i < N:
                item = items[i]
                prof = item["profile"]
                
                ax.plot(prof["layer_rel"], prof["sink_mass"], linewidth=2)
                ax.fill_between(prof["layer_rel"], 0, prof["sink_mass"], alpha=0.2)
                
                ax.set_title(item["name"], fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_xlabel("depth" if i >= (rows_n - 1) * cols else "")
                ax.set_ylabel("sink" if i % cols == 0 else "")
                ax.grid(True, alpha=0.3)
            else:
                ax.axis("off")
        
        grid_path = out_dir / f"layer_grid_{task}.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
