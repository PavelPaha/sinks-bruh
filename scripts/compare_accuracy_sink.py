import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import glob
import argparse
import math
from pathlib import Path

def process_file(filepath):
    try:
        with open(filepath, "r") as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

    model_name = filepath.split("mmlu_accuracy_sink_")[-1].replace(".json", "")
    # Cleanup model name for display
    if "Qwen" in model_name:
        parts = model_name.split("-")
        size = next((p for p in parts if "B" in p), model_name)
        display_name = f"Qwen-{size}"
    elif "Mistral" in model_name:
        if "Nemo" in model_name:
            display_name = "Mistral-Nemo"
        else:
            display_name = "Mistral-7B"
    elif "Llama" in model_name:
        display_name = "Llama-3-8B"
    elif "Yi" in model_name:
        display_name = "Yi-34B"
    else:
        display_name = model_name

    df = pd.DataFrame(results)
    if "sink_mass" not in df.columns or "correct" not in df.columns:
        print(f"Skipping {filepath}: missing required columns")
        return None, None

    # Drop NaNs (some runs can produce NaN attentions)
    n_before = len(df)
    df = df.dropna(subset=["sink_mass", "correct"])
    n_after = len(df)

    print(f"\nProcessing {display_name} ({len(df)} samples):")
    if "sink_tokens" in df.columns:
        try:
            ks = sorted(set(int(x) for x in df["sink_tokens"].dropna().unique()))
            if ks:
                print(f"  Sink window (K tokens): {ks}")
        except Exception:
            pass
    print(f"  Sink Range: [{df['sink_mass'].min():.4f}, {df['sink_mass'].max():.4f}]")
    print(f"  Accuracy: {df['correct'].mean():.2%}")
    if n_after != n_before:
        print(f"  Dropped NaNs: {n_before - n_after}")

    # Quick correlation diagnostics (not perfect, but good sanity check)
    try:
        x = df["sink_mass"].to_numpy(dtype=float)
        y = df["correct"].to_numpy(dtype=float)
        if np.std(x) > 0 and np.std(y) > 0:
            pear = float(np.corrcoef(x, y)[0, 1])
        else:
            pear = float("nan")
        print(f"  Pearson(sink, correct): {pear:+.3f}")
    except Exception as e:
        print(f"  Correlation compute failed: {e}")

    # Quantile Binning
    try:
        df['Bin'] = pd.qcut(df['sink_mass'], q=15, duplicates='drop')
    except ValueError:
        df['Bin'] = pd.cut(df['sink_mass'], bins=15)

    bin_stats = df.groupby('Bin', observed=True).agg({
        'correct': ['mean', 'count'],
        'sink_mass': 'mean'
    }).reset_index()
    
    bin_stats.columns = ['Bin', 'Accuracy', 'Count', 'Sink Center']
    bin_stats = bin_stats[bin_stats['Count'] > 10]
    
    bin_stats['Model'] = display_name
    return bin_stats, df[['sink_mass', 'correct']].assign(Model=display_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default=None, help="Glob for result json files.")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to write plots into.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pattern = args.pattern or str(repo_root / "artifacts" / "results" / "mmlu_accuracy_sink_*.json")
    files = glob.glob(pattern)
    if not files:
        legacy = repo_root / "mmlu_accuracy_sink.json"
        if legacy.exists():
            files.append(str(legacy))
    
    if not files:
        print("No result files found!")
        return

    all_stats = []
    all_raw_data = []

    for f in files:
        stats, raw = process_file(f)
        if stats is not None:
            all_stats.append(stats)
            all_raw_data.append(raw)

    if not all_stats:
        print("No valid data extracted.")
        return

    full_stats = pd.concat(all_stats, ignore_index=True)
    full_raw = pd.concat(all_raw_data, ignore_index=True)

    plt.figure(figsize=(14, 9))
    sns.set_style("whitegrid")
    
    models = full_stats['Model'].unique()
    palette = sns.color_palette("husl", len(models))
    
    # 1. Plot Accuracy Curves
    sns.lineplot(
        data=full_stats, 
        x='Sink Center', 
        y='Accuracy', 
        hue='Model', 
        style='Model', 
        markers=True, 
        dashes=False, 
        linewidth=3, 
        palette=palette,
        markersize=9
    )

    # 2. Confidence Intervals
    for i, model in enumerate(models):
        subset = full_stats[full_stats['Model'] == model]
        p = subset['Accuracy']
        n = subset['Count']
        se = np.sqrt(p * (1 - p) / n)
        plt.fill_between(
            subset['Sink Center'], 
            p - 1.96*se, 
            p + 1.96*se, 
            color=palette[i], 
            alpha=0.1
        )

    # 3. Rug Plot
    sns.rugplot(
        data=full_raw, 
        x='sink_mass', 
        hue='Model', 
        palette=palette, 
        height=0.05, 
        alpha=0.3, 
        legend=False
    )

    plt.title("Accuracy vs Sink Mass: The Sink-Confidence Paradox across Models", fontsize=16)
    plt.xlabel("Sink Mass (Attention to Token 0)", fontsize=14)
    plt.ylabel("Accuracy (MMLU)", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "artifacts" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "accuracy_sink_comparison_full.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved {plot_path}")

if __name__ == "__main__":
    main()
