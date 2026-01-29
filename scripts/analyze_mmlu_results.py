import argparse
import json
import math
import os
from glob import glob
from pathlib import Path

from collections import defaultdict


def _is_finite_number(x) -> bool:
    try:
        xf = float(x)
    except Exception:
        return False
    return math.isfinite(xf)


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    if vx == 0 or vy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (math.sqrt(vx * vy) + 1e-12)


def load_results(path: str):
    with open(path, "r") as f:
        raw = json.load(f)
    # Normalize the shape a bit; keep as list[dict] to avoid heavy deps.
    rows = []
    for r in raw:
        rows.append({**r, "model_file": os.path.basename(path)})
    return rows


def print_model_summary(rows, model_name: str):
    ks = sorted({int(r["sink_tokens"]) for r in rows if isinstance(r, dict) and r.get("sink_tokens") is not None and _is_finite_number(r.get("sink_tokens"))})
    chat_modes = sorted({str(r.get("chat_mode")) for r in rows if isinstance(r, dict) and r.get("chat_mode") is not None})
    xs = []
    ys = []
    ents = []
    ents_y = []
    sink_correct = []
    sink_wrong = []
    ent_correct = []
    ent_wrong = []

    for r in rows:
        correct = bool(r.get("correct"))
        sink = r.get("sink_mass")
        if _is_finite_number(sink):
            sink = float(sink)
            xs.append(sink)
            ys.append(1.0 if correct else 0.0)
            (sink_correct if correct else sink_wrong).append(sink)
        ent = r.get("entropy")
        if _is_finite_number(ent) and _is_finite_number(sink):
            ent = float(ent)
            ents.append(ent)
            ents_y.append(1.0 if correct else 0.0)
            (ent_correct if correct else ent_wrong).append(ent)

    n = len(rows)
    n_valid = len(xs)
    acc = (sum(ys) / len(ys)) if ys else float("nan")
    sink_min = min(xs) if xs else float("nan")
    sink_max = max(xs) if xs else float("nan")
    pear_sink_correct = _pearson(xs, ys) if xs else float("nan")
    mean_sink_correct = (sum(sink_correct) / len(sink_correct)) if sink_correct else float("nan")
    mean_sink_wrong = (sum(sink_wrong) / len(sink_wrong)) if sink_wrong else float("nan")

    print(f"\n== {model_name} ==")
    if ks:
        print(f"sink_tokens(K)={ks}" + (f"  chat_mode={chat_modes}" if chat_modes else ""))
    print(f"n={n}  n_valid={n_valid}  acc={acc:.3f}  sink=[{sink_min:.4f},{sink_max:.4f}]")
    print(f"corr(sink,correct)={pear_sink_correct:+.3f}  mean_sink(correct)={mean_sink_correct:.4f}  mean_sink(wrong)={mean_sink_wrong:.4f}")

    if ents and xs:
        # only for rows where both are present & finite
        sink_for_ent = [float(r["sink_mass"]) for r in rows if _is_finite_number(r.get("sink_mass")) and _is_finite_number(r.get("entropy"))]
        ent_for_sink = [float(r["entropy"]) for r in rows if _is_finite_number(r.get("sink_mass")) and _is_finite_number(r.get("entropy"))]
        pear_sink_entropy = _pearson(sink_for_ent, ent_for_sink) if sink_for_ent else float("nan")
        pear_entropy_correct = _pearson(ents, ents_y) if ents else float("nan")
        print(f"corr(sink,entropy)={pear_sink_entropy:+.3f}  corr(entropy,correct)={pear_entropy_correct:+.3f}")
        print(f"mean_entropy(correct)={(sum(ent_correct)/len(ent_correct)) if ent_correct else float('nan'):.4f}  mean_entropy(wrong)={(sum(ent_wrong)/len(ent_wrong)) if ent_wrong else float('nan'):.4f}")


def plot_subject_breakdown(rows, model_name: str, out_dir: str):
    # Optional feature: only works if matplotlib is installed.
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting disabled (matplotlib not available): {e}")
        return

    # Aggregate per subject
    agg = defaultdict(lambda: {"n": 0, "n_valid": 0, "n_correct": 0, "sink_sum": 0.0})
    for r in rows:
        subject = r.get("subject")
        if not subject:
            continue
        agg[subject]["n"] += 1
        if _is_finite_number(r.get("sink_mass")):
            sm = float(r["sink_mass"])
            agg[subject]["n_valid"] += 1
            agg[subject]["sink_sum"] += sm
            if bool(r.get("correct")):
                agg[subject]["n_correct"] += 1

    stats = []
    for subject, a in agg.items():
        if a["n_valid"] < 25:
            continue
        stats.append(
            {
                "subject": subject,
                "n": a["n_valid"],
                "acc": a["n_correct"] / a["n_valid"] if a["n_valid"] else float("nan"),
                "mean_sink": a["sink_sum"] / a["n_valid"] if a["n_valid"] else float("nan"),
            }
        )

    if not stats:
        return

    stats.sort(key=lambda d: d["mean_sink"], reverse=True)
    subjects = [d["subject"] for d in stats]
    mean_sinks = [d["mean_sink"] for d in stats]
    accs = [d["acc"] for d in stats]

    plt.figure(figsize=(12, max(6, 0.25 * len(stats))))
    plt.scatter(mean_sinks, range(len(stats)), c=accs, s=[max(20, min(200, d["n"])) for d in stats], cmap="viridis")
    plt.yticks(range(len(stats)), subjects)
    plt.xlabel("Mean sink mass")
    plt.ylabel("Subject")
    plt.title(f"MMLU subject breakdown: mean sink vs accuracy ({model_name})")
    plt.colorbar(label="Accuracy")
    plt.tight_layout()

    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mmlu_subject_sink_{safe}.png")
    plt.savefig(out_path, dpi=250)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default=None, help="Glob for result json files.")
    parser.add_argument("--plots", action="store_true", help="Save subject breakdown plots into artifacts/plots/.")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to write plots into (when --plots).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pattern = args.pattern or str(repo_root / "artifacts" / "results" / "mmlu_accuracy_sink_*.json")
    files = sorted(glob(pattern))
    if not files:
        raise SystemExit(f"No files match pattern: {pattern}")

    for path in files:
        rows = load_results(path)
        model_name = os.path.basename(path).replace("mmlu_accuracy_sink_", "").replace(".json", "")
        print_model_summary(rows, model_name)
        if args.plots:
            out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "artifacts" / "plots")
            plot_subject_breakdown(rows, model_name, out_dir=str(out_dir))


if __name__ == "__main__":
    main()

