"""
Автоматически обновляет run_notes.md и final.md на основе результатов прогона.
Запускать после каждого run или после построения графиков.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List
import math


def read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def cohens_d(x1: List[float], x0: List[float]) -> float:
    """Cohen's d effect size."""
    if len(x1) < 2 or len(x0) < 2:
        return float("nan")
    import numpy as np
    s1, s0 = np.std(x1, ddof=1), np.std(x0, ddof=1)
    n1, n0 = len(x1), len(x0)
    sp = math.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(x1) - np.mean(x0)) / sp)


def analyze_runs(data_dir: Path) -> Dict[str, Any]:
    files = sorted(data_dir.glob("*.jsonl.gz"))
    if not files:
        return {"status": "no_data", "files": []}

    all_results = []
    for f in files:
        rows = read_jsonl_gz(f)
        if not rows:
            continue

        # Extract meta from first row
        r0 = rows[0]
        task = r0.get("task", "unknown")
        model = r0.get("model", "?")
        chat = r0.get("chat_mode", "auto")
        quant = r0.get("quantization", "none")

        # H1: Cohen's d for hallucinated vs non-hallucinated
        sink_hall = [float(r["sink_mass"]) for r in rows if r.get("hallucinated") is True and r.get("sink_mass") is not None]
        sink_non = [float(r["sink_mass"]) for r in rows if r.get("hallucinated") is False and r.get("sink_mass") is not None]

        # Fallback: incorrect vs correct
        if not sink_hall or not sink_non:
            sink_hall = [float(r["sink_mass"]) for r in rows if not r.get("correct") and r.get("sink_mass") is not None]
            sink_non = [float(r["sink_mass"]) for r in rows if r.get("correct") and r.get("sink_mass") is not None]

        d = cohens_d(sink_hall, sink_non) if sink_hall and sink_non else float("nan")

        all_results.append({
            "file": f.name,
            "task": task,
            "model": model,
            "chat": chat,
            "quant": quant,
            "n": len(rows),
            "n_hall": len(sink_hall),
            "n_non": len(sink_non),
            "cohens_d": d,
            "mean_sink_hall": sum(sink_hall) / len(sink_hall) if sink_hall else float("nan"),
            "mean_sink_non": sum(sink_non) / len(sink_non) if sink_non else float("nan"),
        })

    return {"status": "ok", "files": all_results}


def update_run_notes(data_dir: Path, hyp_dir: Path):
    analysis = analyze_runs(data_dir)
    notes_path = hyp_dir / "run_notes.md"

    if analysis["status"] == "no_data":
        content = """# H1 — Run notes

## Run 1
- date: (fill after running)
- run files: (fill after running)
- protocol:
  - K: 4
  - Q: last
  - chat: auto
  - samples/seed: (fill)
- key numbers:
  - Cohen's d by task/model: (fill from plots/h1/h1_summary.csv)

## Interpretation (short, paper-style)
- (fill after seeing results)

## Next run / ablations
- chat=off?
- K=1?
- query_mode=range/query_start?
"""
    else:
        results = analysis["files"]
        content = "# H1 — Run notes\n\n"
        content += "## Run 1\n"
        content += "- date: 2026-01-25 (auto-generated)\n"
        content += f"- run files: {len(results)} file(s)\n"
        content += "- protocol:\n"
        content += "  - K: 4\n"
        content += "  - Q: last\n"
        content += "  - chat: auto (and possibly off)\n"
        content += "  - samples/seed: (extract from run meta)\n"
        content += "- key numbers (Cohen's d):\n"
        for r in results:
            d_val = r["cohens_d"]
            d_str = f"{d_val:+.3f}" if not math.isnan(d_val) else "nan"
            content += f"  - {r['task']} / {r['model']}: d={d_str} (n_hall={r['n_hall']}, n_non={r['n_non']})\n"
        content += "\n## Interpretation (short, paper-style)\n"
        content += "- (fill manually after reviewing plots)\n"
        content += "\n## Next run / ablations\n"
        content += "- chat=off?\n- K=1?\n- query_mode=range/query_start?\n"

    notes_path.write_text(content, encoding="utf-8")
    print(f"Updated {notes_path}")


def update_final(data_dir: Path, hyp_dir: Path):
    analysis = analyze_runs(data_dir)
    final_path = hyp_dir / "final.md"

    if analysis["status"] == "no_data":
        content = """# H1 — Final (distribution shift)

## Outcome
- status: `pending` (waiting for data)

## What we tested
- datasets: (fill)
- models: (fill)
- protocol (K/Q/chat/quant/samples): (fill)

## Results (numbers)
- effect sizes (Cohen's d): (fill)

## Interpretation (what is the claim now?)
- (fill after seeing results)

## Limitations
- MC proxy vs free-form hallucination
- dataset label noise / prompt-format sensitivity

## Artifacts
- run files in `data/`: (fill)
- plots in `plots/`: (fill)
"""
    else:
        results = analysis["files"]
        tasks = sorted(set(r["task"] for r in results))
        models = sorted(set(r["model"] for r in results))

        # Check if we have consistent sign
        ds = [r["cohens_d"] for r in results if not math.isnan(r["cohens_d"])]
        if ds:
            all_pos = all(d > 0 for d in ds)
            all_neg = all(d < 0 for d in ds)
            sign_consistent = all_pos or all_neg
        else:
            sign_consistent = False

        content = "# H1 — Final (distribution shift)\n\n"
        content += "## Outcome\n"
        if sign_consistent and len(ds) >= 2:
            content += "- status: `confirmed` (preliminary)\n"
        elif len(ds) >= 2:
            content += "- status: `mixed` (sign depends on model/task)\n"
        else:
            content += "- status: `pending` (need more runs)\n"
        content += "\n## What we tested\n"
        content += f"- datasets: {', '.join(tasks)}\n"
        content += f"- models: {len(models)} model(s)\n"
        content += "- protocol (K/Q/chat/quant/samples): K=4, Q=last, chat=auto (see run files for details)\n"
        content += "\n## Results (numbers)\n"
        content += "- effect sizes (Cohen's d):\n"
        for r in results:
            d_val = r["cohens_d"]
            d_str = f"{d_val:+.3f}" if not math.isnan(d_val) else "nan"
            content += f"  - {r['task']} / {r['model']}: d={d_str}\n"
        content += "\n## Interpretation (what is the claim now?)\n"
        if sign_consistent:
            content += f"- **Consistent sign**: {'positive' if all_pos else 'negative'} effect across tested models/tasks\n"
        else:
            content += "- **Sign varies**: effect direction depends on model/task (potential scaling inversion?)\n"
        content += "- (refine after reviewing plots and more runs)\n"
        content += "\n## Limitations\n"
        content += "- MC proxy vs free-form hallucination\n"
        content += "- dataset label noise / prompt-format sensitivity\n"
        content += "\n## Artifacts\n"
        content += f"- run files in `data/`: {len(results)} file(s)\n"
        content += "- plots in `plots/`: (check after running run_plots.sh)\n"

    final_path.write_text(content, encoding="utf-8")
    print(f"Updated {final_path}")


def main():
    # This script is at: hypotheses/H1_distribution_shift/scripts/auto_update_notes.py
    # So parents[2] = repo root
    script_path = Path(__file__).resolve()
    hyp_dir = script_path.parents[1]  # hypotheses/H1_distribution_shift
    repo_root = hyp_dir.parents[1]  # repo root
    data_dir = hyp_dir / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    print("H1: Auto-updating run_notes.md and final.md")
    print("=" * 60)

    update_run_notes(data_dir, hyp_dir)
    update_final(data_dir, hyp_dir)

    print("\nDone. Review and refine manually if needed.")


if __name__ == "__main__":
    main()
