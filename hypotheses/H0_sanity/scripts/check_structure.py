"""
H0 sanity: проверка структуры данных без запуска моделей.
Если есть готовые .jsonl.gz - проверяем их. Если нет - показываем, что нужно.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:  # только первые 5 строк для sanity
                    break
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except Exception as e:
        print(f"  ERROR reading {path.name}: {e}")
        return []
    return rows


def check_file(path: Path) -> Dict[str, Any]:
    print(f"\n=== {path.name} ===")
    rows = read_jsonl_gz(path)
    if not rows:
        return {"valid": False, "reason": "empty or unreadable"}

    r0 = rows[0]
    keys = sorted(r0.keys())
    print(f"  Keys: {keys}")

    # Check required fields
    required = ["sink_mass", "correct"]
    missing = [k for k in required if k not in keys]
    if missing:
        return {"valid": False, "reason": f"missing keys: {missing}"}

    # Check sink_by_layer_head shape
    has_lh = "sink_by_layer_head" in keys
    if has_lh:
        lh = r0.get("sink_by_layer_head")
        if isinstance(lh, list) and lh:
            if isinstance(lh[0], list):
                L, H = len(lh), len(lh[0]) if lh[0] else 0
                print(f"  sink_by_layer_head: {L} layers × {H} heads")
            else:
                print(f"  sink_by_layer_head: unexpected shape")
        else:
            print(f"  sink_by_layer_head: missing or empty")

    # Check values
    sink_vals = [r.get("sink_mass") for r in rows if r.get("sink_mass") is not None]
    if sink_vals:
        import math
        finite = [v for v in sink_vals if isinstance(v, (int, float)) and math.isfinite(v)]
        print(f"  sink_mass: {len(finite)}/{len(rows)} finite, range=[{min(finite):.4f}, {max(finite):.4f}]" if finite else "  sink_mass: no finite values")

    correct_vals = [r.get("correct") for r in rows if r.get("correct") is not None]
    if correct_vals:
        n_true = sum(1 for v in correct_vals if v is True)
        print(f"  correct: {n_true}/{len(rows)} True ({n_true/len(rows):.1%})")

    return {"valid": True, "n_rows": len(rows), "keys": keys}


def main():
    # Script is at: hypotheses/H0_sanity/scripts/check_structure.py
    # So parents[1] = hypotheses/H0_sanity, parents[2] = hypotheses, parents[3] = repo root
    script_path = Path(__file__).resolve()
    hyp_dir = script_path.parents[1]  # hypotheses/H0_sanity
    repo_root = hyp_dir.parents[1]  # repo root
    data_dir = hyp_dir / "data"
    general_runs = repo_root / "artifacts" / "sink_runs"

    print("H0 Sanity Check: Structure validation")
    print("=" * 60)

    # Check local data/
    local_files = sorted(data_dir.glob("*.jsonl.gz")) if data_dir.exists() else []
    if local_files:
        print(f"\nFound {len(local_files)} file(s) in hypotheses/H0_sanity/data/")
        for f in local_files:
            check_file(f)
    else:
        print("\nNo files in hypotheses/H0_sanity/data/")

    # Check general artifacts/sink_runs/
    general_files = sorted(general_runs.glob("*.jsonl.gz")) if general_runs.exists() else []
    if general_files:
        print(f"\nFound {len(general_files)} file(s) in artifacts/sink_runs/")
        print("(showing first 3 for sanity)")
        for f in general_files[:3]:
            check_file(f)
    else:
        print("\nNo files in artifacts/sink_runs/")

    # Summary
    print("\n" + "=" * 60)
    if not local_files and not general_files:
        print("ACTION NEEDED:")
        print("  1) Run a measurement:")
        print("     python scripts/measure_sink_text.py --task mmlu --split test --model Qwen/Qwen2.5-0.5B-Instruct --samples 100 --sink_tokens 4 --query_mode last --chat auto")
        print("  2) Copy the output .jsonl.gz to hypotheses/H0_sanity/data/")
        print("  3) Re-run this script")
    else:
        print("✓ Data files found. Next: run plotting scripts.")


if __name__ == "__main__":
    main()
