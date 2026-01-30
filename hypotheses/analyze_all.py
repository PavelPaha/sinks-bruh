from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="auto", choices=["auto", "hallucinated", "incorrect"])
    p.add_argument("--include_h7", action="store_true")
    p.add_argument("--only", default=None, help="Comma-separated list like H1,H3,H6")
    p.add_argument("--skip_missing", action="store_true", help="Skip hypothesis if data/ has no *.jsonl.gz")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    hyp_root = repo_root / "hypotheses"
    if not hyp_root.exists():
        raise SystemExit(f"Expected hypotheses/ at {hyp_root}")

    wanted = None
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    ids = ["H0_sanity", "H1_distribution_shift", "H2_predictive", "H3_added_value_entropy", "H4_localization", "H5_chat_sensitivity", "H6_query_set_sensitivity"]
    if args.include_h7:
        ids.append("H7_sink_entropy")

    for hid in ids:
        if wanted is not None and hid.split("_", 1)[0] not in wanted and hid not in wanted:
            continue

        hyp_dir = hyp_root / hid
        if not hyp_dir.exists():
            print(f"[SKIP] missing {hyp_dir}")
            continue

        if args.skip_missing:
            data_dir = hyp_dir / "data"
            if not list(data_dir.glob("*.jsonl.gz")):
                print(f"[SKIP] no data in {data_dir}")
                continue

        run_all = hyp_dir / "scripts" / "run_all.py"
        if not run_all.exists():
            print(f"[SKIP] missing {run_all}")
            continue

        cmd = [sys.executable, str(run_all), "--label", args.label]
        _run(cmd, cwd=repo_root)


if __name__ == "__main__":
    main()

