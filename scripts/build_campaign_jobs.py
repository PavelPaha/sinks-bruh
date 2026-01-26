"""
Build per-GPU job scripts for sink campaigns.

This is used by run_sink_campaign_4gpus.sh to generate 4 queue scripts
from a JSON config (configs/sink_campaign.json).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List


def model_size_b(model_id: str) -> float:
    """
    Best-effort extract of model size (B) from model_id.
    Supports "0.5B", "1.5B", "7B", "32B", etc.
    """
    # Handle MoE like "8x7B" (e.g., Mixtral-8x7B) by treating it as (experts * expert_size).
    moe = re.search(r"(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    if moe:
        try:
            return float(moe.group(1)) * float(moe.group(2))
        except Exception:
            pass

    # Accept both "7B" and "7b" (many repos use lowercase b, e.g. "pythia-2.8b-deduped").
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to configs/sink_campaign.json")
    p.add_argument("--job_paths", nargs=4, required=True, help="Four output job script paths (gpu0..gpu3)")
    args = p.parse_args()

    config_path = Path(args.config)
    job_paths = [Path(x) for x in args.job_paths]

    cfg = json.loads(config_path.read_text())
    defaults = cfg.get("defaults", {})
    datasets = cfg.get("datasets", [])
    models = cfg.get("models", [])

    sink_tokens = int(defaults.get("sink_tokens", 4))
    query_mode = defaults.get("query_mode", "last")
    chat = defaults.get("chat", "auto")
    seed = int(defaults.get("seed", 42))
    lang = defaults.get("language_freshqa", "English")

    samples_small = int(defaults.get("samples_small", 2000))
    samples_large = int(defaults.get("samples_large", 1000))
    large_thr = float(defaults.get("large_threshold_b", 14))
    quant_small = defaults.get("quant_small", "none")
    quant_large = defaults.get("quant_large", "4bit")

    # Initialize job scripts (preserve shebang/strict mode, overwrite content)
    for jp in job_paths:
        jp.write_text("#!/usr/bin/env bash\nset -euo pipefail\n")

    cmds: List[str] = []
    for model in models:
        size = model_size_b(model)
        is_large = size >= large_thr
        samples = samples_large if is_large else samples_small
        quant = quant_large if is_large else quant_small

        for d in datasets:
            task = d["task"]
            split = d.get("split", "test")

            extra = ""
            if task == "freshqa_false_premise":
                extra += f" --language {lang}"

            cmd = (
                "uv run python scripts/measure_sink_text.py"
                f" --task {task}"
                f" --split {split}"
                f" --model \"{model}\""
                f" --samples {samples}"
                f" --seed {seed}"
                f" --sink_tokens {sink_tokens}"
                f" --query_mode {query_mode}"
                f" --chat {chat}"
                f" --quantization {quant}"
                f"{extra}"
            )
            cmds.append(cmd)

    # Round-robin assignment
    for i, cmd in enumerate(cmds):
        gpu = i % 4
        jp = job_paths[gpu]
        line = (
            f'echo "[GPU {gpu}] {cmd}"\n'
            f"{cmd} --cuda_device {gpu} || echo \"[GPU {gpu}] FAILED\"\n"
        )
        jp.write_text(jp.read_text() + line)

    print(f"Wrote {len(cmds)} commands into:")
    for jp in job_paths:
        print(f" - {jp}")


if __name__ == "__main__":
    main()

