from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import ensure_dir, write_json


def run_cmd(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    """
    Run a command while streaming output, but keep a tail for errors.
    This is important for long-running model eval jobs (tqdm) while still
    surfacing useful diagnostics when a subprocess fails.
    """
    tail: List[str] = []
    tail_limit = 200

    # Prefer using a writable, spacious HF cache location.
    # Many cloud notebooks preconfigure caches on small partitions.
    env = dict(os.environ)
    if not env.get("HF_HOME"):
        # Heuristic: on many notebook platforms /home/jupyter/project is the large volume.
        candidate = Path("/home/jupyter/project/hf_home")
        if candidate.parent.exists():
            env["HF_HOME"] = str(candidate)
        else:
            base = (cwd if cwd else Path.cwd()) / ".hf_home"
            env["HF_HOME"] = str(base)
    env.setdefault("HF_HUB_CACHE", str(Path(env["HF_HOME"]) / "hub"))

    # Reduce HF cache-related warnings: TRANSFORMERS_CACHE is deprecated in favor of HF_HOME.
    if env.get("TRANSFORMERS_CACHE") and not env.get("HF_HOME"):
        env["HF_HOME"] = env["TRANSFORMERS_CACHE"]
    # If TRANSFORMERS_CACHE is set, transformers emits a deprecation warning. Prefer HF_HOME.
    if env.get("TRANSFORMERS_CACHE"):
        env.pop("TRANSFORMERS_CACHE", None)

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line.rstrip("\n"))
        if len(tail) > tail_limit:
            tail.pop(0)
    rc = p.wait()
    if rc != 0:
        tail_txt = "\n".join(tail[-50:])
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}\n\n--- last output ---\n{tail_txt}\n")


def run_measure_sink_text(
    repo_root: Path,
    out_dir: Path,
    *,
    task: str,
    split: str,
    model: str,
    samples: int,
    seed: int,
    sink_tokens: int,
    query_mode: str,
    query_start: int = 0,
    chat: str = "auto",
    device: str = "cuda",
    quantization: str = "none",
    revision: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> None:
    ensure_dir(out_dir)
    script = repo_root / "scripts" / "measure_sink_text.py"
    cmd = [
        sys.executable,
        str(script),
        "--task",
        task,
        "--split",
        split,
        "--model",
        model,
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--sink_tokens",
        str(sink_tokens),
        "--query_mode",
        query_mode,
        "--query_start",
        str(query_start),
        "--chat",
        chat,
        "--device",
        device,
        "--quantization",
        quantization,
        "--out_dir",
        str(out_dir),
    ]
    if revision:
        cmd.extend(["--revision", revision])

    extra_args = list(extra_args) if extra_args else []
    # Default to resume-friendly behavior for large sweeps.
    if "--skip_existing" not in extra_args:
        extra_args.append("--skip_existing")
    # Ensure TruthfulQA MC doesn't silently drop most examples:
    # use "sample4" (correct + 3 distractors) instead of "first4".
    if task == "truthfulqa_mc" and "--truthfulqa_mc_policy" not in extra_args:
        extra_args.extend(["--truthfulqa_mc_policy", "sample4"])
    if extra_args:
        cmd.extend(extra_args)

    try:
        run_cmd(cmd, cwd=repo_root)
    except RuntimeError as e:
        msg = str(e)
        # Skip gated repos (e.g. meta-llama/*) instead of failing the whole sweep.
        gated_markers = ("gated repo", "GatedRepoError", "401 Client Error", "Unauthorized")
        if any(m in msg for m in gated_markers):
            print(f"[WARN] Skipping model due to gated/unauthorized access: {model}")
            return
        # Skip when cache disk is full (common on managed notebook environments).
        disk_markers = ("No space left on device", "Not enough free disk space")
        if any(m in msg for m in disk_markers):
            print(f"[WARN] Skipping model due to insufficient disk/cache space: {model}")
            print("       Tip: set HF_HOME to a bigger disk (e.g. /home/jupyter/project/hf_home) and retry.")
            return
        raise


def run_plot_script(repo_root: Path, script_name: str, *, inputs: List[Path], out_dir: Path, extra_args: List[str]) -> None:
    ensure_dir(out_dir)
    script = repo_root / "scripts" / script_name
    cmd = [sys.executable, str(script), "--out_dir", str(out_dir), "--inputs", *[str(p) for p in inputs], *extra_args]
    run_cmd(cmd, cwd=repo_root)


def write_manifest(out_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest = dict(manifest)
    manifest.setdefault("created_ts", time.time())
    write_json(out_dir / "manifest.json", manifest)

