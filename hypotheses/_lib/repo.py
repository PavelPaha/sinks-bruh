from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """
    Walk upwards until we find pyproject.toml.
    Works both locally and in notebook / cloud.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(20):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Could not locate repo root (pyproject.toml not found)")

