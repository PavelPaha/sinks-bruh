from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Json = Dict[str, Any]


def read_jsonl_gz(path: Path) -> List[Json]:
    rows: List[Json] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def count_jsonl_gz(path: Path) -> int:
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def read_json(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_run_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("*.jsonl.gz"))


def matching_meta_for(run_path: Path) -> Optional[Path]:
    # Our measure_sink_text.py writes <base>.meta.json next to <base>.jsonl.gz
    meta = run_path.with_suffix("").with_suffix(".meta.json")
    return meta if meta.exists() else None


def run_id_from_path(run_path: Path) -> str:
    return run_path.name.replace(".jsonl.gz", "")


@dataclass(frozen=True)
class Run:
    path: Path
    rows: List[Json]
    meta_path: Optional[Path]
    meta: Optional[Json]

    @property
    def task(self) -> str:
        return str(self.rows[0].get("task", "unknown")) if self.rows else "unknown"

    @property
    def model(self) -> str:
        return str(self.rows[0].get("model", "?")) if self.rows else "?"

    @property
    def chat_mode(self) -> str:
        return str(self.rows[0].get("chat_mode", "auto")) if self.rows else "auto"

    @property
    def quantization(self) -> str:
        return str(self.rows[0].get("quantization", "none")) if self.rows else "none"


def load_runs(data_dir: Path, *, min_rows: int = 1) -> List[Run]:
    runs: List[Run] = []
    for p in list_run_files(data_dir):
        rows = read_jsonl_gz(p)
        if len(rows) < min_rows:
            continue
        mp = matching_meta_for(p)
        meta = read_json(mp) if mp else None
        runs.append(Run(path=p, rows=rows, meta_path=mp, meta=meta))
    return runs


def require_keys(rows: Iterable[Json], keys: List[str]) -> Tuple[bool, List[str]]:
    missing = []
    for k in keys:
        if any(r.get(k) is not None for r in rows):
            continue
        missing.append(k)
    return (len(missing) == 0), missing

