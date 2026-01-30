from __future__ import annotations

from collections import defaultdict
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:
    from scipy.stats import pearsonr, spearmanr  # type: ignore
except Exception:  # pragma: no cover
    def pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            return float("nan"), float("nan")
        r = float(np.corrcoef(x, y)[0, 1])
        return r, float("nan")

    def spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            return float("nan"), float("nan")
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        r = float(np.corrcoef(rx, ry)[0, 1])
        return r, float("nan")

from hypotheses._lib.metrics import BootstrapCI, bootstrap_ci, choose_label, cohens_d, pr_auc, roc_auc


Json = Dict[str, Any]


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _nanmean_scalar(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    n = int(m.sum())
    if n == 0:
        return float("nan")
    return float(np.nansum(x) / n)


def _nanstd_scalar(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return float("nan")
    return float(np.std(xf))


def _nanmean_axis0(x: np.ndarray) -> np.ndarray:
    """
    NaN-mean over axis=0 without RuntimeWarning on empty slices.
    Works for 2D (-> 1D) and 3D (-> 2D) arrays.
    """
    x = np.asarray(x, dtype=float)
    sums = np.nansum(x, axis=0)
    cnt = np.sum(np.isfinite(x), axis=0)
    out = np.full(sums.shape, np.nan, dtype=float)
    m = cnt > 0
    out[m] = sums[m] / cnt[m]
    return out


def _as_float_array(rows: List[Json], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key)
        if v is None:
            out.append(float("nan"))
            continue
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return np.array(out, dtype=float)


def _as_int_array(rows: List[Json], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key)
        if v is None:
            out.append(-1)
            continue
        try:
            out.append(int(v))
        except Exception:
            out.append(-1)
    return np.array(out, dtype=int)

def _as_optional_bool_array(rows: List[Json], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key)
        if v is None:
            out.append(False)
        else:
            out.append(bool(v))
    return np.array(out, dtype=bool)

def _numeric_scalar_keys(rows: List[Json]) -> List[str]:
    """
    Return keys that look like numeric scalar features (int/float/bool) across rows.
    Excludes obvious identifiers and known large/structured fields.
    """
    banned = {
        # identifiers / bookkeeping
        "idx",
        "text",
        "prompt",
        "question",
        "choices",
        "answer",
        "pred",
        "prediction",
        "logits",
        "tokens",
        "input_ids",
        "attentions",
        "sink_by_layer",
        "sink_by_layer_head",
        "meta",
        "task",
        "model",
        "chat_mode",
        "quantization",
        "query_mode",
        "query_start",
        "run_id",
        "id",
        "sink_tokens",
        # labels (handled separately; do not treat as "features")
        "correct",
        "hallucinated",
        "label",
    }
    keys = set()
    for r in rows:
        for k, v in r.items():
            if k in banned:
                continue
            if isinstance(v, (int, float, bool, np.integer, np.floating)):
                keys.add(k)
    # keep stable order
    return sorted(keys)


def _as_feature_array(rows: List[Json], key: str) -> np.ndarray:
    out: List[float] = []
    for r in rows:
        v = r.get(key)
        if v is None:
            out.append(float("nan"))
            continue
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return np.array(out, dtype=float)


def summarize_numeric_features(rows: List[Json], *, label_mode: str = "auto") -> List[Dict[str, Any]]:
    """
    Build a table over all numeric scalar features found in rows:
    - missingness, mean/std
    - Spearman corr with sink_mass/entropy/seq_len
    - AUROC(feature → label) when label exists
    """
    keys = _numeric_scalar_keys(rows)
    sink = _as_float_array(rows, "sink_mass")
    ent = _as_float_array(rows, "entropy")
    seq_len = _as_int_array(rows, "seq_len").astype(float)
    try:
        label_name, y = choose_label(rows, label_mode)
    except Exception:
        label_name, y = "unknown", np.full(len(rows), -1, dtype=int)

    out: List[Dict[str, Any]] = []
    for k in keys:
        x = _as_feature_array(rows, k)
        fin = np.isfinite(x)
        n_fin = int(fin.sum())
        row: Dict[str, Any] = {
            "feature": k,
            "n": int(len(rows)),
            "n_finite": n_fin,
            "frac_missing": float(1.0 - (n_fin / max(1, len(rows)))),
            "mean": float(np.mean(x[fin])) if n_fin else float("nan"),
            "std": float(np.std(x[fin])) if n_fin else float("nan"),
            "corr_spearman_with_sink": _safe_corr(x, sink, kind="spearman"),
            "corr_spearman_with_entropy": _safe_corr(x, ent, kind="spearman"),
            "corr_spearman_with_seq_len": _safe_corr(x, seq_len, kind="spearman"),
            "label": label_name,
        }
        # AUROC(feature → label)
        m = fin & (y >= 0)
        if int(m.sum()) >= 20 and len(set(int(v) for v in y[m])) >= 2:
            try:
                row["auroc_feature_vs_label"] = float(roc_auc(y[m].astype(int), x[m].astype(float)))
            except Exception:
                row["auroc_feature_vs_label"] = float("nan")
        else:
            row["auroc_feature_vs_label"] = float("nan")
        out.append(row)
    return out


def plot_feature_overview(rows: List[Json], out_dir: Path, *, title: str, label_mode: str = "auto") -> List[Dict[str, Any]]:
    """
    Per-run: auto-discover numeric features and save
    - `features_summary.csv`
    - per-feature plots for all discovered scalar metrics.
    - optional multi-panel plot showing strongest associations (quick overview).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    table = summarize_numeric_features(rows, label_mode=label_mode)
    csv_p = out_dir / "features_summary.csv"
    save_table_csv(csv_p, table)
    items.append({"kind": "table", "path": str(csv_p), "desc": "Auto-discovered numeric feature summary (all scalar metrics)."})
    if plt is None or not table:
        return items

    def _safe_name(s: str, *, max_len: int = 120) -> str:
        # keep filenames stable across OSes
        out = []
        for ch in str(s):
            if ch.isalnum() or ch in ("_", "-", "."):
                out.append(ch)
            else:
                out.append("_")
        name = "".join(out)
        while "__" in name:
            name = name.replace("__", "_")
        name = name.strip("._")
        if not name:
            name = "metric"
        return name[:max_len]

    # Core arrays (used for consistent cross-metric plots)
    sink = _as_float_array(rows, "sink_mass")
    ent = _as_float_array(rows, "entropy")
    seq_len = _as_int_array(rows, "seq_len").astype(float)

    # Label (positive = hallucinated/incorrect)
    try:
        label_name, y = choose_label(rows, label_mode)
        y = y.astype(int)
        has_label = True
    except Exception:
        label_name, y = "unknown", np.full(len(rows), -1, dtype=int)
        has_label = False

    feats_dir = out_dir / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)

    # Per-feature plots for all discovered scalar metrics
    for rr in table:
        feat = str(rr.get("feature", ""))
        if not feat:
            continue
        safe = _safe_name(feat)

        x = _as_feature_array(rows, feat)
        x_fin = _finite(x)
        if x_fin.size == 0:
            continue

        # 1) Histogram
        plt.figure(figsize=(5.4, 3.2))
        plt.hist(x_fin, bins=40)
        plt.title(f"{title}\n{feat}: distribution")
        plt.xlabel(feat)
        plt.ylabel("count")
        p = feats_dir / f"hist__{safe}.png"
        _savefig(p)
        items.append({"kind": "hist", "path": str(p), "desc": f"Histogram: {feat}."})

        # 2) By-label boxplot (when label exists)
        if has_label:
            m = np.isfinite(x) & (y >= 0)
            if int(m.sum()) >= 30 and len(set(int(v) for v in y[m])) >= 2:
                x0 = x[m & (y == 0)]
                x1 = x[m & (y == 1)]
                if x0.size >= 5 and x1.size >= 5:
                    plt.figure(figsize=(4.2, 3.2))
                    plt.boxplot([x0, x1], labels=["0", "1"], showfliers=False)
                    plt.title(f"{title}\n{feat} by label ({label_name})")
                    plt.xlabel("label (1=positive)")
                    plt.ylabel(feat)
                    p = feats_dir / f"box__{safe}__by_label.png"
                    _savefig(p)
                    items.append({"kind": "box", "path": str(p), "desc": f"Boxplot: {feat} by label."})

        # 3) Scatter vs core axes (sink/entropy/seq_len) when available
        if feat != "sink_mass":
            m = np.isfinite(x) & np.isfinite(sink)
            if int(m.sum()) >= 60:
                xs = sink[m]
                ys = x[m]
                rho = _safe_corr(xs, ys, kind="spearman")
                plt.figure(figsize=(4.8, 4.0))
                plt.scatter(xs, ys, s=8, alpha=0.25)
                plt.title(f"{title}\n{feat} vs sink_mass (spearman ρ={rho:.2f})")
                plt.xlabel("sink_mass")
                plt.ylabel(feat)
                p = feats_dir / f"scatter__{safe}__vs_sink_mass.png"
                _savefig(p)
                items.append({"kind": "scatter", "path": str(p), "desc": f"Scatter: {feat} vs sink_mass."})

        if feat != "entropy":
            m = np.isfinite(x) & np.isfinite(ent)
            if int(m.sum()) >= 60:
                xs = ent[m]
                ys = x[m]
                rho = _safe_corr(xs, ys, kind="spearman")
                plt.figure(figsize=(4.8, 4.0))
                plt.scatter(xs, ys, s=8, alpha=0.25)
                plt.title(f"{title}\n{feat} vs entropy (spearman ρ={rho:.2f})")
                plt.xlabel("entropy")
                plt.ylabel(feat)
                p = feats_dir / f"scatter__{safe}__vs_entropy.png"
                _savefig(p)
                items.append({"kind": "scatter", "path": str(p), "desc": f"Scatter: {feat} vs entropy."})

        if feat != "seq_len":
            m = np.isfinite(x) & np.isfinite(seq_len)
            if int(m.sum()) >= 60:
                xs = seq_len[m]
                ys = x[m]
                rho = _safe_corr(xs, ys, kind="spearman")
                plt.figure(figsize=(4.8, 4.0))
                plt.scatter(xs, ys, s=8, alpha=0.25)
                plt.title(f"{title}\n{feat} vs seq_len (spearman ρ={rho:.2f})")
                plt.xlabel("seq_len")
                plt.ylabel(feat)
                p = feats_dir / f"scatter__{safe}__vs_seq_len.png"
                _savefig(p)
                items.append({"kind": "scatter", "path": str(p), "desc": f"Scatter: {feat} vs seq_len."})

    # pick top-k by |corr(feature, sink)| and |corr(feature, entropy)|, excluding trivial ones
    def _top(table: List[Dict[str, Any]], key: str, k: int = 12) -> List[Dict[str, Any]]:
        arr = []
        for r in table:
            v = float(r.get(key, float("nan")))
            if not np.isfinite(v):
                continue
            arr.append((abs(v), r))
        arr.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in arr[:k]]

    top_sink = _top(table, "corr_spearman_with_sink", k=10)
    top_ent = _top(table, "corr_spearman_with_entropy", k=10)
    top_lbl = _top(table, "auroc_feature_vs_label", k=10)

    plt.figure(figsize=(10.5, 6.2))
    plt.suptitle(f"{title}\nAuto feature overview (top associations)")

    ax = plt.subplot(2, 2, 1)
    if top_sink:
        names = [r["feature"] for r in top_sink][::-1]
        vals = [float(r["corr_spearman_with_sink"]) for r in top_sink][::-1]
        ax.barh(names, vals)
        ax.axvline(0, color="gray", linewidth=1)
    ax.set_title("Top |Spearman corr(feature, sink)|")
    ax.set_xlabel("ρ")

    ax = plt.subplot(2, 2, 2)
    if top_ent:
        names = [r["feature"] for r in top_ent][::-1]
        vals = [float(r["corr_spearman_with_entropy"]) for r in top_ent][::-1]
        ax.barh(names, vals)
        ax.axvline(0, color="gray", linewidth=1)
    ax.set_title("Top |Spearman corr(feature, entropy)|")
    ax.set_xlabel("ρ")

    ax = plt.subplot(2, 2, 3)
    if top_lbl:
        # AUROC centered around 0.5
        names = [r["feature"] for r in top_lbl][::-1]
        vals = [float(r["auroc_feature_vs_label"]) - 0.5 for r in top_lbl][::-1]
        ax.barh(names, vals)
        ax.axvline(0, color="gray", linewidth=1)
    ax.set_title("Top |AUROC(feature→label) - 0.5|")
    ax.set_xlabel("ΔAUROC")

    ax = plt.subplot(2, 2, 4)
    # Heatmap among a few core variables
    core = ["sink_mass", "entropy", "seq_len"]
    extra = []
    for r in top_sink[:3] + top_ent[:3]:
        if r["feature"] not in core and r["feature"] not in extra:
            extra.append(r["feature"])
    feats = core + extra
    X = []
    for f in feats:
        if f == "sink_mass":
            X.append(_as_float_array(rows, "sink_mass"))
        elif f == "entropy":
            X.append(_as_float_array(rows, "entropy"))
        elif f == "seq_len":
            X.append(_as_int_array(rows, "seq_len").astype(float))
        else:
            X.append(_as_feature_array(rows, f))
    X = np.stack(X, axis=0)
    # pairwise spearman via rank-corr approximation (corr of ranks)
    R = np.full((len(feats), len(feats)), np.nan, dtype=float)
    for i in range(len(feats)):
        for j in range(len(feats)):
            R[i, j] = _safe_corr(X[i], X[j], kind="spearman")
    vmax = float(np.nanmax(np.abs(R))) if np.isfinite(R).any() else 1.0
    im = ax.imshow(R, vmin=-vmax, vmax=vmax, cmap="coolwarm", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=25, ha="right", fontsize=8)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_title("Core feature correlations")

    p = out_dir / "feature_overview_panel.png"
    _savefig(p)
    items.append({"kind": "plot", "path": str(p), "desc": "Multi-panel: top feature correlations and core heatmap."})
    return items


def _maybe_stack_layer(rows: List[Json], key: str) -> Optional[np.ndarray]:
    """
    Stack variable-length per-layer vectors into [n, L] with NaNs.
    Returns None if no valid rows found.
    """
    vecs: List[np.ndarray] = []
    max_L = 0
    for r in rows:
        v = r.get(key)
        if not isinstance(v, list) or not v:
            continue
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0:
            continue
        vecs.append(a)
        max_L = max(max_L, int(a.size))
    if not vecs or max_L == 0:
        return None
    out = np.full((len(rows), max_L), np.nan, dtype=float)
    for i, r in enumerate(rows):
        v = r.get(key)
        if not isinstance(v, list) or not v:
            continue
        a = np.asarray(v, dtype=float).reshape(-1)
        L = min(max_L, int(a.size))
        out[i, :L] = a[:L]
    return out


def _maybe_stack_layer_head(rows: List[Json], key: str) -> Optional[np.ndarray]:
    """
    Stack per-layer-head matrices into [n, L, H] with NaNs.
    Uses the first observed (L,H) as target; mismatched rows are skipped (NaNs).
    """
    target: Optional[Tuple[int, int]] = None
    for r in rows:
        v = r.get(key)
        if not isinstance(v, list) or not v:
            continue
        a = np.asarray(v, dtype=float)
        if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] > 0:
            target = (int(a.shape[0]), int(a.shape[1]))
            break
    if target is None:
        return None
    L, H = target
    out = np.full((len(rows), L, H), np.nan, dtype=float)
    for i, r in enumerate(rows):
        v = r.get(key)
        if not isinstance(v, list) or not v:
            continue
        a = np.asarray(v, dtype=float)
        if a.shape != (L, H):
            continue
        out[i] = a
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray, *, kind: str) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    xx = x[m]
    yy = y[m]
    if np.all(xx == xx[0]) or np.all(yy == yy[0]):
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(xx, yy)[0])
    if kind == "spearman":
        return float(spearmanr(xx, yy)[0])
    raise ValueError(f"Unknown corr kind: {kind}")


def _partial_corr_residualize(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, kind: str) -> float:
    """
    Partial correlation by residualization: corr(resid(x~z), resid(y~z)).
    - If kind="spearman", we apply Spearman on residuals (NOT true partial-Spearman).
    """
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(m.sum()) < 5:
        return float("nan")
    x = x[m]
    y = y[m]
    z = z[m]
    if np.all(z == z[0]):
        return _safe_corr(x, y, kind=kind)

    Z = np.stack([np.ones_like(z), z], axis=1)  # intercept + z
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    return _safe_corr(rx, ry, kind=kind)

@dataclass(frozen=True)
class RunQuickSummary:
    run_id: str
    task: str
    model: str
    chat_mode: str
    quantization: str
    sink_tokens: Optional[int]
    query_mode: Optional[str]
    query_start: Optional[int]
    n: int
    acc: float
    label: str
    pos_rate: float
    sink_mean: float
    sink_std: float
    seq_len_mean: float
    seq_len_std: float
    entropy_mean: float
    entropy_std: float
    corr_sink_entropy_pearson: float
    corr_sink_entropy_spearman: float
    partial_sink_entropy_spearman_seq_len: float
    auroc_sink_vs_label: float
    auprc_sink_vs_label: float
    cohens_d_sink_pos_minus_neg: float


def summarize_run(rows: List[Json], *, run_id: str, label_mode: str = "auto") -> RunQuickSummary:
    # Minimal metadata from first row (all runs are homogeneous).
    r0 = rows[0] if rows else {}
    task = str(r0.get("task", "unknown"))
    model = str(r0.get("model", "?"))
    chat_mode = str(r0.get("chat_mode", "auto"))
    quantization = str(r0.get("quantization", "none"))
    sink_tokens = int(r0.get("sink_tokens")) if r0.get("sink_tokens") is not None else None
    query_mode = str(r0.get("query_mode")) if r0.get("query_mode") is not None else None
    query_start = int(r0.get("query_start")) if r0.get("query_start") is not None else None

    # Core arrays
    sink = _as_float_array(rows, "sink_mass")
    ent = _as_float_array(rows, "entropy")
    correct = _as_optional_bool_array(rows, "correct")
    seq_len = _as_int_array(rows, "seq_len").astype(float)

    # label (positive = hallucinated/incorrect)
    # IMPORTANT: do not ignore the requested label mode.
    label_name, y = choose_label(rows, label_mode)
    y = y.astype(int)

    # row filter for metrics
    m = np.isfinite(sink) & np.isfinite(ent)
    n = int(len(rows))
    acc = float(np.mean(correct)) if n else float("nan")
    pos_rate = float(np.mean(y)) if n else float("nan")

    sink_m = _nanmean_scalar(sink)
    sink_s = _nanstd_scalar(sink)
    seq_m = _nanmean_scalar(seq_len)
    seq_s = _nanstd_scalar(seq_len)
    ent_m = _nanmean_scalar(ent)
    ent_s = _nanstd_scalar(ent)

    corr_p = _safe_corr(sink, ent, kind="pearson")
    corr_s = _safe_corr(sink, ent, kind="spearman")
    partial_s = _partial_corr_residualize(sink, ent, seq_len, kind="spearman")

    # label-related
    m_lbl = np.isfinite(sink)
    y_lbl = y[m_lbl]
    sink_lbl = sink[m_lbl]
    auroc = float(roc_auc(y_lbl, sink_lbl)) if len(y_lbl) else float("nan")
    auprc = float(pr_auc(y_lbl, sink_lbl)) if len(y_lbl) else float("nan")
    x1 = sink_lbl[y_lbl == 1]
    x0 = sink_lbl[y_lbl == 0]
    d = float(cohens_d(x1, x0))

    return RunQuickSummary(
        run_id=run_id,
        task=task,
        model=model,
        chat_mode=chat_mode,
        quantization=quantization,
        sink_tokens=sink_tokens,
        query_mode=query_mode,
        query_start=query_start,
        n=n,
        acc=acc,
        label=label_name,
        pos_rate=pos_rate,
        sink_mean=sink_m,
        sink_std=sink_s,
        seq_len_mean=seq_m,
        seq_len_std=seq_s,
        entropy_mean=ent_m,
        entropy_std=ent_s,
        corr_sink_entropy_pearson=corr_p,
        corr_sink_entropy_spearman=corr_s,
        partial_sink_entropy_spearman_seq_len=partial_s,
        auroc_sink_vs_label=auroc,
        auprc_sink_vs_label=auprc,
        cohens_d_sink_pos_minus_neg=d,
    )


def write_manifest(plots_dir: Path, *, hypothesis_id: str, inputs: Sequence[Path], items: Sequence[Dict[str, Any]]) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "hypothesis_id": hypothesis_id,
        "generated_at": time.time(),
        "inputs": [str(p) for p in inputs],
        "items": list(items),
    }
    (plots_dir / "manifest.json").write_text(json_dumps(obj), encoding="utf-8")


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)


def save_table_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # stable field order: sorted union
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _savefig(path: Path) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_basic_run_diagnostics(rows: List[Json], out_dir: Path, *, title: str, label_mode: str = "auto") -> List[Dict[str, Any]]:
    """
    Produces a set of per-run diagnostic plots.
    Returns manifest items.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    if plt is None:
        return items

    # Minimal required key
    if not any(r.get("sink_mass") is not None for r in rows):
        return items

    sink = _as_float_array(rows, "sink_mass")
    ent = _as_float_array(rows, "entropy")
    seq_len = _as_int_array(rows, "seq_len").astype(float)
    correct = _as_optional_bool_array(rows, "correct")

    # Label
    try:
        label_name, y = choose_label(rows, label_mode)
    except Exception:
        label_name = "unknown"
        y = np.full(len(rows), -1, dtype=int)

    # 1) sink histogram
    plt.figure(figsize=(5.2, 3.2))
    plt.hist(_finite(sink), bins=40)
    plt.title(f"{title}\nSink mass distribution")
    plt.xlabel("sink_mass")
    plt.ylabel("count")
    p = out_dir / "hist_sink_mass.png"
    _savefig(p)
    items.append({"kind": "hist", "path": str(p), "desc": "Histogram of sink_mass."})

    # 2) entropy histogram
    if int(np.isfinite(ent).sum()) >= 30:
        plt.figure(figsize=(5.2, 3.2))
        plt.hist(_finite(ent), bins=40)
        plt.title(f"{title}\nNext-token entropy distribution")
        plt.xlabel("entropy")
        plt.ylabel("count")
        p = out_dir / "hist_entropy.png"
        _savefig(p)
        items.append({"kind": "hist", "path": str(p), "desc": "Histogram of entropy."})

    # 2b) seq_len histogram
    if int(np.isfinite(seq_len).sum()) >= 30:
        plt.figure(figsize=(5.2, 3.2))
        plt.hist(_finite(seq_len), bins=40)
        plt.title(f"{title}\nSequence length distribution")
        plt.xlabel("seq_len")
        plt.ylabel("count")
        p = out_dir / "hist_seq_len.png"
        _savefig(p)
        items.append({"kind": "hist", "path": str(p), "desc": "Histogram of seq_len."})

    # 3) sink vs entropy scatter (+ fit)
    m_se = np.isfinite(sink) & np.isfinite(ent)
    if int(m_se.sum()) >= 50:
        x = sink[m_se]
        yv = ent[m_se]
        r_p = _safe_corr(x, yv, kind="pearson")
        r_s = _safe_corr(x, yv, kind="spearman")
        plt.figure(figsize=(4.8, 4.0))
        plt.scatter(x, yv, s=8, alpha=0.25)
        try:
            z = np.polyfit(x, yv, 1)
            xx = np.linspace(np.min(x), np.max(x), 50)
            plt.plot(xx, z[0] * xx + z[1], color="red", linewidth=1.5)
        except Exception:
            pass
        plt.title(f"{title}\nEntropy vs sink (pearson r={r_p:.2f}, spearman ρ={r_s:.2f})")
        plt.xlabel("sink_mass")
        plt.ylabel("entropy")
        p = out_dir / "scatter_sink_vs_entropy.png"
        _savefig(p)
        items.append({"kind": "scatter", "path": str(p), "desc": "Scatter of entropy vs sink_mass with linear fit."})

    # 3b) sink vs seq_len scatter
    m_sl = np.isfinite(sink) & np.isfinite(seq_len)
    if int(m_sl.sum()) >= 80:
        x = seq_len[m_sl]
        yv = sink[m_sl]
        r_s = _safe_corr(x, yv, kind="spearman")
        plt.figure(figsize=(4.8, 4.0))
        plt.scatter(x, yv, s=8, alpha=0.25)
        plt.title(f"{title}\nSink vs seq_len (spearman ρ={r_s:.2f})")
        plt.xlabel("seq_len")
        plt.ylabel("sink_mass")
        p = out_dir / "scatter_sink_vs_seq_len.png"
        _savefig(p)
        items.append({"kind": "scatter", "path": str(p), "desc": "Scatter of sink_mass vs seq_len."})

    # 3c) entropy vs seq_len scatter
    m_el = np.isfinite(ent) & np.isfinite(seq_len)
    if int(m_el.sum()) >= 80:
        x = seq_len[m_el]
        yv = ent[m_el]
        r_s = _safe_corr(x, yv, kind="spearman")
        plt.figure(figsize=(4.8, 4.0))
        plt.scatter(x, yv, s=8, alpha=0.25)
        plt.title(f"{title}\nEntropy vs seq_len (spearman ρ={r_s:.2f})")
        plt.xlabel("seq_len")
        plt.ylabel("entropy")
        p = out_dir / "scatter_entropy_vs_seq_len.png"
        _savefig(p)
        items.append({"kind": "scatter", "path": str(p), "desc": "Scatter of entropy vs seq_len."})

    # 4) accuracy vs sink bins (if correct exists)
    m_sc = np.isfinite(sink)
    if int(m_sc.sum()) >= 80:
        s = sink[m_sc]
        c = correct[m_sc].astype(float)
        # quantile bins by count
        order = np.argsort(s)
        s = s[order]
        c = c[order]
        B = 12
        edges = np.linspace(0, len(s), B + 1).astype(int)
        xs = []
        ys = []
        ns = []
        for i in range(B):
            a, b = edges[i], edges[i + 1]
            if b - a < 10:
                continue
            xs.append(float(np.mean(s[a:b])))
            ys.append(float(np.mean(c[a:b])))
            ns.append(int(b - a))
        if len(xs) >= 4:
            plt.figure(figsize=(5.6, 3.2))
            plt.plot(xs, ys, marker="o")
            plt.axhline(float(np.mean(c)), color="gray", linestyle="--", linewidth=1)
            plt.ylim(0, 1)
            plt.title(f"{title}\nAccuracy vs sink_mass (binned)")
            plt.xlabel("mean sink_mass in bin")
            plt.ylabel("accuracy")
            p = out_dir / "curve_accuracy_vs_sink_bins.png"
            _savefig(p)
            items.append({"kind": "curve", "path": str(p), "desc": "Accuracy as function of sink_mass quantile bin."})

    # 4b) accuracy vs entropy bins
    m_ec = np.isfinite(ent)
    if int(m_ec.sum()) >= 80:
        e = ent[m_ec]
        c = correct[m_ec].astype(float)
        order = np.argsort(e)
        e = e[order]
        c = c[order]
        B = 12
        edges = np.linspace(0, len(e), B + 1).astype(int)
        xs = []
        ys = []
        for i in range(B):
            a, b = edges[i], edges[i + 1]
            if b - a < 10:
                continue
            xs.append(float(np.mean(e[a:b])))
            ys.append(float(np.mean(c[a:b])))
        if len(xs) >= 4:
            plt.figure(figsize=(5.6, 3.2))
            plt.plot(xs, ys, marker="o")
            plt.axhline(float(np.mean(c)), color="gray", linestyle="--", linewidth=1)
            plt.ylim(0, 1)
            plt.title(f"{title}\nAccuracy vs entropy (binned)")
            plt.xlabel("mean entropy in bin")
            plt.ylabel("accuracy")
            p = out_dir / "curve_accuracy_vs_entropy_bins.png"
            _savefig(p)
            items.append({"kind": "curve", "path": str(p), "desc": "Accuracy as function of entropy quantile bin."})

    # 4c) accuracy vs seq_len bins
    m_lc = np.isfinite(seq_len)
    if int(m_lc.sum()) >= 80:
        Ls = seq_len[m_lc]
        c = correct[m_lc].astype(float)
        order = np.argsort(Ls)
        Ls = Ls[order]
        c = c[order]
        B = 12
        edges = np.linspace(0, len(Ls), B + 1).astype(int)
        xs = []
        ys = []
        for i in range(B):
            a, b = edges[i], edges[i + 1]
            if b - a < 10:
                continue
            xs.append(float(np.mean(Ls[a:b])))
            ys.append(float(np.mean(c[a:b])))
        if len(xs) >= 4:
            plt.figure(figsize=(5.6, 3.2))
            plt.plot(xs, ys, marker="o")
            plt.axhline(float(np.mean(c)), color="gray", linestyle="--", linewidth=1)
            plt.ylim(0, 1)
            plt.title(f"{title}\nAccuracy vs seq_len (binned)")
            plt.xlabel("mean seq_len in bin")
            plt.ylabel("accuracy")
            p = out_dir / "curve_accuracy_vs_seq_len_bins.png"
            _savefig(p)
            items.append({"kind": "curve", "path": str(p), "desc": "Accuracy as function of seq_len quantile bin."})

    # 5) label-wise boxplots (if label is defined)
    if label_name != "unknown":
        m_lbl = np.isfinite(sink) & (y >= 0)
        if int(m_lbl.sum()) >= 50 and len(set(int(v) for v in y[m_lbl])) >= 2:
            s1 = sink[m_lbl & (y == 1)]
            s0 = sink[m_lbl & (y == 0)]
            if len(s1) >= 5 and len(s0) >= 5:
                plt.figure(figsize=(4.0, 3.2))
                plt.boxplot([s0, s1], labels=["0", "1"], showfliers=False)
                plt.title(f"{title}\nSink by label (label={label_name})")
                plt.xlabel("label (1=positive)")
                plt.ylabel("sink_mass")
                p = out_dir / "box_sink_by_label.png"
                _savefig(p)
                items.append({"kind": "box", "path": str(p), "desc": "Boxplot of sink_mass by label."})

            m_ent = np.isfinite(ent) & (y >= 0)
            if int(m_ent.sum()) >= 50 and len(set(int(v) for v in y[m_ent])) >= 2:
                e1 = ent[m_ent & (y == 1)]
                e0 = ent[m_ent & (y == 0)]
                if len(e1) >= 5 and len(e0) >= 5:
                    plt.figure(figsize=(4.0, 3.2))
                    plt.boxplot([e0, e1], labels=["0", "1"], showfliers=False)
                    plt.title(f"{title}\nEntropy by label (label={label_name})")
                    plt.xlabel("label (1=positive)")
                    plt.ylabel("entropy")
                    p = out_dir / "box_entropy_by_label.png"
                    _savefig(p)
                    items.append({"kind": "box", "path": str(p), "desc": "Boxplot of entropy by label."})

    # 6) layer/head diagnostics if present
    layer = _maybe_stack_layer(rows, "sink_by_layer")
    if layer is not None and layer.shape[1] >= 2:
        mean_layer = _nanmean_axis0(layer)
        plt.figure(figsize=(6.4, 3.2))
        plt.plot(np.arange(len(mean_layer)), mean_layer, marker="o", linewidth=1)
        plt.title(f"{title}\nMean sink_by_layer")
        plt.xlabel("layer")
        plt.ylabel("mean sink mass to sink tokens")
        p = out_dir / "layer_profile_mean.png"
        _savefig(p)
        items.append({"kind": "curve", "path": str(p), "desc": "Mean sink_by_layer (pooled)."})

        if label_name != "unknown" and np.isfinite(layer).any():
            m_lbl = (y >= 0)
            if int(m_lbl.sum()) >= 50 and len(set(int(v) for v in y[m_lbl])) >= 2:
                mean0 = _nanmean_axis0(layer[m_lbl & (y == 0)])
                mean1 = _nanmean_axis0(layer[m_lbl & (y == 1)])
                plt.figure(figsize=(6.4, 3.2))
                plt.plot(mean0, label="label=0", linewidth=1)
                plt.plot(mean1, label="label=1", linewidth=1)
                plt.title(f"{title}\nMean sink_by_layer by label ({label_name})")
                plt.xlabel("layer")
                plt.ylabel("mean sink mass to sink tokens")
                plt.legend(fontsize=8)
                p = out_dir / "layer_profile_by_label.png"
                _savefig(p)
                items.append({"kind": "curve", "path": str(p), "desc": "Mean sink_by_layer per label."})

                plt.figure(figsize=(6.4, 3.2))
                plt.plot(mean1 - mean0, linewidth=1)
                plt.axhline(0, color="gray", linestyle="--", linewidth=1)
                plt.title(f"{title}\nΔ sink_by_layer (label1 - label0) ({label_name})")
                plt.xlabel("layer")
                plt.ylabel("delta sink mass")
                p = out_dir / "layer_profile_delta_label.png"
                _savefig(p)
                items.append({"kind": "curve", "path": str(p), "desc": "Delta sink_by_layer between labels."})

    layer_head = _maybe_stack_layer_head(rows, "sink_by_layer_head")
    if layer_head is not None and layer_head.ndim == 3:
        mean_lh = _nanmean_axis0(layer_head)  # [L,H]
        plt.figure(figsize=(6.8, 4.2))
        plt.imshow(mean_lh, aspect="auto", interpolation="nearest")
        plt.colorbar(label="mean sink mass")
        plt.title(f"{title}\nMean sink_by_layer_head")
        plt.xlabel("head")
        plt.ylabel("layer")
        p = out_dir / "heatmap_layer_head_mean.png"
        _savefig(p)
        items.append({"kind": "heatmap", "path": str(p), "desc": "Heatmap: mean sink_by_layer_head."})

        if label_name != "unknown":
            m_lbl = (y >= 0)
            if int(m_lbl.sum()) >= 50 and len(set(int(v) for v in y[m_lbl])) >= 2:
                mean0 = _nanmean_axis0(layer_head[m_lbl & (y == 0)])
                mean1 = _nanmean_axis0(layer_head[m_lbl & (y == 1)])
                delta = mean1 - mean0
                plt.figure(figsize=(6.8, 4.2))
                vmax = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else 1.0
                plt.imshow(delta, aspect="auto", interpolation="nearest", vmin=-vmax, vmax=vmax, cmap="coolwarm")
                plt.colorbar(label="Δ mean sink mass (label1 - label0)")
                plt.title(f"{title}\nΔ sink_by_layer_head by label ({label_name})")
                plt.xlabel("head")
                plt.ylabel("layer")
                p = out_dir / "heatmap_layer_head_delta_label.png"
                _savefig(p)
                items.append({"kind": "heatmap", "path": str(p), "desc": "Heatmap: delta sink_by_layer_head between labels."})

                # top heads by |delta|
                L, H = delta.shape
                flat = np.abs(delta).ravel()
                idx = np.argsort(-flat)[: min(25, flat.size)]
                top = []
                for j in idx:
                    l = int(j // H)
                    h = int(j % H)
                    top.append({"layer": l, "head": h, "delta": float(delta[l, h]), "abs_delta": float(abs(delta[l, h]))})
                csv_p = out_dir / "top_heads_by_abs_delta.csv"
                save_table_csv(csv_p, top)
                items.append({"kind": "table", "path": str(csv_p), "desc": "Top (layer,head) by |Δ sink_by_layer_head|."})

    # 7) Auto feature overview (all numeric scalar metrics found in rows)
    try:
        items.extend(plot_feature_overview(rows, out_dir, title=title, label_mode=label_mode))
    except Exception:
        # Never fail the run diagnostics due to feature discovery.
        pass

    return items


def plot_metrics_boxgrid_by_model(summaries: Sequence[RunQuickSummary], out_dir: Path, *, title: str) -> List[Dict[str, Any]]:
    """
    Aggregate: for each model, show distributions of key metrics across runs.
    This is the "consistency across models" view the paper needs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    if plt is None or not summaries:
        return items

    models = sorted({s.model for s in summaries})
    if not models:
        return items

    def _vals(getter):
        out = []
        for m in models:
            xs = [getter(s) for s in summaries if s.model == m]
            arr = np.array(xs, dtype=float)
            out.append(arr[np.isfinite(arr)])
        return out

    # Build metric list automatically (avoid silently ignoring summary metrics).
    # Exclude identifiers/categorical context fields; also exclude constant run settings.
    exclude = {
        "run_id",
        "task",
        "model",
        "chat_mode",
        "quantization",
        "label",
        "query_mode",
        "sink_tokens",
        "query_start",
    }
    all_fields = sorted({k for s in summaries for k in s.__dict__.keys()})
    metric_fields = [f for f in all_fields if f not in exclude]
    if not metric_fields:
        return items

    per_page = 8
    n_pages = int(math.ceil(len(metric_fields) / per_page))
    fig_w = max(10.5, 0.5 * len(models) + 6.0)
    fig_h = 8.2

    for page in range(n_pages):
        chunk = metric_fields[page * per_page : (page + 1) * per_page]
        plt.figure(figsize=(fig_w, fig_h))
        plt.suptitle(f"{title}\nDistributions by model (across runs) — page {page+1}/{n_pages}")

        for i, field in enumerate(chunk, start=1):
            ax = plt.subplot(2, 4, i)
            data = _vals(lambda s, f=field: float(getattr(s, f)))
            if any(len(d) for d in data):
                ax.boxplot(data, showfliers=False)
            ax.set_title(field)
            ax.set_xticks(range(1, len(models) + 1))
            ax.set_xticklabels(models, rotation=60, ha="right", fontsize=7)
            lf = field.lower()
            if "auroc" in lf or "auprc" in lf:
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            if "corr" in lf or "cohens_d" in lf or "partial" in lf:
                ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)

        p = out_dir / (f"boxgrid_metrics_by_model_p{page+1:02d}.png" if n_pages > 1 else "boxgrid_metrics_by_model.png")
        _savefig(p)
        items.append({"kind": "plot", "path": str(p), "desc": "Boxgrid of per-run summary metrics by model."})

    return items


def plot_aggregate_summary_table(summaries: Sequence[RunQuickSummary], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [s.__dict__ for s in summaries]
    p = out_dir / "runs_summary.csv"
    save_table_csv(p, rows)
    return p


def plot_signflip_and_effects(summaries: Sequence[RunQuickSummary], out_dir: Path, *, title: str) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    if plt is None:
        return items
    if not summaries:
        return items

    acc_all = np.array([float(s.acc) for s in summaries], dtype=float)
    acc_fin = acc_all[np.isfinite(acc_all)]
    if acc_fin.size:
        plt.figure(figsize=(5.0, 3.0))
        plt.hist(acc_fin, bins=25)
        plt.title(f"{title}\nAccuracy distribution across runs")
        plt.xlabel("accuracy")
        plt.ylabel("count")
        p = out_dir / "hist_accuracy_across_runs.png"
        _savefig(p)
        items.append({"kind": "hist", "path": str(p), "desc": "Histogram of accuracy across runs."})

    # sign flip counts for label effect (Cohen's d)
    d_all = np.array([float(s.cohens_d_sink_pos_minus_neg) for s in summaries], dtype=float)
    d = d_all[np.isfinite(d_all)]
    if d.size:
        plt.figure(figsize=(5.0, 3.0))
        plt.hist(d, bins=25)
        plt.axvline(0, color="black", linewidth=1)
        plt.title(f"{title}\nCohen's d distribution across runs")
        plt.xlabel("d = mean(sink|pos) - mean(sink|neg), pooled std")
        plt.ylabel("count")
        p = out_dir / "hist_cohens_d_across_runs.png"
        _savefig(p)
        items.append({"kind": "hist", "path": str(p), "desc": "Histogram of Cohen's d across runs."})

        # d vs accuracy
        if np.isfinite(acc_all).sum() >= 3:
            plt.figure(figsize=(5.0, 3.6))
            plt.scatter(acc_all, d_all, s=20, alpha=0.7)
            plt.axhline(0, color="gray", linestyle="--", linewidth=1)
            plt.xlabel("run accuracy")
            plt.ylabel("Cohen's d (sink|pos - sink|neg)")
            plt.title(f"{title}\nEffect size vs accuracy")
            p = out_dir / "scatter_d_vs_accuracy.png"
            _savefig(p)
            items.append({"kind": "scatter", "path": str(p), "desc": "Scatter: effect size vs accuracy."})

        # AUROC vs d
        auroc_all = np.array([float(s.auroc_sink_vs_label) for s in summaries], dtype=float)
        m = np.isfinite(auroc_all) & np.isfinite(d_all)
        if int(m.sum()) >= 5:
            plt.figure(figsize=(5.0, 3.6))
            plt.scatter(d_all[m], auroc_all[m], s=22, alpha=0.7)
            plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
            plt.xlabel("Cohen's d (sink|pos - sink|neg)")
            plt.ylabel("AUROC(sink→label)")
            plt.title(f"{title}\nAUROC vs effect size")
            p = out_dir / "scatter_auroc_vs_d.png"
            _savefig(p)
            items.append({"kind": "scatter", "path": str(p), "desc": "Scatter: AUROC vs Cohen's d across runs."})

    # sink-entropy correlations
    rho = np.array([float(s.corr_sink_entropy_spearman) for s in summaries], dtype=float)
    rho = rho[np.isfinite(rho)]
    if rho.size:
        plt.figure(figsize=(5.0, 3.0))
        plt.hist(rho, bins=25)
        plt.axvline(0, color="black", linewidth=1)
        plt.title(f"{title}\nSpearman ρ(sink, entropy) across runs")
        plt.xlabel("Spearman ρ")
        plt.ylabel("count")
        p = out_dir / "hist_spearman_sink_entropy_across_runs.png"
        _savefig(p)
        items.append({"kind": "hist", "path": str(p), "desc": "Histogram of Spearman correlation across runs."})

    prho = np.array([float(s.partial_sink_entropy_spearman_seq_len) for s in summaries], dtype=float)
    raw_all = np.array([float(s.corr_sink_entropy_spearman) for s in summaries], dtype=float)
    m = np.isfinite(prho) & np.isfinite(raw_all)
    if int(m.sum()) >= 5:
        raw = raw_all[m]
        part = prho[m]
        plt.figure(figsize=(4.8, 4.0))
        plt.scatter(raw, part, s=25, alpha=0.7)
        lo = float(np.nanmin([raw.min(), part.min()]))
        hi = float(np.nanmax([raw.max(), part.max()]))
        plt.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
        plt.xlabel("Spearman ρ(sink, entropy)")
        plt.ylabel("Partial corr(resid by seq_len)")
        plt.title(f"{title}\nPartial-vs-raw correlation (seq_len control)")
        p = out_dir / "scatter_partial_vs_spearman.png"
        _savefig(p)
        items.append({"kind": "scatter", "path": str(p), "desc": "Scatter: partial vs raw correlation."})

        # histogram of difference (partial - raw)
        diff = part - raw
        plt.figure(figsize=(5.0, 3.0))
        plt.hist(diff[np.isfinite(diff)], bins=25)
        plt.axvline(0, color="black", linewidth=1)
        plt.title(f"{title}\nΔ (partial - raw) Spearman correlation")
        plt.xlabel("Δρ (partial - raw)")
        plt.ylabel("count")
        p = out_dir / "hist_delta_partial_minus_raw.png"
        _savefig(p)
        items.append({"kind": "hist", "path": str(p), "desc": "Histogram of (partial - raw) correlation differences."})

    # Cross-model consistency view (many runs per model -> boxplots)
    items.extend(plot_metrics_boxgrid_by_model(summaries, out_dir, title=title))

    return items


def plot_paired_chat_deltas(summaries: Sequence[RunQuickSummary], out_dir: Path, *, title: str) -> List[Dict[str, Any]]:
    """
    H5-style aggregate: for matched (task,model,quant,K,Q,query_start,label), compute deltas between chat modes.
    We define Δ = off - auto.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []

    # Build pairs
    groups: Dict[Tuple[Any, ...], Dict[str, RunQuickSummary]] = {}
    for s in summaries:
        cm = str(s.chat_mode)
        if cm not in ("auto", "off"):
            continue
        key = (s.task, s.model, s.quantization, s.sink_tokens, s.query_mode, s.query_start, s.label)
        groups.setdefault(key, {})[cm] = s

    rows: List[Dict[str, Any]] = []
    for key, by_cm in groups.items():
        if "auto" not in by_cm or "off" not in by_cm:
            continue
        a = by_cm["auto"]
        o = by_cm["off"]
        rows.append(
            {
                "task": key[0],
                "model": key[1],
                "quantization": key[2],
                "sink_tokens": key[3],
                "query_mode": key[4],
                "query_start": key[5],
                "label": key[6],
                "delta_acc": float(o.acc - a.acc),
                "delta_pos_rate": float(o.pos_rate - a.pos_rate),
                "delta_sink_mean": float(o.sink_mean - a.sink_mean),
                "delta_entropy_mean": float(o.entropy_mean - a.entropy_mean),
                "delta_seq_len_mean": float(o.seq_len_mean - a.seq_len_mean),
                "delta_d": float(o.cohens_d_sink_pos_minus_neg - a.cohens_d_sink_pos_minus_neg),
                "delta_auroc": float(o.auroc_sink_vs_label - a.auroc_sink_vs_label),
                "delta_auprc": float(o.auprc_sink_vs_label - a.auprc_sink_vs_label),
                "delta_rho": float(o.corr_sink_entropy_spearman - a.corr_sink_entropy_spearman),
                "delta_partial_rho": float(o.partial_sink_entropy_spearman_seq_len - a.partial_sink_entropy_spearman_seq_len),
                "auto_run_id": a.run_id,
                "off_run_id": o.run_id,
            }
        )

    csv_p = out_dir / "paired_chat_deltas.csv"
    save_table_csv(csv_p, rows)
    items.append({"kind": "table", "path": str(csv_p), "desc": "Paired deltas (off - auto) across matched runs."})

    if plt is None or not rows:
        return items

    def _col(name: str) -> np.ndarray:
        return np.array([float(r.get(name, float("nan"))) for r in rows], dtype=float)

    d = _col("delta_d")
    auroc = _col("delta_auroc")
    rho = _col("delta_rho")
    prho = _col("delta_partial_rho")

    plt.figure(figsize=(9.2, 6.2))
    plt.suptitle(f"{title}\nPaired chat deltas (off - auto)")

    ax = plt.subplot(2, 2, 1)
    dd = _finite(d)
    if dd.size:
        ax.hist(dd, bins=20)
        ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Δ Cohen's d")
    ax.set_xlabel("Δd")
    ax.set_ylabel("count")

    ax = plt.subplot(2, 2, 2)
    da = _finite(auroc)
    if da.size:
        ax.hist(da, bins=20)
        ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Δ AUROC")
    ax.set_xlabel("ΔAUROC")
    ax.set_ylabel("count")

    ax = plt.subplot(2, 2, 3)
    mr = np.isfinite(d) & np.isfinite(auroc)
    if int(mr.sum()) >= 3:
        ax.scatter(d[mr], auroc[mr], s=25, alpha=0.75)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("ΔAUROC vs Δd")
    ax.set_xlabel("Δd")
    ax.set_ylabel("ΔAUROC")

    ax = plt.subplot(2, 2, 4)
    mpr = np.isfinite(rho) & np.isfinite(prho)
    if int(mpr.sum()) >= 3:
        ax.scatter(rho[mpr], prho[mpr], s=25, alpha=0.75)
        lo = float(np.nanmin([rho[mpr].min(), prho[mpr].min()]))
        hi = float(np.nanmax([rho[mpr].max(), prho[mpr].max()]))
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
    ax.set_title("Δρ partial vs raw")
    ax.set_xlabel("Δρ raw")
    ax.set_ylabel("Δρ partial(seq_len)")

    p = out_dir / "paired_chat_deltas_panel.png"
    _savefig(p)
    items.append({"kind": "plot", "path": str(p), "desc": "Multi-panel: chat deltas (off-auto) for d/AUROC/corr."})
    return items


def plot_query_sensitivity_aggregate(
    summaries: Sequence[RunQuickSummary],
    out_dir: Path,
    *,
    title: str,
    max_context_panels: int = 6,
) -> List[Dict[str, Any]]:
    """
    H6-style aggregate: visualize how metrics vary with (query_mode, query_start).
    Produces stacked lines and heatmaps per (task, chat_mode, quantization, sink_tokens, label).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    if plt is None or not summaries:
        return items

    # Partition into contexts so we don't mix different K/chat/quant/label.
    buckets: Dict[Tuple[str, str, str, Optional[int], str], List[RunQuickSummary]] = defaultdict(list)
    for s in summaries:
        buckets[(s.task, str(s.chat_mode), str(s.quantization), s.sink_tokens, str(s.label))].append(s)

    contexts = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max_context_panels]
    if not contexts:
        return items

    # Metric accessors
    def get_metric(s: RunQuickSummary, name: str) -> float:
        if name == "auroc":
            return float(s.auroc_sink_vs_label)
        if name == "d":
            return float(s.cohens_d_sink_pos_minus_neg)
        if name == "rho":
            return float(s.corr_sink_entropy_spearman)
        if name == "sink_mean":
            return float(s.sink_mean)
        raise ValueError(name)

    for (task, chat_mode, quant, K, label), ss in contexts:
        # Determine x labels (last + range starts)
        starts = sorted({int(s.query_start) for s in ss if str(s.query_mode) == "range" and s.query_start is not None and int(s.query_start) > 0})
        xlabels = ["last"] + [f"range@{k}" for k in starts]
        xs = [0] + starts  # map last to 0 for plotting convenience

        # Build model list
        models = sorted({s.model for s in ss})

        def build_matrix(metric_name: str) -> np.ndarray:
            M = np.full((len(models), len(xs)), np.nan, dtype=float)
            # index runs
            idx: Dict[Tuple[str, str, int], float] = {}
            for s in ss:
                qm = str(s.query_mode or "last")
                if qm == "last":
                    idx[(s.model, "last", 0)] = get_metric(s, metric_name)
                elif qm == "range" and s.query_start is not None:
                    idx[(s.model, "range", int(s.query_start))] = get_metric(s, metric_name)
            for i, m in enumerate(models):
                M[i, 0] = idx.get((m, "last", 0), np.nan)
                for j, st in enumerate(starts, start=1):
                    M[i, j] = idx.get((m, "range", int(st)), np.nan)
            return M

        for metric_name, cmap, vline in [
            ("auroc", "viridis", 0.5),
            ("d", "coolwarm", 0.0),
            ("rho", "coolwarm", 0.0),
        ]:
            mat = build_matrix(metric_name)
            if not np.isfinite(mat).any():
                continue
            plt.figure(figsize=(max(6.5, 0.55 * len(xs) + 3.5), max(3.6, 0.32 * len(models) + 2.2)))
            plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)
            plt.colorbar(label=metric_name)
            plt.yticks(range(len(models)), models, fontsize=7)
            plt.xticks(range(len(xs)), xlabels, rotation=25, ha="right")
            plt.title(f"{title}\n{task} | chat={chat_mode} | quant={quant} | K={K} | label={label}\nHeatmap: {metric_name} by query spec")
            p = out_dir / f"heatmap_{metric_name}_by_model_query__{task}__chat{chat_mode}__K{K}.png"
            _savefig(p)
            items.append({"kind": "heatmap", "path": str(p), "desc": f"Heatmap of {metric_name} by (model, query spec)."})

        # Stacked line plot: AUROC vs query_start (range) for each model (plus last point at 0)
        mat_auroc = build_matrix("auroc")
        if np.isfinite(mat_auroc).any():
            plt.figure(figsize=(7.8, 4.4))
            for i, m in enumerate(models):
                y = mat_auroc[i]
                if not np.isfinite(y).any():
                    continue
                plt.plot(xs, y, marker="o", linewidth=1.2, alpha=0.85, label=m)
            plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            plt.xticks(xs, xlabels, rotation=25, ha="right")
            plt.ylabel("AUROC(sink→label)")
            plt.title(f"{title}\n{task} | chat={chat_mode} | quant={quant} | K={K} | label={label}\nStacked: AUROC vs query spec")
            plt.legend(fontsize=6, ncol=2, frameon=False)
            p = out_dir / f"line_auroc_vs_query__{task}__chat{chat_mode}__K{K}.png"
            _savefig(p)
            items.append({"kind": "plot", "path": str(p), "desc": "Stacked lines: AUROC vs query spec per model."})

    return items


def plot_top_heads_summary(metrics: Sequence[Json], out_dir: Path, *, title: str) -> List[Dict[str, Any]]:
    """
    H4-style aggregate: summarize top heads across runs (using plots/metrics.json content).
    Works even when attention shapes differ across models by aggregating at the *layer index* level.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []

    # Flatten top10 list into a long table
    long_rows: List[Dict[str, Any]] = []
    for m in metrics:
        run = str(m.get("run", ""))
        task = str(m.get("task", ""))
        model = str(m.get("model", ""))
        chat_mode = str(m.get("chat_mode", "auto"))
        quant = str(m.get("quantization", "none"))
        shape = m.get("shape")
        tops = m.get("top10", [])
        if not isinstance(tops, list):
            continue
        for rank, t in enumerate(tops, start=1):
            try:
                layer = int(t.get("layer"))
                head = int(t.get("head"))
                delta = float(t.get("delta"))
                abs_delta = float(t.get("abs_delta", abs(delta)))
            except Exception:
                continue
            long_rows.append(
                {
                    "run": run,
                    "task": task,
                    "model": model,
                    "chat_mode": chat_mode,
                    "quantization": quant,
                    "shape": str(shape),
                    "rank": int(rank),
                    "layer": layer,
                    "head": head,
                    "delta": delta,
                    "abs_delta": abs_delta,
                }
            )

    csv_p = out_dir / "top_heads_long.csv"
    save_table_csv(csv_p, long_rows)
    items.append({"kind": "table", "path": str(csv_p), "desc": "Long table of top heads (flattened from per-run top10)."})

    if plt is None or not long_rows:
        return items

    layers = np.array([float(r["layer"]) for r in long_rows], dtype=float)
    absd = np.array([float(r["abs_delta"]) for r in long_rows], dtype=float)
    ranks = np.array([float(r["rank"]) for r in long_rows], dtype=float)

    plt.figure(figsize=(9.0, 3.6))
    plt.suptitle(f"{title}\nTop-head localization summary")

    ax = plt.subplot(1, 2, 1)
    m1 = np.isfinite(layers)
    if int(m1.sum()):
        ax.hist(layers[m1], bins=30)
    ax.set_title("Layer index frequency (top10 pooled)")
    ax.set_xlabel("layer")
    ax.set_ylabel("count")

    ax = plt.subplot(1, 2, 2)
    m2 = np.isfinite(layers) & np.isfinite(absd) & np.isfinite(ranks)
    if int(m2.sum()):
        # emphasize top-1 points
        s = np.where(ranks[m2] <= 1.0, 28.0, 14.0)
        ax.scatter(layers[m2], absd[m2], s=s, alpha=0.65)
    ax.set_title("|Δ sink| vs layer (top10 pooled)")
    ax.set_xlabel("layer")
    ax.set_ylabel("|Δ|")

    p = out_dir / "top_heads_localization_panel.png"
    _savefig(p)
    items.append({"kind": "plot", "path": str(p), "desc": "Multi-panel: layer frequency + |Δ| vs layer for top heads."})
    return items


def plot_aggregate_layer_profiles(
    runs: Sequence[Tuple[str, List[Json]]],
    out_dir: Path,
    *,
    title: str,
    label_mode: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Stacked aggregate for hypotheses that produce `sink_by_layer`.
    - Plot many per-run mean layer profiles on one canvas.
    - Plot many per-run (label1-label0) delta layer profiles on one canvas.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    if plt is None or not runs:
        return items

    means: List[np.ndarray] = []
    deltas: List[np.ndarray] = []
    labels: List[str] = []

    for rid, rows in runs:
        layer = _maybe_stack_layer(rows, "sink_by_layer")
        if layer is None:
            continue
        # mean profile
        mean = _nanmean_axis0(layer)
        means.append(mean)
        labels.append(str(rid))

        # delta profile by label
        label_name, y = choose_label(rows, label_mode)
        y = y.astype(int)
        m = np.isfinite(layer).any(axis=1)
        if int(m.sum()) < 10:
            continue
        layer_m = layer[m]
        y_m = y[m]
        if int((y_m == 1).sum()) < 5 or int((y_m == 0).sum()) < 5:
            continue
        mean1 = _nanmean_axis0(layer_m[y_m == 1])
        mean0 = _nanmean_axis0(layer_m[y_m == 0])
        deltas.append(mean1 - mean0)

    if means:
        plt.figure(figsize=(7.8, 4.2))
        for v in means:
            if np.isfinite(v).any():
                plt.plot(v, linewidth=1.0, alpha=0.6)
        plt.title(f"{title}\nStacked mean sink_by_layer profiles")
        plt.xlabel("layer")
        plt.ylabel("mean sink_by_layer")
        p = out_dir / "stacked_layer_mean_profiles.png"
        _savefig(p)
        items.append({"kind": "plot", "path": str(p), "desc": "Stacked lines: mean sink_by_layer per run."})

    if deltas:
        plt.figure(figsize=(7.8, 4.2))
        for v in deltas:
            if np.isfinite(v).any():
                plt.plot(v, linewidth=1.0, alpha=0.6)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title(f"{title}\nStacked Δ sink_by_layer profiles (label1-label0)")
        plt.xlabel("layer")
        plt.ylabel("Δ mean sink_by_layer")
        p = out_dir / "stacked_layer_delta_profiles.png"
        _savefig(p)
        items.append({"kind": "plot", "path": str(p), "desc": "Stacked lines: delta sink_by_layer between labels per run."})

    return items

