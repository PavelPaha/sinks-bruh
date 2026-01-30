from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Key:
    task: str
    sink_tokens: int
    query_mode: str
    query_start: int
    chat_mode: str


def _read_metrics_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r = dict(r)
            for k in ("sink_tokens", "query_start", "n"):
                r[k] = int(r[k])  # type: ignore[assignment]
            for k in ("sink_mean", "entropy_mean", "pearson", "spearman", "partial_seq_len"):
                r[k] = float(r[k])  # type: ignore[assignment]
            rows.append(r)
    return rows


def _safe_f(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _median(xs: Sequence[float]) -> float:
    xs2 = [x for x in xs if math.isfinite(x)]
    if not xs2:
        return float("nan")
    xs2.sort()
    n = len(xs2)
    if n % 2 == 1:
        return float(xs2[n // 2])
    return float(0.5 * (xs2[n // 2 - 1] + xs2[n // 2]))


def _key(r: Dict[str, object]) -> Key:
    return Key(
        task=str(r["task"]),
        sink_tokens=int(r["sink_tokens"]),
        query_mode=str(r["query_mode"]),
        query_start=int(r["query_start"]),
        chat_mode=str(r["chat_mode"]),
    )


def _cond_label(k: Key) -> str:
    q = "last" if k.query_mode == "last" else f"range@{k.query_start}"
    return f"K{k.sink_tokens}/{q}/chat={k.chat_mode}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_manifest(out_dir: Path, items: List[Dict[str, object]]) -> None:
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "hypothesis_id": "H7_sink_entropy",
                "generated_by": str(Path(__file__).name),
                "items": items,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _fit_line(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit y = a*x + b and return (a,b,r2). NaNs must be removed by caller.
    """
    if xs.size < 3:
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(xs, ys, 1)
    yhat = a * xs + b
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return float(a), float(b), r2


def _hist(out_path: Path, *, xs: List[float], title: str, xlabel: str, vline0: bool = True) -> None:
    x = np.array(xs, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    plt.figure(figsize=(5.6, 3.6))
    plt.hist(x, bins=18, alpha=0.85, color="slateblue")
    if vline0:
        plt.axvline(0.0, color="black", linewidth=1)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _box_by_condition(
    out_path: Path,
    *,
    groups: Dict[str, List[float]],
    title: str,
    ylabel: str,
) -> None:
    labels = list(groups.keys())
    data = []
    for k in labels:
        v = np.array(groups[k], dtype=float)
        v = v[np.isfinite(v)]
        data.append(v)
    if not any(len(v) for v in data):
        return
    plt.figure(figsize=(max(6.5, 1.2 * len(labels)), 3.8))
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _scatter_diag(
    out_path: Path,
    *,
    xs: List[float],
    ys: List[float],
    labels: Optional[List[str]],
    title: str,
    xlabel: str,
    ylabel: str,
    lim: Tuple[float, float],
    annotate_topk: int = 5,
    diag: bool = True,
) -> Dict[str, float]:
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    labs = None
    if labels is not None:
        labs = [labels[i] for i in range(len(labels)) if m[i]]

    plt.figure(figsize=(5.8, 4.6))
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    plt.axvline(0.0, color="black", linewidth=1, alpha=0.4)
    plt.scatter(x, y, s=38, alpha=0.85, color="teal")
    if diag:
        plt.plot([lim[0], lim[1]], [lim[0], lim[1]], color="gray", linewidth=1, linestyle="--", alpha=0.8)
    a, b, r2 = _fit_line(x, y)
    if np.isfinite(a) and np.isfinite(b):
        xx = np.linspace(lim[0], lim[1], 100)
        plt.plot(xx, a * xx + b, color="tomato", linewidth=1.7)
        plt.text(
            0.02,
            0.98,
            f"fit: y={a:+.2f}x{b:+.2f}, R²={r2:.2f}",
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, linewidth=0.5),
        )

    # annotate top-k by |delta|
    if labs is not None and annotate_topk > 0:
        d = np.abs(y - x)
        top = np.argsort(-d)[: min(int(annotate_topk), int(len(d)))]
        for i in top:
            plt.text(x[i], y[i], labs[i], fontsize=8, alpha=0.9, ha="left", va="bottom")

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(*lim)
    plt.ylim(*lim)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return {"slope": a, "intercept": b, "r2": r2, "n": float(len(x))}


def _bar_counts(out_path: Path, *, labels: List[str], counts: List[int], title: str, ylabel: str) -> None:
    x = np.arange(len(labels))
    plt.figure(figsize=(max(6.5, 1.1 * len(labels)), 3.6))
    plt.bar(x, counts, color="darkorange", alpha=0.85)
    plt.xticks(x, labels, rotation=25, ha="right", fontsize=9)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _heatmap(
    out_path: Path, *,
    models: List[str],
    cols: List[Key],
    value_by: Dict[Tuple[str, Key], float],
    title: str,
    vmin: float = -0.3,
    vmax: float = 0.3,
) -> None:
    M = np.full((len(models), len(cols)), np.nan, dtype=float)
    for i, m in enumerate(models):
        for j, c in enumerate(cols):
            v = value_by.get((m, c))
            if v is not None and math.isfinite(float(v)):
                M[i, j] = float(v)

    plt.figure(figsize=(max(7, 1.4 * len(cols)), max(3, 0.45 * len(models) + 1.5)))
    cmap = plt.get_cmap("coolwarm")
    im = plt.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.03, pad=0.02, label="Spearman ρ (sink_mass, entropy)")
    plt.yticks(range(len(models)), [m.split("/")[-1] for m in models], fontsize=9)
    plt.xticks(range(len(cols)), [_cond_label(c) for c in cols], rotation=35, ha="right", fontsize=9)
    plt.title(title, fontsize=12, fontweight="bold")

    # annotate values
    for i in range(len(models)):
        for j in range(len(cols)):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            plt.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _bar_deltas(
    out_path: Path, *,
    models: List[str],
    deltas: Dict[str, float],
    title: str,
    xlabel: str,
) -> None:
    xs = [deltas.get(m, float("nan")) for m in models]
    plt.figure(figsize=(max(8, 0.7 * len(models)), 3.8))
    x = np.arange(len(models))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.bar(x, [0 if not math.isfinite(v) else v for v in xs], color="slateblue", alpha=0.85)
    plt.xticks(x, [m.split("/")[-1] for m in models], rotation=30, ha="right", fontsize=9)
    plt.ylabel(xlabel)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _scatter(
    out_path: Path, *,
    points: List[Tuple[float, float, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    # points: (x,y,label)
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[m]
    ys = ys[m]
    labs = [points[i][2] for i in range(len(points)) if m[i]]
    plt.figure(figsize=(5.5, 4.2))
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    plt.scatter(xs, ys, s=35, alpha=0.85, color="teal")
    for x, y, lab in zip(xs, ys, labs):
        plt.text(x, y, lab, fontsize=8, alpha=0.9, ha="left", va="bottom")
    if len(xs) >= 3:
        try:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            xx = np.linspace(float(xs.min()), float(xs.max()), 50)
            plt.plot(xx, p(xx), color="tomato", linewidth=1.5)
        except Exception:
            pass
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots_dir", default=None, help="Override plots dir (defaults to hypotheses/H7_sink_entropy/plots)")
    ap.add_argument("--metric", default="spearman", choices=["spearman", "pearson", "partial_seq_len"])
    args = ap.parse_args()

    hyp_dir = Path(__file__).resolve().parents[1]
    plots_dir = Path(args.plots_dir) if args.plots_dir else (hyp_dir / "plots")
    metrics_csv = plots_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise SystemExit(f"Missing {metrics_csv}. Expected metrics from H7 run_all.py.")

    rows = _read_metrics_csv(metrics_csv)
    if not rows:
        raise SystemExit("Empty metrics.csv")

    models = sorted({str(r["model"]) for r in rows})
    keys = sorted(
        {_key(r) for r in rows},
        key=lambda k: (k.task, k.sink_tokens, k.query_mode, k.query_start, k.chat_mode),
    )

    value_by: Dict[Tuple[str, Key], float] = {}
    for r in rows:
        value_by[(str(r["model"]), _key(r))] = _safe_f(r.get(args.metric))

    out_dir = _ensure_dir(plots_dir / "agg")
    manifest_items: List[Dict[str, object]] = []

    # 1) heatmap for chosen metric
    heatmap_path = out_dir / f"heatmap_{args.metric}.png"
    _heatmap(
        heatmap_path,
        models=models,
        cols=keys,
        value_by=value_by,
        title=f"H7: {args.metric} correlation heatmap (by model × condition)",
        vmin=-0.35,
        vmax=0.35,
    )
    manifest_items.append({"type": "heatmap", "path": str(heatmap_path), "metric": args.metric})

    # helpers for deltas (we focus on the common grid K∈{1,4}, q∈{last, range@32})
    def get(model: str, sink_tokens: int, query_mode: str, query_start: int, chat_mode: str = "auto") -> Optional[float]:
        k = Key(task="truthfulqa_mc", sink_tokens=sink_tokens, query_mode=query_mode, query_start=query_start, chat_mode=chat_mode)
        v = value_by.get((model, k))
        if v is None or not math.isfinite(v):
            return None
        return float(v)

    # 2) query effect: (last - range@32), separately for K=1 and K=4
    for K in (1, 4):
        deltas = {}
        for m in models:
            a = get(m, K, "last", 0)
            b = get(m, K, "range", 32)
            if a is None or b is None:
                continue
            deltas[m] = a - b
        p = out_dir / f"delta_query_last_minus_range__K{K}__{args.metric}.png"
        _bar_deltas(
            p,
            models=models,
            deltas=deltas,
            title=f"H7: query-set effect (last − range@32), K={K} [{args.metric}]",
            xlabel="Δ correlation",
        )
        manifest_items.append({"type": "bar_delta", "path": str(p), "metric": args.metric, "delta": "last-range@32", "K": K})

    # 3) K effect: (K4 - K1), separately for q=last and q=range@32
    for qmode, qstart in (("last", 0), ("range", 32)):
        deltas = {}
        for m in models:
            a = get(m, 4, qmode, qstart)
            b = get(m, 1, qmode, qstart)
            if a is None or b is None:
                continue
            deltas[m] = a - b
        qlbl = "last" if qmode == "last" else "range@32"
        p = out_dir / f"delta_K4_minus_K1__q{qlbl}__{args.metric}.png"
        _bar_deltas(
            p,
            models=models,
            deltas=deltas,
            title=f"H7: sink-window effect (K4 − K1), q={qlbl} [{args.metric}]",
            xlabel="Δ correlation",
        )
        manifest_items.append({"type": "bar_delta", "path": str(p), "metric": args.metric, "delta": "K4-K1", "q": qlbl})

    # 4) scatter: sink_mean vs correlation (per condition)
    for k in keys:
        pts = []
        for m in models:
            # locate row to get sink_mean/entropy_mean (stable per run)
            rr = next((r for r in rows if str(r["model"]) == m and _key(r) == k), None)
            if rr is None:
                continue
            x = float(rr["sink_mean"])
            y = _safe_f(rr.get(args.metric))
            pts.append((x, y, m.split("/")[-1]))
        if len(pts) >= 3:
            _scatter(
                out_dir / f"scatter_sinkmean_vs_{args.metric}__{_cond_label(k).replace('/','_')}.png",
                points=pts,
                title=f"H7: sink_mean vs {args.metric} ({_cond_label(k)})",
                xlabel="mean(sink_mass)",
                ylabel=args.metric,
            )

    # 5) scatter: entropy_mean vs correlation (per condition)
    for k in keys:
        pts = []
        for m in models:
            rr = next((r for r in rows if str(r["model"]) == m and _key(r) == k), None)
            if rr is None:
                continue
            x = float(rr["entropy_mean"])
            y = _safe_f(rr.get(args.metric))
            pts.append((x, y, m.split("/")[-1]))
        if len(pts) >= 3:
            _scatter(
                out_dir / f"scatter_entropymean_vs_{args.metric}__{_cond_label(k).replace('/','_')}.png",
                points=pts,
                title=f"H7: entropy_mean vs {args.metric} ({_cond_label(k)})",
                xlabel="mean(entropy)",
                ylabel=args.metric,
            )

    # 6) partial-vs-raw scatter on per-run basis (if metric is spearman)
    if args.metric == "spearman":
        pts = []
        xs = []
        ys = []
        labs = []
        for r in rows:
            x = _safe_f(r.get("spearman"))
            y = _safe_f(r.get("partial_seq_len"))
            lab = f"{str(r['model']).split('/')[-1]} K{int(r['sink_tokens'])}/{str(r['query_mode'])}"
            pts.append((x, y, lab))
            xs.append(x)
            ys.append(y)
            labs.append(lab)

        # (a) detailed scatter with diagonal + fit
        sp = out_dir / "scatter_partial_vs_spearman.png"
        fit = _scatter_diag(
            sp,
            xs=xs,
            ys=ys,
            labels=labs,
            title="H7: partial(ρ|seq_len) vs raw Spearman ρ",
            xlabel="Spearman ρ(sink_mass, entropy)",
            ylabel="partial ρ(sink_mass, entropy | seq_len)",
            lim=(-0.35, 0.35),
            annotate_topk=7,
            diag=True,
        )
        manifest_items.append({"type": "scatter", "path": str(sp), "kind": "partial_vs_spearman", "fit": fit})

        # (b) delta histogram
        deltas = [(_safe_f(r.get("partial_seq_len")) - _safe_f(r.get("spearman"))) for r in rows]
        dh = out_dir / "hist_delta_partial_minus_spearman.png"
        _hist(
            dh,
            xs=deltas,
            title="H7: distribution of Δ = partial(ρ|len) − Spearman ρ",
            xlabel="Δ",
            vline0=True,
        )
        manifest_items.append({"type": "hist", "path": str(dh), "kind": "delta_partial_minus_spearman"})

        # (c) boxplots by condition (raw vs partial)
        groups_raw: Dict[str, List[float]] = defaultdict(list)
        groups_part: Dict[str, List[float]] = defaultdict(list)
        groups_delta: Dict[str, List[float]] = defaultdict(list)
        signflip_counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            k = _key(r)
            lbl = _cond_label(k)
            raw = _safe_f(r.get("spearman"))
            par = _safe_f(r.get("partial_seq_len"))
            if np.isfinite(raw):
                groups_raw[lbl].append(raw)
            if np.isfinite(par):
                groups_part[lbl].append(par)
            if np.isfinite(raw) and np.isfinite(par):
                groups_delta[lbl].append(par - raw)
                if raw == 0.0:
                    continue
                if (raw > 0) != (par > 0):
                    signflip_counts[lbl] += 1

        bp1 = out_dir / "box_spearman_by_condition.png"
        _box_by_condition(
            bp1,
            groups=dict(sorted(groups_raw.items(), key=lambda kv: kv[0])),
            title="H7: Spearman ρ by condition",
            ylabel="Spearman ρ(sink_mass, entropy)",
        )
        manifest_items.append({"type": "box", "path": str(bp1), "kind": "spearman_by_condition"})

        bp2 = out_dir / "box_partial_by_condition.png"
        _box_by_condition(
            bp2,
            groups=dict(sorted(groups_part.items(), key=lambda kv: kv[0])),
            title="H7: partial(ρ|seq_len) by condition",
            ylabel="partial ρ(sink_mass, entropy | seq_len)",
        )
        manifest_items.append({"type": "box", "path": str(bp2), "kind": "partial_by_condition"})

        bp3 = out_dir / "box_delta_partial_minus_spearman_by_condition.png"
        _box_by_condition(
            bp3,
            groups=dict(sorted(groups_delta.items(), key=lambda kv: kv[0])),
            title="H7: Δ = partial − raw by condition",
            ylabel="Δ",
        )
        manifest_items.append({"type": "box", "path": str(bp3), "kind": "delta_by_condition"})

        # (d) sign-flip counts per condition
        sfc_labels = list(sorted(signflip_counts.keys()))
        sfc_counts = [int(signflip_counts[k]) for k in sfc_labels]
        sfp = out_dir / "bar_signflip_counts_by_condition.png"
        _bar_counts(
            sfp,
            labels=sfc_labels,
            counts=sfc_counts,
            title="H7: how often partial flips sign vs raw (per condition)",
            ylabel="#sign flips",
        )
        manifest_items.append({"type": "bar", "path": str(sfp), "kind": "signflip_counts"})

        # (e) spearman vs pearson scatter (run-level)
        sp2 = out_dir / "scatter_spearman_vs_pearson.png"
        x2 = [float(r.get("pearson", float("nan"))) for r in rows]
        y2 = [float(r.get("spearman", float("nan"))) for r in rows]
        labs2 = [f"{str(r['model']).split('/')[-1]} K{int(r['sink_tokens'])}/{str(r['query_mode'])}" for r in rows]
        _scatter_diag(
            sp2,
            xs=x2,
            ys=y2,
            labels=labs2,
            title="H7: Spearman ρ vs Pearson r (run-level)",
            xlabel="Pearson r(sink_mass, entropy)",
            ylabel="Spearman ρ(sink_mass, entropy)",
            lim=(-0.35, 0.35),
            annotate_topk=0,
            diag=True,
        )
        manifest_items.append({"type": "scatter", "path": str(sp2), "kind": "spearman_vs_pearson"})

        # (f) K-effect scatter: rho(K1) vs rho(K4) for q=last and q=range@32
        def _k_scatter(qmode: str, qstart: int) -> None:
            ptsx = []
            ptsy = []
            labs3 = []
            for mname in models:
                r1 = get(mname, 1, qmode, qstart)
                r4 = get(mname, 4, qmode, qstart)
                if r1 is None or r4 is None:
                    continue
                ptsx.append(r1)
                ptsy.append(r4)
                labs3.append(mname.split("/")[-1])
            if len(ptsx) < 3:
                return
            qlbl2 = "last" if qmode == "last" else "range@32"
            outp = out_dir / f"scatter_rho_K1_vs_K4__q{qlbl2}.png"
            _scatter_diag(
                outp,
                xs=ptsx,
                ys=ptsy,
                labels=labs3,
                title=f"H7: model-wise ρ(K1) vs ρ(K4), q={qlbl2}",
                xlabel="Spearman ρ at K=1",
                ylabel="Spearman ρ at K=4",
                lim=(-0.35, 0.35),
                annotate_topk=5,
                diag=True,
            )
            manifest_items.append({"type": "scatter", "path": str(outp), "kind": "rho_K1_vs_K4", "q": qlbl2})

        _k_scatter("last", 0)
        _k_scatter("range", 32)

    # summary json for final.md
    by_key: Dict[Key, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_key[_key(r)].append(r)

    summary = {
        "hypothesis_id": "H7_sink_entropy",
        "source": str(metrics_csv),
        "n_rows": len(rows),
        "n_models": len(models),
        "keys": [
            {
                "key": k.__dict__,
                "n_models": len(by_key[k]),
                "median_spearman": _median([_safe_f(r.get("spearman")) for r in by_key[k]]),
                "median_pearson": _median([_safe_f(r.get("pearson")) for r in by_key[k]]),
                "median_partial_seq_len": _median([_safe_f(r.get("partial_seq_len")) for r in by_key[k]]),
            }
            for k in keys
        ],
    }
    (out_dir / "agg_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_items.append({"type": "data", "path": str(out_dir / "agg_metrics.json"), "kind": "agg_metrics"})

    _write_manifest(out_dir, manifest_items)

    print(f"[OK] wrote plots into {out_dir}")


if __name__ == "__main__":
    main()

