#!/usr/bin/env python

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


@dataclass(frozen=True)
class Point:
    pipeline: str
    aggregation: str
    t1: float
    t2: float
    tzsr: float
    created_at: str
    metrics: Dict[str, Optional[float]]


DEFAULT_METRIC_STYLES: Dict[str, Any] = {
    "accuracy": "-",
    "precision": "--",
    "recall": "-.",
    "sensitivity": "-.",
    "specificity": ":",
    "f1": (0, (5, 1)),
}

DEFAULT_METRICS: List[str] = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
]

AGG_MARKERS: Dict[str, str] = {
    "mean": "x",
    "max": "o",
}


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _iter_overview_files(results_root: str) -> Iterable[str]:
    root = os.path.abspath(results_root)
    for dirpath, _, filenames in os.walk(root):
        if dirpath == root:
            continue
        if "overview.json" not in filenames:
            continue
        yield os.path.join(dirpath, "overview.json")


def _load_points_from_overview(path: str) -> List[Point]:
    with open(path, "r") as f:
        obj = json.load(f)

    created_at = str(obj.get("created_at", ""))
    pipelines = obj.get("pipelines")
    if not isinstance(pipelines, list):
        return []

    points: List[Point] = []
    for entry in pipelines:
        if not isinstance(entry, dict):
            continue
        pipeline = str(entry.get("pipeline", ""))
        aggregation = str(entry.get("aggregation", "mean"))
        thresholds = entry.get("thresholds")
        metrics = entry.get("metrics")
        if (
            not pipeline
            or not isinstance(thresholds, dict)
            or not isinstance(metrics, dict)
        ):
            continue

        t1 = _as_float(thresholds.get("t1"))
        t2 = _as_float(thresholds.get("t2"))
        tzsr = _as_float(thresholds.get("tzsr"))
        if t1 is None or t2 is None or tzsr is None:
            continue

        m: Dict[str, Optional[float]] = {}
        for k in DEFAULT_METRIC_STYLES.keys():
            v = metrics.get(k)
            if v is None and k == "sensitivity":
                v = metrics.get("recall")
            if v is None and k == "recall":
                v = metrics.get("sensitivity")
            m[k] = _as_float(v)

        points.append(
            Point(
                pipeline=pipeline,
                aggregation=aggregation,
                t1=float(t1),
                t2=float(t2),
                tzsr=float(tzsr),
                created_at=created_at,
                metrics=m,
            )
        )

    return points


def _parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_csv_strings(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [p.strip() for p in s.split(",") if p.strip()]


def _label(pipeline: str, t1: float) -> str:
    return f"{pipeline}"
    # return f"{pipeline}@t1={t1:.2f}"


def _select_latest(points: Sequence[Point]) -> Point:
    # created_at is YYYYmmdd_HHMMSS; lexicographic compare works
    return sorted(points, key=lambda p: p.created_at)[-1]


def _collect_points(
    *,
    results_root: str,
    pipelines: Optional[Sequence[str]],
    aggregations: Optional[Sequence[str]],
    t1_values: Optional[Sequence[float]],
    t2_values: Optional[Sequence[float]],
    tzsr_values: Optional[Sequence[float]],
) -> List[Point]:
    all_points: List[Point] = []
    for fp in _iter_overview_files(results_root):
        all_points.extend(_load_points_from_overview(fp))

    if pipelines is not None:
        allowed = set(pipelines)
        all_points = [p for p in all_points if p.pipeline in allowed]

    if aggregations is not None:
        allowed = set(aggregations)
        all_points = [p for p in all_points if p.aggregation in allowed]

    if t1_values is not None:
        allowed = {float(x) for x in t1_values}
        all_points = [p for p in all_points if p.t1 in allowed]

    if t2_values is not None:
        allowed = {float(x) for x in t2_values}
        all_points = [p for p in all_points if p.t2 in allowed]

    if tzsr_values is not None:
        allowed = {float(x) for x in tzsr_values}
        all_points = [p for p in all_points if p.tzsr in allowed]

    # Deduplicate exact (pipeline, aggregation, t1, t2, tzsr) by choosing latest created_at.
    grouped: Dict[Tuple[str, str, float, float, float], List[Point]] = {}
    for p in all_points:
        key = (p.pipeline, p.aggregation, p.t1, p.t2, p.tzsr)
        grouped.setdefault(key, []).append(p)

    return [_select_latest(v) for v in grouped.values()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot metrics vs tzsr for patient-level ZSR summaries (overview.json runs)."
    )
    ap.add_argument(
        "--results-root",
        default=os.path.join("analysis", "outputs", "patient_level_zsr"),
        help="Root dir containing run folders (default: analysis/outputs/patient_level_zsr)",
    )
    ap.add_argument(
        "--pipelines",
        default=None,
        help="Comma-separated list of pipelines to include (default: all)",
    )
    ap.add_argument(
        "--aggregations",
        default=None,
        help="Comma-separated list of aggregations to include (e.g. mean,max). Default: all",
    )
    ap.add_argument(
        "--t1-values",
        default=None,
        help="Comma-separated list of t1 values to include (e.g. 0.5,0.8)",
    )
    ap.add_argument(
        "--t2-values",
        default=None,
        help="Comma-separated list of t2 values to include (default: all)",
    )
    ap.add_argument(
        "--tzsr-values",
        default=None,
        help="Comma-separated list of tzsr values to include (default: all)",
    )
    ap.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=(
            "Comma-separated list of metrics to plot (default: accuracy,precision,recall,specificity,f1). "
            "Use 'all' to plot all known metrics."
        ),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(
            "analysis", "outputs", "patient_level_zsr", "zsr_sweep_metrics.png"
        ),
    )
    ap.add_argument(
        "--xtick-step",
        type=float,
        default=None,
        help="Optional major tick step for x-axis (e.g. 0.1). Default: auto",
    )
    ap.add_argument(
        "--legend-inside",
        action="store_true",
        help="If set, place legends inside the axes (on top of the grid) instead of below the plot.",
    )
    ap.add_argument(
        "--legend-inside-loc",
        choices=["upper", "lower"],
        default="upper",
        help="Legend placement when --legend-inside is set (default: upper).",
    )
    ap.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="Optional font size for x/y axis labels (e.g. 14). Default: matplotlib default",
    )
    ap.add_argument(
        "--title-fontsize",
        type=float,
        default=None,
        help="Optional font size for title (e.g. 16). Default: matplotlib default",
    )
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    pipelines = _parse_csv_strings(args.pipelines)
    aggregations = _parse_csv_strings(args.aggregations)
    t1_values = _parse_csv_floats(args.t1_values)
    t2_values = _parse_csv_floats(args.t2_values)
    tzsr_values = _parse_csv_floats(args.tzsr_values)

    if aggregations is not None:
        unknown_aggs = [a for a in aggregations if a not in AGG_MARKERS]
        if unknown_aggs:
            allowed = ",".join(sorted(AGG_MARKERS.keys()))
            raise SystemExit(
                f"Unknown aggregation(s): {unknown_aggs}. Allowed: {allowed}"
            )

    if str(args.metrics).strip().lower() == "all":
        selected_metrics = list(DEFAULT_METRIC_STYLES.keys())
    else:
        selected_metrics = _parse_csv_strings(args.metrics) or []

    unknown = [m for m in selected_metrics if m not in DEFAULT_METRIC_STYLES]
    if unknown:
        allowed = ",".join(sorted(DEFAULT_METRIC_STYLES.keys()))
        raise SystemExit(f"Unknown metric(s): {unknown}. Allowed: {allowed}")

    points = _collect_points(
        results_root=args.results_root,
        pipelines=pipelines,
        aggregations=aggregations,
        t1_values=t1_values,
        t2_values=t2_values,
        tzsr_values=tzsr_values,
    )

    if not points:
        raise SystemExit(f"No points found under: {os.path.abspath(args.results_root)}")

    # group by (pipeline, t1, aggregation)
    groups: Dict[Tuple[str, float, str], List[Point]] = {}
    for p in points:
        groups.setdefault((p.pipeline, p.t1, p.aggregation), []).append(p)

    group_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2]))

    cmap = plt.get_cmap("tab10")
    color_for: Dict[Tuple[str, float], Any] = {}
    base_keys = sorted({(k[0], k[1]) for k in group_keys}, key=lambda k: (k[0], k[1]))
    for i, k in enumerate(base_keys):
        color_for[k] = cmap(i % 10)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for gk in group_keys:
        pts = sorted(groups[gk], key=lambda p: p.tzsr)
        color = color_for[(gk[0], gk[1])]
        marker = AGG_MARKERS.get(gk[2], "x")

        for metric_name in selected_metrics:
            ls = DEFAULT_METRIC_STYLES[metric_name]
            ys: List[float] = []
            xs_used: List[float] = []
            for p in pts:
                v = p.metrics.get(metric_name)
                if v is None:
                    continue
                xs_used.append(p.tzsr)
                ys.append(float(v))
            if not ys:
                continue
            ax.plot(
                xs_used,
                ys,
                linestyle=ls,
                color=color,
                linewidth=2.2,
                marker=marker,
                markersize=7,
            )

    if args.label_fontsize is not None:
        ax.set_xlabel("ZSR threshold (tzsr)", fontsize=float(args.label_fontsize))
        ax.set_ylabel("Metric value", fontsize=float(args.label_fontsize))
    else:
        ax.set_xlabel("ZSR threshold (tzsr)")
        ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.01)
    if args.xtick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(float(args.xtick_step)))
    ax.grid(True, linestyle="--", alpha=0.4)

    # ax.axvline(0.7, color="green", linestyle="--", linewidth=2.0, alpha=0.9) # only for publication once

    if args.title:
        if args.title_fontsize is not None:
            ax.set_title(str(args.title), fontsize=float(args.title_fontsize))
        else:
            ax.set_title(str(args.title))

    # Legends: one for pipeline/t1 colors, one for metric linestyles.
    color_handles = [
        Patch(facecolor=color_for[k], edgecolor="none", label=_label(k[0], k[1]))
        for k in base_keys
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=DEFAULT_METRIC_STYLES[name],
            linewidth=2.2,
            label=name,
        )
        for name in selected_metrics
    ]

    # aggregation legend (marker-only)
    used_aggs = sorted({k[2] for k in group_keys})
    agg_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="none",
            marker=AGG_MARKERS.get(a, "x"),
            markersize=7,
            label=a,
        )
        for a in used_aggs
    ]

    if args.legend_inside:
        if args.legend_inside_loc == "lower":
            pipe_loc = "lower left"
            pipe_anchor = (0.01, 0.01)
            metric_loc = "lower right"
            metric_anchor = (0.99, 0.01)
        else:
            pipe_loc = "upper left"
            pipe_anchor = (0.01, 0.99)
            metric_loc = "upper right"
            metric_anchor = (0.99, 0.99)

        leg1 = ax.legend(
            handles=color_handles,
            title="Pipeline",
            loc=pipe_loc,
            bbox_to_anchor=pipe_anchor,
            ncol=2,
            frameon=True,
            framealpha=0.9,
        )
        ax.add_artist(leg1)

        leg2 = ax.legend(
            handles=style_handles,
            title="Metric",
            loc=metric_loc,
            bbox_to_anchor=metric_anchor,
            ncol=2,
            frameon=True,
            framealpha=0.9,
        )
    else:
        legend_y = -0.20

        leg1 = ax.legend(
            handles=color_handles,
            title="Pipeline",
            loc="lower left",
            bbox_to_anchor=(0.0, legend_y),
            ncol=2,
            frameon=True,
        )
        ax.add_artist(leg1)

        leg2 = ax.legend(
            handles=style_handles,
            title="Metric",
            loc="lower right",
            bbox_to_anchor=(1.0, legend_y),
            ncol=2,
            frameon=True,
        )

    # if agg_handles:
    #     ax.add_artist(leg2)
    #     ax.legend(
    #         handles=agg_handles,
    #         title="Aggregation",
    #         loc="lower center",
    #         bbox_to_anchor=(0.5, -0.34),
    #         ncol=2,
    #         frameon=True,
    #     )

    plt.tight_layout()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))

    if out_path.lower().endswith(".png"):
        pdf_path = out_path[:-4] + ".pdf"
        fig.savefig(pdf_path)


if __name__ == "__main__":
    main()
