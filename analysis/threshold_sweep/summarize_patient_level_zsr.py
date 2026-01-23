#!/usr/bin/env python

import argparse
import csv
import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class TrackDecision:
    path: str
    recording_id: str
    n_windows: int
    n_swallow: int
    n_zenker: int
    zsr: float


@dataclass(frozen=True)
class PatientDecision:
    pipeline: str
    aggregation: str
    fold: str
    patient_id: str
    y_true: int
    y_pred: int
    zsr_track: List[float]
    zsr_mean: float
    zsr_max: float
    zsr_sum: float
    zsr_patient: float
    tracks: List[TrackDecision]


def _parse_csv_strings(s: str) -> List[str]:
    return [p.strip() for p in str(s).split(",") if p.strip()]


def _safe_slug(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _run_name(pipelines: Sequence[str], t1: float, t2: float, tzsr: float) -> str:
    if len(pipelines) == 1:
        pipe_part = _safe_slug(pipelines[0])
    else:
        pipe_part = _safe_slug("_".join(pipelines))
        if len(pipe_part) > 80:
            pipe_part = pipe_part[:80] + "_etc"

    thr_part = f"t1{t1:.2f}_t2{t2:.2f}_tzsr_{tzsr:.2f}"
    return f"{pipe_part}_{thr_part}"


def _as_str(x: Any) -> str:
    if isinstance(x, np.ndarray) and getattr(x, "shape", None) == ():
        x = x.item()
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    try:
        return {k: data[k] for k in data.files}
    finally:
        data.close()


def _compute_track_zsr(
    *, p_swallow: np.ndarray, p_zenker: np.ndarray, t1: float, t2: float
) -> Tuple[int, int, float]:
    p_sw = np.asarray(p_swallow, dtype=np.float32)
    p_ze = np.asarray(p_zenker, dtype=np.float32)

    swallow_mask = p_sw >= float(t1)
    zenker_mask = p_ze >= float(t2)

    forwarded_mask = swallow_mask & np.isfinite(p_ze)
    zenker_over_swallow = forwarded_mask & zenker_mask

    n_swallow = int(swallow_mask.sum())
    n_zenker = int(zenker_over_swallow.sum())

    zsr = float(n_zenker / n_swallow) if n_swallow > 0 else 0.0
    return n_swallow, n_zenker, zsr


def _infer_true_label_from_long_root(
    long_audio_root: str, patient_id: str
) -> Optional[int]:
    root = os.path.abspath(long_audio_root)

    def _exists(cls: str, pid: str) -> bool:
        return os.path.isdir(os.path.join(root, cls, pid))

    candidates = [patient_id]
    if patient_id.isdigit():
        candidates.append(patient_id.zfill(3))

    for pid in candidates:
        if _exists("Zenker", pid):
            return 1
        if _exists("Healthy", pid):
            return 0

    return None


def _find_pipelines(cache_root: str) -> List[str]:
    root = os.path.abspath(cache_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"cache_root not found: {root}")
    items = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted(items)


def _collect_patient_npzs(
    cache_root: str, pipeline: str
) -> Dict[Tuple[str, str], List[str]]:
    """Return mapping (fold, patient_id) -> list of npz paths."""

    base = os.path.join(os.path.abspath(cache_root), pipeline)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"pipeline dir not found: {base}")

    out: Dict[Tuple[str, str], List[str]] = {}

    for fold_name in sorted(os.listdir(base)):
        fold_dir = os.path.join(base, fold_name)
        if not os.path.isdir(fold_dir):
            continue
        if not fold_name.startswith("fold"):
            continue

        for patient_id in sorted(os.listdir(fold_dir)):
            patient_dir = os.path.join(fold_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            npzs = [
                os.path.join(patient_dir, f)
                for f in os.listdir(patient_dir)
                if f.endswith("_window_probs.npz")
            ]
            npzs = sorted(npzs)
            if not npzs:
                continue

            out[(fold_name, patient_id)] = npzs

    return out


def _confusion_counts(rows: Sequence[PatientDecision]) -> Dict[str, int]:
    tp = sum(1 for r in rows if r.y_true == 1 and r.y_pred == 1)
    tn = sum(1 for r in rows if r.y_true == 0 and r.y_pred == 0)
    fp = sum(1 for r in rows if r.y_true == 0 and r.y_pred == 1)
    fn = sum(1 for r in rows if r.y_true == 1 and r.y_pred == 0)
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def _confusion_indicators(*, y_true: int, y_pred: int) -> Dict[str, int]:
    yt = int(y_true)
    yp = int(y_pred)
    return {
        "tp": int(yt == 1 and yp == 1),
        "tn": int(yt == 0 and yp == 0),
        "fp": int(yt == 0 and yp == 1),
        "fn": int(yt == 1 and yp == 0),
    }


def _safe_div(n: float, d: float) -> Optional[float]:
    if d == 0:
        return None
    return float(n / d)


def _compute_metrics(counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp = float(counts["tp"])
    tn = float(counts["tn"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))

    bal_acc = None
    if recall is not None and specificity is not None:
        bal_acc = float(0.5 * (recall + specificity))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": bal_acc,
    }


def _compute_cv_summary(rows: Sequence[PatientDecision]) -> Dict[str, Any]:
    by_fold: Dict[str, List[PatientDecision]] = {}
    for r in rows:
        by_fold.setdefault(str(r.fold), []).append(r)

    per_fold: Dict[str, Any] = {}
    metric_names: List[str] = []

    for fold, fold_rows in sorted(by_fold.items(), key=lambda kv: kv[0]):
        counts = _confusion_counts(fold_rows)
        metrics = _compute_metrics(counts)
        per_fold[fold] = {
            "num_patients": int(len(fold_rows)),
            "confusion": counts,
            "metrics": metrics,
        }
        for k in metrics.keys():
            if k not in metric_names:
                metric_names.append(k)

    mean: Dict[str, Optional[float]] = {}
    std: Dict[str, Optional[float]] = {}

    for m in metric_names:
        vals: List[float] = []
        for fold in per_fold.keys():
            v = per_fold[fold]["metrics"].get(m)
            if v is None:
                continue
            vals.append(float(v))

        if not vals:
            mean[m] = None
            std[m] = None
            continue

        mean[m] = float(np.mean(vals))
        if len(vals) > 1:
            std[m] = float(np.std(vals, ddof=1))
        else:
            std[m] = 0.0

    return {"per_fold": per_fold, "metrics": {"mean": mean, "std": std}}


def _compute_roc_pr(
    *, y_true: Sequence[int], y_score: Sequence[float]
) -> Optional[Dict[str, Any]]:
    yt = np.asarray(list(y_true), dtype=np.int32)
    ys = np.asarray(list(y_score), dtype=np.float64)
    ok = np.isfinite(ys)
    yt = yt[ok]
    ys = ys[ok]

    if yt.size == 0:
        return None
    if len(np.unique(yt)) < 2:
        return None

    fpr, tpr, _thr = roc_curve(yt, ys)
    precision, recall, _thr2 = precision_recall_curve(yt, ys)

    out: Dict[str, Any] = {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auroc": float(roc_auc_score(yt, ys)),
        },
        "pr": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "auprc": float(auc(recall, precision)),
            "average_precision": float(average_precision_score(yt, ys)),
        },
        "n": int(yt.size),
        "n_pos": int((yt == 1).sum()),
        "n_neg": int((yt == 0).sum()),
    }
    return out


def _plot_roc(*, curves: Dict[str, Dict[str, Any]], title: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    for label, c in curves.items():
        roc = c.get("roc", {})
        fpr = roc.get("fpr")
        tpr = roc.get("tpr")
        auroc_v = roc.get("auroc")
        if fpr is None or tpr is None:
            continue
        if auroc_v is None:
            plt.plot(fpr, tpr, linewidth=2, label=label)
        else:
            plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUROC={auroc_v:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_pr(*, curves: Dict[str, Dict[str, Any]], title: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    for label, c in curves.items():
        pr = c.get("pr", {})
        precision = pr.get("precision")
        recall = pr.get("recall")
        auprc_v = pr.get("average_precision")
        if precision is None or recall is None:
            continue
        if auprc_v is None:
            plt.plot(recall, precision, linewidth=2, label=label)
        else:
            plt.plot(
                recall,
                precision,
                linewidth=2,
                label=f"{label} (AP={auprc_v:.3f})",
            )

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_pipeline(
    *,
    cache_root: str,
    pipeline: str,
    long_audio_root: str,
    t1: float,
    t2: float,
    tzsr: float,
    aggregation: str,
    skip_patients: Sequence[str],
    compute_curves: bool,
) -> Tuple[List[PatientDecision], Dict[str, Any]]:
    mapping = _collect_patient_npzs(cache_root, pipeline)

    patients: List[PatientDecision] = []

    for (fold_name, patient_id), npz_paths in sorted(mapping.items()):
        if patient_id in set(skip_patients):
            continue

        y_true = _infer_true_label_from_long_root(long_audio_root, patient_id)
        if y_true is None:
            # Fallback to cache field, if present.
            try:
                first = _load_npz(npz_paths[0])
                gt = _as_str(first.get("ground_truth", ""))
                if gt.lower().startswith("zenker"):
                    y_true = 1
                elif gt.lower().startswith("healthy"):
                    y_true = 0
            except Exception:
                y_true = None

        if y_true is None:
            # Cannot evaluate this patient
            continue

        track_decisions: List[TrackDecision] = []
        zsr_track: List[float] = []
        sum_swallow = 0
        sum_zenker = 0

        for npz_path in npz_paths:
            d = _load_npz(npz_path)
            p_sw = d.get("p_swallow")
            p_ze = d.get("p_zenker")
            if p_sw is None or p_ze is None:
                continue

            n_swallow, n_zenker, zsr = _compute_track_zsr(
                p_swallow=p_sw, p_zenker=p_ze, t1=t1, t2=t2
            )

            sum_swallow += int(n_swallow)
            sum_zenker += int(n_zenker)

            recording_id = _as_str(d.get("recording_id", os.path.basename(npz_path)))
            track_decisions.append(
                TrackDecision(
                    path=npz_path,
                    recording_id=recording_id,
                    n_windows=int(np.asarray(p_sw).shape[0]),
                    n_swallow=n_swallow,
                    n_zenker=n_zenker,
                    zsr=zsr,
                )
            )
            zsr_track.append(zsr)

        if not track_decisions:
            continue

        zsr_mean = float(np.mean(zsr_track)) if zsr_track else 0.0
        zsr_max = float(np.max(zsr_track)) if zsr_track else 0.0
        zsr_sum = float(sum_zenker / sum_swallow) if sum_swallow > 0 else 0.0
        if aggregation == "mean":
            zsr_patient = zsr_mean
        elif aggregation == "max":
            zsr_patient = zsr_max
        elif aggregation == "sum":
            zsr_patient = zsr_sum
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        y_pred = int(zsr_patient >= float(tzsr))

        patients.append(
            PatientDecision(
                pipeline=pipeline,
                aggregation=aggregation,
                fold=fold_name,
                patient_id=patient_id,
                y_true=int(y_true),
                y_pred=y_pred,
                zsr_track=zsr_track,
                zsr_mean=zsr_mean,
                zsr_max=zsr_max,
                zsr_sum=zsr_sum,
                zsr_patient=zsr_patient,
                tracks=track_decisions,
            )
        )

    counts = _confusion_counts(patients)
    metrics = _compute_metrics(counts)
    cv = _compute_cv_summary(patients)

    curve_summary: Optional[Dict[str, Any]] = None
    if compute_curves:
        y_true = [r.y_true for r in patients]
        y_score = [r.zsr_patient for r in patients]
        curves = _compute_roc_pr(y_true=y_true, y_score=y_score)
        if curves is not None:
            curve_summary = {
                "auroc": curves.get("roc", {}).get("auroc"),
                "auprc": curves.get("pr", {}).get("auprc"),
                "average_precision": curves.get("pr", {}).get("average_precision"),
                "n": curves.get("n"),
                "n_pos": curves.get("n_pos"),
                "n_neg": curves.get("n_neg"),
            }

    summary = {
        "pipeline": pipeline,
        "aggregation": aggregation,
        "thresholds": {"t1": float(t1), "t2": float(t2), "tzsr": float(tzsr)},
        "num_patients": len(patients),
        "confusion": counts,
        "metrics": metrics,
        "cv": cv,
        "curves": curve_summary,
    }

    return patients, summary


def _write_csv(path: str, rows: Sequence[PatientDecision]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pipeline",
                "aggregation",
                "fold",
                "patient_id",
                "y_true",
                "y_pred",
                "tp",
                "tn",
                "fp",
                "fn",
                "zsr_track_0",
                "zsr_track_1",
                "zsr_mean",
                "zsr_max",
                "zsr_sum",
                "zsr_patient",
                "track_count",
                "recording_ids",
            ]
        )
        for r in rows:
            z0 = r.zsr_track[0] if len(r.zsr_track) > 0 else None
            z1 = r.zsr_track[1] if len(r.zsr_track) > 1 else None
            rec_ids = ";".join(t.recording_id for t in r.tracks)
            cm = _confusion_indicators(y_true=r.y_true, y_pred=r.y_pred)
            w.writerow(
                [
                    r.pipeline,
                    r.aggregation,
                    r.fold,
                    r.patient_id,
                    r.y_true,
                    r.y_pred,
                    cm["tp"],
                    cm["tn"],
                    cm["fp"],
                    cm["fn"],
                    z0,
                    z1,
                    r.zsr_mean,
                    r.zsr_max,
                    r.zsr_sum,
                    r.zsr_patient,
                    len(r.tracks),
                    rec_ids,
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize patient-level ZSR classification from cached *_window_probs.npz files."
    )
    ap.add_argument(
        "--cache-root",
        default=os.path.join("caches", "ablation_study"),
        help="Root directory containing pipelines (default: caches/ablation_study)",
    )
    ap.add_argument(
        "--long-audio-root",
        required=True,
        help="Path to New_SwallowSet/Long (used to infer true label from Healthy/ vs Zenker/)",
    )
    ap.add_argument("--pipelines", nargs="*", default=None)

    ap.add_argument("--t1", type=float, default=0.5)
    ap.add_argument("--t2", type=float, default=0.5)
    ap.add_argument("--tzsr", type=float, default=0.5)

    ap.add_argument(
        "--aggregations",
        default="mean,max",
        help="Comma-separated patient-level aggregation(s) for zsr across tracks (default: mean,max)",
    )

    ap.add_argument(
        "--skip-patients",
        default="004",
        help="Comma-separated patient IDs to skip (default: 004)",
    )

    ap.add_argument(
        "--out-dir",
        default=os.path.join("analysis", "outputs", "patient_level_zsr"),
    )
    ap.add_argument(
        "--plot-curves",
        action="store_true",
        help="If set, write ROC and PR curve plots (AUROC/AUPRC) using zsr_patient as score.",
    )
    ap.add_argument(
        "--compute-curves",
        action="store_true",
        help="If set, compute AUROC/AUPRC/AP using zsr_patient as score and store them in summary JSON (no plots).",
    )
    args = ap.parse_args()

    pipelines = args.pipelines
    if not pipelines:
        pipelines = _find_pipelines(args.cache_root)

    skip_patients = [p.strip() for p in str(args.skip_patients).split(",") if p.strip()]
    aggregations = _parse_csv_strings(args.aggregations)
    unknown_aggs = [a for a in aggregations if a not in ("mean", "max", "sum")]
    if unknown_aggs:
        raise SystemExit(
            f"Unknown aggregation(s): {unknown_aggs}. Allowed: mean,max,sum"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = os.path.abspath(args.out_dir)
    os.makedirs(base_out_dir, exist_ok=True)

    run_dir = os.path.join(
        base_out_dir, _run_name(pipelines, args.t1, args.t2, args.tzsr)
    )
    os.makedirs(run_dir, exist_ok=True)

    overall = {
        "run_dir": run_dir,
        "created_at": ts,
        "pipelines": [],
        "thresholds": {"t1": args.t1, "t2": args.t2, "tzsr": args.tzsr},
    }

    comparison_curves: Dict[str, Dict[str, Dict[str, Any]]] = {
        agg: {} for agg in aggregations
    }

    compute_curves = bool(args.plot_curves or args.compute_curves)

    for pipeline in pipelines:
        pipe_dir = os.path.join(run_dir, pipeline)
        os.makedirs(pipe_dir, exist_ok=True)

        for agg in aggregations:
            rows, summary = summarize_pipeline(
                cache_root=args.cache_root,
                pipeline=pipeline,
                long_audio_root=args.long_audio_root,
                t1=args.t1,
                t2=args.t2,
                tzsr=args.tzsr,
                aggregation=agg,
                skip_patients=skip_patients,
                compute_curves=compute_curves,
            )

            overall["pipelines"].append(summary)

            with open(os.path.join(pipe_dir, f"summary_{agg}.json"), "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            if agg == "mean":
                with open(os.path.join(pipe_dir, "summary.json"), "w") as f:
                    json.dump(summary, f, indent=2, sort_keys=True)

            with open(os.path.join(pipe_dir, f"patients_{agg}.json"), "w") as f:
                json.dump(
                    {
                        "rows": [
                            {
                                "pipeline": r.pipeline,
                                "aggregation": r.aggregation,
                                "fold": r.fold,
                                "patient_id": r.patient_id,
                                "y_true": r.y_true,
                                "y_pred": r.y_pred,
                                **_confusion_indicators(
                                    y_true=r.y_true, y_pred=r.y_pred
                                ),
                                "zsr_track": r.zsr_track,
                                "zsr_mean": r.zsr_mean,
                                "zsr_max": r.zsr_max,
                                "zsr_sum": r.zsr_sum,
                                "zsr_patient": r.zsr_patient,
                                "tracks": [
                                    {
                                        "recording_id": t.recording_id,
                                        "path": t.path,
                                        "n_windows": t.n_windows,
                                        "n_swallow": t.n_swallow,
                                        "n_zenker": t.n_zenker,
                                        "zsr": t.zsr,
                                    }
                                    for t in r.tracks
                                ],
                            }
                            for r in rows
                        ]
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )

            if agg == "mean":
                with open(os.path.join(pipe_dir, "patients.json"), "w") as f:
                    json.dump(
                        {
                            "rows": [
                                {
                                    "pipeline": r.pipeline,
                                    "aggregation": r.aggregation,
                                    "fold": r.fold,
                                    "patient_id": r.patient_id,
                                    "y_true": r.y_true,
                                    "y_pred": r.y_pred,
                                    **_confusion_indicators(
                                        y_true=r.y_true, y_pred=r.y_pred
                                    ),
                                    "zsr_track": r.zsr_track,
                                    "zsr_mean": r.zsr_mean,
                                    "zsr_max": r.zsr_max,
                                    "zsr_sum": r.zsr_sum,
                                    "zsr_patient": r.zsr_patient,
                                    "tracks": [
                                        {
                                            "recording_id": t.recording_id,
                                            "path": t.path,
                                            "n_windows": t.n_windows,
                                            "n_swallow": t.n_swallow,
                                            "n_zenker": t.n_zenker,
                                            "zsr": t.zsr,
                                        }
                                        for t in r.tracks
                                    ],
                                }
                                for r in rows
                            ]
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )

            _write_csv(os.path.join(pipe_dir, f"patients_{agg}.csv"), rows)

            if agg == "mean":
                _write_csv(os.path.join(pipe_dir, "patients.csv"), rows)

            if args.plot_curves:
                y_true = [r.y_true for r in rows]
                y_score = [r.zsr_patient for r in rows]
                curves = _compute_roc_pr(y_true=y_true, y_score=y_score)
                if curves is not None:
                    with open(os.path.join(pipe_dir, f"curves_{agg}.json"), "w") as f:
                        json.dump(curves, f, indent=2, sort_keys=True)

                    _plot_roc(
                        curves={pipeline: curves},
                        title=f"ROC ({pipeline}, agg={agg})",
                        out_path=os.path.join(pipe_dir, f"roc_{agg}.png"),
                    )
                    _plot_pr(
                        curves={pipeline: curves},
                        title=f"Precision-Recall ({pipeline}, agg={agg})",
                        out_path=os.path.join(pipe_dir, f"pr_{agg}.png"),
                    )

                    comparison_curves[agg][pipeline] = curves

    if args.plot_curves:
        curves_dir = os.path.join(run_dir, "curves")
        os.makedirs(curves_dir, exist_ok=True)
        for agg in aggregations:
            if not comparison_curves.get(agg):
                continue
            _plot_roc(
                curves=comparison_curves[agg],
                title=f"ROC comparison (agg={agg})",
                out_path=os.path.join(curves_dir, f"roc_compare_{agg}.png"),
            )
            _plot_pr(
                curves=comparison_curves[agg],
                title=f"Precision-Recall comparison (agg={agg})",
                out_path=os.path.join(curves_dir, f"pr_compare_{agg}.png"),
            )
            with open(os.path.join(curves_dir, f"curves_compare_{agg}.json"), "w") as f:
                json.dump(comparison_curves[agg], f, indent=2, sort_keys=True)

    overview_path = os.path.join(run_dir, "overview.json")
    with open(overview_path, "w") as f:
        json.dump(overall, f, indent=2, sort_keys=True)

    print(f"Wrote outputs to: {run_dir}")
    if args.plot_curves:
        print(f"Wrote curve plots to: {os.path.join(run_dir, 'curves')}")


if __name__ == "__main__":
    main()
