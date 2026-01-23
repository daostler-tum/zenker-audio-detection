#!/usr/bin/env python3
"""
Confounding analysis utilities for the Zenker diverticulum (ZD) screening paper.

What this script does
- Reads a patient-level CSV containing at least: Age, gt (ground truth), ratio (ZSR).
- Computes patient-level predictions from a ratio threshold (default 0.5).
- Produces:
  (1) Scatter plot: Age vs ratio with the decision threshold line.
  (2) Stratified confusion matrices (markdown) and a metrics CSV containing tp/tn/fp/fn.
      - Overall
      - Stratified by an age cut (default 60 years)
      - Optionally by custom age bins

Example
    python analysis_confounds.py \
        --csv Zenker_ID_Age_Analysis.csv \
        --threshold 0.5 \
        --age-cut 60 \
        --outdir outputs \
        --age-bins 0 40 50 60 70 120

Dependencies
    pandas, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score


POS_LABEL = "Zenker"
NEG_LABEL = "Healthy"


def _normalize_label(x: str) -> str:
    """Normalize string labels to canonical {Zenker, Healthy}."""
    if pd.isna(x):
        return str(x)
    s = str(x).strip().lower()
    if s in {"zenker", "zd", "diverticulum", "diverticule"}:
        return POS_LABEL
    if s in {"healthy", "control", "normal"}:
        return NEG_LABEL
    return str(x).strip()


def compute_predictions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add canonicalized gt and a derived prediction column based on ratio >= threshold."""
    out = df.copy()
    required = {"gt", "ratio", "Age"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    out["gt_norm"] = out["gt"].apply(_normalize_label)
    out["pred_from_ratio"] = np.where(
        out["ratio"].astype(float) >= threshold, POS_LABEL, NEG_LABEL
    )

    bad = set(out["gt_norm"].unique()) - {POS_LABEL, NEG_LABEL}
    if bad:
        raise ValueError(
            f"Unexpected labels in gt after normalization: {bad}. "
            f"Please update _normalize_label()."
        )
    return out


@dataclass
class Metrics:
    n: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    sensitivity: float  # recall for Zenker
    specificity: float
    precision: float
    f1: float
    auroc_ratio: Optional[float]
    auroc_age: Optional[float]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else float("nan")


def compute_metrics(df: pd.DataFrame) -> Metrics:
    """Compute confusion matrix + metrics for gt_norm vs pred_from_ratio."""
    y_true = (df["gt_norm"] == POS_LABEL).astype(int).to_numpy()
    y_pred = (df["pred_from_ratio"] == POS_LABEL).astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = int(tp + tn + fp + fn)

    acc = _safe_div(tp + tn, n)
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    prec = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * prec * sens, prec + sens)

    auroc_ratio = None
    auroc_age = None
    if len(np.unique(y_true)) == 2:
        auroc_ratio = float(roc_auc_score(y_true, df["ratio"].astype(float)))
        auroc_age = float(roc_auc_score(y_true, df["Age"].astype(float)))

    return Metrics(
        n=n,
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        accuracy=float(acc),
        sensitivity=float(sens),
        specificity=float(spec),
        precision=float(prec),
        f1=float(f1),
        auroc_ratio=auroc_ratio,
        auroc_age=auroc_age,
    )


def metrics_to_frame(name: str, m: Metrics) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group": name,
                "n": m.n,
                "tp": m.tp,
                "tn": m.tn,
                "fp": m.fp,
                "fn": m.fn,
                "accuracy": m.accuracy,
                "sensitivity": m.sensitivity,
                "specificity": m.specificity,
                "precision": m.precision,
                "f1": m.f1,
                "auroc_ratio": m.auroc_ratio,
                "auroc_age": m.auroc_age,
            }
        ]
    )


def confusion_matrix_markdown(df: pd.DataFrame, title: str) -> str:
    """Return a small markdown confusion matrix for a subset."""
    y_true = (df["gt_norm"] == POS_LABEL).astype(int).to_numpy()
    y_pred = (df["pred_from_ratio"] == POS_LABEL).astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    md = []
    md.append(f"### {title} (n={len(df)})")
    md.append("")
    md.append("|  | Pred Healthy | Pred Zenker |")
    md.append("|---|---:|---:|")
    md.append(f"| GT Healthy | {tn} | {fp} |")
    md.append(f"| GT Zenker | {fn} | {tp} |")
    md.append("")
    return "\n".join(md)


def scatter_age_ratio(
    df: pd.DataFrame, threshold: float, out_png: Path, out_pdf: Optional[Path] = None
) -> None:
    """Scatter plot of Age vs ratio with a horizontal threshold line."""
    df_pos = df[df["gt_norm"] == POS_LABEL]
    df_neg = df[df["gt_norm"] == NEG_LABEL]

    plt.figure(figsize=(7, 4.5))
    plt.scatter(df_neg["Age"], df_neg["ratio"], label=NEG_LABEL, marker="o")
    plt.scatter(df_pos["Age"], df_pos["ratio"], label=POS_LABEL, marker="^")
    plt.axhline(threshold, linestyle="--", linewidth=1)

    plt.xlabel("Age (years)")
    plt.ylabel("Zenker-window ratio (ZSR)")
    plt.title("Age vs ZSR (patient-level)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if out_pdf is not None:
        plt.savefig(out_pdf)
    plt.close()


def stratify_by_age_cut(df: pd.DataFrame, age_cut: float) -> Dict[str, pd.DataFrame]:
    return {
        f"Age ≤ {age_cut}": df[df["Age"].astype(float) <= age_cut],
        f"Age > {age_cut}": df[df["Age"].astype(float) > age_cut],
    }


def stratify_by_bins(
    df: pd.DataFrame, bins: Sequence[float]
) -> Dict[str, pd.DataFrame]:
    bins = list(bins)
    if len(bins) < 2:
        raise ValueError("Need at least two bin edges.")
    labels = [f"[{bins[i]}, {bins[i + 1]})" for i in range(len(bins) - 1)]
    binned = pd.cut(
        df["Age"].astype(float),
        bins=bins,
        right=False,
        labels=labels,
        include_lowest=True,
    )
    out: Dict[str, pd.DataFrame] = {}
    for lab in labels:
        out[str(lab)] = df[binned == lab]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str, help="Path to patient-level CSV.")
    ap.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Ratio threshold for Zenker classification.",
    )
    ap.add_argument(
        "--age-cut", default=60.0, type=float, help="Age cut for stratified reporting."
    )
    ap.add_argument(
        "--age-bins",
        nargs="*",
        type=float,
        default=None,
        help="Optional explicit age bin edges, e.g. --age-bins 0 40 50 60 70 120",
    )
    ap.add_argument("--outdir", default="outputs", type=str, help="Output directory.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = compute_predictions(df, threshold=args.threshold)

    # 1) Scatter plot
    scatter_age_ratio(
        df,
        threshold=args.threshold,
        out_png=outdir / "age_vs_zsr.png",
        out_pdf=outdir / "age_vs_zsr.pdf",
    )

    # 2) Metrics overall and stratified
    frames: List[pd.DataFrame] = []
    frames.append(metrics_to_frame("Overall", compute_metrics(df)))

    for name, sub in stratify_by_age_cut(df, age_cut=args.age_cut).items():
        if len(sub) == 0:
            continue
        frames.append(metrics_to_frame(name, compute_metrics(sub)))

    if args.age_bins is not None and len(args.age_bins) >= 2:
        for name, sub in stratify_by_bins(df, bins=args.age_bins).items():
            if len(sub) == 0:
                continue
            frames.append(metrics_to_frame(f"Bin {name}", compute_metrics(sub)))

    report = pd.concat(frames, ignore_index=True)
    report.to_csv(outdir / "confusion_metrics_stratified.csv", index=False)

    # Raw per-patient predictions for traceability
    cols = [
        c
        for c in [
            "patient_id",
            "Fold",
            "Age",
            "Gender",
            "gt",
            "gt_norm",
            "ratio",
            "pred_from_ratio",
        ]
        if c in df.columns
    ]
    df[cols].to_csv(outdir / "patient_level_predictions.csv", index=False)

    # Confusion matrices in markdown (human-readable)
    md_blocks: List[str] = []
    md_blocks.append("# Stratified confusion matrices\n")
    md_blocks.append(confusion_matrix_markdown(df, "Overall"))
    for name, sub in stratify_by_age_cut(df, age_cut=args.age_cut).items():
        if len(sub) == 0:
            continue
        md_blocks.append(confusion_matrix_markdown(sub, name))
    if args.age_bins is not None and len(args.age_bins) >= 2:
        for name, sub in stratify_by_bins(df, bins=args.age_bins).items():
            if len(sub) == 0:
                continue
            md_blocks.append(confusion_matrix_markdown(sub, f"Bin {name}"))

    (outdir / "confusion_matrices_stratified.md").write_text("\n\n".join(md_blocks))

    # Console summary
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(report.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
