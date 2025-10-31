#!/usr/bin/env python
"""Aggregate per-patient two-stage inference JSON outputs into global metrics.

Scans an output directory for files named *_2stage.json (excluding batch summary files
like batch_fold*_2stage.json) that have the structure produced by run_batch_simple_2stage.py
or test_long_audio_windows_2stage.py (per-patient variant), e.g. 217_2stage.json.

Classification Logic (default threshold = 0.5):
  * Each JSON's ground-truth class is inferred from the first path in aggregate.files_used:
       contains '/Healthy/' -> Healthy
       contains '/Zenker/'  -> Zenker
       otherwise -> Unknown (ignored)
  * Predicted positive (Zenker) if overall_zenker_ratio_over_swallow >= threshold.
  * Predicted negative (Healthy) otherwise.
  * Confusion matrix mapping:
       GT Healthy & Pred Healthy => TN
       GT Healthy & Pred Zenker  => FP
       GT Zenker  & Pred Zenker  => TP
       GT Zenker  & Pred Healthy => FN
  * Files with missing ratio (None) or unknown GT are skipped (reported separately).

Outputs:
  * Prints confusion matrix + metrics to stdout.
  * Optionally writes a CSV of per-patient rows.
  * Optionally writes a JSON summary.

Usage:
  python aggregate_2stage_results.py \
      --outputs-dir outputs \
      --threshold 0.5 \
      --csv per_patient_results.csv \
      --json aggregate_summary.json

"""

from __future__ import annotations
import os
import json
import argparse
import glob
import csv
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class PatientResult:
    patient_id: str
    gt: str  # Healthy | Zenker | Unknown
    ratio: Optional[float]
    predicted_label: Optional[str]  # Healthy | Zenker | None
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    swallow_windows: Optional[int] = None
    zenker_windows: Optional[int] = None
    healthy_windows: Optional[int] = None
    total_windows: Optional[int] = None
    json_path: str = ""


def infer_ground_truth(files_used: List[str]) -> str:
    if not files_used:
        return "Unknown"
    first = files_used[0]
    lower = first.lower()
    if "/healthy/" in lower:
        return "Healthy"
    if "/zenker/" in lower:
        return "Zenker"
    return "Unknown"


def classify_result(
    gt: str, ratio: Optional[float], threshold: float
) -> (Optional[str], Dict[str, int]):
    if ratio is None or gt == "Unknown":
        return None, {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    pred = "Zenker" if ratio >= threshold else "Healthy"
    if gt == "Healthy" and pred == "Healthy":
        return pred, {"tp": 0, "tn": 1, "fp": 0, "fn": 0}
    if gt == "Healthy" and pred == "Zenker":
        return pred, {"tp": 0, "tn": 0, "fp": 1, "fn": 0}
    if gt == "Zenker" and pred == "Zenker":
        return pred, {"tp": 1, "tn": 0, "fp": 0, "fn": 0}
    if gt == "Zenker" and pred == "Healthy":
        return pred, {"tp": 0, "tn": 0, "fp": 0, "fn": 1}
    return pred, {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def parse_patient_id(filename: str) -> str:
    base = os.path.basename(filename)
    if base.endswith("_2stage.json"):
        return base[: -len("_2stage.json")]
    return os.path.splitext(base)[0]


def aggregate(args):
    pattern = os.path.join(args.outputs_dir, "*_2stage.json")
    files = sorted(glob.glob(pattern))
    patient_results: List[PatientResult] = []
    skipped_no_ratio = 0
    skipped_unknown_gt = 0

    for path in files:
        # Skip batch-level summary files
        bname = os.path.basename(path)
        if bname.startswith("batch_fold"):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            if args.verbose:
                print(f"[WARN] Failed to read {path}: {e}")
            continue
        # Expect structure with aggregate
        agg = data.get("aggregate", {})
        ratio = agg.get("overall_zenker_ratio_over_swallow")
        files_used = agg.get("files_used") or []
        gt = infer_ground_truth(files_used)
        patient_id = parse_patient_id(path)
        swallow_windows = agg.get("total_swallow_windows") or agg.get(
            "total_swallow_windows", None
        )
        zenker_windows = agg.get("total_zenker_windows")
        healthy_windows = agg.get("total_healthy_windows")
        total_windows = agg.get("total_windows")
        pred, cm = classify_result(gt, ratio, args.threshold)
        if ratio is None:
            skipped_no_ratio += 1
        if gt == "Unknown":
            skipped_unknown_gt += 1
        pr = PatientResult(
            patient_id=patient_id,
            gt=gt,
            ratio=ratio,
            predicted_label=pred,
            swallow_windows=swallow_windows,
            zenker_windows=zenker_windows,
            healthy_windows=healthy_windows,
            total_windows=total_windows,
            json_path=path,
            **cm,
        )
        patient_results.append(pr)

    # Compute confusion matrix totals
    tp = sum(r.tp for r in patient_results)
    tn = sum(r.tn for r in patient_results)
    fp = sum(r.fp for r in patient_results)
    fn = sum(r.fn for r in patient_results)
    evaluated = tp + tn + fp + fn
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else None
    )
    balanced_accuracy = (
        ((recall or 0.0) + (specificity or 0.0)) / 2
        if (recall is not None and specificity is not None)
        else None
    )

    summary = {
        "outputs_dir": args.outputs_dir,
        "threshold": args.threshold,
        "num_files_found": len(files),
        "num_patient_results": len(patient_results),
        "skipped_no_ratio": skipped_no_ratio,
        "skipped_unknown_gt": skipped_unknown_gt,
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
        },
    }

    # Print summary
    print(json.dumps(summary, indent=2))

    # Optionally write per-patient CSV
    if args.store_output:
        csv_path = os.path.join(args.outputs_dir, "per_patient_results.csv")
        json_path = os.path.join(args.outputs_dir, "aggregate_summary.json")

    if args.csv or args.store_output:
        fieldnames = (
            list(asdict(patient_results[0]).keys())
            if patient_results
            else [
                "patient_id",
                "gt",
                "ratio",
                "predicted_label",
                "tp",
                "tn",
                "fp",
                "fn",
                "swallow_windows",
                "zenker_windows",
                "healthy_windows",
                "total_windows",
                "json_path",
            ]
        )
        with open(args.csv or csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in patient_results:
                writer.writerow(asdict(r))
        if args.verbose:
            print(f"[INFO] Wrote per-patient CSV: {args.csv}")

    # Optionally write JSON summary
    if args.json or args.store_output:
        out_obj = {
            "summary": summary,
            "patients": [asdict(r) for r in patient_results],
        }
        with open(args.json or json_path, "w") as jf:
            json.dump(out_obj, jf, indent=2)
        if args.verbose:
            print(f"[INFO] Wrote aggregate JSON: {args.json}")


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Aggregate two-stage per-patient inference JSON outputs."
    )
    ap.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing *_2stage.json files.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Zenker ratio threshold for positive prediction.",
    )
    ap.add_argument("--csv", help="Optional CSV path for per-patient rows.")
    ap.add_argument(
        "--json", help="Optional JSON path for full summary + per-patient data."
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    ap.add_argument(
        "--store-output",
        action="store_true",
        help="Store json and csv with default name in output folder.",
    )
    return ap


def main():
    args = build_arg_parser().parse_args()
    aggregate(args)


if __name__ == "__main__":  # pragma: no cover
    main()
