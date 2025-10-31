#!/usr/bin/env python3
"""
Extract optimal thresholds PER FOLD from validation ROC/PR analysis.

This creates a fold-specific threshold config that ensures each fold's test set
uses only the threshold derived from that fold's validation set.

Usage:
    python extract_thresholds_per_fold.py \
        --stage2-metrics runs/results/ROC_PR_stage2/validation_metrics.json \
        --output-config optimal_thresholds_per_fold.json
"""

import json
import argparse
import os


def extract_per_fold_thresholds(metrics_path, stage_name):
    """Extract per-fold best F1 thresholds from ROC/PR metrics file."""
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found")
        return None

    with open(metrics_path, "r") as f:
        data = json.load(f)

    fold_reports = data.get("fold_reports", [])
    if not fold_reports:
        print(f"Warning: No fold_reports found in {metrics_path}")
        return None

    per_fold = {}
    for report in fold_reports:
        fold = report.get("fold")
        if fold is None:
            continue

        threshold = report.get("best_f1_threshold")
        f1 = report.get("best_f1")
        precision = report.get("best_f1_precision")
        recall = report.get("best_f1_recall")

        if threshold is None:
            print(f"Warning: No best_f1_threshold for fold {fold}")
            continue

        per_fold[fold] = {
            "threshold": float(threshold),
            "validation_f1": float(f1) if f1 is not None else None,
            "validation_precision": float(precision) if precision is not None else None,
            "validation_recall": float(recall) if recall is not None else None,
        }

    # Also extract aggregate for reference
    aggregate = data.get("aggregate", {})
    agg_threshold = aggregate.get("best_f1_threshold")
    if agg_threshold is not None:
        per_fold["aggregate"] = {
            "threshold": float(agg_threshold),
            "validation_f1": float(aggregate.get("best_f1", 0)),
            "validation_precision": float(aggregate.get("best_f1_precision", 0)),
            "validation_recall": float(aggregate.get("best_f1_recall", 0)),
            "note": "Aggregate across all folds (use fold-specific thresholds instead)",
        }

    return per_fold


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-fold optimal thresholds from validation ROC/PR metrics"
    )
    parser.add_argument(
        "--stage1-metrics",
        type=str,
        help="Path to Stage 1 validation ROC/PR metrics JSON",
    )
    parser.add_argument(
        "--stage2-metrics",
        type=str,
        required=True,
        help="Path to Stage 2 validation ROC/PR metrics JSON",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="optimal_thresholds_per_fold.json",
        help="Output config file path",
    )
    args = parser.parse_args()

    config = {
        "description": "Per-fold optimal thresholds from validation set ROC/PR analysis",
        "methodology": "Best F1 operating point from precision-recall curves, derived independently per fold",
        "note": "Each fold uses threshold from its own validation set only (no cross-fold contamination)",
        "folds": {},
    }

    # Extract Stage 2 thresholds (required)
    stage2_per_fold = extract_per_fold_thresholds(args.stage2_metrics, "Stage 2")
    if not stage2_per_fold:
        print(f"\nERROR: Could not extract Stage 2 thresholds from {args.stage2_metrics}")
        return

    # Extract Stage 1 thresholds if provided
    stage1_per_fold = None
    if args.stage1_metrics:
        stage1_per_fold = extract_per_fold_thresholds(args.stage1_metrics, "Stage 1")

    # Build per-fold config (use string keys for JSON compatibility)
    for fold in sorted([k for k in stage2_per_fold.keys() if isinstance(k, int)]):
        fold_key = str(fold)  # Explicitly use string keys for JSON
        config["folds"][fold_key] = {"stage2": stage2_per_fold[fold]}
        if stage1_per_fold and fold in stage1_per_fold:
            config["folds"][fold_key]["stage1"] = stage1_per_fold[fold]

    # Add aggregate for reference
    if "aggregate" in stage2_per_fold:
        config["aggregate_reference"] = {"stage2": stage2_per_fold["aggregate"]}
        if stage1_per_fold and "aggregate" in stage1_per_fold:
            config["aggregate_reference"]["stage1"] = stage1_per_fold["aggregate"]

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"PER-FOLD THRESHOLDS EXTRACTED")
    print(f"{'=' * 70}")

    for fold in sorted(config["folds"].keys()):
        fold_data = config["folds"][fold]
        print(f"\nFold {fold}:")
        if "stage1" in fold_data:
            print(
                f"  Stage 1: {fold_data['stage1']['threshold']:.4f} "
                f"(val F1={fold_data['stage1']['validation_f1']:.4f})"
            )
        print(
            f"  Stage 2: {fold_data['stage2']['threshold']:.4f} "
            f"(val F1={fold_data['stage2']['validation_f1']:.4f})"
        )

    if "aggregate_reference" in config:
        print(f"\nAggregate (for reference):")
        agg = config["aggregate_reference"]["stage2"]
        print(f"  Stage 2: {agg['threshold']:.4f} (val F1={agg['validation_f1']:.4f})")

    # Save config
    with open(args.output_config, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"CONFIG SAVED: {args.output_config}")
    print(f"{'=' * 70}")

    # Print usage
    print(f"\nUSAGE:")
    print(f"  python run_batch_simple_2stage.py \\")
    print(f"    --fold 1 \\")
    print(f"    --threshold-config {args.output_config} \\")
    print(f"    ... (other args)")
    print(f"\n  The script will automatically use fold-specific thresholds!")


if __name__ == "__main__":
    main()
