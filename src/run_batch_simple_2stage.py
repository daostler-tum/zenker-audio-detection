#!/usr/bin/env python
"""Simple wrapper to run the existing two‑stage window inference script for all test IDs of a fold.

This keeps logic minimal:
  * Reads test IDs from data_ast_stage2/test_ids_fold<FOLD>.txt
  * Extracts the leaf patient ID after the last '/' (e.g. Healthy/224 -> 224)
  * Invokes test_long_audio_windows_2stage.py once per patient ID
  * Skips IDs already processed unless --force is set

Assumptions:
  * Exactly two long audio files for each patient are discoverable by the existing script.
  * The existing script resolves model roots from --fold.

Example:
  python run_batch_simple_2stage.py --fold 1 \
    --long-audio-root /path/to/your/long/audio/directory \
    --pattern "*.wav" --window-sec 1.0 --hop-sec 0.5 --plot

You can pass extra args to the underlying script via --extra, e.g.:
  --extra "--stage1-threshold 0.55 --flatten-json"

Outputs are whatever test_long_audio_windows_2stage.py produces (JSON + optional plots).
"""

import argparse
import os
import subprocess
import shlex
import json
from typing import List, Optional

SCRIPT_NAME = "test_long_audio_windows_2stage_cache.py"


def get_script_path():
    """Get the absolute path to the target script in the same directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, SCRIPT_NAME)


def get_default_ids_root():
    """Get the default path to data_ast_stage2 directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "data_ast_stage2")


def read_ids(ids_path: str) -> List[str]:
    patients = []
    with open(ids_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            leaf = line.split("/")[-1]
            patients.append(leaf)
    return patients


def load_threshold_config(config_path: str) -> Optional[dict]:
    """Load threshold configuration from JSON file."""
    if not config_path or not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def build_cmd(
    base_args: argparse.Namespace,
    patient_id: str,
    threshold_config: Optional[dict] = None,
) -> List[str]:
    cmd = [
        "python",
        get_script_path(),
        "--long-audio-root",
        base_args.long_audio_root,
        "--pattern",
        base_args.pattern,
        "--fold",
        str(base_args.fold),
        "--patient-id",
        patient_id,
        "--window-sec",
        str(base_args.window_sec),
        "--hop-sec",
        str(base_args.hop_sec),
    ]

    # Add custom model roots if provided
    if base_args.stage1_model_root:
        cmd.extend(["--stage1-model-root", base_args.stage1_model_root])
    if base_args.stage2_model_root:
        cmd.extend(["--stage2-model-root", base_args.stage2_model_root])

    # Add thresholds from config if available
    if threshold_config:
        # Check if per-fold thresholds are available
        folds = threshold_config.get("folds", {})
        fold_key = str(base_args.fold)  # JSON keys are strings
        if folds and fold_key in folds:
            # Use fold-specific thresholds
            fold_thresholds = folds[fold_key]
            if "stage1" in fold_thresholds:
                stage1_thr = fold_thresholds["stage1"]["threshold"]
                cmd.extend(["--stage1-threshold", str(stage1_thr)])
            if "stage2" in fold_thresholds:
                stage2_thr = fold_thresholds["stage2"]["threshold"]
                cmd.extend(["--stage2-threshold", str(stage2_thr)])
        else:
            # Fall back to single-threshold config format
            thresholds = threshold_config.get("thresholds", {})
            if "stage1" in thresholds:
                stage1_thr = thresholds["stage1"]["threshold"]
                cmd.extend(["--stage1-threshold", str(stage1_thr)])
            if "stage2" in thresholds:
                stage2_thr = thresholds["stage2"]["threshold"]
                cmd.extend(["--stage2-threshold", str(stage2_thr)])

    if base_args.stage1_forward_min_prob is not None:
        cmd.extend([
            "--stage1-forward-min-prob",
            str(base_args.stage1_forward_min_prob),
        ])

    if base_args.stage2_argmax:
        cmd.append("--stage2-argmax")

    # Handle output directory for JSON and plots
    if base_args.output_dir:
        # Generate output JSON path
        output_json = os.path.join(base_args.output_dir, f"{patient_id}_2stage.json")
        cmd.extend(["--output-json", output_json])
        # Set plot directory
        cmd.extend(["--plot-dir", base_args.output_dir])

    if base_args.plot:
        cmd.append("--plot")
    if base_args.extra:
        cmd.extend(shlex.split(base_args.extra))
    return cmd


def main():
    ap = argparse.ArgumentParser(
        description="Simple batch launcher for two-stage window inference."
    )
    ap.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold number to use (resolves model roots).",
    )
    ap.add_argument(
        "--ids-root",
        default=None,  # Will be set to default value after parsing
        help="Directory containing test_ids_fold<fold>.txt",
    )
    ap.add_argument(
        "--long-audio-root",
        required=True,
        help="Root directory containing patient subfolders.",
    )
    ap.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern for audio files passed to underlying script.",
    )
    ap.add_argument("--window-sec", type=float, default=1.0)
    ap.add_argument("--hop-sec", type=float, default=0.5)
    ap.add_argument(
        "--plot", action="store_true", help="Forward --plot to underlying script."
    )
    ap.add_argument(
        "--output-dir",
        help="Directory for output JSONs and plots (default: outputs/).",
    )
    ap.add_argument(
        "--threshold-config",
        help="Path to threshold config JSON (from extract_thresholds_to_config.py).",
    )
    ap.add_argument(
        "--stage1-model-root",
        help="Custom Stage 1 model root (default: runs/ast_classifier_stage1/fold{fold}/best).",
    )
    ap.add_argument(
        "--stage2-model-root",
        help="Custom Stage 2 model root (default: runs/ast_classifier_stage2/fold{fold}/best).",
    )
    ap.add_argument(
        "--stage1-forward-min-prob",
        type=float,
        help="Only forward Stage1 windows to Stage2 when p_swallow exceeds this value.",
    )
    ap.add_argument(
        "--stage2-argmax",
        action="store_true",
        help="Use argmax for Stage2 classification instead of threshold (passed to underlying script).",
    )
    ap.add_argument(
        "--extra", help="Additional raw arguments string appended to each invocation."
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if per-patient JSON already exists in output-dir.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing."
    )
    args = ap.parse_args()

    # Set default ids_root if not provided
    if args.ids_root is None:
        args.ids_root = get_default_ids_root()

    # Load threshold config if provided
    threshold_config = None
    if args.threshold_config:
        threshold_config = load_threshold_config(args.threshold_config)
        if threshold_config:
            print(f"Loaded threshold config from: {args.threshold_config}")

            # Check if per-fold config
            folds = threshold_config.get("folds", {})
            if folds:
                print(f"  Per-fold thresholds detected (using fold {args.fold})")
                # JSON keys are strings, so convert fold to string for lookup
                fold_key = str(args.fold)
                if fold_key in folds:
                    fold_thresholds = folds[fold_key]
                    if "stage1" in fold_thresholds:
                        print(
                            f"  Fold {args.fold} Stage 1: {fold_thresholds['stage1']['threshold']:.4f}"
                        )
                    if "stage2" in fold_thresholds:
                        print(
                            f"  Fold {args.fold} Stage 2: {fold_thresholds['stage2']['threshold']:.4f}"
                        )
                else:
                    print(f"  Warning: No thresholds for fold {args.fold} in config")
            else:
                # Single threshold config
                thresholds = threshold_config.get("thresholds", {})
                if "stage1" in thresholds:
                    print(
                        f"  Stage 1 threshold: {thresholds['stage1']['threshold']:.4f}"
                    )
                if "stage2" in thresholds:
                    print(
                        f"  Stage 2 threshold: {thresholds['stage2']['threshold']:.4f}"
                    )
        else:
            print(
                f"Warning: Could not load threshold config from {args.threshold_config}"
            )

    ids_path = os.path.join(args.ids_root, f"test_ids_fold{args.fold}.txt")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"IDs file not found: {ids_path}")

    patients = read_ids(ids_path)
    if not patients:
        print("No patient IDs found; exiting.")
        return
    print(f"Read {len(patients)} patient IDs from {ids_path}")

    # Setup output directory for per-patient JSONs and plots
    out_dir = args.output_dir or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    for pid in patients:
        expected_json = os.path.join(out_dir, f"{pid}_2stage.json")
        if os.path.exists(expected_json) and not args.force:
            print(f"[SKIP] {pid} (exists: {expected_json})")
            continue
        cmd = build_cmd(args, pid, threshold_config)
        print("[RUN]", " ".join(shlex.quote(c) for c in cmd))
        if args.dry_run:
            continue
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        # result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"[ERROR] patient {pid} return code {result.returncode}")
            print("STDOUT:\n" + result.stdout)
            print("STDERR:\n" + result.stderr)
        else:
            print(f"[DONE] {pid} OK")
    print("Batch complete.")


if __name__ == "__main__":
    main()
