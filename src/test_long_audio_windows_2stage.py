#!/usr/bin/env python
"""Two-stage windowed inference (Stage1: Idle vs Swallow, Stage2: Healthy vs Zenker) on exactly two long audio files.

Process:
  1. Discover or accept two long audio file paths.
  2. Slice into 1.0s (configurable) windows, default 50% overlap via 0.5s hop.
  3. Run Stage1 model to classify each window as Idle (0) or Swallow (1).
  4. For windows classified Swallow, run Stage2 model to classify Healthy (0) vs Zenker (1).
  5. Aggregate and report per-file + overall statistics and (optionally) save JSON.

Assumptions:
  - Stage1 model directory has HF artifacts with label order [Idle, Swallow].
  - Stage2 model directory has HF artifacts with label order [Healthy, Zenker].
  - Models stored e.g. runs/ast_classifier_stage1/fold1/best , runs/ast_classifier_stage2/fold1/best

Example:
  python test_long_audio_windows_2stage.py \
      --stage1-model-root runs/ast_classifier_stage1/fold1/best \
      --stage2-model-root runs/ast_classifier_stage2/fold1/best \
      --patient-id 006 \
      --long-audio-root /data/Long/Zenker/006 \
      --output-json outputs/p006_2stage.json

Or explicit files:
  python test_long_audio_windows_2stage.py \
      --stage1-model-root runs/ast_classifier_stage1/fold1/best \
      --stage2-model-root runs/ast_classifier_stage2/fold1/best \
      --file-a A.wav --file-b B.wav
"""

import os
import argparse
import glob
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torchaudio
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification

try:
    import matplotlib.pyplot as plt  # Optional for plotting
except ImportError:  # pragma: no cover
    plt = None

SAMPLING_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Audio Helpers -----------------


def load_audio(path: str, target_sr: int = SAMPLING_RATE) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()


def window_audio(
    audio: np.ndarray, window_sec: float, hop_sec: float, sr: int = SAMPLING_RATE
) -> List[np.ndarray]:
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    out = []
    for start in range(0, max(1, len(audio) - win + 1), hop):
        segment = audio[start : start + win]
        if len(segment) < win:
            pad = np.zeros(win, dtype=audio.dtype)
            pad[: len(segment)] = segment
            segment = pad
        out.append(segment)
    return out


def batch_iter(items: List[np.ndarray], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ----------------- Model Loading -----------------


def load_stage_model(
    model_root: str, label_order: List[str]
) -> Tuple[ASTFeatureExtractor, ASTForAudioClassification]:
    fx = ASTFeatureExtractor.from_pretrained(model_root)
    config = ASTConfig.from_pretrained(model_root)
    label2id = {lbl: i for i, lbl in enumerate(label_order)}
    config.label2id = label2id
    config.id2label = {v: k for k, v in label2id.items()}
    model = ASTForAudioClassification.from_pretrained(model_root, config=config).to(
        DEVICE
    )
    model.eval()
    return fx, model


# ----------------- Inference -----------------


def forward_probs(model, fx, windows: List[np.ndarray], batch_size: int) -> np.ndarray:
    probs_all = []
    with torch.inference_mode():
        for batch in batch_iter(windows, batch_size):
            inputs = fx(batch, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            feats = inputs[fx.model_input_names[0]].to(DEVICE)
            logits = model(feats).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
    return np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0,))


# ----------------- File Discovery -----------------


def discover_two_files(root: str, patient_id: str, pattern: str) -> List[str]:
    base = os.path.abspath(root)
    matches = []
    for dirpath, _, filenames in os.walk(base):
        if patient_id not in dirpath:
            continue
        for fn in filenames:
            if glob.fnmatch.fnmatch(fn, pattern):
                matches.append(os.path.join(dirpath, fn))
    matches = sorted(matches)
    if len(matches) > 2:
        lengths = []
        for p in matches:
            try:
                info = torchaudio.info(p)
                lengths.append((p, info.num_frames))
            except Exception:
                lengths.append((p, 0))
        matches = [p for p, _ in sorted(lengths, key=lambda x: x[1], reverse=True)[:2]]
    if len(matches) != 2:
        raise ValueError(
            f"Expected exactly 2 files for patient {patient_id}, found {len(matches)}: {matches}"
        )
    return matches


# ----------------- Aggregation -----------------


def summarize_stage_outputs(
    stage1_probs: np.ndarray,
    stage2_probs_or_none: List[Tuple[int, np.ndarray]],
    stage1_label_order: List[str],
    stage2_label_order: List[str],
    stage2_threshold: float = 0.5,
) -> Dict[str, Any]:
    # stage1_probs shape (N,2) ; stage2_probs_or_none list of (index_in_all_windows, probs[2]) for swallow windows
    stage1_preds = stage1_probs.argmax(axis=1)  # 0=Idle, 1=Swallow
    swallow_indices = np.where(stage1_preds == 1)[0]
    # Build aligned arrays for stage2 (None for idle)
    stage2_aligned = [None] * len(stage1_preds)
    for idx, probs in stage2_probs_or_none:
        stage2_aligned[idx] = probs
    # Counts
    idle_count = int((stage1_preds == 0).sum())
    swallow_count = int((stage1_preds == 1).sum())
    # Apply stage2 threshold: only call Zenker if p_zenker >= threshold
    healthy_count = int(
        sum(1 for p in stage2_aligned if p is not None and p[1] < stage2_threshold)
    )
    zenker_count = int(
        sum(1 for p in stage2_aligned if p is not None and p[1] >= stage2_threshold)
    )
    return {
        "num_windows": int(len(stage1_preds)),
        "stage1_idle_windows": idle_count,
        "stage1_swallow_windows": swallow_count,
        "stage1_swallow_ratio": (swallow_count / len(stage1_preds))
        if len(stage1_preds)
        else 0.0,
        "stage1_mean_probs": stage1_probs.mean(axis=0).tolist()
        if len(stage1_probs)
        else None,
        "stage2_mean_probs_over_swallow": np.mean(
            [p for p in stage2_aligned if p is not None], axis=0
        ).tolist()
        if swallow_count
        else None,
        "stage2_swallow_windows_evaluated": int(
            len([p for p in stage2_aligned if p is not None])
        ),
        "stage2_healthy_windows": healthy_count,
        "stage2_zenker_windows": zenker_count,
        "stage2_zenker_ratio_over_swallow": (zenker_count / swallow_count)
        if swallow_count
        else None,
    }


# ----------------- CLI -----------------


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Two-stage AST inference over two long audio files (windowed)."
    )
    ap.add_argument(
        "--stage1-model-root",
        help="Path to Stage1 model (Idle vs Swallow). If omitted and --fold given, auto: <project_root>/runs/ast_classifier_stage1/fold<FOLD>/best",
    )
    ap.add_argument(
        "--stage2-model-root",
        help="Path to Stage2 model (Healthy vs Zenker). If omitted and --fold given, auto: <project_root>/runs/ast_classifier_stage2/fold<FOLD>/best",
    )
    ap.add_argument("--fold", type=int, help="Fold number to auto-resolve model roots.")
    ap.add_argument("--file-a", help="Explicit path to first audio file.")
    ap.add_argument("--file-b", help="Explicit path to second audio file.")
    ap.add_argument("--patient-id", help="Patient/specimen identifier for discovery.")
    ap.add_argument(
        "--long-audio-root", help="Root directory searched recursively for patient id."
    )
    ap.add_argument("--pattern", default="*.wav", help="Glob pattern for discovery.")
    ap.add_argument("--window-sec", type=float, default=1.0)
    ap.add_argument("--hop-sec", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--stage1-threshold",
        type=float,
        default=0.5,
        help="Probability threshold (p_swallow) to retain swallow classification.",
    )
    ap.add_argument(
        "--stage2-threshold",
        type=float,
        default=0.5,
        help="Probability threshold (p_zenker) for Zenker classification. Use validation-derived optimal threshold.",
    )
    ap.add_argument("--output-json", help="Optional output JSON path.")
    ap.add_argument("--show-first-n", type=int, default=5)
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Generate waveform overlay plot (requires matplotlib).",
    )
    ap.add_argument(
        "--plot-dir",
        default="outputs",
        help="Directory to save plot if --plot enabled.",
    )
    return ap


# ----------------- Main -----------------


def main():
    args = build_arg_parser().parse_args()

    if args.file_a and args.file_b:
        files = [args.file_a, args.file_b]
    else:
        if not (args.patient_id and args.long_audio_root):
            raise ValueError(
                "Provide either --file-a & --file-b or (--patient-id and --long-audio-root)."
            )
        files = discover_two_files(args.long_audio_root, args.patient_id, args.pattern)

    print(f"Using files:\n  A: {files[0]}\n  B: {files[1]}")

    # Resolve model roots if fold specified
    if args.fold is not None:
        if not args.stage1_model_root:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            args.stage1_model_root = os.path.join(project_root, "runs", "ast_classifier_stage1", f"fold{args.fold}", "best")
        if not args.stage2_model_root:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            args.stage2_model_root = os.path.join(project_root, "runs", "ast_classifier_stage2", f"fold{args.fold}", "best")

    if not (args.stage1_model_root and args.stage2_model_root):
        raise ValueError(
            "Model roots must be provided either explicitly or via --fold."
        )

    # Load models
    stage1_label_order = ["Idle", "Swallow"]
    stage2_label_order = ["Healthy", "Zenker"]
    fx_s1, model_s1 = load_stage_model(args.stage1_model_root, stage1_label_order)
    fx_s2, model_s2 = load_stage_model(args.stage2_model_root, stage2_label_order)

    # Verify window/hop sanity
    if args.window_sec <= 0 or args.hop_sec <= 0:
        raise ValueError("window-sec and hop-sec must be > 0")
    if args.hop_sec > args.window_sec:
        print(
            "[WARN] hop-sec larger than window-sec; windows will be disjoint with gaps."
        )

    per_file = {}
    plot_assets = []  # collect (audio, stage1_preds, stage2_aligned_classes, file_label)

    for idx, path in enumerate(files):
        audio = load_audio(path)
        windows = window_audio(audio, args.window_sec, args.hop_sec)
        print(f"File {idx}: {len(windows)} windows of {args.window_sec}s")

        # Stage1 inference
        s1_probs = forward_probs(
            model_s1, fx_s1, windows, args.batch_size
        )  # shape (N,2)
        if s1_probs.ndim != 2 or s1_probs.shape[1] != 2:
            raise RuntimeError("Stage1 output shape unexpected; expected (N,2)")
        p_swallow = s1_probs[:, 1]
        s1_preds = s1_probs.argmax(axis=1)  # 0=Idle 1=Swallow
        # Apply threshold: if argmax==Swallow but prob < threshold, revert to Idle
        s1_preds = np.where(
            (s1_preds == 1) & (p_swallow >= args.stage1_threshold), 1, 0
        )

        # Stage2 only on swallow windows
        swallow_indices = np.where(s1_preds == 1)[0]
        stage2_results: List[Tuple[int, np.ndarray]] = []
        if len(swallow_indices):
            swallow_windows = [windows[i] for i in swallow_indices]
            s2_probs = forward_probs(model_s2, fx_s2, swallow_windows, args.batch_size)
            if s2_probs.ndim != 2 or s2_probs.shape[1] != 2:
                raise RuntimeError("Stage2 output shape unexpected; expected (K,2)")
            for local_i, global_idx in enumerate(swallow_indices):
                stage2_results.append((int(global_idx), s2_probs[local_i]))

        # Build aligned stage2 class vector for plotting (-1 idle, 0 healthy, 1 zenker)
        # Apply stage2 threshold to probabilities
        stage2_aligned_classes = np.full(len(s1_preds), -1, dtype=int)
        for gidx, probs in stage2_results:
            # probs = [p_healthy, p_zenker]
            p_zenker = probs[1]
            # Apply threshold: only call Zenker if p_zenker >= threshold
            if p_zenker >= args.stage2_threshold:
                stage2_aligned_classes[gidx] = 1  # Zenker
            else:
                stage2_aligned_classes[gidx] = 0  # Healthy

        summary = summarize_stage_outputs(
            s1_probs, stage2_results, stage1_label_order, stage2_label_order, args.stage2_threshold
        )
        per_file[f"file_{idx}"] = {"path": path, **summary}
        plot_assets.append(
            (audio, s1_preds.copy(), stage2_aligned_classes.copy(), f"file_{idx}", path)
        )

        # Show a preview
        if args.show_first_n > 0 and len(windows):
            first_n = min(args.show_first_n, len(windows))
            print(f"First {first_n} stage1 preds: {s1_preds[:first_n].tolist()}")
            swallow_preview = [r for r in stage2_results if r[0] < first_n]
            if swallow_preview:
                print("Stage2 probs for swallow windows within preview:")
                for gidx, probs in swallow_preview:
                    print(f"  window {gidx}: probs_healthy_zenker={probs.tolist()}")

    # Aggregate across both files
    total_idle = sum(f["stage1_idle_windows"] for f in per_file.values())
    total_swallow = sum(f["stage1_swallow_windows"] for f in per_file.values())
    total_swallow_eval = sum(
        f["stage2_swallow_windows_evaluated"] for f in per_file.values()
    )
    total_healthy = sum(f["stage2_healthy_windows"] for f in per_file.values())
    total_zenker = sum(f["stage2_zenker_windows"] for f in per_file.values())
    aggregate = {
        "files_used": files,
        "total_windows": int(sum(f["num_windows"] for f in per_file.values())),
        "total_idle_windows": int(total_idle),
        "total_swallow_windows": int(total_swallow),
        "total_swallow_ratio": (
            total_swallow / max(1, sum(f["num_windows"] for f in per_file.values()))
        ),
        "total_swallow_windows_evaluated_stage2": int(total_swallow_eval),
        "total_healthy_windows": int(total_healthy),
        "total_zenker_windows": int(total_zenker),
        "overall_zenker_ratio_over_swallow": (total_zenker / total_swallow)
        if total_swallow
        else None,
    }

    output = {
        "config": {
            "stage1_model_root": args.stage1_model_root,
            "stage2_model_root": args.stage2_model_root,
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "batch_size": args.batch_size,
            "stage1_threshold": args.stage1_threshold,
            "files": files,
        },
        "per_file": per_file,
        "aggregate": aggregate,
    }

    # Auto-generate output path if not provided and patient-id given
    if not args.output_json and args.patient_id:
        auto_dir = "outputs"
        os.makedirs(auto_dir, exist_ok=True)
        args.output_json = os.path.join(auto_dir, f"{args.patient_id}_2stage.json")

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved JSON: {args.output_json}")

    # Plotting (combined figure with one subplot per file)
    if args.plot:
        if plt is None:
            print("[WARN] matplotlib not installed; cannot generate plot.")
        else:
            os.makedirs(args.plot_dir, exist_ok=True)
            
            # Set larger font sizes for better readability
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12
            })
            
            fig, axes = plt.subplots(
                len(plot_assets), 1, figsize=(14, 4 * len(plot_assets)), sharex=False
            )
            if len(plot_assets) == 1:
                axes = [axes]
            for ax, (audio_arr, s1_preds_arr, s2_classes_arr, label, path) in zip(
                axes, plot_assets
            ):
                duration = len(audio_arr) / SAMPLING_RATE
                time_axis = np.linspace(0, duration, len(audio_arr))
                ax.plot(time_axis, audio_arr, color="blue", linewidth=0.6)

                ax.set_ylabel("Amplitude")
                # Overlay windows coloring
                window_sec = args.window_sec
                hop_sec = args.hop_sec
                for w_idx, cls1 in enumerate(s1_preds_arr):
                    start_t = w_idx * hop_sec
                    end_t = start_t + window_sec
                    if end_t > duration:
                        end_t = duration
                    if cls1 == 0:
                        continue  # Idle -> no shading
                    # swallow window -> classify by stage2 if available
                    cls2 = s2_classes_arr[w_idx]
                    if cls2 == 0:
                        color = "#a4e5a4"  # healthy
                        zlabel = "Healthy"
                    elif cls2 == 1:
                        color = "#f5a3a3"  # zenker
                        zlabel = "Zenker"
                    else:
                        color = "#ffd27f"  # swallow w/out stage2 (unlikely)
                        zlabel = "Swallow"
                    ax.axvspan(start_t, end_t, color=color, alpha=0.35, linewidth=0)
                ax.set_xlim(0, duration)

                # Extract ground truth from path (check if 'healthy' or 'zenker' in path)
                path_lower = path.lower()
                if "zenker" in path_lower:
                    ground_truth = "Zenker"
                elif "healthy" in path_lower:
                    ground_truth = "Healthy"
                else:
                    ground_truth = "Unknown"

                # Count detections

                num_healthy = int(np.sum(s2_classes_arr == 0))
                num_zenker = int(np.sum(s2_classes_arr == 1))
                num_swallow = int(np.sum(s1_preds_arr == 1))

                # Calculate zenker/swallow ratio
                if num_swallow > 0:
                    zenker_ratio = num_zenker / num_swallow
                    ratio_str = f", Ratio Z/Sw: {zenker_ratio:.2f}"
                else:
                    ratio_str = ", Ratio: N/A" if num_zenker > 0 else ""

                ax.set_title(
                    f"{label}: {os.path.basename(path)} [GT: {ground_truth}] | Detected: {num_healthy} Healthy, {num_zenker} Zenker{ratio_str}"
                )

            axes[-1].set_xlabel("Time (s)")

            # Legend patches
            from matplotlib.patches import Patch

            legend_elems = [
                Patch(
                    facecolor="#a4e5a4",
                    edgecolor="none",
                    alpha=0.35,
                    label="Swallow→Healthy",
                ),
                Patch(
                    facecolor="#f5a3a3",
                    edgecolor="none",
                    alpha=0.35,
                    label="Swallow→Zenker",
                ),
                # Patch(facecolor='#ffd27f', edgecolor='none', alpha=0.35, label='Swallow (no stage2)'),
            ]
            axes[0].legend(handles=legend_elems, loc="upper right")
            # Save both PNG and PDF
            base_id = args.patient_id if args.patient_id else "pair"
            plot_path_png = os.path.join(args.plot_dir, f"{base_id}_2stage_plot.png")
            plot_path_pdf = os.path.join(args.plot_dir, f"{base_id}_2stage_plot.pdf")
            fig.tight_layout()
            fig.savefig(plot_path_png, dpi=150)
            fig.savefig(plot_path_pdf, format='pdf', bbox_inches='tight')
            print(f"Saved plot: {plot_path_png}")
            print(f"Saved PDF: {plot_path_pdf}")
            plt.close(fig)

    print("\n=== Aggregate (Two-Stage) Summary ===")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
