#!/usr/bin/env python
"""Two-stage windowed inference with feature caching (Stage1: Idle vs Swallow, Stage2: Healthy vs Zenker).

Compared to ``test_long_audio_windows_2stage.py`` this variant optionally caches the
log-mel features produced by the AST feature extractor so repeated runs (or swapping
classifier heads) can skip the expensive preprocessing step.
"""

import argparse
import glob
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

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


# ----------------- Feature Caching -----------------


def get_fx_fingerprint(fx: ASTFeatureExtractor) -> str:
    serialized = json.dumps(fx.to_dict(), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_cache_path(
    cache_dir: str,
    audio_path: str,
    window_sec: float,
    hop_sec: float,
    sr: int,
    fx_fingerprint: str,
) -> str:
    audio_abs = os.path.abspath(audio_path)
    audio_stats = f"{os.path.getsize(audio_abs)}_{int(os.path.getmtime(audio_abs))}"
    key = f"{audio_abs}|{window_sec}|{hop_sec}|{sr}|{fx_fingerprint}|{audio_stats}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(audio_abs))[0]
    filename = f"{base}_{digest}.pt"
    return os.path.join(cache_dir, filename)


def build_base_metadata(
    audio_path: str,
    window_sec: float,
    hop_sec: float,
    num_windows: int,
    sr: int,
    fx_fingerprint: str,
) -> Dict[str, Any]:
    audio_abs = os.path.abspath(audio_path)
    return {
        "audio_path": audio_abs,
        "audio_size": os.path.getsize(audio_abs),
        "audio_mtime": int(os.path.getmtime(audio_abs)),
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "num_windows": num_windows,
        "sampling_rate": sr,
        "extractor_fingerprint": fx_fingerprint,
    }


def compute_features(
    fx: ASTFeatureExtractor, windows: List[np.ndarray], batch_size: int
) -> torch.Tensor:
    feature_name = fx.model_input_names[0]
    chunks: List[torch.Tensor] = []
    for batch in batch_iter(windows, batch_size):
        inputs = fx(batch, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        feats = inputs[feature_name]
        chunks.append(feats)
    if not chunks:
        raise RuntimeError("Feature extraction yielded no data; check window setup.")
    features = torch.cat(chunks, dim=0)
    return features.to(torch.float32).contiguous()


def load_or_compute_features(
    audio_path: str,
    windows: List[np.ndarray],
    fx: ASTFeatureExtractor,
    window_sec: float,
    hop_sec: float,
    batch_size: int,
    cache_dir: Optional[str],
    disable_cache: bool,
    refresh_cache: bool,
    stage_label: str,
) -> torch.Tensor:
    fx_fingerprint = get_fx_fingerprint(fx)
    base_meta = build_base_metadata(
        audio_path, window_sec, hop_sec, len(windows), SAMPLING_RATE, fx_fingerprint
    )

    if disable_cache or not cache_dir:
        print(f"[cache:{stage_label}] Computing features (cache disabled).")
        return compute_features(fx, windows, batch_size)

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = build_cache_path(
        cache_dir, audio_path, window_sec, hop_sec, SAMPLING_RATE, fx_fingerprint
    )

    if not refresh_cache and os.path.exists(cache_path):
        try:
            bundle = torch.load(cache_path, map_location="cpu")
            metadata = bundle.get("metadata", {})
            if all(metadata.get(k) == v for k, v in base_meta.items()):
                features = bundle["features"].to(torch.float32).contiguous()
                print(f"[cache:{stage_label}] Loaded {cache_path}")
                return features
            print(
                f"[cache:{stage_label}] Metadata mismatch for {cache_path}; recomputing."
            )
        except Exception as exc:  # pragma: no cover - best-effort warning
            print(f"[cache:{stage_label}] Failed to load {cache_path}: {exc}; recomputing.")

    features = compute_features(fx, windows, batch_size)
    full_meta = dict(base_meta)
    full_meta["feature_shape"] = list(features.shape)

    try:
        torch.save({"metadata": full_meta, "features": features.cpu()}, cache_path)
        print(f"[cache:{stage_label}] Saved {cache_path}")
    except Exception as exc:  # pragma: no cover - best-effort warning
        print(f"[cache:{stage_label}] Failed to save {cache_path}: {exc}")

    return features


# ----------------- Inference -----------------


def forward_probs_from_features(
    model: ASTForAudioClassification, features: torch.Tensor, batch_size: int
) -> np.ndarray:
    probs_all = []
    with torch.inference_mode():
        for start in range(0, features.size(0), batch_size):
            batch_feats = features[start : start + batch_size].to(DEVICE)
            logits = model(batch_feats).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
    return np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 0))


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
            except Exception:  # pragma: no cover - best effort gathering
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
    use_argmax: bool = False,
) -> Dict[str, Any]:
    stage1_preds = stage1_probs.argmax(axis=1)  # 0=Idle, 1=Swallow
    stage2_aligned = [None] * len(stage1_preds)
    for idx, probs in stage2_probs_or_none:
        stage2_aligned[idx] = probs
    idle_count = int((stage1_preds == 0).sum())
    swallow_count = int((stage1_preds == 1).sum())
    
    if use_argmax:
        # Use argmax for Stage2 classification
        healthy_count = int(
            sum(1 for p in stage2_aligned if p is not None and np.argmax(p) == 0)
        )
        zenker_count = int(
            sum(1 for p in stage2_aligned if p is not None and np.argmax(p) == 1)
        )
    else:
        # Use threshold for Stage2 classification
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
        description="Two-stage AST inference over two long audio files (windowed) with feature caching."
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
        "--stage1-forward-min-prob",
        type=float,
        default=None,
        help="Minimum p_swallow required to forward a window to Stage2 (defaults to no extra filter).",
    )
    ap.add_argument(
        "--stage2-threshold",
        type=float,
        default=0.5,
        help="Probability threshold (p_zenker) for Zenker classification. Use validation-derived optimal threshold.",
    )
    ap.add_argument(
        "--stage2-argmax",
        action="store_true",
        help="Use argmax for Stage2 classification instead of threshold (consistent with training evaluation).",
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
    ap.add_argument(
        "--feature-cache-dir",
        default=os.path.join(".cache", "ast_features"),
        help="Directory to store cached AST features (per audio file).",
    )
    ap.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable feature caching and always recompute features.",
    )
    ap.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force recomputation and overwrite any existing cached features.",
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

    same_feature_extractors = (
        fx_s1.__class__ is fx_s2.__class__ and fx_s1.to_dict() == fx_s2.to_dict()
    )
    if same_feature_extractors:
        print("[cache] Stage1 and Stage2 share the same feature extractor; caches will be reused.")

    if args.window_sec <= 0 or args.hop_sec <= 0:
        raise ValueError("window-sec and hop-sec must be > 0")
    if args.hop_sec > args.window_sec:
        print(
            "[WARN] hop-sec larger than window-sec; windows will be disjoint with gaps."
        )

    cache_dir = None
    if not args.disable_cache and args.feature_cache_dir:
        cache_dir = os.path.abspath(args.feature_cache_dir)

    per_file = {}
    plot_assets = []  # collect (audio, stage1_preds, stage2_aligned_classes, file_label, path)

    for idx, path in enumerate(files):
        audio = load_audio(path)
        windows = window_audio(audio, args.window_sec, args.hop_sec)
        print(f"File {idx}: {len(windows)} windows of {args.window_sec}s")

        # Stage1 features (cached if available)
        stage1_features_cpu = load_or_compute_features(
            path,
            windows,
            fx_s1,
            args.window_sec,
            args.hop_sec,
            args.batch_size,
            cache_dir,
            args.disable_cache,
            args.refresh_cache,
            stage_label="stage1",
        )

        # Stage1 inference
        s1_probs = forward_probs_from_features(
            model_s1, stage1_features_cpu, args.batch_size
        )  # shape (N,2)
        if s1_probs.ndim != 2 or s1_probs.shape[1] != 2:
            raise RuntimeError("Stage1 output shape unexpected; expected (N,2)")
        p_swallow = s1_probs[:, 1]
        s1_preds = s1_probs.argmax(axis=1)  # 0=Idle 1=Swallow
        s1_preds = np.where(
            (s1_preds == 1) & (p_swallow >= args.stage1_threshold), 1, 0
        )

        # Stage2 only on swallow windows
        swallow_indices = np.where(s1_preds == 1)[0]
        if args.stage1_forward_min_prob is not None and len(swallow_indices):
            keep_mask = p_swallow[swallow_indices] >= args.stage1_forward_min_prob
            filtered = len(swallow_indices) - int(keep_mask.sum())
            if filtered:
                print(
                    f"[stage1] Filtering out {filtered} swallow windows with p_swallow < {args.stage1_forward_min_prob:.2f}"
                )
            swallow_indices = swallow_indices[keep_mask]
        stage2_results: List[Tuple[int, np.ndarray]] = []
        stage2_feature_source: Optional[torch.Tensor] = (
            stage1_features_cpu if same_feature_extractors else None
        )

        if len(swallow_indices):
            if stage2_feature_source is None:
                stage2_feature_source = load_or_compute_features(
                    path,
                    windows,
                    fx_s2,
                    args.window_sec,
                    args.hop_sec,
                    args.batch_size,
                    cache_dir,
                    args.disable_cache,
                    args.refresh_cache,
                    stage_label="stage2",
                )

            index_tensor = torch.as_tensor(swallow_indices, dtype=torch.long)
            feats_swallow = stage2_feature_source.index_select(0, index_tensor)
            s2_probs = forward_probs_from_features(
                model_s2, feats_swallow, args.batch_size
            )
            if s2_probs.ndim != 2 or s2_probs.shape[1] != 2:
                raise RuntimeError("Stage2 output shape unexpected; expected (K,2)")
            for local_i, global_idx in enumerate(swallow_indices):
                stage2_results.append((int(global_idx), s2_probs[local_i]))

        # Build aligned stage2 class vector for plotting (-1 idle, 0 healthy, 1 zenker)
        stage2_aligned_classes = np.full(len(s1_preds), -1, dtype=int)
        for gidx, probs in stage2_results:
            if args.stage2_argmax:
                # Use argmax classification (consistent with training evaluation)
                pred_class = int(np.argmax(probs))  # 0=Healthy, 1=Zenker
                stage2_aligned_classes[gidx] = pred_class
            else:
                # Use threshold-based classification
                p_zenker = probs[1]
                if p_zenker >= args.stage2_threshold:
                    stage2_aligned_classes[gidx] = 1  # Zenker
                else:
                    stage2_aligned_classes[gidx] = 0  # Healthy

        summary = summarize_stage_outputs(
            s1_probs,
            stage2_results,
            stage1_label_order,
            stage2_label_order,
            args.stage2_threshold,
            args.stage2_argmax,
        )
        per_file[f"file_{idx}"] = {"path": path, **summary}
        plot_assets.append(
            (audio, s1_preds.copy(), stage2_aligned_classes.copy(), f"file_{idx}", path)
        )

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
            "stage1_forward_min_prob": args.stage1_forward_min_prob,
            "stage2_threshold": args.stage2_threshold,
            "stage2_argmax": args.stage2_argmax,
            "files": files,
            "feature_cache_dir": cache_dir,
            "disable_cache": args.disable_cache,
            "refresh_cache": args.refresh_cache,
        },
        "per_file": per_file,
        "aggregate": aggregate,
    }

    if not args.output_json and args.patient_id:
        auto_dir = "outputs"
        os.makedirs(auto_dir, exist_ok=True)
        args.output_json = os.path.join(auto_dir, f"{args.patient_id}_2stage_cached.json")

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved JSON: {args.output_json}")

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
                window_sec = args.window_sec
                hop_sec = args.hop_sec
                for w_idx, cls1 in enumerate(s1_preds_arr):
                    start_t = w_idx * hop_sec
                    end_t = start_t + window_sec
                    if end_t > duration:
                        end_t = duration
                    if cls1 == 0:
                        continue
                    cls2 = s2_classes_arr[w_idx]
                    if cls2 == 0:
                        color = "#a4e5a4"  # healthy
                    elif cls2 == 1:
                        color = "#f5a3a3"  # zenker
                    else:
                        color = "#ffd27f"  # swallow without stage2
                    ax.axvspan(start_t, end_t, color=color, alpha=0.35, linewidth=0)
                ax.set_xlim(0, duration)

                path_lower = path.lower()
                if "zenker" in path_lower:
                    ground_truth = "Zenker"
                elif "healthy" in path_lower:
                    ground_truth = "Healthy"
                else:
                    ground_truth = "Unknown"

                num_healthy = int(np.sum(s2_classes_arr == 0))
                num_zenker = int(np.sum(s2_classes_arr == 1))
                num_swallow = int(np.sum(s1_preds_arr == 1))

                if num_swallow > 0:
                    zenker_ratio = num_zenker / num_swallow
                    ratio_str = f", Ratio Z/Sw: {zenker_ratio:.2f}"
                else:
                    ratio_str = ", Ratio: N/A" if num_zenker > 0 else ""

                ax.set_title(
                    f"{label}: {os.path.basename(path)} [GT: {ground_truth}] | Detected: {num_healthy} Healthy, {num_zenker} Zenker{ratio_str}"
                )

            axes[-1].set_xlabel("Time (s)")

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
            ]
            axes[0].legend(handles=legend_elems, loc="upper right")
            base_id = args.patient_id if args.patient_id else "pair"
            plot_path_png = os.path.join(args.plot_dir, f"{base_id}_2stage_plot_cached.png")
            plot_path_pdf = os.path.join(args.plot_dir, f"{base_id}_2stage_plot_cached.pdf")
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
