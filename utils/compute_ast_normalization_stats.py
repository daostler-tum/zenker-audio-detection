import os
import json
import argparse
import numpy as np
from transformers import ASTFeatureExtractor, set_seed
import torch
import librosa
import soundfile as sf

"""Compute normalization (mean/std) statistics for AST feature extractor across all CV folds.

Enhancements:
    - Batch processing of raw audio -> feature extraction (reduces Python overhead).
    - Optional --stage flag: stage1 -> data_ast_stage1, stage2 -> data_ast_stage2.
    - Fast accumulation using running sum and squared sum (exact, numerically stable with float64).

Assumptions:
    - Fold data prepared by PrepareTrainingData_AST.py or PrepareTrainingData_AST_2stage.py producing:
            train_x_fold{1..K}.npy (arrays of audio file paths)
    - Only TRAIN splits per fold are used.
    - do_normalize=False during extraction so we capture raw log-mel features.

Outputs (written to --output-dir):
    stats_per_fold.json : list of dicts with per-fold mean/std/count
    stats_aggregate.json: global weighted mean/std across folds
    stats_all.npz       : serialized numpy arrays (optional convenience)

NOTE: Because CV folds reuse specimens across folds, the aggregate reflects the average
            training distribution view rather than a deduplicated corpus. For a deduplicated
            corpus statistic, run once on a single chosen training split.
"""

NUM_FOLDS = 5
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLING_RATE = 16000
SEED = 42

set_seed(SEED)

def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load audio file, resampling if needed (mono)."""
    try:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)
    except Exception:
        # Fallback to librosa entirely if soundfile fails
        audio, _ = librosa.load(path, sr=target_sr, mono=True)
        return audio.astype(np.float32)


def compute_fold_stats(data_dir: str, fold: int, batch_size: int):
    train_x_path = os.path.join(data_dir, f"train_x_fold{fold}.npy")
    if not os.path.exists(train_x_path):
        raise FileNotFoundError(f"Missing fold {fold} train data. Expected {train_x_path}")

    train_x = np.load(train_x_path).tolist()
    if len(train_x) == 0:
        return {"fold": fold, "mean": 0.0, "std": 0.0, "count": 0}

    feature_extractor = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL)
    feature_extractor.do_normalize = False  # ensure raw features
    model_input_name = feature_extractor.model_input_names[0]

    total_count = 0  # total number of feature elements
    running_sum = 0.0
    running_sq_sum = 0.0

    # Batch iteration over file paths
    for start in range(0, len(train_x), batch_size):
        batch_paths = train_x[start:start + batch_size]
        wavs = [load_audio(p, SAMPLING_RATE) for p in batch_paths]
        feats = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")[model_input_name]  # (B, time, freq)
        flat = feats.view(feats.size(0), -1).to(torch.float64)  # keep precision for sums
        running_sum += flat.sum().item()
        running_sq_sum += (flat ** 2).sum().item()
        total_count += flat.numel()

    if total_count == 0:
        return {"fold": fold, "mean": 0.0, "std": 0.0, "count": 0}

    mean = running_sum / total_count
    # population variance first
    var_pop = running_sq_sum / total_count - mean * mean
    var_pop = max(var_pop, 0.0)
    # convert to unbiased sample variance if possible
    if total_count > 1:
        var = var_pop * (total_count / (total_count - 1))
    else:
        var = 0.0
    std = var ** 0.5
    return {"fold": fold, "mean": float(mean), "std": float(std), "count": total_count}


def aggregate_stats(per_fold):
    # Weighted mean
    total_count = sum(d["count"] for d in per_fold)
    if total_count == 0:
        return {"mean": 0.0, "std": 0.0, "total_count": 0}
    weighted_mean = sum(d["mean"] * d["count"] for d in per_fold) / total_count
    # Combine variances: Σ( (n_k -1)*s_k^2 + n_k*(μ_k - μ)^2 ) / (N - 1)
    numerator = 0.0
    for d in per_fold:
        n = d["count"]
        if n < 2:
            continue
        numerator += (n - 1) * (d["std"] ** 2) + n * (d["mean"] - weighted_mean) ** 2
    global_variance = numerator / (total_count - 1) if total_count > 1 else 0.0
    global_std = float(global_variance ** 0.5)
    return {"mean": float(weighted_mean), "std": global_std, "total_count": total_count}


def main():
    parser = argparse.ArgumentParser(description="Compute AST normalization stats across CV folds (batched)")
    parser.add_argument("--data-dir", default="data_ast_cv", help="Directory with train_x_fold*.npy files (overridden by --stage if provided)")
    parser.add_argument("--folds", type=int, default=NUM_FOLDS, help="Number of folds")
    parser.add_argument("--output-dir", default="data_ast_cv", help="Where to write stats files")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of audio files per feature extraction batch")
    parser.add_argument("--stage", choices=["stage1", "stage2"], help="Shortcut to use data_ast_stage1 or data_ast_stage2 directories")
    args = parser.parse_args()

    if args.stage:
        script_dir = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        mapped = os.path.join(project_root, f"data_ast_{args.stage}")
        print(f"[Info] Using stage alias '{args.stage}' -> data/output dir '{mapped}'")
        args.data_dir = mapped
        # Always store results in the canonical stage directory when --stage is used
        if args.output_dir != mapped:
            print(f"[Info] Overriding --output-dir to '{mapped}' for stage consistency")
        args.output_dir = mapped

    per_fold_stats = []
    for fold in range(1, args.folds + 1):
        print(f"Computing stats for fold {fold} (batch_size={args.batch_size}) ...")
        stats = compute_fold_stats(args.data_dir, fold, args.batch_size)
        print(f"  Fold {fold}: mean={stats['mean']:.6f} std={stats['std']:.6f} (count={stats['count']})")
        per_fold_stats.append(stats)

    aggregate = aggregate_stats(per_fold_stats)
    print("\nWeighted aggregate (training folds, with repetition):")
    print(f"  mean={aggregate['mean']:.6f} std={aggregate['std']:.6f} (total_count={aggregate['total_count']})")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "stats_per_fold.json"), "w") as f:
        json.dump(per_fold_stats, f, indent=2)
    with open(os.path.join(args.output_dir, "stats_aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    np.savez(os.path.join(args.output_dir, "stats_all.npz"), per_fold=per_fold_stats, aggregate=aggregate)

    print(f"\nSaved per-fold and aggregate stats to {args.output_dir}")
    print("Use aggregate mean/std in your training script if desired.")

if __name__ == "__main__":
    main()
