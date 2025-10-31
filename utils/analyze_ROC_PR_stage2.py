import argparse
import json
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification

import matplotlib.pyplot as plt

SAMPLING_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ROC/PR curves for Stage 2 AST classifier on held-out data."
    )
    parser.add_argument(
        "--data-dir", default="data_ast_stage2", help="Directory with fold npy files."
    )
    parser.add_argument(
        "--model-root-template",
        default="runs/ast_classifier_stage2/fold{fold}/best",
        help="Format string resolving the model directory per fold.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Held-out split to evaluate (falls back to test if val missing).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Specific fold to analyze (1-based). If omitted, loop over all.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Total number of cross-validation folds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction/model forward.",
    )
    parser.add_argument(
        "--decision-thresholds",
        type=float,
        nargs="*",
        default=[0.5],
        help="Optional list of probability thresholds to report confusion metrics for.",
    )
    parser.add_argument(
        "--output-json", help="Optional path to save raw scores and curve points."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Save ROC/PR plots (requires matplotlib)."
    )
    parser.add_argument(
        "--plot-dir", default="analysis_plots", help="Output directory for plots."
    )
    parser.add_argument(
        "--plot-combined",
        action="store_true",
        help="Plot all folds together in one ROC and one PR plot.",
    )
    parser.add_argument(
        "--plot-individual", action="store_true", help="Plot each fold separately."
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        help="DPI for saved plots (default: 150, use 300-600 for publication).",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf", "both"],
        default="png",
        help="Output format for plots (default: png).",
    )
    return parser.parse_args()


def save_figure(fig, base_path: str, dpi: int, fmt: str):
    """Save figure in requested format(s)."""
    import os

    base_name = os.path.splitext(base_path)[0]  # Remove extension

    if fmt in ["png", "both"]:
        png_path = f"{base_name}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved plot to {png_path}")

    if fmt in ["pdf", "both"]:
        pdf_path = f"{base_name}.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"Saved plot to {pdf_path}")


def load_split(
    data_dir: str, fold: int, preferred_split: str
) -> Tuple[List, List, str]:
    candidates = [preferred_split, "test"] if preferred_split == "val" else ["test"]
    for split in candidates:
        x_path = os.path.join(data_dir, f"{split}_x_fold{fold}.npy")
        y_path = os.path.join(data_dir, f"{split}_y_fold{fold}.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            X = np.load(x_path, allow_pickle=True).tolist()
            y = np.load(y_path).astype(int).tolist()
            return X, y, split
    raise FileNotFoundError(
        f"No {preferred_split} or test split found for fold {fold} in {data_dir}."
    )


def to_waveform(entry) -> np.ndarray:
    if isinstance(entry, np.ndarray):
        return entry.astype(np.float32)
    if isinstance(entry, dict):
        arr = entry.get("array") or entry.get("audio") or entry.get("values")
        if arr is None:
            raise ValueError("Unsupported dict payload for audio sample.")
        arr = np.asarray(arr, dtype=np.float32)
        sr = (
            entry.get("sampling_rate") or entry.get("sampling_rate_hz") or SAMPLING_RATE
        )
        if sr != SAMPLING_RATE:
            arr = torchaudio.functional.resample(
                torch.from_numpy(arr), sr, SAMPLING_RATE
            ).numpy()
        return arr
    if isinstance(entry, str):
        wav, sr = torchaudio.load(entry)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLING_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
        return wav.squeeze(0).numpy()
    raise TypeError(f"Unsupported audio payload type: {type(entry)}")


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def run_inference(
    model_dir: str,
    X: List,
    batch_size: int,
) -> np.ndarray:
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_dir)
    config = ASTConfig.from_pretrained(model_dir)
    model = ASTForAudioClassification.from_pretrained(model_dir, config=config).to(
        DEVICE
    )
    model.eval()

    scores = []
    with torch.inference_mode():
        for batch_entries in batched(X, batch_size):
            wavs = [to_waveform(entry) for entry in batch_entries]
            inputs = feature_extractor(
                wavs,
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt",
                padding=True,
            )
            feats = inputs[feature_extractor.model_input_names[0]].to(DEVICE)
            logits = model(feats).logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # probability of Zenker
            scores.append(probs.cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)


def bootstrap_ci(y_true, y_scores, metric_func, n_bootstrap=2000, seed=42):
    """Calculate bootstrap 95% CI for a given metric."""
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    np.random.seed(seed)
    n_samples = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            score = metric_func(y_true_boot, y_scores_boot)
            scores.append(score)
        except Exception:
            continue

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return lower, upper


def evaluate_fold(
    fold: int, args: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray, dict]:
    X, y_true, used_split = load_split(args.data_dir, fold, args.split)
    model_dir = args.model_root_template.format(fold=fold)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Fold {fold}: model directory missing at {model_dir}")
    y_scores = run_inference(model_dir, X, args.batch_size)

    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # Calculate 95% CIs
    roc_ci = bootstrap_ci(y_true, y_scores, roc_auc_score, n_bootstrap=2000, seed=42)
    pr_ci = bootstrap_ci(
        y_true, y_scores, average_precision_score, n_bootstrap=2000, seed=42
    )

    metrics = {
        "fold": fold,
        "split": used_split,
        "roc_auc": roc_auc,
        "roc_auc_ci_lower": roc_ci[0],
        "roc_auc_ci_upper": roc_ci[1],
        "pr_auc": pr_auc,
        "pr_auc_ci_lower": pr_ci[0],
        "pr_auc_ci_upper": pr_ci[1],
    }

    # Report optional operating points
    for thr in args.decision_thresholds:
        preds = (y_scores >= thr).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        report = classification_report(
            y_true,
            preds,
            target_names=["Healthy", "Zenker"],
            output_dict=True,
            zero_division=0,
        )
        metrics[f"thr_{thr}_confusion"] = cm.tolist()
        metrics[f"thr_{thr}_precision"] = report["Zenker"]["precision"]
        metrics[f"thr_{thr}_recall"] = report["Zenker"]["recall"]
        metrics[f"thr_{thr}_f1"] = report["Zenker"]["f1-score"]

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    metrics["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
    }
    metrics["pr_curve"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
    }

    # Best F1 from PR curve thresholds (exclude last precision point without threshold)
    if len(pr_thresholds):
        f1_scores = (2 * precision[:-1] * recall[:-1]) / np.clip(
            precision[:-1] + recall[:-1], 1e-8, None
        )
        best_idx = int(np.argmax(f1_scores))
        metrics["best_f1_threshold"] = float(pr_thresholds[best_idx])
        metrics["best_f1"] = float(f1_scores[best_idx])
        metrics["best_f1_precision"] = float(precision[best_idx])
        metrics["best_f1_recall"] = float(recall[best_idx])

    return np.asarray(y_true), np.asarray(y_scores), metrics


def main():
    args = parse_args()
    target_folds = [args.fold] if args.fold else list(range(1, args.num_folds + 1))

    all_true: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    reports = []

    for fold in target_folds:
        print(f"[Fold {fold}] evaluating...")
        y_true, y_scores, fold_metrics = evaluate_fold(fold, args)
        all_true.append(y_true)
        all_scores.append(y_scores)
        reports.append(fold_metrics)
        roc_ci_str = f"[{fold_metrics['roc_auc_ci_lower']:.3f}, {fold_metrics['roc_auc_ci_upper']:.3f}]"
        pr_ci_str = f"[{fold_metrics['pr_auc_ci_lower']:.3f}, {fold_metrics['pr_auc_ci_upper']:.3f}]"
        print(
            f"  ROC-AUC: {fold_metrics['roc_auc']:.4f} {roc_ci_str} | PR-AUC: {fold_metrics['pr_auc']:.4f} {pr_ci_str} | "
            f"Best-F1 Threshold: {fold_metrics.get('best_f1_threshold', float('nan')):.3f}"
        )

    # Aggregate across folds
    y_true_all = np.concatenate(all_true)
    y_scores_all = np.concatenate(all_scores)

    agg_roc_auc = roc_auc_score(y_true_all, y_scores_all)
    agg_pr_auc = average_precision_score(y_true_all, y_scores_all)

    # Calculate CIs for aggregate
    agg_roc_ci = bootstrap_ci(
        y_true_all, y_scores_all, roc_auc_score, n_bootstrap=2000, seed=42
    )
    agg_pr_ci = bootstrap_ci(
        y_true_all, y_scores_all, average_precision_score, n_bootstrap=2000, seed=42
    )

    agg = {
        "roc_auc": agg_roc_auc,
        "roc_auc_ci_lower": agg_roc_ci[0],
        "roc_auc_ci_upper": agg_roc_ci[1],
        "pr_auc": agg_pr_auc,
        "pr_auc_ci_lower": agg_pr_ci[0],
        "pr_auc_ci_upper": agg_pr_ci[1],
    }
    fpr, tpr, roc_thresholds = roc_curve(y_true_all, y_scores_all)
    precision, recall, pr_thresholds = precision_recall_curve(y_true_all, y_scores_all)
    agg["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
    }
    agg["pr_curve"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
    }

    if len(pr_thresholds):
        f1_scores = (2 * precision[:-1] * recall[:-1]) / np.clip(
            precision[:-1] + recall[:-1], 1e-8, None
        )
        best_idx = int(np.argmax(f1_scores))
        agg["best_f1_threshold"] = float(pr_thresholds[best_idx])
        agg["best_f1"] = float(f1_scores[best_idx])
        agg["best_f1_precision"] = float(precision[best_idx])
        agg["best_f1_recall"] = float(recall[best_idx])

    print("\n=== Aggregate ===")
    print(
        f"ROC-AUC: {agg['roc_auc']:.4f} [{agg['roc_auc_ci_lower']:.3f}, {agg['roc_auc_ci_upper']:.3f}]"
    )
    print(
        f"PR-AUC:  {agg['pr_auc']:.4f} [{agg['pr_auc_ci_lower']:.3f}, {agg['pr_auc_ci_upper']:.3f}]"
    )
    if "best_f1_threshold" in agg:
        print(
            f"Best F1 threshold: {agg['best_f1_threshold']:.3f} "
            f"(precision={agg['best_f1_precision']:.3f}, recall={agg['best_f1_recall']:.3f}, F1={agg['best_f1']:.3f})"
        )

    if args.output_json:
        payload = {
            "fold_reports": reports,
            "aggregate": agg,
            "decision_thresholds_evaluated": args.decision_thresholds,
        }
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved analysis JSON to {args.output_json}")
    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        # Default behavior: use combined if neither flag is set, or use both if --plot is set
        plot_combined = args.plot_combined or (
            not args.plot_individual and not args.plot_combined
        )
        plot_individual = args.plot_individual

        # Combined plots: all folds in one plot
        if plot_combined and len(reports) > 1:
            # ROC plot
            fig, ax = plt.subplots(figsize=(6, 5))
            for fold_metrics in reports:
                rc = fold_metrics["roc_curve"]
                ax.plot(
                    rc["fpr"],
                    rc["tpr"],
                    alpha=0.5,
                    label=f"Fold {fold_metrics['fold']} (AUC={fold_metrics['roc_auc']:.3f})",
                )
            agg_rc = agg["roc_curve"]
            ax.plot(
                agg_rc["fpr"],
                agg_rc["tpr"],
                color="black",
                linewidth=2,
                label=f"Aggregate (AUC={agg['roc_auc']:.3f})",
            )
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Stage 2 ROC Curves - All Folds")
            ax.legend(loc="lower right", fontsize="small")
            roc_path = os.path.join(args.plot_dir, "stage2_roc_combined.png")
            fig.tight_layout()
            save_figure(fig, roc_path, args.plot_dpi, args.plot_format)
            plt.close(fig)

            # PR plot
            fig, ax = plt.subplots(figsize=(6, 5))
            for fold_metrics in reports:
                pr = fold_metrics["pr_curve"]
                ax.plot(
                    pr["recall"],
                    pr["precision"],
                    alpha=0.5,
                    label=f"Fold {fold_metrics['fold']} (AP={fold_metrics['pr_auc']:.3f})",
                )
            agg_pr = agg["pr_curve"]
            ax.plot(
                agg_pr["recall"],
                agg_pr["precision"],
                color="black",
                linewidth=2,
                label=f"Aggregate (AP={agg['pr_auc']:.3f})",
            )
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_ylim(0.0, 1.05)
            ax.set_title("Stage 2 Precision-Recall Curves - All Folds")
            ax.legend(loc="lower right", fontsize="small")
            pr_path = os.path.join(args.plot_dir, "stage2_pr_combined.png")
            fig.tight_layout()
            save_figure(fig, pr_path, args.plot_dpi, args.plot_format)
            plt.close(fig)

        # Individual plots: one plot per fold
        if plot_individual:
            for fold_metrics in reports:
                fold = fold_metrics["fold"]
                # ROC plot for individual fold
                fig, ax = plt.subplots(figsize=(6, 5))
                rc = fold_metrics["roc_curve"]
                ax.plot(
                    rc["fpr"],
                    rc["tpr"],
                    linewidth=2,
                    label=f"Fold {fold} (AUC={fold_metrics['roc_auc']:.3f})",
                )
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"Stage 2 ROC Curve - Fold {fold}")
                ax.legend(loc="lower right")
                roc_path = os.path.join(args.plot_dir, f"stage2_roc_fold{fold}.png")
                fig.tight_layout()
                save_figure(fig, roc_path, args.plot_dpi, args.plot_format)
                plt.close(fig)

                # PR plot for individual fold
                fig, ax = plt.subplots(figsize=(6, 5))
                pr = fold_metrics["pr_curve"]
                ax.plot(
                    pr["recall"],
                    pr["precision"],
                    linewidth=2,
                    label=f"Fold {fold} (AP={fold_metrics['pr_auc']:.3f})",
                )
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_ylim(0.0, 1.05)
                ax.set_title(f"Stage 2 Precision-Recall Curve - Fold {fold}")
                ax.legend(loc="lower right")
                pr_path = os.path.join(args.plot_dir, f"stage2_pr_fold{fold}.png")
                fig.tight_layout()
                save_figure(fig, pr_path, args.plot_dpi, args.plot_format)
                plt.close(fig)


if __name__ == "__main__":
    main()
