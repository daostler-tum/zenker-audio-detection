#!/usr/bin/env python3
"""
Generate confusion matrices using seaborn with overlay technique.
(found on https://stackoverflow.com/questions/77849503/how-to-plot-accuracy-precision-and-recall-in-confusion-matrix-plot-using-seabor)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_confusion_matrix_overlay(cm, class_names, title, output_path):
    """
    Plot confusion matrix with precision/recall overlaid using masking technique.

    Args:
        cm: 2x2 confusion matrix [[TN, FP], [FN, TP]]
        class_names: List of class names [class0, class1]
        title: Plot title
        output_path: Where to save the plot
    """
    # Create labels for main confusion matrix (counts + percentages)
    total = np.sum(cm)
    labels = [[f"{val:0.0f}\n{val / total:.1%}" for val in row] for row in cm]

    # Create main heatmap
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=labels,
        cmap="Blues",
        fmt="",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        linewidths=1,
        linecolor="black",
        ax=ax,
        annot_kws={"fontsize": 20, "fontweight": "bold"},
    )

    ax.set_xlabel("Predicted Class", fontweight="bold", fontsize=16)
    ax.set_ylabel("True Class", fontweight="bold", fontsize=16)
    ax.set_title(title, fontweight="bold", fontsize=18, pad=20)
    ax.tick_params(labeltop=False, labelbottom=True, labelsize=14, length=0)

    # Create extended matrix for precision/recall overlay
    f_mat = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1))

    # Fill recall column (right side) - percentage of each true class correctly predicted
    f_mat[:-1, -1] = np.diag(cm) / np.sum(cm, axis=1)

    # Fill precision row (bottom) - percentage of each predicted class that is correct
    f_mat[-1, :-1] = np.diag(cm) / np.sum(cm, axis=0)

    # Overall accuracy (bottom-right corner)
    f_mat[-1, -1] = np.trace(cm) / np.sum(cm)

    # Create mask - only show last row and column
    f_mask = np.ones_like(f_mat)
    f_mask[:, -1] = 0  # Unmask last column (recall)
    f_mask[-1, :] = 0  # Unmask last row (precision)

    # Color matrix - different color for accuracy cell
    f_color = np.ones_like(f_mat)
    f_color[-1, -1] = 0  # Accuracy gets different color

    # Annotations for precision/recall
    f_annot = [[f"{val:0.1%}" for val in row] for row in f_mat]
    f_annot[-1][-1] = f"Acc:\n{f_mat[-1, -1]:0.1%}"

    # Overlay precision/recall on same axis
    sns.heatmap(
        f_color,
        mask=f_mask,
        annot=f_annot,
        fmt="",
        xticklabels=list(class_names) + ["Recall"],
        yticklabels=list(class_names) + ["Precision"],
        cmap=ListedColormap(["lightblue", "lightgrey"]),
        cbar=False,
        ax=ax,
        linewidths=2,
        linecolor="black",
        annot_kws={"fontsize": 16, "fontweight": "bold"},
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), facecolor="white")
    plt.close()

    print(f"Saved confusion matrix to {output_path}")
    print(f"Saved confusion matrix to {output_path.replace('.png', '.pdf')}")


def load_confusion_matrix_from_results(stage, model_root=None):
    """
    Load confusion matrix from aggregate classification results.

    Args:
        stage: 'stage1' or 'stage2'
        model_root: Root directory containing model results (default: runs)

    Returns:
        Confusion matrix as numpy array
    """
    if model_root is None:
        model_root = os.path.join(SCRIPT_DIR, "../runs")
    
    cm_path = os.path.join(
        model_root,
        f"ast_classifier_{stage}",
        "cv_aggregate_evaluation/confusion_matrix.npy",
    )

    if os.path.exists(cm_path):
        return np.load(cm_path)

    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices from model results"
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default=None,
        help="Root directory containing model results (default: runs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: MODEL_ROOT/results/)",
    )
    args = parser.parse_args()
    
    # Determine model root
    model_root = args.model_root
    if model_root is None:
        model_root = os.path.join(SCRIPT_DIR, "../runs")
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(model_root, "results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading confusion matrices from: {model_root}")
    print(f"Output directory: {output_dir}")

    # Try to load confusion matrices from files
    cm_stage1 = load_confusion_matrix_from_results("stage1", model_root)
    cm_stage2 = load_confusion_matrix_from_results("stage2", model_root)

    # Fallback to hardcoded values if files not found
    if cm_stage1 is None:
        print("Using hardcoded Stage 1 confusion matrix")
        cm_stage1 = np.array(
            [
                [2590, 229],  # True Idle
                [108, 1324],  # True Swallow
            ]
        )

    if cm_stage2 is None:
        print("Using hardcoded Stage 2 confusion matrix")
        cm_stage2 = np.array(
            [
                [473, 261],  # True Healthy
                [117, 581],  # True Zenker
            ]
        )

    plot_confusion_matrix_overlay(
        cm_stage1,
        class_names=["Idle", "Swallow"],
        title="Summed 5-Fold Confusion Matrix: Stage 1 (Idle vs. Swallow)",
        output_path=os.path.join(output_dir, "stage1_confusion_matrix_clean.png"),
    )

    plot_confusion_matrix_overlay(
        cm_stage2,
        class_names=["Healthy", "Zenker"],
        title="Summed 5-Fold Confusion Matrix: Stage 2 (Healthy vs. Zenker)",
        output_path=os.path.join(output_dir, "stage2_confusion_matrix_clean.png"),
    )

    print("\n" + "=" * 70)
    print("Confusion matrices generated")
    print("=" * 70)

    # Print statistics
    print("\nStage 1 (Idle vs. Swallow):")
    print(f"  Total samples: {cm_stage1.sum()}")
    print(
        f"  Accuracy: {(cm_stage1[0, 0] + cm_stage1[1, 1]) / cm_stage1.sum() * 100:.2f}%"
    )
    print(f"  Swallow Recall: {cm_stage1[1, 1] / cm_stage1[1, :].sum() * 100:.2f}%")
    print(f"  Swallow Precision: {cm_stage1[1, 1] / cm_stage1[:, 1].sum() * 100:.2f}%")

    print("\nStage 2 (Healthy vs. Zenker):")
    print(f"  Total samples: {cm_stage2.sum()}")
    print(
        f"  Accuracy: {(cm_stage2[0, 0] + cm_stage2[1, 1]) / cm_stage2.sum() * 100:.2f}%"
    )
    print(f"  Zenker Recall: {cm_stage2[1, 1] / cm_stage2[1, :].sum() * 100:.2f}%")
    print(f"  Zenker Precision: {cm_stage2[1, 1] / cm_stage2[:, 1].sum() * 100:.2f}%")


if __name__ == "__main__":
    main()
