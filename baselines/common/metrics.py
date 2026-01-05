from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0


def confusion_counts_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    counts = confusion_counts_binary(y_true, y_pred)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    bal_acc = 0.5 * (recall + specificity)

    out: Dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion": counts,
        "support": int(tp + tn + fp + fn),
    }

    if y_proba is not None:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            out["auroc"] = float(roc_auc_score(y_true, y_proba))
            out["auprc"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            out["auroc"] = None
            out["auprc"] = None

    return out


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    out: Dict[str, Any] = {"support": int(len(y_true))}

    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        out["recall_macro"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        out["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        out["confusion"] = confusion_matrix(y_true, y_pred).tolist()
    except Exception:
        out["accuracy"] = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
        out["balanced_accuracy"] = None
        out["precision_macro"] = None
        out["recall_macro"] = None
        out["f1_macro"] = None
        out["confusion"] = None

    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score

            out["auroc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except Exception:
            out["auroc_ovr_macro"] = None

    return out


def aggregate_by_patient(
    *,
    patient_ids: Sequence[str],
    y_true: Sequence[int],
    y_proba: Sequence[float],
    method: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Aggregate snippet predictions to patient-level."""

    if method not in ("mean_prob", "majority_vote"):
        raise ValueError(f"Unknown aggregation method: {method}")

    patient_ids = np.asarray(patient_ids)
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    patients = sorted(set(patient_ids.tolist()))

    yt: List[int] = []
    yp: List[int] = []
    pp: List[float] = []

    for pid in patients:
        idx = np.where(patient_ids == pid)[0]
        if len(idx) == 0:
            continue

        p_mean = float(np.mean(y_proba[idx]))
        if method == "mean_prob":
            p_pred = int(p_mean >= threshold)
        else:
            vote = np.mean((y_proba[idx] >= threshold).astype(float))
            p_pred = int(vote >= 0.5)

        vals, counts = np.unique(y_true[idx], return_counts=True)
        p_true = int(vals[np.argmax(counts)])

        yt.append(p_true)
        yp.append(p_pred)
        pp.append(p_mean)

    return np.asarray(yt), np.asarray(yp), np.asarray(pp), patients


def aggregate_by_patient_multiclass(
    *,
    patient_ids: Sequence[str],
    y_true: Sequence[int],
    y_proba: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if method not in ("mean_prob", "majority_vote"):
        raise ValueError(f"Unknown aggregation method: {method}")

    patient_ids = np.asarray(patient_ids)
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    patients = sorted(set(patient_ids.tolist()))

    yt: List[int] = []
    yp: List[int] = []
    pp: List[np.ndarray] = []

    for pid in patients:
        idx = np.where(patient_ids == pid)[0]
        if len(idx) == 0:
            continue

        mean_proba = np.mean(y_proba[idx], axis=0)
        if method == "mean_prob":
            pred = int(np.argmax(mean_proba))
        else:
            votes = np.argmax(y_proba[idx], axis=1)
            vals, counts = np.unique(votes, return_counts=True)
            pred = int(vals[np.argmax(counts)])

        vals, counts = np.unique(y_true[idx], return_counts=True)
        true = int(vals[np.argmax(counts)])

        yt.append(true)
        yp.append(pred)
        pp.append(mean_proba)

    if len(pp) == 0:
        proba_out = np.zeros((0, y_proba.shape[1]), dtype=float)
    else:
        proba_out = np.stack(pp, axis=0)

    return np.asarray(yt), np.asarray(yp), proba_out, patients


def find_best_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    candidates = np.unique(np.round(y_proba, 4))
    if len(candidates) == 0:
        return 0.5

    best_t = 0.5
    best_f1 = -1.0

    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        m = binary_metrics(y_true, y_pred)
        if m["f1"] > best_f1:
            best_f1 = float(m["f1"])
            best_t = float(t)

    return best_t


def classification_report_dict(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    target_names: Sequence[str],
) -> Optional[Dict[str, Any]]:
    try:
        from sklearn.metrics import classification_report

        return classification_report(
            y_true,
            y_pred,
            labels=list(range(len(target_names))),
            target_names=list(target_names),
            digits=4,
            output_dict=True,
            zero_division=0,
        )
    except Exception:
        return None


def save_confusion_matrix_png(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    out_path: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(title)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        return


def save_roc_pr_curves(
    *,
    y_true: Sequence[int],
    y_proba: Sequence[float],
    out_dir: str,
    prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from sklearn.metrics import precision_recall_curve, roc_curve

        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba).astype(float)

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        p, r, _ = precision_recall_curve(y_true, y_proba)

        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_roc.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(r, p)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_pr.png"), dpi=200)
        plt.close(fig)
    except Exception:
        return
