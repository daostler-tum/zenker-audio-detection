import os
import json
import shutil
from datetime import datetime

import numpy as np
import evaluate
from sklearn.metrics import confusion_matrix, classification_report
from datasets import Dataset, Audio, ClassLabel, Features
from transformers import (
    ASTFeatureExtractor,
    ASTConfig,
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from audiomentations import (
    Compose,
    AddGaussianSNR,
    GainTransition,
    Gain,
    ClippingDistortion,
    TimeStretch,
    PitchShift,
    TimeMask,
)
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False

"""Cross-validation training script for Stage 1 (Idle vs Swallow).
Data directory produced by utils/PrepareTrainingData_AST_cv_2stage.py:
    ../data_ast_stage1/train_x_fold{k}.npy, train_y_fold{k}.npy, test_x_fold{k}.npy, test_y_fold{k}.npy

Label mapping (binary):
    0 -> Idle
    1 -> Swallow (Healthy or Zenker)

Notes:
    - Binary metrics treat label '1' (Swallow) as the positive class by default.
    - Ensure upstream preparation used Idle=0, Swallow=1 (PrepareTrainingData_AST_cv_2stage.py).
"""


class FocalLossTrainer(Trainer):
    """Custom Trainer with Focal Loss support for handling class imbalance."""

    def __init__(self, *args, focal_gamma=0.0, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.label_smoothing_factor = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply focal loss if gamma > 0
        if self.focal_gamma > 0:
            ce_loss = F.cross_entropy(
                logits,
                labels,
                reduction="none",
                label_smoothing=self.label_smoothing_factor,
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
            loss = focal_loss
        else:
            # Standard cross-entropy with optional label smoothing
            loss = F.cross_entropy(
                logits, labels, label_smoothing=self.label_smoothing_factor
            )

        return (loss, outputs) if return_outputs else loss


NUM_FOLDS = 5

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(project_root, "data_ast_stage1")
OUTPUT_ROOT = os.path.join(project_root, "runs", "ast_classifier_stage1")
LOG_DIR = os.path.join(project_root, "logs", "ast_classifier", "stage1")
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLING_RATE = 16000
SEED = 42
WANDB_PROJECT = "zenker_detect_AST_2stage"
WANDB_GROUP = "stage1-cv"

NUM_EPOCHS = 10

# Default fallback normalization values (historical reference)
FEATURE_EXTRACTOR_MEAN = -1.1509622
FEATURE_EXTRACTOR_STD = 3.5340312


def build_run_config(
    *,
    args: Any,
    target_folds: List[int],
    dry_run: bool,
    enable_early_stopping: bool,
    use_wandb: bool,
    run_id: str,
    run_started_at: datetime,
) -> Dict[str, Any]:
    checkpoint_limit = 1 if dry_run else max(2, (NUM_EPOCHS + 1) // 4)
    return {
        "run_id": run_id,
        "timestamp": run_started_at.isoformat(),
        "script": "train_ast_stage1_cross_validation",
        "stage": "stage1",
        "pretrained_model": PRETRAINED_MODEL,
        "seed": SEED,
        "num_epochs": 1 if dry_run else NUM_EPOCHS,
        "per_device_train_batch_size": 16,
        "learning_rate": args.learning_rate,
        "optimizer": {
            "name": args.optim,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "adam_beta2": args.adam_beta2,
        },
        "loss": {
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
        },
        "dry_run": dry_run,
        "target_folds": target_folds,
        "fold_requested": args.fold,
        "early_stopping": {
            "enabled": enable_early_stopping,
            "patience": 2,
        },
        "checkpoint_limit": checkpoint_limit,
        "paths": {
            "data_dir": DATA_DIR,
            "output_root": OUTPUT_ROOT,
            "log_dir": LOG_DIR,
        },
        "wandb": {
            "enabled": use_wandb,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "per_fold": args.wandb_per_fold,
            "offline": args.wandb_offline,
        },
    }


def log_config_to_wandb(
    run,
    *,
    config_path: str,
    artifact_name: str,
    config_payload: Dict[str, Any],
):
    if run is None or not _WANDB_AVAILABLE:
        return
    try:
        run.config.update(config_payload, allow_val_change=True)
    except Exception as exc:
        print(
            f"[WARN] Failed to update W&B config for artifact '{artifact_name}': {exc}"
        )
    try:
        artifact = wandb.Artifact(artifact_name, type="run-config")
        artifact.add_file(config_path, name=os.path.basename(config_path))
        run.log_artifact(artifact)
    except Exception as exc:
        print(
            f"[WARN] Failed to log run-config artifact to W&B ('{artifact_name}'): {exc}"
        )


def backup_existing_run_dir(path: str) -> str | None:
    """Copy non-empty existing run directory to a timestamped backup."""

    if not os.path.isdir(path):
        return None

    try:
        has_contents = any(os.scandir(path))
    except FileNotFoundError:
        return None

    if not has_contents:
        return None

    try:
        stat_info = os.stat(path)
    except OSError:
        return None

    source_ts = getattr(stat_info, "st_mtime", None)
    if source_ts is None:
        source_ts = getattr(stat_info, "st_ctime", None)

    if source_ts is not None:
        timestamp = datetime.fromtimestamp(source_ts).strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.dirname(os.path.normpath(path))
    base_name = os.path.basename(os.path.normpath(path))
    backup_base = f"{base_name}_{timestamp}"
    backup_path = os.path.join(base_dir, backup_base)
    counter = 1

    while os.path.exists(backup_path):
        backup_path = os.path.join(base_dir, f"{backup_base}_{counter}")
        counter += 1

    print(f"[RunBackup] Existing run dir detected; copying '{path}' -> '{backup_path}'")
    try:
        shutil.copytree(path, backup_path)
    except Exception as exc:
        print(f"[RunBackup][WARN] Failed to copy existing run dir: {exc}")
        return None
    return backup_path


def load_fold_normalization(fold: int):
    """Load per-fold normalization stats if available, else aggregate, else defaults.

    Priority:
      1. stats_per_fold.json entry with matching fold number (key 'fold').
      2. stats_aggregate.json (single mean/std for all folds).
      3. Hardcoded FEATURE_EXTRACTOR_MEAN / FEATURE_EXTRACTOR_STD.
    """
    import json

    per_fold_path = os.path.join(DATA_DIR, "stats_per_fold.json")
    if os.path.exists(per_fold_path):
        try:
            with open(per_fold_path, "r") as f:
                entries = json.load(f)
            # entries may be list[dict]
            if isinstance(entries, list):
                for entry in entries:
                    if (
                        isinstance(entry, dict)
                        and entry.get("fold") == fold
                        and "mean" in entry
                        and "std" in entry
                    ):
                        print(
                            f"[Normalization] Using per-fold stats for fold {fold}: mean={entry['mean']:.6f} std={entry['std']:.6f}"
                        )
                        return float(entry["mean"]), float(entry["std"])
        except Exception as e:
            print(
                f"[Normalization] Failed reading per-fold stats ({e}); will try aggregate."
            )
    agg_path = os.path.join(DATA_DIR, "stats_aggregate.json")
    if os.path.exists(agg_path):
        try:
            with open(agg_path, "r") as f:
                agg = json.load(f)
            if "mean" in agg and "std" in agg:
                print(
                    f"[Normalization] Using aggregate stats: mean={agg['mean']:.6f} std={agg['std']:.6f}"
                )
                return float(agg["mean"]), float(agg["std"])
        except Exception as e:
            print(
                f"[Normalization] Failed reading aggregate stats ({e}); falling back to defaults."
            )
    print("[Normalization] Using hardcoded default stats.")
    return FEATURE_EXTRACTOR_MEAN, FEATURE_EXTRACTOR_STD


set_seed(SEED)

class_labels = ClassLabel(names=["Idle", "Swallow"])  # index 0=Idle, 1=Swallow
features = Features({"audio": Audio(), "labels": class_labels})

# Augmentations
audio_augmentations = Compose(
    [
        AddGaussianSNR(min_snr_db=10, max_snr_db=20),
        Gain(min_gain_db=-6, max_gain_db=6),
        GainTransition(
            min_gain_db=-6,
            max_gain_db=6,
            min_duration=0.01,
            max_duration=0.3,
            duration_unit="fraction",
        ),
        ClippingDistortion(
            min_percentile_threshold=0, max_percentile_threshold=30, p=0.5
        ),
        TimeStretch(min_rate=0.8, max_rate=1.2),
        PitchShift(min_semitones=-4, max_semitones=4),
        TimeMask(min_band_part=0.01, max_band_part=0.2),
    ],
    p=0.8,
    shuffle=True,
)

accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")

label2id = {"Idle": 0, "Swallow": 1}
AVERAGE = "binary"


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    predictions = np.argmax(logits, axis=1)
    metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    metrics.update(
        precision.compute(
            predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
        )
    )
    metrics.update(
        recall.compute(
            predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
        )
    )
    metrics.update(
        f1.compute(
            predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
        )
    )
    return metrics


def build_datasets(fold: int, feature_extractor: ASTFeatureExtractor, dry_run: bool):
    train_x = np.load(os.path.join(DATA_DIR, f"train_x_fold{fold}.npy")).tolist()
    train_y = np.load(os.path.join(DATA_DIR, f"train_y_fold{fold}.npy")).tolist()
    test_x = np.load(os.path.join(DATA_DIR, f"test_x_fold{fold}.npy")).tolist()
    test_y = np.load(os.path.join(DATA_DIR, f"test_y_fold{fold}.npy")).tolist()
    val_x_path = os.path.join(DATA_DIR, f"val_x_fold{fold}.npy")
    val_y_path = os.path.join(DATA_DIR, f"val_y_fold{fold}.npy")
    has_val = os.path.exists(val_x_path) and os.path.exists(val_y_path)
    if has_val:
        val_x = np.load(val_x_path).tolist()
        val_y = np.load(val_y_path).tolist()
    # Dry run downsizing
    if dry_run:
        train_x, train_y = train_x[:32], train_y[:32]
        test_x, test_y = test_x[:32], test_y[:32]
        if has_val:
            val_x, val_y = val_x[:32], val_y[:32]
    # Basic label sanity
    for name, arr in [("train_y", train_y), ("test_y", test_y)] + (
        [("val_y", val_y)] if has_val else []
    ):
        uniq = sorted(set(arr))
        if any(l not in (0, 1) for l in uniq):
            raise ValueError(f"Unexpected labels in {name} fold {fold}: {uniq}")
        if len(uniq) < 2:
            print(f"[WARN] Fold {fold} {name} single class: {uniq}")
    dataset_train = Dataset.from_dict(
        {"audio": train_x, "labels": train_y}, features=features
    )
    dataset_test = Dataset.from_dict(
        {"audio": test_x, "labels": test_y}, features=features
    )
    if has_val:
        dataset_val = Dataset.from_dict(
            {"audio": val_x, "labels": val_y}, features=features
        )
    # Cast audio column
    dataset_train = dataset_train.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    if has_val:
        dataset_val = dataset_val.cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )

    # Preprocessing (batched map)
    def preprocess_train(batch):
        wavs = [
            audio_augmentations(a["array"], sample_rate=SAMPLING_RATE)
            for a in batch["audio"]
        ]
        out = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="np")
        return {"input_values": out["input_values"]}

    def preprocess_eval(batch):
        wavs = [a["array"] for a in batch["audio"]]
        out = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="np")
        return {"input_values": out["input_values"]}

    dataset_train = dataset_train.map(
        preprocess_train, batched=True, remove_columns=["audio"]
    )
    dataset_test = dataset_test.map(
        preprocess_eval, batched=True, remove_columns=["audio"]
    )
    if has_val:
        dataset_val = dataset_val.map(
            preprocess_eval, batched=True, remove_columns=["audio"]
        )
    return dataset_train, (dataset_val if has_val else None), dataset_test


def train_fold(
    fold: int,
    wandb_run=None,
    dry_run: bool = False,
    enable_early_stopping: bool = True,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.0,
    adam_beta2: float = 0.999,
    optim_name: str = "adamw_torch_fused",
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
    config_file_path: str | None = None,
) -> Dict[str, float]:
    print(f"\n===== Stage1 Fold {fold} / {NUM_FOLDS} =====")
    fold_output_dir = os.path.join(OUTPUT_ROOT, f"fold{fold}")
    backup_path = backup_existing_run_dir(fold_output_dir)
    if backup_path:
        try:
            shutil.rmtree(fold_output_dir)
            print(
                f"[RunBackup] Cleared original run dir '{fold_output_dir}' after backup."
            )
        except Exception as exc:
            print(
                f"[RunBackup][WARN] Failed to clear '{fold_output_dir}' after backup: {exc}"
            )
    os.makedirs(fold_output_dir, exist_ok=True)
    if config_file_path is not None:
        try:
            shutil.copy2(
                config_file_path, os.path.join(fold_output_dir, "run_config.json")
            )
        except Exception as exc:
            print(
                f"[WARN] Failed to copy run configuration into '{fold_output_dir}': {exc}"
            )

    feature_extractor = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL)
    mean, std = load_fold_normalization(fold)
    feature_extractor.mean = mean
    feature_extractor.std = std

    config = ASTConfig.from_pretrained(PRETRAINED_MODEL)
    config.num_labels = len(label2id)
    config.label2id = label2id
    config.id2label = {v: k for k, v in label2id.items()}

    model = ASTForAudioClassification.from_pretrained(
        PRETRAINED_MODEL, config=config, ignore_mismatched_sizes=True
    )
    model.init_weights()

    dataset_train, dataset_val, dataset_test = build_datasets(
        fold, feature_extractor, dry_run=dry_run
    )

    checkpoint_limit = 1 if dry_run else max(2, (NUM_EPOCHS + 1) // 2)

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        logging_dir=os.path.join(LOG_DIR, f"fold{fold}"),
        learning_rate=learning_rate,
        push_to_hub=False,
        num_train_epochs=1 if dry_run else NUM_EPOCHS,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=1,
        save_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=5 if dry_run else 20,
        seed=SEED,
        save_total_limit=checkpoint_limit,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        adam_beta2=adam_beta2,
        optim=optim_name,
    )

    callbacks = []
    if dataset_val is not None and enable_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=2, early_stopping_threshold=0.001
            )
        )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val if dataset_val is not None else dataset_test,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )

    trainer.train()

    best_dir = os.path.join(fold_output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    feature_extractor.save_pretrained(best_dir)

    metrics = {}
    # Validation metrics (during training eval_dataset)
    eval_metrics = trainer.evaluate()
    prefix_val = "val" if dataset_val is not None else "test_during_train"
    for k, v in eval_metrics.items():
        if isinstance(v, (int, float)):
            metrics[f"fold{fold}_{prefix_val}_{k}"] = v
    # Final test metrics (always)
    test_metrics = trainer.evaluate(eval_dataset=dataset_test)
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            metrics[f"fold{fold}_test_{k}"] = v

    # Confusion matrices (skip in dry run for speed)
    if not dry_run:

        def save_cm(split_name: str, ds, wandb_run=None):
            try:
                preds_output = trainer.predict(ds)
                y_pred = preds_output.predictions.argmax(axis=1)
                y_true = preds_output.label_ids
                class_names = ["Idle", "Swallow"]
                cm = confusion_matrix(
                    y_true, y_pred, labels=list(range(len(class_names)))
                )
                eval_dir = os.path.join(
                    fold_output_dir, "best", f"evaluation_{split_name}"
                )
                os.makedirs(eval_dir, exist_ok=True)
                np.save(os.path.join(eval_dir, "confusion_matrix.npy"), cm)
                report_text = classification_report(
                    y_true, y_pred, target_names=class_names, digits=4
                )
                with open(
                    os.path.join(eval_dir, "classification_report.txt"), "w"
                ) as f:
                    f.write(report_text)

                if wandb_run is not None and _WANDB_AVAILABLE:
                    try:
                        cm_plot = wandb.plot.confusion_matrix(
                            preds=y_pred,
                            y_true=y_true,
                            class_names=class_names,
                            title=f"Fold {fold} {split_name} confusion matrix",
                        )
                        wandb_run.log(
                            {f"fold{fold}/{split_name}_confusion_matrix": cm_plot}
                        )
                    except Exception as wandb_exc:
                        print(
                            f"[WARN] Failed to log confusion matrix to W&B ({split_name}) fold {fold}: {wandb_exc}"
                        )

                    try:
                        report_dict = classification_report(
                            y_true,
                            y_pred,
                            target_names=class_names,
                            digits=4,
                            output_dict=True,
                            zero_division=0,
                        )
                        table = wandb.Table(
                            columns=["label", "precision", "recall", "f1", "support"]
                        )
                        for label in class_names:
                            entry = report_dict.get(label, {})
                            table.add_data(
                                label,
                                float(entry.get("precision", 0.0)),
                                float(entry.get("recall", 0.0)),
                                float(entry.get("f1-score", 0.0)),
                                int(entry.get("support", 0)),
                            )
                        for label in ["macro avg", "weighted avg"]:
                            entry = report_dict.get(label)
                            if isinstance(entry, dict):
                                table.add_data(
                                    label,
                                    float(entry.get("precision", 0.0)),
                                    float(entry.get("recall", 0.0)),
                                    float(entry.get("f1-score", 0.0)),
                                    int(entry.get("support", 0)),
                                )
                        accuracy_value = report_dict.get("accuracy")
                        if isinstance(accuracy_value, (int, float)):
                            table.add_data(
                                "accuracy",
                                float(accuracy_value),
                                float(accuracy_value),
                                float(accuracy_value),
                                int(
                                    sum(
                                        report_dict.get(lbl, {}).get("support", 0)
                                        for lbl in class_names
                                    )
                                ),
                            )
                        wandb_run.log(
                            {f"fold{fold}/{split_name}_classification_report": table}
                        )
                        wandb_run.log(
                            {
                                f"fold{fold}/{split_name}_classification_report_text": wandb.Html(
                                    f"<pre>{report_text}</pre>"
                                )
                            }
                        )
                    except Exception as wandb_exc:
                        print(
                            f"[WARN] Failed to log classification report to W&B ({split_name}) fold {fold}: {wandb_exc}"
                        )
            except Exception as e:
                print(f"[WARN] Failed confusion matrix ({split_name}) fold {fold}: {e}")

        if dataset_val is not None:
            save_cm("val", dataset_val, wandb_run=wandb_run)
        save_cm("test", dataset_test, wandb_run=wandb_run)
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Stage1 AST classifier (Idle vs Swallow) with CV"
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Train only a specific fold (1-based). If omitted, trains all folds.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (enabled by default)",
    )
    parser.add_argument(
        "--wandb-project", default=WANDB_PROJECT, help="W&B project name"
    )
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/user/team")
    parser.add_argument("--wandb-group", default=WANDB_GROUP, help="W&B run group")
    parser.add_argument(
        "--wandb-offline", action="store_true", help="Use W&B offline mode (no network)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fast sanity run (small subset, 1 epoch).",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Train full epoch schedule without early stopping.",
    )
    parser.add_argument(
        "--wandb-per-fold",
        action="store_true",
        help="Create a distinct W&B run for each fold to visualize metrics separately.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay applied by AdamW (default: 0.01).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Linear warmup ratio for the scheduler (default: 0.1).",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.98,
        help="Adam beta2 coefficient (default: 0.98 for faster adaptation).",
    )
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="Optimizer identifier for Hugging Face Trainer (default: adamw_torch_fused).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma parameter (0.0 = no focal loss, 2.0-5.0 typical). Helps with class imbalance.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (0.0-0.3, default 0.0 = none). Reduces overconfidence.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for training (default: 5e-5).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Custom output root directory (default: runs/ast_classifier_stage1).",
    )
    args = parser.parse_args()
    use_wandb = not args.no_wandb
    if use_wandb:
        print("[Config] W&B logging ENABLED (pass --no-wandb to disable).")
    else:
        print("[Config] W&B logging disabled via --no-wandb.")
    setattr(args, "wandb", use_wandb)
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup-ratio must be in [0.0, 1.0)")
    if not (0.0 < args.adam_beta2 < 1.0):
        raise ValueError("--adam-beta2 must be in (0.0, 1.0)")
    missing = []
    target_folds = [args.fold] if args.fold else list(range(1, NUM_FOLDS + 1))
    for fold in target_folds:
        for prefix in ["train_x", "train_y", "test_x", "test_y"]:
            path = os.path.join(DATA_DIR, f"{prefix}_fold{fold}.npy")
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing fold data files for Stage1. Run PrepareTrainingData_AST_2stage.py first. Missing: "
            + ", ".join(missing)
        )

    run_started_at = datetime.now()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    dry_run = args.dry_run
    enable_early_stopping = not args.disable_early_stopping

    # Override OUTPUT_ROOT if custom path provided
    global OUTPUT_ROOT
    if args.output_root:
        OUTPUT_ROOT = args.output_root
        print(f"[Config] Using custom output root: {OUTPUT_ROOT}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    config_snapshot = build_run_config(
        args=args,
        target_folds=target_folds,
        dry_run=dry_run,
        enable_early_stopping=enable_early_stopping,
        use_wandb=use_wandb,
        run_id=run_id,
        run_started_at=run_started_at,
    )
    config_path = os.path.join(OUTPUT_ROOT, f"run_config_{run_id}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, sort_keys=True)
    print(f"[Config] Snapshot saved to {config_path}")

    # Setup wandb (one run per full CV or per single fold)
    wandb_run = None
    start_wandb_run = None
    if use_wandb:
        if not _WANDB_AVAILABLE:
            raise RuntimeError(
                "wandb not installed. Install with 'pip install wandb' or pass --no-wandb."
            )
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

        base_wandb_config = {
            "stage": "stage1",
            "num_folds": NUM_FOLDS,
            "fold_mode": "single" if args.fold else "all",
            "label_mapping": {"Idle": 0, "Swallow": 1},
        }

        def _start_wandb_run(
            run_name: str,
            extra_config: Dict[str, object] | None = None,
            *,
            reinit: bool = False,
        ):
            cfg = dict(base_wandb_config)
            if extra_config:
                cfg.update(extra_config)
            return wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                group=args.wandb_group,
                config=cfg,
                reinit=reinit,
            )

        start_wandb_run = _start_wandb_run

        if not args.wandb_per_fold:
            run_name = (
                f"ast-stage1-fold{args.fold}" if args.fold else "ast-stage1-all-folds"
            )
            wandb_run = start_wandb_run(run_name)
            if wandb_run is not None:
                log_config_to_wandb(
                    wandb_run,
                    config_path=config_path,
                    artifact_name=f"stage1-config-{run_id}",
                    config_payload=dict(config_snapshot),
                )

    all_metrics: List[Dict[str, float]] = []
    if not enable_early_stopping:
        print(
            "[Config] Early stopping disabled for this run; full epoch schedule will train."
        )
    if dry_run:
        print("[DryRun] Enabled: limiting samples and epochs.")
    print(
        f"[Config] Optimizer={args.optim} weight_decay={args.weight_decay} "
        f"warmup_ratio={args.warmup_ratio} beta2={args.adam_beta2}"
    )
    for fold in target_folds:
        current_wandb_run = wandb_run
        if start_wandb_run is not None and args.wandb_per_fold:
            current_wandb_run = start_wandb_run(
                f"ast-stage1-fold{fold}",
                extra_config={"active_fold": fold},
                reinit=True,
            )
            if current_wandb_run is not None:
                fold_payload = dict(config_snapshot)
                fold_payload["active_fold"] = fold
                log_config_to_wandb(
                    current_wandb_run,
                    config_path=config_path,
                    artifact_name=f"stage1-fold{fold}-config-{run_id}",
                    config_payload=fold_payload,
                )

        metrics = train_fold(
            fold,
            wandb_run=current_wandb_run,
            dry_run=dry_run,
            enable_early_stopping=enable_early_stopping,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            adam_beta2=args.adam_beta2,
            optim_name=args.optim,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            config_file_path=config_path,
        )
        all_metrics.append(metrics)
        if current_wandb_run is not None:
            current_wandb_run.log(metrics)
            if args.wandb_per_fold:
                current_wandb_run.finish()

    # Aggregate ONLY test metrics
    aggregate: Dict[str, float] = {}
    test_metric_names = set()
    for d in all_metrics:
        for k in d:
            if "_test_" in k:
                test_metric_names.add(k.split("_test_", 1)[1])
    for name in test_metric_names:
        # FIX: use d directly (m was undefined)
        vals = [d[k] for d in all_metrics for k in d if k.endswith(f"_test_{name}")]
        if vals:
            aggregate[f"{name}_mean"] = float(np.mean(vals))
            aggregate[f"{name}_std"] = float(np.std(vals))

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    np.save(
        os.path.join(OUTPUT_ROOT, "cv_metrics.npy"),
        {"per_fold": all_metrics, "aggregate": aggregate},
    )
    with open(os.path.join(OUTPUT_ROOT, "cv_metrics.txt"), "w") as f:
        f.write("Per-fold metrics:\n")
        for m in all_metrics:
            f.write(str(m) + "\n")
        f.write("\nAggregate metrics:\n")
        f.write(str(aggregate) + "\n")

    print("\nStage 1 cross-validation complete")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}")

    aggregate_payload = {f"aggregate/{k}": v for k, v in aggregate.items()}

    if wandb_run is not None and aggregate_payload:
        # Log aggregate separately
        wandb_run.log(aggregate_payload)
        wandb_run.finish()
    elif start_wandb_run is not None and args.wandb_per_fold and aggregate_payload:
        summary_name = (
            "ast-stage1-summary"
            if args.fold is None
            else f"ast-stage1-fold{args.fold}-summary"
        )
        summary_run = start_wandb_run(
            summary_name, extra_config={"summary_only": True}, reinit=True
        )
        if summary_run is not None:
            summary_payload = dict(config_snapshot)
            summary_payload["summary_only"] = True
            log_config_to_wandb(
                summary_run,
                config_path=config_path,
                artifact_name=f"stage1-summary-config-{run_id}",
                config_payload=summary_payload,
            )
            summary_run.log(aggregate_payload)
            summary_run.finish()


if __name__ == "__main__":
    main()
