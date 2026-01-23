import os
import json
import shutil
from datetime import datetime

import numpy as np
import evaluate
import torch
import torch.nn.functional as F
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
from typing import Any, Dict, List, Optional

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False

"""5-Fold cross validation training script (multiclass Idle/Healthy/Zenker).

Expected upstream data (created by PrepareTrainingData_AST_cv.py or equivalent):
    data_ast_cv/train_x_fold{k}.npy, train_y_fold{k}.npy, test_x_fold{k}.npy, test_y_fold{k}.npy

Label mapping dynamic load:
    Attempts to read data_ast_cv/class_mapping.json (default) unless overridden via --class-mapping-path.
    Fallback mapping (if file absent): {"Idle":0, "Healthy":1, "Zenker":2}

Weights & Biases (optional): enable with --wandb; run naming:
    Single fold: ast-multiclass-fold{k}
    All folds:   ast-multiclass-all-folds

Adds per-fold confusion matrix & classification report saving and logging.
"""

NUM_FOLDS = 5
NUM_EPOCHS = 12

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

DATA_DIR = os.path.join(project_root, "data_ast_cv")
OUTPUT_ROOT = os.path.join(project_root, "runs", "ast_classifier_multiclass")
LOG_DIR = os.path.join(project_root, "logs", "ast_classifier", "multiclass")
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLING_RATE = 16000
SEED = 42
WANDB_PROJECT = "zenker_detect_AST_multiclass"
WANDB_GROUP = "multiclass-cv"

# Pre-computed normalization stats (same as single split script)
FEATURE_EXTRACTOR_MEAN = -1.1509622
FEATURE_EXTRACTOR_STD = 3.5340312

# Attempt to override normalization stats with aggregated fold statistics if available


def build_run_config(
    *,
    args: Any,
    target_folds: List[int],
    dry_run: bool,
    enable_early_stopping: bool,
    use_wandb: bool,
    run_id: str,
    run_started_at: datetime,
    class_names: List[str],
    use_class_weights: bool,
) -> Dict[str, Any]:
    checkpoint_limit = 1 if dry_run else max(2, (NUM_EPOCHS + 1) // 4)
    return {
        "run_id": run_id,
        "timestamp": run_started_at.isoformat(),
        "script": "train_ast_cross_validation",
        "stage": "multiclass",
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
            "class_weights_enabled": use_class_weights,
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
        "class_mapping_path": args.class_mapping_path,
        "class_names": class_names,
        "num_classes": len(class_names),
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
    """Load per-fold normalization stats if available."""

    per_fold_path = os.path.join(DATA_DIR, "stats_per_fold.json")
    if os.path.exists(per_fold_path):
        try:
            with open(per_fold_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
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
        except Exception as exc:
            print(
                f"[Normalization] Failed reading per-fold stats ({exc}); will try aggregate."
            )

    agg_path = os.path.join(DATA_DIR, "stats_aggregate.json")
    if os.path.exists(agg_path):
        try:
            with open(agg_path, "r", encoding="utf-8") as f:
                agg = json.load(f)
            if "mean" in agg and "std" in agg:
                print(
                    f"[Normalization] Using aggregate stats: mean={agg['mean']:.6f} std={agg['std']:.6f}"
                )
                return float(agg["mean"]), float(agg["std"])
        except Exception as exc:
            print(
                f"[Normalization] Failed reading aggregate stats ({exc}); falling back to defaults."
            )

    print("[Normalization] Using hardcoded default stats.")
    return FEATURE_EXTRACTOR_MEAN, FEATURE_EXTRACTOR_STD


set_seed(SEED)

# Will be populated after loading mapping
CLASS_NAMES: List[str] = []
features: Optional[Features] = None
label2id: Dict[str, int] = {}
AVERAGE = "macro"


def load_class_mapping(path_override: Optional[str]) -> List[str]:
    """Load class mapping JSON (name->index) else fallback.

    Fallback ordering: Idle=0, Healthy=1, Zenker=2.
    Updates globals CLASS_NAMES, label2id, features, AVERAGE and returns class order.
    """

    global CLASS_NAMES, label2id, features, AVERAGE

    default_path = os.path.join(DATA_DIR, "class_mapping.json")
    target = path_override if path_override else default_path
    mapping: Dict[str, int] | None = None

    if os.path.exists(target):
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and all(
                isinstance(k, str) and isinstance(v, (int, float))
                for k, v in data.items()
            ):
                mapping = {k: int(v) for k, v in data.items()}
                print(f"[Mapping] Loaded mapping from {target}: {mapping}")
            else:
                print(
                    f"[Mapping] Invalid mapping structure in {target}; using fallback."
                )
        except Exception as exc:  # pragma: no cover
            print(f"[Mapping] Failed to read {target} ({exc}); using fallback.")

    if mapping is None:
        mapping = {"Idle": 0, "Healthy": 1, "Zenker": 2}
        print(f"[Mapping] Using fallback mapping {mapping}")

    CLASS_NAMES = [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    label2id = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    features = Features({"audio": Audio(), "labels": ClassLabel(names=CLASS_NAMES)})
    AVERAGE = "macro" if len(CLASS_NAMES) > 2 else "binary"
    print(f"[Mapping] Class order: {CLASS_NAMES}")
    return CLASS_NAMES


# Augmentations (applied only to training data)
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

# Metrics (lazy reused objects)
accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")


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


def build_datasets(fold: int, feature_extractor: ASTFeatureExtractor, *, dry_run: bool):
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

    if dry_run:
        train_x, train_y = train_x[:32], train_y[:32]
        test_x, test_y = test_x[:32], test_y[:32]
        if has_val:
            val_x, val_y = val_x[:32], val_y[:32]

    # Label sanity checks
    for name, arr in [
        ("train_y", train_y),
        ("test_y", test_y),
    ] + ([("val_y", val_y)] if has_val else []):
        uniq = sorted(set(arr))
        if any(label not in range(len(CLASS_NAMES)) for label in uniq):
            raise ValueError(f"Unexpected labels in {name} fold {fold}: {uniq}")
        if len(uniq) < len(CLASS_NAMES):
            print(f"[WARN] Fold {fold} {name} missing classes: {uniq}")

    dataset_train = Dataset.from_dict(
        {"audio": train_x, "labels": train_y}, features=features
    )
    dataset_test = Dataset.from_dict(
        {"audio": test_x, "labels": test_y}, features=features
    )
    dataset_val = None
    if has_val:
        dataset_val = Dataset.from_dict(
            {"audio": val_x, "labels": val_y}, features=features
        )

    dataset_train = dataset_train.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    if has_val:
        dataset_val = dataset_val.cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )

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

    class_counts = np.bincount(train_y, minlength=len(label2id))
    total = class_counts.sum()
    class_weights = [
        float(total / max(1, len(class_counts) * count)) for count in class_counts
    ]

    return (
        dataset_train,
        (dataset_val if has_val else None),
        dataset_test,
        class_weights,
    )


class MulticlassWeightedTrainer(Trainer):
    """Trainer supporting focal loss, label smoothing, and optional class weights."""

    def __init__(
        self,
        *,
        class_weights=None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        use_focal_loss: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.use_focal_loss:
            loss = self._focal_loss_with_smoothing(logits, labels)
        else:
            if self.class_weights is not None:
                if self.class_weights.device != logits.device:
                    self.class_weights = self.class_weights.to(logits.device)
                loss_fn = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights, label_smoothing=self.label_smoothing
                )
            else:
                loss_fn = torch.nn.CrossEntropyLoss(
                    label_smoothing=self.label_smoothing
                )
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def _focal_loss_with_smoothing(self, logits, labels):
        num_classes = logits.size(-1)

        smooth_labels = torch.zeros_like(logits)
        smooth_labels.fill_(self.label_smoothing / max(1, num_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.focal_gamma

        ce_loss = -(smooth_labels * log_probs).sum(dim=-1)
        focal_loss = focal_weight * ce_loss

        if self.class_weights is not None:
            if self.class_weights.device != logits.device:
                self.class_weights = self.class_weights.to(logits.device)
            weight_per_sample = self.class_weights[labels]
            focal_loss = focal_loss * weight_per_sample

        if self.focal_alpha is not None:
            focal_loss = focal_loss * self.focal_alpha

        return focal_loss.mean()


def train_fold(
    fold: int,
    *,
    wandb_run=None,
    dry_run: bool = False,
    enable_early_stopping: bool = True,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.0,
    adam_beta2: float = 0.999,
    optim_name: str = "adamw_torch_fused",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    use_focal_loss: bool = True,
    use_class_weights: bool = True,
    config_file_path: str | None = None,
) -> Dict[str, float]:
    print(f"\n===== Multiclass Fold {fold} / {NUM_FOLDS} =====")

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

    dataset_train, dataset_val, dataset_test, class_weights = build_datasets(
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
        optim=optim_name,
        adam_beta2=adam_beta2,
        report_to=["wandb"] if wandb_run is not None else [],
    )

    callbacks = []
    if dataset_val is not None and enable_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=2, early_stopping_threshold=0.001
            )
        )

    trainer = MulticlassWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val if dataset_val is not None else dataset_test,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights if use_class_weights else None,
        focal_alpha=0.25,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        use_focal_loss=use_focal_loss,
    )

    trainer.train()

    best_dir = os.path.join(fold_output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    feature_extractor.save_pretrained(best_dir)

    metrics: Dict[str, float] = {}

    eval_metrics = trainer.evaluate()
    prefix_val = "val" if dataset_val is not None else "test_during_train"
    for key, value in eval_metrics.items():
        if isinstance(value, (int, float)):
            metrics[f"fold{fold}_{prefix_val}_{key}"] = value

    test_metrics = trainer.evaluate(eval_dataset=dataset_test)
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            metrics[f"fold{fold}_test_{key}"] = value
            metric_name = key.replace("eval_", "")
            metrics[f"test_{metric_name}"] = value

    if not dry_run:

        def save_cm(split_name: str, ds, wandb_run=None):
            try:
                predictions = trainer.predict(ds)
                y_pred = predictions.predictions.argmax(axis=1)
                y_true = predictions.label_ids
                cm = confusion_matrix(
                    y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
                )
                report_text = classification_report(
                    y_true, y_pred, target_names=CLASS_NAMES, digits=4
                )
                eval_dir = os.path.join(
                    fold_output_dir, "best", f"evaluation_{split_name}"
                )
                os.makedirs(eval_dir, exist_ok=True)
                np.save(os.path.join(eval_dir, "confusion_matrix.npy"), cm)
                with open(
                    os.path.join(eval_dir, "classification_report.txt"), "w"
                ) as f:
                    f.write(report_text)

                if wandb_run is not None and _WANDB_AVAILABLE:
                    try:
                        cm_plot = wandb.plot.confusion_matrix(
                            preds=y_pred,
                            y_true=y_true,
                            class_names=CLASS_NAMES,
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
                            target_names=CLASS_NAMES,
                            digits=4,
                            output_dict=True,
                            zero_division=0,
                        )
                        table = wandb.Table(
                            columns=["label", "precision", "recall", "f1", "support"]
                        )
                        for label in CLASS_NAMES:
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
                                        report_dict.get(name, {}).get("support", 0)
                                        for name in CLASS_NAMES
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
            except Exception as exc:
                print(
                    f"[WARN] Failed confusion matrix ({split_name}) fold {fold}: {exc}"
                )

        if dataset_val is not None:
            save_cm("val", dataset_val, wandb_run=wandb_run)
        save_cm("test", dataset_test, wandb_run=wandb_run)

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train multiclass AST classifier (Idle/Healthy/Zenker) with CV"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing fold npy files (default: data_ast_cv).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Train only a specific fold (1-based). If omitted, trains all folds.",
    )
    parser.add_argument(
        "--class-mapping-path",
        default=None,
        help="Path to class_mapping.json overriding default",
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
        "--wandb-offline", action="store_true", help="Use W&B offline mode"
    )
    parser.add_argument(
        "--wandb-per-fold",
        action="store_true",
        help="Create a distinct W&B run for each fold to visualize metrics separately.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Fast sanity run (subset, 1 epoch)."
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Train full schedule without early stopping.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Linear warmup ratio (default: 0.1)",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.98,
        help="Adam beta2 coefficient (default: 0.98)",
    )
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="Optimizer name for Hugging Face Trainer (default: adamw_torch_fused)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (default: 2.0)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1)",
    )
    parser.add_argument(
        "--no-focal-loss",
        action="store_true",
        help="Disable focal loss and use standard cross-entropy.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable dynamic class weighting.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Custom output root directory (default: runs/ast_classifier_multiclass).",
    )
    args = parser.parse_args()

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = os.path.abspath(args.data_dir)
        print(f"[Config] Using data dir: {DATA_DIR}")

    use_wandb = not args.no_wandb
    setattr(args, "wandb", use_wandb)
    if use_wandb:
        print("[Config] W&B logging ENABLED (pass --no-wandb to disable).")
    else:
        print("[Config] W&B logging disabled via --no-wandb.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup-ratio must be in [0.0, 1.0)")
    if not (0.0 < args.adam_beta2 < 1.0):
        raise ValueError("--adam-beta2 must be in (0.0, 1.0)")

    class_names = load_class_mapping(args.class_mapping_path)

    if args.fold is not None:
        if not (1 <= args.fold <= NUM_FOLDS):
            raise ValueError(
                f"--fold must be between 1 and {NUM_FOLDS}, got {args.fold}"
            )
    target_folds = [args.fold] if args.fold else list(range(1, NUM_FOLDS + 1))
    print(f"[Config] Selected folds: {target_folds} (requested fold={args.fold})")

    missing = []
    for fold in target_folds:
        for prefix in ["train_x", "train_y", "test_x", "test_y"]:
            path = os.path.join(DATA_DIR, f"{prefix}_fold{fold}.npy")
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing fold data files. Run PrepareTrainingData_AST_cv.py first. Missing: "
            + ", ".join(missing)
        )

    run_started_at = datetime.now()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    dry_run = args.dry_run
    enable_early_stopping = not args.disable_early_stopping
    use_focal_loss = not args.no_focal_loss
    use_class_weights = not args.no_class_weights

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
        class_names=class_names,
        use_class_weights=use_class_weights,
    )
    config_path = os.path.join(OUTPUT_ROOT, f"run_config_{run_id}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, sort_keys=True)
    print(f"[Config] Snapshot saved to {config_path}")

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
            "stage": "multiclass",
            "num_folds": NUM_FOLDS,
            "fold_mode": "single" if args.fold else "all",
            "label_mapping": {name: idx for idx, name in enumerate(class_names)},
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
            if wandb.run is not None:
                print("[Sweep] Using existing wandb run from sweep agent")
                wandb_run = wandb.run
                try:
                    wandb_run.config.update(base_wandb_config, allow_val_change=True)
                except Exception:
                    pass
            else:
                run_name = (
                    f"ast-multiclass-fold{args.fold}"
                    if args.fold
                    else "ast-multiclass-all-folds"
                )
                wandb_run = start_wandb_run(run_name)

            if wandb_run is not None:
                log_config_to_wandb(
                    wandb_run,
                    config_path=config_path,
                    artifact_name=f"multiclass-config-{run_id}",
                    config_payload=dict(config_snapshot),
                )

    all_metrics: List[Dict[str, float]] = []
    if not enable_early_stopping:
        print("[Config] Early stopping disabled; training full schedule.")
    if dry_run:
        print("[DryRun] Enabled: limiting samples and epochs.")
    print(
        f"[Config] Optimizer={args.optim} lr={args.learning_rate} weight_decay={args.weight_decay} "
        f"warmup_ratio={args.warmup_ratio} beta2={args.adam_beta2}"
    )
    print(
        f"[Config] Focal Loss={'ENABLED' if use_focal_loss else 'DISABLED'} gamma={args.focal_gamma} "
        f"label_smoothing={args.label_smoothing} class_weights={'ON' if use_class_weights else 'OFF'}"
    )

    for fold in target_folds:
        current_wandb_run = wandb_run
        if start_wandb_run is not None and args.wandb_per_fold:
            current_wandb_run = start_wandb_run(
                f"ast-multiclass-fold{fold}",
                extra_config={"active_fold": fold},
                reinit=True,
            )
            if current_wandb_run is not None:
                fold_payload = dict(config_snapshot)
                fold_payload["active_fold"] = fold
                log_config_to_wandb(
                    current_wandb_run,
                    config_path=config_path,
                    artifact_name=f"multiclass-fold{fold}-config-{run_id}",
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
            use_focal_loss=use_focal_loss,
            use_class_weights=use_class_weights,
            config_file_path=config_path,
        )
        all_metrics.append(metrics)
        if current_wandb_run is not None:
            current_wandb_run.log(metrics)
            if args.wandb_per_fold:
                current_wandb_run.finish()

    aggregate: Dict[str, float] = {}
    test_metric_names = set()
    for metric_dict in all_metrics:
        for key in metric_dict:
            if "_test_" in key:
                test_metric_names.add(key.split("_test_", 1)[1])
    for name in test_metric_names:
        values = [
            metrics[key]
            for metrics in all_metrics
            for key in metrics
            if key.endswith(f"_test_{name}")
        ]
        if values:
            aggregate[f"{name}_mean"] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(np.std(values))

    np.save(
        os.path.join(OUTPUT_ROOT, "cv_metrics.npy"),
        {"per_fold": all_metrics, "aggregate": aggregate},
    )
    with open(os.path.join(OUTPUT_ROOT, "cv_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Per-fold metrics:\n")
        for metrics in all_metrics:
            f.write(str(metrics) + "\n")
        f.write("\nAggregate metrics:\n")
        f.write(str(aggregate) + "\n")

    print("\nMulticlass cross-validation complete")
    for key, value in aggregate.items():
        print(f"  {key}: {value:.4f}")

    aggregate_payload = {f"aggregate/{k}": v for k, v in aggregate.items()}
    if wandb_run is not None and aggregate_payload:
        wandb_run.log(aggregate_payload)
        wandb_run.finish()
    elif start_wandb_run is not None and args.wandb_per_fold and aggregate_payload:
        summary_name = (
            "ast-multiclass-summary"
            if args.fold is None
            else f"ast-multiclass-fold{args.fold}-summary"
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
                artifact_name=f"multiclass-summary-config-{run_id}",
                config_payload=summary_payload,
            )
            summary_run.log(aggregate_payload)
            summary_run.finish()


if __name__ == "__main__":
    main()
