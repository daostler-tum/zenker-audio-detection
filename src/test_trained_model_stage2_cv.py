import os
import json
import argparse
import numpy as np
import evaluate
from datasets import Dataset, Audio, ClassLabel, Features
from transformers import (
    ASTFeatureExtractor,
    ASTConfig,
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report, confusion_matrix

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False

"""Evaluation script for Stage 2 (Healthy vs Zenker) cross-validation models.
Models expected under runs/ast_classifier_stage2/fold{k}/best
Data expected under data_ast_stage2/test_x_fold{k}.npy etc.
Labels: 0 -> Healthy, 1 -> Zenker
"""

NUM_FOLDS = 5
# Get the directory where the script is located and build relative path from there
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data_ast_stage2")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "runs", "ast_classifier_stage2")
SAMPLING_RATE = 16000
CLASS_NAMES = None  # dynamic
DEFAULT_MAPPING_PATH = os.path.join(PROJECT_ROOT, "data_ast_stage2", "class_mapping.json")


def load_class_names(mapping_path: str | None) -> list[str]:
    target = (
        mapping_path
        if mapping_path and os.path.isabs(mapping_path)
        else (
            os.path.join(PROJECT_ROOT, mapping_path)
            if mapping_path
            else DEFAULT_MAPPING_PATH
        )
    )
    if os.path.exists(target):
        try:
            with open(target, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                sorted_items = sorted(data.items(), key=lambda kv: kv[1])
                names = [k for k, _ in sorted_items]
                print(
                    f"Loaded class mapping from {target}: {data} -> class order {names}"
                )
                return names
            else:
                print(
                    f"[WARN] Mapping file {target} not a dict; using fallback Healthy/Zenker"
                )
        except Exception as e:
            print(f"[WARN] Failed reading class mapping {target}: {e}")
    else:
        print(
            f"[INFO] Class mapping file not found at {target}; using fallback Healthy/Zenker"
        )
    return ["Healthy", "Zenker"]


DEFAULT_MEAN = -1.1509622
DEFAULT_STD = 3.5340312


def load_mean_std(fold: int, use_aggregate: bool):
    agg_path = os.path.join(DATA_DIR, "stats_aggregate.json")
    per_path = os.path.join(DATA_DIR, "stats_per_fold.json")
    if use_aggregate and os.path.exists(agg_path):
        try:
            with open(agg_path, "r") as f:
                agg = json.load(f)
            return float(agg["mean"]), float(agg["std"])
        except Exception:
            pass
    if os.path.exists(per_path):
        try:
            with open(per_path, "r") as f:
                per = json.load(f)
            for d in per:
                if int(d.get("fold", -1)) == fold:
                    return float(d["mean"]), float(d["std"])
        except Exception:
            pass
    return DEFAULT_MEAN, DEFAULT_STD


def build_dataset(test_x, test_y, feature_extractor, class_names):
    class_labels = ClassLabel(names=class_names)
    features = Features({"audio": Audio(), "labels": class_labels})
    dataset_test = Dataset.from_dict(
        {"audio": test_x, "labels": test_y}, features=features
    )
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    dataset_test = dataset_test.rename_column("audio", "input_values")

    model_input_name = feature_extractor.model_input_names[0]

    def preprocess_audio(batch):
        wavs = [audio["array"] for audio in batch["input_values"]]
        inputs = feature_extractor(
            wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        )
        return {
            model_input_name: inputs.get(model_input_name),
            "labels": list(batch["labels"]),
        }

    dataset_test.set_transform(preprocess_audio, output_all_columns=False)
    return dataset_test


def evaluate_fold(fold: int, use_aggregate_stats: bool, wandb_run=None):
    print(f"\n=== Stage2 Evaluating fold {fold} ===")
    test_x_path = os.path.join(DATA_DIR, f"test_x_fold{fold}.npy")
    test_y_path = os.path.join(DATA_DIR, f"test_y_fold{fold}.npy")
    if not (os.path.exists(test_x_path) and os.path.exists(test_y_path)):
        raise FileNotFoundError(f"Missing test data for fold {fold}")
    test_x = np.load(test_x_path).tolist()
    test_y = np.load(test_y_path).tolist()

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    mean, std = load_mean_std(fold, use_aggregate_stats)
    feature_extractor.mean = mean
    feature_extractor.std = std
    print(f"Using normalization mean={mean:.6f} std={std:.6f}")

    dataset_test = build_dataset(test_x, test_y, feature_extractor, CLASS_NAMES)

    model_dir = os.path.join(MODEL_ROOT, f"fold{fold}", "best")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found for fold {fold}: {model_dir}"
        )

    config = ASTConfig.from_pretrained(model_dir)
    config.num_labels = len(CLASS_NAMES)
    config.label2id = {n: i for i, n in enumerate(CLASS_NAMES)}
    config.id2label = {i: n for i, n in enumerate(CLASS_NAMES)}

    model = ASTForAudioClassification.from_pretrained(model_dir, config=config)

    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "eval_tmp"),
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=False,
        logging_strategy="no",
    )

    trainer = Trainer(model=model, args=training_args, eval_dataset=dataset_test)
    predictions = trainer.predict(test_dataset=dataset_test)
    y_pred = predictions.predictions.argmax(axis=1)
    y_true = predictions.label_ids

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cr = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    eval_dir = os.path.join(model_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    np.save(os.path.join(eval_dir, "confusion_matrix.npy"), cm)
    with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
        f.write(cr + "\n")
    np.save(os.path.join(eval_dir, "y_true.npy"), y_true)
    np.save(os.path.join(eval_dir, "y_pred.npy"), y_pred)

    # Also write classification report to central results directory
    central_results_dir = os.path.join(PROJECT_ROOT, "results", "stage2")
    os.makedirs(central_results_dir, exist_ok=True)
    central_report_path = os.path.join(
        central_results_dir, f"fold{fold}_classification_report.txt"
    )
    try:
        with open(central_report_path, "w") as cf:
            cf.write(cr + "\n")
    except Exception as e:
        print(
            f"[WARN] Failed writing central classification report for fold {fold}: {e}"
        )

    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    if wandb_run is not None:
        try:
            cm_plot = wandb.plot.confusion_matrix(
                preds=y_pred,
                y_true=y_true,
                class_names=CLASS_NAMES,
                title=f"Stage2 Fold {fold} Confusion Matrix",
            )
            wandb_run.log({f"fold{fold}/confusion_matrix": cm_plot})
            wandb_run.log(
                {
                    f"fold{fold}/confusion_matrix_counts": wandb.Table(
                        columns=["class"] + CLASS_NAMES,
                        data=[[CLASS_NAMES[i]] + list(row) for i, row in enumerate(cm)],
                    )
                }
            )
        except Exception as e:
            print(f"[wandb] Failed to log fold {fold} confusion matrix: {e}")

    return {
        "fold": fold,
        "confusion_matrix": cm,
        "classification_report": cr,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage2 AST models on CV folds"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int, help="Evaluate a specific fold (1-based)")
    group.add_argument(
        "--all", action="store_true", help="Evaluate all folds and aggregate"
    )
    parser.add_argument(
        "--use-aggregate-stats",
        action="store_true",
        help="Use aggregated normalization stats instead of per-fold",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--wandb-project",
        default="ast-stage2-cv-test",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="Weights & Biases entity (team/user)"
    )
    parser.add_argument(
        "--wandb-run-name", default=None, help="Optional custom run name"
    )
    parser.add_argument(
        "--wandb-group", default="cv-eval-stage2", help="Optional run group"
    )
    parser.add_argument(
        "--class-mapping-path",
        default=None,
        help="Path to class_mapping.json (overrides default)",
    )
    parser.add_argument(
        "--model-root", default=None, help="Root directory for model checkpoints"
    )
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        if not _WANDB_AVAILABLE:
            raise RuntimeError(
                "wandb not installed. Install with 'pip install wandb' or omit --wandb."
            )
        run_name = args.wandb_run_name or (
            f"stage2-cv-eval-fold{args.fold}" if args.fold else "stage2-cv-eval-all"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            config={
                "evaluation_mode": "all" if args.all else "single_fold",
                "use_aggregate_stats": args.use_aggregate_stats,
                "num_folds": NUM_FOLDS,
            },
        )

    global CLASS_NAMES
    CLASS_NAMES = load_class_names(args.class_mapping_path)
    if wandb_run is not None:
        try:
            wandb_run.config.update(
                {"label_mapping": {n: i for i, n in enumerate(CLASS_NAMES)}},
                allow_val_change=True,
            )
        except Exception:
            pass
    # Override MODEL_ROOT if custom path provided
    global MODEL_ROOT
    if args.model_root is not None:
        MODEL_ROOT = args.model_root
        print(f"[Config] Using custom model root: {MODEL_ROOT}")
               
    results = []
    if args.all:
        for f in range(1, NUM_FOLDS + 1):
            results.append(evaluate_fold(f, args.use_aggregate_stats, wandb_run))
        all_true = np.concatenate([r["y_true"] for r in results])
        all_pred = np.concatenate([r["y_pred"] for r in results])
        agg_cm = confusion_matrix(
            all_true, all_pred, labels=list(range(len(CLASS_NAMES)))
        )
        agg_cr = classification_report(
            all_true, all_pred, target_names=CLASS_NAMES, digits=4
        )
        print("\n=== Stage2 Aggregate Across Folds ===")
        print("Confusion Matrix:\n", agg_cm)
        print("Classification Report:\n", agg_cr)
        agg_dir = os.path.join(MODEL_ROOT, "cv_aggregate_evaluation")
        os.makedirs(agg_dir, exist_ok=True)
        np.save(os.path.join(agg_dir, "confusion_matrix.npy"), agg_cm)
        with open(os.path.join(agg_dir, "classification_report.txt"), "w") as f:
            f.write(agg_cr + "\n")
        # Write aggregate classification report to central results directory
        central_results_dir = os.path.join(MODEL_ROOT, "results", "stage2")
        os.makedirs(central_results_dir, exist_ok=True)
        aggregate_report_path = os.path.join(
            central_results_dir, "aggregate_classification_report.txt"
        )
        try:
            with open(aggregate_report_path, "w") as cf:
                cf.write(agg_cr + "\n")
        except Exception as e:
            print(f"[WARN] Failed writing central aggregate classification report: {e}")
        if wandb_run is not None:
            try:
                agg_plot = wandb.plot.confusion_matrix(
                    preds=all_pred,
                    y_true=all_true,
                    class_names=CLASS_NAMES,
                    title="Stage2 Aggregate Confusion Matrix",
                )
                wandb_run.log({"aggregate/confusion_matrix": agg_plot})
                wandb_run.log(
                    {
                        "aggregate/confusion_matrix_counts": wandb.Table(
                            columns=["class"] + CLASS_NAMES,
                            data=[
                                [CLASS_NAMES[i]] + list(row)
                                for i, row in enumerate(agg_cm)
                            ],
                        )
                    }
                )
            except Exception as e:
                print(f"[wandb] Failed to log aggregate confusion matrix: {e}")
    else:
        if args.fold < 1 or args.fold > NUM_FOLDS:
            raise ValueError(f"Fold must be between 1 and {NUM_FOLDS}")
        evaluate_fold(args.fold, args.use_aggregate_stats, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
