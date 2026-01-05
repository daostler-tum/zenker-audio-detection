import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from keras.utils import Sequence as KerasSequence
except Exception:  # pragma: no cover
    KerasSequence = object  # type: ignore

try:
    from baselines.common import data, metrics, utils
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from baselines.common import data, metrics, utils


def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "se_resnet baseline requires tensorflow (not included in requirements.txt). "
            "Install tensorflow to run this baseline."
        ) from exc


def _load_predefined_folds(task: str) -> Optional[str]:
    if task == "binary_stage2":
        return "data_ast_stage2"
    if task == "binary_stage1":
        return "data_ast_stage1"
    if task == "multiclass":
        return "data_ast_cv"
    return None


def _wav_to_spec_candidates(wav_path: str, spectrogram_root: str) -> List[str]:
    parts = list(os.path.normpath(wav_path).split(os.sep))
    class_idx = None
    for i, token in enumerate(parts):
        if token in ("Idle", "Healthy", "Zenker"):
            class_idx = i
            break

    stem = os.path.splitext(os.path.basename(wav_path))[0]

    if class_idx is None or class_idx + 1 >= len(parts):
        return [os.path.join(spectrogram_root, stem + ".npy")]

    cls = parts[class_idx]
    patient = parts[class_idx + 1]
    rest = parts[class_idx + 2 : -1]

    candidates = []
    candidates.append(os.path.join(spectrogram_root, cls, patient, stem + ".npy"))
    if rest:
        candidates.append(
            os.path.join(spectrogram_root, cls, patient, *rest, stem + ".npy")
        )

    return candidates


def _wav_paths_to_specs(wav_paths: List[str], spectrogram_root: str) -> List[str]:
    out: List[str] = []
    missing: List[Tuple[str, List[str]]] = []
    for wp in wav_paths:
        cands = _wav_to_spec_candidates(wp, spectrogram_root)
        found = None
        for c in cands:
            if os.path.exists(c):
                found = c
                break
        if found is None:
            missing.append((wp, cands))
            continue
        out.append(found)

    if missing:
        first = missing[0]
        raise FileNotFoundError(
            "Could not map some wav paths to spectrogram .npy files. "
            f"Example wav: {first[0]} candidates: {first[1]}"
        )

    return out


class SpectrogramSequence(KerasSequence):
    def __init__(
        self,
        *,
        spec_paths: List[str],
        labels: List[int],
        batch_size: int,
        n_classes: int,
        shuffle: bool,
        height: int,
        width: int,
        pad_value: float,
        normalize: bool,
        mean: float,
        std: float,
        tf,
    ):
        try:
            super().__init__()
        except Exception:
            pass
        self.spec_paths = list(spec_paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.n_classes = int(n_classes)
        self.shuffle = bool(shuffle)
        self.height = int(height)
        self.width = int(width)
        self.pad_value = float(pad_value)
        self.normalize = bool(normalize)
        self.mean = float(mean)
        self.std = float(std)
        self.tf = tf
        self.indexes = np.arange(len(self.spec_paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.spec_paths) / float(self.batch_size)))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx: int):
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        x_batch = np.zeros(
            (len(batch_idx), self.height, self.width, 1), dtype=np.float32
        )
        y_batch = self.labels[batch_idx]

        for i, j in enumerate(batch_idx):
            arr = np.load(self.spec_paths[int(j)]).astype(np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D spectrogram array, got shape {arr.shape}"
                )

            h, w = arr.shape
            if h != self.height:
                if h < self.height:
                    pad_h = self.height - h
                    arr = np.pad(
                        arr,
                        ((0, pad_h), (0, 0)),
                        mode="constant",
                        constant_values=self.pad_value,
                    )
                else:
                    arr = arr[: self.height, :]

            if w != self.width:
                if w < self.width:
                    pad_w = self.width - w
                    arr = np.pad(
                        arr,
                        ((0, 0), (0, pad_w)),
                        mode="constant",
                        constant_values=self.pad_value,
                    )
                else:
                    arr = arr[:, : self.width]

            if self.normalize:
                arr = (arr - self.mean) / self.std

            x_batch[i, :, :, 0] = arr

        return x_batch, y_batch


@dataclass
class FoldResult:
    fold: int
    threshold_fixed: Optional[float]
    threshold_train_best: Optional[float]
    metrics_snippet_fixed: Dict[str, Any]
    metrics_patient_fixed_mean: Dict[str, Any]
    metrics_patient_fixed_vote: Dict[str, Any]
    metrics_snippet_best: Dict[str, Any]
    metrics_patient_best_mean: Dict[str, Any]
    metrics_patient_best_vote: Dict[str, Any]


def run_fold(
    *,
    fold: int,
    train_wavs: List[str],
    train_y: List[int],
    train_pids: List[str],
    val_wavs: Optional[List[str]],
    val_y: Optional[List[int]],
    val_pids: Optional[List[str]],
    test_wavs: List[str],
    test_y: List[int],
    test_pids: List[str],
    cfg: Dict[str, Any],
    run_dir: str,
    spectrogram_root: str,
    seed: int,
    compute_patient_metrics: bool,
    tf,
) -> FoldResult:
    from baselines.se_resnet.resnet import ResNet18

    fold_dir = os.path.join(run_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    task = str(cfg.get("task", "binary_stage2"))
    if task not in ("binary_stage2", "binary_stage1", "multiclass"):
        raise ValueError(f"Unsupported task: {task}")

    if task == "binary_stage2":
        labels_plot = ["Healthy", "Zenker"]
    elif task == "binary_stage1":
        labels_plot = ["Idle", "Swallow"]
    else:
        labels_plot = ["Idle", "Healthy", "Zenker"]

    if task == "multiclass":
        n_classes = 3
    else:
        n_classes = 2

    spec_cfg = cfg.get("spectrogram", {})
    height = int(spec_cfg.get("height", 256))
    width = int(spec_cfg.get("width", 346))
    pad_value = float(spec_cfg.get("pad_value", -80.0))
    normalize = bool(spec_cfg.get("normalize", True))
    mean = float(spec_cfg.get("mean", -63.333866))
    std = float(spec_cfg.get("std", 17.661556))

    train_specs = _wav_paths_to_specs(train_wavs, spectrogram_root)
    test_specs = _wav_paths_to_specs(test_wavs, spectrogram_root)

    if task == "binary_stage2":
        example_paths = (train_wavs[:16] if len(train_wavs) >= 16 else train_wavs) + (
            test_wavs[:16] if len(test_wavs) >= 16 else test_wavs
        )
        for p, y in zip(example_paths, (train_y[:16] + test_y[:16])):
            if os.sep + "Zenker" + os.sep in p and int(y) != 1:
                raise ValueError(
                    "Expected label mapping for binary_stage2 to be Healthy=0, Zenker=1 (Zenker positive), "
                    f"but found Zenker path with label={y}: {p}"
                )
            if os.sep + "Healthy" + os.sep in p and int(y) != 0:
                raise ValueError(
                    "Expected label mapping for binary_stage2 to be Healthy=0, Zenker=1 (Zenker positive), "
                    f"but found Healthy path with label={y}: {p}"
                )

    model_cfg = cfg.get("model", {})
    use_se = bool(model_cfg.get("use_se", True))

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 100))
    lr = float(train_cfg.get("learning_rate", 5e-6))
    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    es_patience = int(es_cfg.get("patience", 10))
    val_split = float(train_cfg.get("val_split_if_missing", 0.1))

    utils.set_seed(seed)
    tf.random.set_seed(seed)

    train_specs_for_threshold = train_specs
    train_y_for_threshold = train_y

    seq_train = SpectrogramSequence(
        spec_paths=train_specs,
        labels=train_y,
        batch_size=batch_size,
        n_classes=n_classes,
        shuffle=True,
        height=height,
        width=width,
        pad_value=pad_value,
        normalize=normalize,
        mean=mean,
        std=std,
        tf=tf,
    )

    seq_val = None
    if val_wavs is not None and val_y is not None:
        val_specs = _wav_paths_to_specs(val_wavs, spectrogram_root)
        seq_val = SpectrogramSequence(
            spec_paths=val_specs,
            labels=val_y,
            batch_size=batch_size,
            n_classes=n_classes,
            shuffle=False,
            height=height,
            width=width,
            pad_value=pad_value,
            normalize=normalize,
            mean=mean,
            std=std,
            tf=tf,
        )
    elif val_split > 0:
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            tr_idx, va_idx = next(
                sgkf.split(
                    np.zeros(len(train_y)),
                    np.asarray(train_y),
                    groups=np.asarray(train_pids),
                )
            )
        except Exception:
            tr_idx = np.arange(len(train_y))
            rng = np.random.RandomState(seed)
            rng.shuffle(tr_idx)
            n_val = max(1, int(round(len(train_y) * val_split)))
            va_idx = tr_idx[:n_val]
            tr_idx = tr_idx[n_val:]

        tr_idx = np.asarray(tr_idx)
        va_idx = np.asarray(va_idx)
        if len(va_idx) > 0 and len(tr_idx) > 0:
            train_specs_split = [train_specs[int(i)] for i in tr_idx]
            train_y_split = [int(train_y[int(i)]) for i in tr_idx]
            val_specs_split = [train_specs[int(i)] for i in va_idx]
            val_y_split = [int(train_y[int(i)]) for i in va_idx]

            train_specs_for_threshold = train_specs_split
            train_y_for_threshold = train_y_split

            seq_train = SpectrogramSequence(
                spec_paths=train_specs_split,
                labels=train_y_split,
                batch_size=batch_size,
                n_classes=n_classes,
                shuffle=True,
                height=height,
                width=width,
                pad_value=pad_value,
                normalize=normalize,
                mean=mean,
                std=std,
                tf=tf,
            )

            seq_val = SpectrogramSequence(
                spec_paths=val_specs_split,
                labels=val_y_split,
                batch_size=batch_size,
                n_classes=n_classes,
                shuffle=False,
                height=height,
                width=width,
                pad_value=pad_value,
                normalize=normalize,
                mean=mean,
                std=std,
                tf=tf,
            )

    seq_train_eval = SpectrogramSequence(
        spec_paths=train_specs_for_threshold,
        labels=train_y_for_threshold,
        batch_size=batch_size,
        n_classes=n_classes,
        shuffle=False,
        height=height,
        width=width,
        pad_value=pad_value,
        normalize=normalize,
        mean=mean,
        std=std,
        tf=tf,
    )
    seq_test = SpectrogramSequence(
        spec_paths=test_specs,
        labels=test_y,
        batch_size=batch_size,
        n_classes=n_classes,
        shuffle=False,
        height=height,
        width=width,
        pad_value=pad_value,
        normalize=normalize,
        mean=mean,
        std=std,
        tf=tf,
    )

    model = ResNet18(input_shape=(height, width, 1), classes=n_classes, use_se=use_se)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = []
    if es_enabled:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=es_patience, restore_best_weights=True
            )
        )

    if seq_val is not None:
        model.fit(
            seq_train,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=seq_val,
            verbose=1,
        )
    else:
        model.fit(seq_train, epochs=epochs, callbacks=callbacks, verbose=1)

    model.save(os.path.join(fold_dir, "model.keras"))

    y_pred = model.predict(seq_test)
    y_true = np.asarray(test_y, dtype=int)

    if n_classes == 2:
        y_proba = y_pred[:, 1]
        threshold_fixed = 0.5
        y_score_train = model.predict(seq_train_eval)[:, 1]
        if hasattr(metrics, "find_best_threshold_f1"):
            threshold_best = metrics.find_best_threshold_f1(
                np.asarray(train_y_for_threshold, dtype=int),
                y_score_train,
            )
        elif hasattr(metrics, "find_best_threshold"):
            threshold_best = metrics.find_best_threshold(
                y_true=np.asarray(train_y_for_threshold, dtype=int),
                y_score=y_score_train,
            )
        else:
            threshold_best = None

        threshold_best_eff = (
            threshold_best if threshold_best is not None else threshold_fixed
        )

        y_pred_fixed = (y_proba >= threshold_fixed).astype(int)
        y_pred_best = (y_proba >= threshold_best_eff).astype(int)

        snippet_fixed = metrics.binary_metrics(
            y_true=y_true, y_pred=y_pred_fixed, y_proba=y_proba
        )
        snippet_best = metrics.binary_metrics(
            y_true=y_true, y_pred=y_pred_best, y_proba=y_proba
        )

        report_fixed = metrics.classification_report_dict(
            y_true=y_true.tolist(),
            y_pred=y_pred_fixed.tolist(),
            target_names=labels_plot,
        )
        report_best = metrics.classification_report_dict(
            y_true=y_true.tolist(),
            y_pred=y_pred_best.tolist(),
            target_names=labels_plot,
        )
        if report_fixed is not None:
            utils.save_json(
                report_fixed,
                os.path.join(fold_dir, "classification_report_snippet_fixed.json"),
            )
        if report_best is not None:
            utils.save_json(
                report_best,
                os.path.join(fold_dir, "classification_report_snippet_best.json"),
            )

        patient_fixed_mean: Dict[str, Any] = {}
        patient_fixed_vote: Dict[str, Any] = {}
        patient_best_mean: Dict[str, Any] = {}
        patient_best_vote: Dict[str, Any] = {}

        if compute_patient_metrics:
            p_yt_mean_fixed, p_yp_mean_fixed, p_pp_mean_fixed, _ = (
                metrics.aggregate_by_patient(
                    patient_ids=test_pids,
                    y_true=y_true.tolist(),
                    y_proba=y_proba.tolist(),
                    method="mean_prob",
                    threshold=threshold_fixed,
                )
            )
            p_yt_vote_fixed, p_yp_vote_fixed, p_pp_vote_fixed, _ = (
                metrics.aggregate_by_patient(
                    patient_ids=test_pids,
                    y_true=y_true.tolist(),
                    y_proba=y_proba.tolist(),
                    method="majority_vote",
                    threshold=threshold_fixed,
                )
            )
            p_yt_mean_best, p_yp_mean_best, p_pp_mean_best, _ = (
                metrics.aggregate_by_patient(
                    patient_ids=test_pids,
                    y_true=y_true.tolist(),
                    y_proba=y_proba.tolist(),
                    method="mean_prob",
                    threshold=threshold_best_eff,
                )
            )
            p_yt_vote_best, p_yp_vote_best, p_pp_vote_best, _ = (
                metrics.aggregate_by_patient(
                    patient_ids=test_pids,
                    y_true=y_true.tolist(),
                    y_proba=y_proba.tolist(),
                    method="majority_vote",
                    threshold=threshold_best_eff,
                )
            )

            patient_fixed_mean = metrics.binary_metrics(
                y_true=p_yt_mean_fixed,
                y_pred=p_yp_mean_fixed,
                y_proba=p_pp_mean_fixed,
            )
            patient_fixed_vote = metrics.binary_metrics(
                y_true=p_yt_vote_fixed,
                y_pred=p_yp_vote_fixed,
                y_proba=p_pp_vote_fixed,
            )
            patient_best_mean = metrics.binary_metrics(
                y_true=p_yt_mean_best,
                y_pred=p_yp_mean_best,
                y_proba=p_pp_mean_best,
            )
            patient_best_vote = metrics.binary_metrics(
                y_true=p_yt_vote_best,
                y_pred=p_yp_vote_best,
                y_proba=p_pp_vote_best,
            )

        metrics.save_confusion_matrix_png(
            y_true=y_true.tolist(),
            y_pred=y_pred_fixed.tolist(),
            labels=labels_plot,
            out_path=os.path.join(fold_dir, "confusion_snippet_fixed.png"),
            title=f"Fold {fold}",
        )
        metrics.save_confusion_matrix_png(
            y_true=y_true.tolist(),
            y_pred=y_pred_best.tolist(),
            labels=labels_plot,
            out_path=os.path.join(fold_dir, "confusion_snippet_best.png"),
            title=f"Fold {fold}",
        )

        metrics.save_roc_pr_curves(
            y_true=y_true.tolist(),
            y_proba=y_proba.tolist(),
            out_dir=fold_dir,
            prefix="snippet",
        )

        utils.save_json(snippet_fixed, os.path.join(fold_dir, "metrics_snippet.json"))
        if compute_patient_metrics:
            utils.save_json(
                patient_fixed_mean, os.path.join(fold_dir, "metrics_patient_mean.json")
            )
            utils.save_json(
                patient_fixed_vote, os.path.join(fold_dir, "metrics_patient_vote.json")
            )

        return FoldResult(
            fold=fold,
            threshold_fixed=threshold_fixed,
            threshold_train_best=(
                float(threshold_best) if threshold_best is not None else None
            ),
            metrics_snippet_fixed=snippet_fixed,
            metrics_patient_fixed_mean=patient_fixed_mean,
            metrics_patient_fixed_vote=patient_fixed_vote,
            metrics_snippet_best=snippet_best,
            metrics_patient_best_mean=patient_best_mean,
            metrics_patient_best_vote=patient_best_vote,
        )

    y_pred_labels = y_pred.argmax(axis=1)
    snippet = metrics.multiclass_metrics(
        y_true=y_true, y_pred=y_pred_labels, y_proba=y_pred
    )

    report_mc = metrics.classification_report_dict(
        y_true=y_true.tolist(),
        y_pred=y_pred_labels.tolist(),
        target_names=labels_plot,
    )
    if report_mc is not None:
        utils.save_json(
            report_mc,
            os.path.join(fold_dir, "classification_report_snippet.json"),
        )

    patient_mean: Dict[str, Any] = {}
    patient_vote: Dict[str, Any] = {}
    if compute_patient_metrics:
        p_yt_mean, p_yp_mean, p_pp_mean, _ = metrics.aggregate_by_patient_multiclass(
            patient_ids=test_pids,
            y_true=y_true.tolist(),
            y_proba=y_pred,
            method="mean_prob",
        )
        p_yt_vote, p_yp_vote, p_pp_vote, _ = metrics.aggregate_by_patient_multiclass(
            patient_ids=test_pids,
            y_true=y_true.tolist(),
            y_proba=y_pred,
            method="majority_vote",
        )

        patient_mean = metrics.multiclass_metrics(
            y_true=p_yt_mean,
            y_pred=p_yp_mean,
            y_proba=p_pp_mean,
        )
        patient_vote = metrics.multiclass_metrics(
            y_true=p_yt_vote,
            y_pred=p_yp_vote,
            y_proba=p_pp_vote,
        )

    metrics.save_confusion_matrix_png(
        y_true=y_true.tolist(),
        y_pred=y_pred_labels.tolist(),
        labels=labels_plot,
        out_path=os.path.join(fold_dir, "confusion_snippet.png"),
        title=f"Fold {fold}",
    )

    utils.save_json(snippet, os.path.join(fold_dir, "metrics_snippet.json"))
    if compute_patient_metrics:
        utils.save_json(
            patient_mean, os.path.join(fold_dir, "metrics_patient_mean.json")
        )
        utils.save_json(
            patient_vote, os.path.join(fold_dir, "metrics_patient_vote.json")
        )

    return FoldResult(
        fold=fold,
        threshold_fixed=None,
        threshold_train_best=None,
        metrics_snippet_fixed=snippet,
        metrics_patient_fixed_mean=patient_mean,
        metrics_patient_fixed_vote=patient_vote,
        metrics_snippet_best=snippet,
        metrics_patient_best_mean=patient_mean,
        metrics_patient_best_vote=patient_vote,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--predefined_folds_dir", default=None)
    ap.add_argument(
        "--task", choices=["binary_stage2", "binary_stage1", "multiclass"], default=None
    )
    ap.add_argument(
        "--spectrogram_root",
        default="/home/ksvoai/source/datasets/spectrograms/New_SwallowSet_Test/",
    )
    ap.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    ap.add_argument("--patient_metrics", action="store_true")
    ap.add_argument("--patient-metrics", dest="patient_metrics", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    tf = _require_tensorflow()

    cfg = utils.load_yaml(args.config)
    task = str(args.task or cfg.get("task", "binary_stage2"))
    cfg["task"] = task

    n_folds = int(args.n_folds)
    if args.fold is not None and not (1 <= int(args.fold) <= n_folds):
        raise ValueError(f"--fold must be in [1, {n_folds}]")

    utils.set_seed(int(args.seed))
    tf.random.set_seed(int(args.seed))

    run_dir = utils.resolve_run_dir(args.output_dir, args.run_name)
    logger = utils.setup_logger(os.path.join(run_dir, "train.log"))
    utils.save_yaml(cfg, os.path.join(run_dir, "config_used.yaml"))

    predefined_dir = args.predefined_folds_dir
    if predefined_dir is None:
        default = _load_predefined_folds(task)
        if default is not None:
            candidate = os.path.join(os.path.dirname(__file__), "..", "..", default)
            if os.path.exists(candidate):
                predefined_dir = os.path.abspath(candidate)

    if predefined_dir is None or not os.path.exists(predefined_dir):
        raise ValueError(
            "se_resnet baseline requires predefined folds (data_ast_stage1/2 or data_ast_cv) via --predefined_folds_dir."
        )

    fold_results: List[FoldResult] = []
    folds_to_run = (
        [int(args.fold)] if args.fold is not None else list(range(1, n_folds + 1))
    )

    for fold in folds_to_run:
        (x_tr, y_tr), val, (x_te, y_te) = data.load_predefined_fold_splits(
            folds_dir=predefined_dir,
            fold=fold,
            task=task,
            dry_run=bool(args.dry_run),
        )

        p_tr = [data.parse_patient_id(p) for p in x_tr]
        p_te = [data.parse_patient_id(p) for p in x_te]

        x_val: Optional[List[str]] = None
        y_val: Optional[List[int]] = None
        p_val: Optional[List[str]] = None
        if val is not None:
            x_val, y_val = val
            p_val = [data.parse_patient_id(p) for p in x_val]

        fold_results.append(
            run_fold(
                fold=fold,
                train_wavs=x_tr,
                train_y=y_tr,
                train_pids=p_tr,
                val_wavs=x_val,
                val_y=y_val,
                val_pids=p_val,
                test_wavs=x_te,
                test_y=y_te,
                test_pids=p_te,
                cfg=cfg,
                run_dir=run_dir,
                spectrogram_root=str(args.spectrogram_root),
                seed=int(args.seed),
                compute_patient_metrics=bool(args.patient_metrics),
                tf=tf,
            )
        )

    rows = []
    for fr in fold_results:
        if task in ("binary_stage2", "binary_stage1"):
            row = {
                "fold": fr.fold,
                "snippet_acc_fixed": fr.metrics_snippet_fixed.get("accuracy"),
                "snippet_bal_acc_fixed": fr.metrics_snippet_fixed.get(
                    "balanced_accuracy"
                ),
                "snippet_f1_fixed": fr.metrics_snippet_fixed.get("f1"),
                "snippet_auroc": fr.metrics_snippet_fixed.get("auroc"),
                "snippet_auprc": fr.metrics_snippet_fixed.get("auprc"),
                "threshold_train_best": fr.threshold_train_best,
            }
            if bool(args.patient_metrics):
                row["patient_mean_f1_fixed"] = fr.metrics_patient_fixed_mean.get("f1")
                row["patient_vote_f1_fixed"] = fr.metrics_patient_fixed_vote.get("f1")
            rows.append(row)
        else:
            row = {
                "fold": fr.fold,
                "snippet_acc": fr.metrics_snippet_fixed.get("accuracy"),
                "snippet_bal_acc": fr.metrics_snippet_fixed.get("balanced_accuracy"),
                "snippet_f1_macro": fr.metrics_snippet_fixed.get("f1_macro"),
            }
            if bool(args.patient_metrics):
                row["patient_mean_f1_macro"] = fr.metrics_patient_fixed_mean.get(
                    "f1_macro"
                )
                row["patient_vote_f1_macro"] = fr.metrics_patient_fixed_vote.get(
                    "f1_macro"
                )
            rows.append(row)

    utils.save_json({"rows": rows}, os.path.join(run_dir, "summary.json"))
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    except Exception:
        pass

    logger.info("Done")


if __name__ == "__main__":
    main()
