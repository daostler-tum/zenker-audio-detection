import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from baselines.common import audio_features, data, metrics, utils
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from baselines.common import audio_features, data, metrics, utils


@dataclass
class FoldSummary:
    fold: int
    threshold_fixed: Optional[float]
    threshold_train_best: Optional[float]
    metrics_snippet_fixed: Dict[str, Any]
    metrics_patient_fixed_mean: Dict[str, Any]
    metrics_patient_fixed_vote: Dict[str, Any]
    metrics_snippet_best: Dict[str, Any]
    metrics_patient_best_mean: Dict[str, Any]
    metrics_patient_best_vote: Dict[str, Any]


def _load_yamnet():
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_hub as hub  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Baseline A requires tensorflow and tensorflow_hub. "
            "Install them in your environment to use YAMNet."
        ) from exc

    model = hub.load("https://tfhub.dev/google/yamnet/1")
    return tf, model


def _embedding_for_file(
    *,
    fp: str,
    tf,
    model,
    cfg_audio: audio_features.AudioConfig,
    aggregate: List[str],
    cache_dir: Optional[str],
) -> np.ndarray:
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"yamnet_{utils.stable_hash(fp)}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

    y, _ = audio_features.load_audio_mono(fp, cfg=cfg_audio)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)

    scores, embeddings, _ = model(waveform)
    emb = embeddings.numpy()

    parts: List[np.ndarray] = []
    if "mean" in aggregate:
        parts.append(np.mean(emb, axis=0))
    if "max" in aggregate:
        parts.append(np.max(emb, axis=0))
    if len(parts) == 0:
        parts.append(np.mean(emb, axis=0))

    v = np.concatenate(parts, axis=0).astype(np.float32)

    if cache_dir is not None:
        np.save(cache_path, v)

    return v


def _extract_matrix(
    filepaths: List[str],
    *,
    tf,
    model,
    cfg_audio: audio_features.AudioConfig,
    aggregate: List[str],
    cache_dir: Optional[str],
) -> np.ndarray:
    feats: List[np.ndarray] = []
    for fp in filepaths:
        feats.append(
            _embedding_for_file(
                fp=fp,
                tf=tf,
                model=model,
                cfg_audio=cfg_audio,
                aggregate=aggregate,
                cache_dir=cache_dir,
            )
        )
    return np.stack(feats, axis=0)


def run_fold(
    *,
    fold: int,
    x_train: List[str],
    y_train: List[int],
    p_train: List[str],
    x_test: List[str],
    y_test: List[int],
    p_test: List[str],
    cfg: Dict[str, Any],
    run_dir: str,
    seed: int,
    tf,
    model,
    logger,
) -> FoldSummary:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    task = str(cfg.get("task", "binary_stage2"))
    if task not in ("binary_stage2", "multiclass"):
        raise ValueError(f"Unsupported task: {task}")

    cfg_audio = audio_features.AudioConfig(
        sample_rate=int(cfg["sample_rate"]),
        clip_duration_s=float(cfg["clip_duration_s"]),
    )

    fold_dir = os.path.join(run_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    aggregate = list(cfg.get("embedding", {}).get("aggregate", ["mean", "max"]))
    cache_enabled = bool(cfg.get("embedding", {}).get("cache", True))

    cache_dir = os.path.join(run_dir, "cache", f"fold{fold}") if cache_enabled else None

    X_train = _extract_matrix(
        x_train,
        tf=tf,
        model=model,
        cfg_audio=cfg_audio,
        aggregate=aggregate,
        cache_dir=os.path.join(cache_dir, "train") if cache_dir else None,
    )
    X_test = _extract_matrix(
        x_test,
        tf=tf,
        model=model,
        cfg_audio=cfg_audio,
        aggregate=aggregate,
        cache_dir=os.path.join(cache_dir, "test") if cache_dir else None,
    )

    y_train_np = np.asarray(y_train).astype(int)
    y_test_np = np.asarray(y_test).astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if task == "binary_stage2":
        clf = LogisticRegression(
            C=float(cfg["classifier"].get("C", 1.0)),
            class_weight=cfg["classifier"].get("class_weight", "balanced"),
            solver="liblinear",
            random_state=seed,
            max_iter=2000,
        )
    else:
        clf = LogisticRegression(
            C=float(cfg["classifier"].get("C", 1.0)),
            class_weight=cfg["classifier"].get("class_weight", "balanced"),
            solver="lbfgs",
            multi_class="multinomial",
            random_state=seed,
            max_iter=2000,
        )
    clf.fit(X_train_s, y_train_np)

    if task == "binary_stage2":
        proba_train = clf.predict_proba(X_train_s)[:, 1]
        proba_test = clf.predict_proba(X_test_s)[:, 1]

        threshold_fixed = float(cfg["patient_aggregation"]["threshold"])
        threshold_best = 0.5
        if bool(cfg.get("threshold_tuning", {}).get("enabled", True)):
            threshold_best = metrics.find_best_threshold_f1(y_train_np, proba_train)

        pred_test_fixed = (proba_test >= threshold_fixed).astype(int)
        pred_test_best = (proba_test >= threshold_best).astype(int)

        snippet_fixed = metrics.binary_metrics(y_test_np, pred_test_fixed, proba_test)
        snippet_best = metrics.binary_metrics(y_test_np, pred_test_best, proba_test)

        yt_mean, yp_mean, pp_mean, _ = metrics.aggregate_by_patient(
            patient_ids=p_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="mean_prob",
            threshold=threshold_fixed,
        )
        patient_fixed_mean = metrics.binary_metrics(yt_mean, yp_mean, pp_mean)

        yt_vote, yp_vote, pp_vote, _ = metrics.aggregate_by_patient(
            patient_ids=p_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="majority_vote",
            threshold=threshold_fixed,
        )
        patient_fixed_vote = metrics.binary_metrics(yt_vote, yp_vote, pp_vote)

        yt_mean_b, yp_mean_b, pp_mean_b, _ = metrics.aggregate_by_patient(
            patient_ids=p_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="mean_prob",
            threshold=threshold_best,
        )
        patient_best_mean = metrics.binary_metrics(yt_mean_b, yp_mean_b, pp_mean_b)

        yt_vote_b, yp_vote_b, pp_vote_b, _ = metrics.aggregate_by_patient(
            patient_ids=p_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="majority_vote",
            threshold=threshold_best,
        )
        patient_best_vote = metrics.binary_metrics(yt_vote_b, yp_vote_b, pp_vote_b)
    else:
        proba_train = clf.predict_proba(X_train_s)
        proba_test = clf.predict_proba(X_test_s)

        pred_test = np.argmax(proba_test, axis=1).astype(int)
        snippet_fixed = metrics.multiclass_metrics(
            y_test_np, pred_test, y_proba=proba_test
        )
        snippet_best = {"note": "threshold tuning not applicable to multiclass"}

        yt_mean, yp_mean, pp_mean, _ = metrics.aggregate_by_patient_multiclass(
            patient_ids=p_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="mean_prob",
        )
        patient_fixed_mean = metrics.multiclass_metrics(
            yt_mean, yp_mean, y_proba=pp_mean
        )

        yt_vote, yp_vote, pp_vote, _ = metrics.aggregate_by_patient_multiclass(
            patient_ids=p_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="majority_vote",
        )
        patient_fixed_vote = metrics.multiclass_metrics(
            yt_vote, yp_vote, y_proba=pp_vote
        )

        patient_best_mean = {"note": "threshold tuning not applicable to multiclass"}
        patient_best_vote = {"note": "threshold tuning not applicable to multiclass"}
        threshold_fixed = None
        threshold_best = None

    utils.save_json(
        {
            "fold": fold,
            "threshold_fixed": threshold_fixed,
            "threshold_train_best": float(threshold_best)
            if threshold_best is not None
            else None,
            "fixed": snippet_fixed,
            "train_best": snippet_best,
        },
        os.path.join(fold_dir, "metrics_snippet.json"),
    )

    utils.save_json(
        {
            "fold": fold,
            "threshold_fixed": threshold_fixed,
            "threshold_train_best": float(threshold_best),
            "fixed": {
                "mean_prob": patient_fixed_mean,
                "majority_vote": patient_fixed_vote,
            },
            "train_best": {
                "mean_prob": patient_best_mean,
                "majority_vote": patient_best_vote,
            },
        },
        os.path.join(fold_dir, "metrics_patient.json"),
    )

    if task == "binary_stage2":
        y_pred_plot = (
            np.asarray(proba_test) >= float(cfg["patient_aggregation"]["threshold"])
        ).astype(int)
        labels_plot = ["Healthy", "Zenker"]
        title = f"Snippet Confusion (fold {fold}, thr=0.5)"
        metrics.save_roc_pr_curves(
            y_true=y_test_np, y_proba=proba_test, out_dir=fold_dir, prefix="snippet"
        )
    else:
        y_pred_plot = np.argmax(np.asarray(proba_test), axis=1).astype(int)
        labels_plot = ["Idle", "Healthy", "Zenker"]
        title = f"Snippet Confusion (fold {fold})"

    metrics.save_confusion_matrix_png(
        y_true=y_test_np,
        y_pred=y_pred_plot,
        labels=labels_plot,
        out_path=os.path.join(fold_dir, "confusion_snippet_fixed.png"),
        title=title,
    )

    return FoldSummary(
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


def _load_predefined_folds(task: str) -> Optional[str]:
    if task == "binary_stage2":
        return "data_ast_stage2"
    if task == "multiclass":
        return "data_ast_cv"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--label_csv", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_splits", type=int, dest="n_folds", help=argparse.SUPPRESS)
    ap.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If set, run only this 1-based fold index (for predefined folds or generated splits).",
    )
    ap.add_argument("--run_name", default=None)
    ap.add_argument(
        "--task",
        choices=["binary_stage2", "multiclass"],
        default=None,
        help="Override task from config (binary_stage2=Healthy vs Zenker, multiclass=Idle/Healthy/Zenker).",
    )
    ap.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    ap.add_argument("--predefined_folds_dir", default=None)
    args = ap.parse_args()

    cfg = utils.load_yaml(args.config)
    task = str(args.task or cfg.get("task", "binary_stage2"))
    if task not in ("binary_stage2", "multiclass"):
        raise ValueError(f"Unsupported task: {task}")

    cfg["task"] = task

    n_folds = int(args.n_folds)
    if args.fold is not None and not (1 <= int(args.fold) <= n_folds):
        raise ValueError(f"--fold must be in [1, {n_folds}]")

    utils.set_seed(int(args.seed))
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

    tf, model = _load_yamnet()

    fold_summaries: List[FoldSummary] = []

    if predefined_dir is not None and os.path.exists(predefined_dir):
        logger.info(f"Using predefined folds from: {predefined_dir}")
        folds_to_run = (
            [int(args.fold)] if args.fold is not None else list(range(1, n_folds + 1))
        )
        for fold in folds_to_run:
            (x_tr, y_tr), _, (x_te, y_te) = data.load_predefined_fold_splits(
                folds_dir=predefined_dir,
                fold=fold,
                task=task,
                dry_run=bool(args.dry_run),
            )
            p_tr = [data.parse_patient_id(p) for p in x_tr]
            p_te = [data.parse_patient_id(p) for p in x_te]

            fold_summaries.append(
                run_fold(
                    fold=fold,
                    x_train=x_tr,
                    y_train=y_tr,
                    p_train=p_tr,
                    x_test=x_te,
                    y_test=y_te,
                    p_test=p_te,
                    cfg=cfg,
                    run_dir=run_dir,
                    seed=int(args.seed),
                    tf=tf,
                    model=model,
                    logger=logger,
                )
            )
    else:
        if args.data_root is None:
            raise ValueError(
                "No predefined folds found (e.g. data_ast_stage2/ or data_ast_cv/) and --data_root was not provided."
            )
        logger.info("Using on-the-fly patient-level folds")
        ds = data.index_dataset(
            data_root=args.data_root, label_csv=args.label_csv, task=task
        )
        splits = data.make_splits(
            ds.filepaths,
            ds.labels,
            ds.patient_ids,
            n_splits=n_folds,
            seed=int(args.seed),
        )

        if args.fold is not None:
            splits = [splits[int(args.fold) - 1]]

        start_fold = 1 if args.fold is None else int(args.fold)
        for fold, (tr_idx, te_idx) in enumerate(splits, start=start_fold):
            x_tr = [ds.filepaths[i] for i in tr_idx]
            y_tr = [ds.labels[i] for i in tr_idx]
            p_tr = [ds.patient_ids[i] for i in tr_idx]

            x_te = [ds.filepaths[i] for i in te_idx]
            y_te = [ds.labels[i] for i in te_idx]
            p_te = [ds.patient_ids[i] for i in te_idx]

            fold_summaries.append(
                run_fold(
                    fold=fold,
                    x_train=x_tr,
                    y_train=y_tr,
                    p_train=p_tr,
                    x_test=x_te,
                    y_test=y_te,
                    p_test=p_te,
                    cfg=cfg,
                    run_dir=run_dir,
                    seed=int(args.seed),
                    tf=tf,
                    model=model,
                    logger=logger,
                )
            )

    rows = []
    for fr in fold_summaries:
        if task == "binary_stage2":
            rows.append(
                {
                    "fold": fr.fold,
                    "snippet_acc_fixed": fr.metrics_snippet_fixed["accuracy"],
                    "snippet_bal_acc_fixed": fr.metrics_snippet_fixed[
                        "balanced_accuracy"
                    ],
                    "snippet_f1_fixed": fr.metrics_snippet_fixed["f1"],
                    "snippet_auroc": fr.metrics_snippet_fixed.get("auroc"),
                    "snippet_auprc": fr.metrics_snippet_fixed.get("auprc"),
                    "patient_mean_f1_fixed": fr.metrics_patient_fixed_mean["f1"],
                    "patient_vote_f1_fixed": fr.metrics_patient_fixed_vote["f1"],
                    "threshold_train_best": fr.threshold_train_best,
                    "snippet_f1_train_best": fr.metrics_snippet_best["f1"],
                    "patient_mean_f1_train_best": fr.metrics_patient_best_mean["f1"],
                    "patient_vote_f1_train_best": fr.metrics_patient_best_vote["f1"],
                }
            )
        else:
            rows.append(
                {
                    "fold": fr.fold,
                    "snippet_acc": fr.metrics_snippet_fixed.get("accuracy"),
                    "snippet_bal_acc": fr.metrics_snippet_fixed.get(
                        "balanced_accuracy"
                    ),
                    "snippet_f1_macro": fr.metrics_snippet_fixed.get("f1_macro"),
                    "patient_mean_f1_macro": fr.metrics_patient_fixed_mean.get(
                        "f1_macro"
                    ),
                    "patient_vote_f1_macro": fr.metrics_patient_fixed_vote.get(
                        "f1_macro"
                    ),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    utils.save_json({"rows": rows}, os.path.join(run_dir, "summary.json"))
    logger.info("Done")


if __name__ == "__main__":
    main()
