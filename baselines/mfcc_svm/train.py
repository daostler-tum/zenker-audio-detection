import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


def _extract_features(
    filepaths: List[str],
    *,
    cfg_audio: audio_features.AudioConfig,
    n_mfcc: int,
    include_deltas: bool,
    n_fft: int,
    hop_length: int,
    cache_dir: str,
) -> np.ndarray:
    feats: List[np.ndarray] = []
    os.makedirs(cache_dir, exist_ok=True)

    for fp in filepaths:
        cache_path = os.path.join(cache_dir, f"mfcc_{utils.stable_hash(fp)}.npy")
        if os.path.exists(cache_path):
            feats.append(np.load(cache_path))
            continue

        y, sr = audio_features.load_audio_mono(fp, cfg=cfg_audio)
        mfcc = audio_features.mfcc_features(
            y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            include_deltas=include_deltas,
        )
        v = audio_features.aggregate_time_stats(mfcc)
        np.save(cache_path, v)
        feats.append(v)

    return np.stack(feats, axis=0)


def _inner_group_split(
    patient_ids: List[str], labels: List[int], *, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    splits = data.make_splits(
        np.zeros(len(labels)), labels, patient_ids, n_splits=5, seed=seed
    )
    tr, te = splits[0]
    return np.asarray(tr), np.asarray(te)


def run_fold(
    *,
    fold: int,
    filepaths_train: List[str],
    y_train: List[int],
    pids_train: List[str],
    filepaths_test: List[str],
    y_test: List[int],
    pids_test: List[str],
    cfg: Dict[str, Any],
    run_dir: str,
    seed: int,
) -> FoldResult:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    task = str(cfg.get("task", "binary_stage2"))
    if task not in ("binary_stage2", "multiclass"):
        raise ValueError(f"Unsupported task: {task}")

    cfg_audio = audio_features.AudioConfig(
        sample_rate=int(cfg["sample_rate"]),
        clip_duration_s=float(cfg["clip_duration_s"]),
    )

    cache_dir = os.path.join(run_dir, "cache", f"fold{fold}")

    X_train = _extract_features(
        filepaths_train,
        cfg_audio=cfg_audio,
        n_mfcc=int(cfg["n_mfcc"]),
        include_deltas=bool(cfg["include_deltas"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        cache_dir=os.path.join(cache_dir, "train"),
    )
    X_test = _extract_features(
        filepaths_test,
        cfg_audio=cfg_audio,
        n_mfcc=int(cfg["n_mfcc"]),
        include_deltas=bool(cfg["include_deltas"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        cache_dir=os.path.join(cache_dir, "test"),
    )

    y_train_np = np.asarray(y_train).astype(int)
    y_test_np = np.asarray(y_test).astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_params: Dict[str, Any] = {}
    if bool(cfg["svm"]["grid_search"]):
        tr_idx, va_idx = _inner_group_split(pids_train, y_train, seed=seed)
        X_tr, y_tr = X_train_s[tr_idx], y_train_np[tr_idx]
        X_va, y_va = X_train_s[va_idx], y_train_np[va_idx]

        best_score = -1.0
        for C in cfg["svm"]["param_grid"]["C"]:
            for gamma in cfg["svm"]["param_grid"]["gamma"]:
                clf = SVC(
                    C=float(C),
                    gamma=gamma,
                    kernel="rbf",
                    probability=bool(cfg["svm"]["probability"]),
                    class_weight=cfg["svm"]["class_weight"],
                    random_state=seed,
                )
                clf.fit(X_tr, y_tr)
                proba_va = clf.predict_proba(X_va)[:, 1]
                pred_va = (proba_va >= 0.5).astype(int)
                m = metrics.binary_metrics(y_va, pred_va, proba_va)
                score = float(m["balanced_accuracy"])
                if score > best_score:
                    best_score = score
                    best_params = {"C": float(C), "gamma": gamma}

    clf = SVC(
        C=float(best_params.get("C", 10.0)),
        gamma=best_params.get("gamma", "scale"),
        kernel="rbf",
        probability=bool(cfg["svm"]["probability"]),
        class_weight=cfg["svm"]["class_weight"],
        random_state=seed,
    )
    clf.fit(X_train_s, y_train_np)

    if task == "binary_stage2":
        proba_train = clf.predict_proba(X_train_s)[:, 1]
        proba_test = clf.predict_proba(X_test_s)[:, 1]

        threshold_fixed = float(cfg["patient_aggregation"]["threshold"])
        threshold_best = 0.5
        if bool(cfg["threshold_tuning"]["enabled"]):
            threshold_best = metrics.find_best_threshold_f1(y_train_np, proba_train)

        pred_test_fixed = (proba_test >= threshold_fixed).astype(int)
        pred_test_best = (proba_test >= threshold_best).astype(int)

        snippet_fixed = metrics.binary_metrics(y_test_np, pred_test_fixed, proba_test)
        snippet_best = metrics.binary_metrics(y_test_np, pred_test_best, proba_test)

        yt_mean, yp_mean, pp_mean, _ = metrics.aggregate_by_patient(
            patient_ids=pids_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="mean_prob",
            threshold=threshold_fixed,
        )
        patient_fixed_mean = metrics.binary_metrics(yt_mean, yp_mean, pp_mean)

        yt_vote, yp_vote, pp_vote, _ = metrics.aggregate_by_patient(
            patient_ids=pids_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="majority_vote",
            threshold=threshold_fixed,
        )
        patient_fixed_vote = metrics.binary_metrics(yt_vote, yp_vote, pp_vote)

        yt_mean_b, yp_mean_b, pp_mean_b, _ = metrics.aggregate_by_patient(
            patient_ids=pids_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="mean_prob",
            threshold=threshold_best,
        )
        patient_best_mean = metrics.binary_metrics(yt_mean_b, yp_mean_b, pp_mean_b)

        yt_vote_b, yp_vote_b, pp_vote_b, _ = metrics.aggregate_by_patient(
            patient_ids=pids_test,
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
            patient_ids=pids_test,
            y_true=y_test_np,
            y_proba=proba_test,
            method="mean_prob",
        )
        patient_fixed_mean = metrics.multiclass_metrics(
            yt_mean, yp_mean, y_proba=pp_mean
        )

        yt_vote, yp_vote, pp_vote, _ = metrics.aggregate_by_patient_multiclass(
            patient_ids=pids_test,
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

    fold_dir = os.path.join(run_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    utils.save_json(
        {
            "fold": fold,
            "best_params": best_params,
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
            "best_params": best_params,
            "threshold_fixed": threshold_fixed,
            "threshold_train_best": threshold_best,
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


def _load_predefined_folds(task: str, *, n_splits: int) -> Optional[str]:
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
    cfg_task = str(args.task or cfg.get("task", "binary_stage2"))
    if cfg_task not in ("binary_stage2", "multiclass"):
        raise ValueError(f"Unsupported task: {cfg_task}")

    cfg["task"] = cfg_task

    n_folds = int(args.n_folds)
    if args.fold is not None and not (1 <= int(args.fold) <= n_folds):
        raise ValueError(f"--fold must be in [1, {n_folds}]")

    utils.set_seed(int(args.seed))
    run_dir = utils.resolve_run_dir(args.output_dir, args.run_name)
    logger = utils.setup_logger(os.path.join(run_dir, "train.log"))

    utils.save_yaml(cfg, os.path.join(run_dir, "config_used.yaml"))

    predefined_dir = args.predefined_folds_dir
    if predefined_dir is None:
        default = _load_predefined_folds(cfg_task, n_splits=n_folds)
        if default is not None:
            candidate = os.path.join(os.path.dirname(__file__), "..", "..", default)
            if os.path.exists(candidate):
                predefined_dir = os.path.abspath(candidate)

    fold_results: List[FoldResult] = []

    if predefined_dir is not None and os.path.exists(predefined_dir):
        logger.info(f"Using predefined folds from: {predefined_dir}")
        folds_to_run = (
            [int(args.fold)] if args.fold is not None else list(range(1, n_folds + 1))
        )
        for fold in folds_to_run:
            x_tr, y_tr = data.load_predefined_fold_from_numpy(
                folds_dir=predefined_dir, fold=fold, split="train"
            )
            x_te, y_te = data.load_predefined_fold_from_numpy(
                folds_dir=predefined_dir, fold=fold, split="test"
            )

            p_tr = [data.parse_patient_id(p) for p in x_tr]
            p_te = [data.parse_patient_id(p) for p in x_te]

            fold_results.append(
                run_fold(
                    fold=fold,
                    filepaths_train=x_tr,
                    y_train=y_tr,
                    pids_train=p_tr,
                    filepaths_test=x_te,
                    y_test=y_te,
                    pids_test=p_te,
                    cfg=cfg,
                    run_dir=run_dir,
                    seed=int(args.seed),
                )
            )
    else:
        if args.data_root is None:
            raise ValueError(
                "No predefined folds found (e.g. data_ast_stage2/ or data_ast_cv/) and --data_root was not provided."
            )
        logger.info("Using on-the-fly patient-level folds")
        ds = data.index_dataset(
            data_root=args.data_root, label_csv=args.label_csv, task=cfg_task
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
        for i, (tr_idx, te_idx) in enumerate(splits, start=start_fold):
            x_tr = [ds.filepaths[j] for j in tr_idx]
            y_tr = [ds.labels[j] for j in tr_idx]
            p_tr = [ds.patient_ids[j] for j in tr_idx]

            x_te = [ds.filepaths[j] for j in te_idx]
            y_te = [ds.labels[j] for j in te_idx]
            p_te = [ds.patient_ids[j] for j in te_idx]

            fold_results.append(
                run_fold(
                    fold=i,
                    filepaths_train=x_tr,
                    y_train=y_tr,
                    pids_train=p_tr,
                    filepaths_test=x_te,
                    y_test=y_te,
                    pids_test=p_te,
                    cfg=cfg,
                    run_dir=run_dir,
                    seed=int(args.seed),
                )
            )

    rows = []
    for fr in fold_results:
        if cfg_task == "binary_stage2":
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
