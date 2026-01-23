import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from baselines.common import audio_features, data, metrics, utils
    from baselines.resnet_mel.model import resnet18
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from baselines.common import audio_features, data, metrics, utils
    from baselines.resnet_mel.model import resnet18


@dataclass
class FoldSummary:
    fold: int
    threshold_fixed: float
    threshold_train_best: float
    metrics_snippet_fixed: Dict[str, Any]
    metrics_patient_fixed_mean: Dict[str, Any]
    metrics_patient_fixed_vote: Dict[str, Any]
    metrics_snippet_best: Dict[str, Any]
    metrics_patient_best_mean: Dict[str, Any]
    metrics_patient_best_vote: Dict[str, Any]


class MelDataset(Dataset):
    def __init__(
        self,
        filepaths: List[str],
        labels: List[int],
        *,
        cfg_audio: audio_features.AudioConfig,
        mel_cfg: Dict[str, Any],
        normalize_stats: Tuple[float, float],
        augment: bool,
        aug_cfg: Dict[str, Any],
        cache_dir: str,
    ):
        self.filepaths = filepaths
        self.labels = np.asarray(labels).astype(int)
        self.cfg_audio = cfg_audio
        self.mel_cfg = mel_cfg
        self.mean, self.std = normalize_stats
        self.augment = augment
        self.aug_cfg = aug_cfg
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.filepaths)

    def _mask(self, x: np.ndarray) -> np.ndarray:
        if not self.augment:
            return x

        out = x.copy()
        n_mels, n_frames = out.shape

        t_frac = float(self.aug_cfg.get("time_mask", 0.0))
        f_frac = float(self.aug_cfg.get("freq_mask", 0.0))

        if t_frac > 0 and n_frames > 1:
            w = max(1, int(round(t_frac * n_frames)))
            start = np.random.randint(0, max(1, n_frames - w + 1))
            out[:, start : start + w] = float(np.mean(out))

        if f_frac > 0 and n_mels > 1:
            w = max(1, int(round(f_frac * n_mels)))
            start = np.random.randint(0, max(1, n_mels - w + 1))
            out[start : start + w, :] = float(np.mean(out))

        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fp = self.filepaths[idx]
        cache_path = os.path.join(self.cache_dir, f"mel_{utils.stable_hash(fp)}.npy")
        if os.path.exists(cache_path):
            mel = np.load(cache_path)
        else:
            y, sr = audio_features.load_audio_mono(fp, cfg=self.cfg_audio)
            mel = audio_features.log_mel_spectrogram(
                y,
                sr=sr,
                n_mels=int(self.mel_cfg["n_mels"]),
                n_fft=int(self.mel_cfg["n_fft"]),
                hop_length=int(self.mel_cfg["hop_length"]),
            )
            np.save(cache_path, mel)

        mel = self._mask(mel)
        mel = (mel - self.mean) / (self.std + 1e-8)
        x = torch.from_numpy(mel).unsqueeze(0).float()
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def _infer_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_norm_stats(
    filepaths: List[str],
    *,
    cfg_audio: audio_features.AudioConfig,
    mel_cfg: Dict[str, Any],
    cache_dir: str,
) -> Tuple[float, float]:
    vals: List[float] = []
    os.makedirs(cache_dir, exist_ok=True)

    for fp in filepaths:
        cache_path = os.path.join(cache_dir, f"mel_{utils.stable_hash(fp)}.npy")
        if os.path.exists(cache_path):
            mel = np.load(cache_path)
        else:
            y, sr = audio_features.load_audio_mono(fp, cfg=cfg_audio)
            mel = audio_features.log_mel_spectrogram(
                y,
                sr=sr,
                n_mels=int(mel_cfg["n_mels"]),
                n_fft=int(mel_cfg["n_fft"]),
                hop_length=int(mel_cfg["hop_length"]),
            )
            np.save(cache_path, mel)
        vals.append(float(np.mean(mel)))
        vals.append(float(np.std(mel)))

    mean = float(np.mean([v for i, v in enumerate(vals) if i % 2 == 0]))
    std = float(np.mean([v for i, v in enumerate(vals) if i % 2 == 1]))
    if std <= 0:
        std = 1.0
    return mean, std


def _split_train_val_by_patient(
    filepaths: List[str],
    labels: List[int],
    patient_ids: List[str],
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splits = data.make_splits(filepaths, labels, patient_ids, n_splits=5, seed=seed)
    tr_idx, va_idx = splits[0]
    return np.asarray(tr_idx), np.asarray(va_idx)


def _predict_proba(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[int] = []
    probs: List[float] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            ys.extend(y.numpy().astype(int).tolist())
            probs.extend(p.cpu().numpy().astype(float).tolist())

    return np.asarray(ys).astype(int), np.asarray(probs).astype(float)


def train_fold(
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
    logger,
) -> FoldSummary:
    cfg_audio = audio_features.AudioConfig(
        sample_rate=int(cfg["sample_rate"]),
        clip_duration_s=float(cfg["clip_duration_s"]),
    )

    device = _infer_device(str(cfg["train"]["device"]))
    mel_cfg = cfg["mel"]

    fold_dir = os.path.join(run_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    cache_dir = os.path.join(run_dir, "cache", f"fold{fold}")

    tr_idx, va_idx = _split_train_val_by_patient(x_train, y_train, p_train, seed=seed)

    x_tr = [x_train[i] for i in tr_idx]
    y_tr = [y_train[i] for i in tr_idx]
    x_va = [x_train[i] for i in va_idx]
    y_va = [y_train[i] for i in va_idx]

    mean, std = _compute_norm_stats(
        x_tr,
        cfg_audio=cfg_audio,
        mel_cfg=mel_cfg,
        cache_dir=os.path.join(cache_dir, "norm"),
    )

    ds_tr = MelDataset(
        x_tr,
        y_tr,
        cfg_audio=cfg_audio,
        mel_cfg=mel_cfg,
        normalize_stats=(mean, std),
        augment=bool(cfg["augmentation"]["enabled"]),
        aug_cfg=cfg["augmentation"],
        cache_dir=os.path.join(cache_dir, "train"),
    )
    ds_va = MelDataset(
        x_va,
        y_va,
        cfg_audio=cfg_audio,
        mel_cfg=mel_cfg,
        normalize_stats=(mean, std),
        augment=False,
        aug_cfg=cfg["augmentation"],
        cache_dir=os.path.join(cache_dir, "val"),
    )
    ds_te = MelDataset(
        x_test,
        y_test,
        cfg_audio=cfg_audio,
        mel_cfg=mel_cfg,
        normalize_stats=(mean, std),
        augment=False,
        aug_cfg=cfg["augmentation"],
        cache_dir=os.path.join(cache_dir, "test"),
    )

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=nw)

    model = resnet18(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    best_val = -1.0
    best_path = os.path.join(fold_dir, "best.pt")
    patience = int(cfg["train"]["patience"])
    bad = 0

    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(cfg["train"]["num_epochs"]) + 1):
        model.train()
        losses: List[float] = []

        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))

        y_va_true, y_va_proba = _predict_proba(model, dl_va, device)
        y_va_pred = (y_va_proba >= 0.5).astype(int)
        m_va = metrics.binary_metrics(y_va_true, y_va_pred, y_va_proba)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "val_bal_acc": m_va.get("balanced_accuracy"),
                "val_f1": m_va.get("f1"),
            }
        )

        score = float(m_va.get("balanced_accuracy", 0.0) or 0.0)
        logger.info(
            f"[fold {fold}] epoch {epoch} train_loss={history[-1]['train_loss']:.4f} val_bal_acc={score:.4f}"
        )

        if score > best_val:
            best_val = score
            bad = 0
            torch.save(
                {"model": model.state_dict(), "mean": mean, "std": std}, best_path
            )
        else:
            bad += 1
            if bad >= patience:
                break

    pd.DataFrame(history).to_csv(
        os.path.join(fold_dir, "train_history.csv"), index=False
    )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    y_te_true, y_te_proba = _predict_proba(model, dl_te, device)

    threshold_fixed = float(cfg["patient_aggregation"]["threshold"])
    threshold_best = metrics.find_best_threshold_f1(y_va_true, y_va_proba)

    pred_te_fixed = (y_te_proba >= threshold_fixed).astype(int)
    pred_te_best = (y_te_proba >= threshold_best).astype(int)

    snippet_fixed = metrics.binary_metrics(y_te_true, pred_te_fixed, y_te_proba)
    snippet_best = metrics.binary_metrics(y_te_true, pred_te_best, y_te_proba)

    yt_mean, yp_mean, pp_mean, _ = metrics.aggregate_by_patient(
        patient_ids=p_test,
        y_true=y_te_true,
        y_proba=y_te_proba,
        method="mean_prob",
        threshold=threshold_fixed,
    )
    patient_fixed_mean = metrics.binary_metrics(yt_mean, yp_mean, pp_mean)

    yt_vote, yp_vote, pp_vote, _ = metrics.aggregate_by_patient(
        patient_ids=p_test,
        y_true=y_te_true,
        y_proba=y_te_proba,
        method="majority_vote",
        threshold=threshold_fixed,
    )
    patient_fixed_vote = metrics.binary_metrics(yt_vote, yp_vote, pp_vote)

    yt_mean_b, yp_mean_b, pp_mean_b, _ = metrics.aggregate_by_patient(
        patient_ids=p_test,
        y_true=y_te_true,
        y_proba=y_te_proba,
        method="mean_prob",
        threshold=threshold_best,
    )
    patient_best_mean = metrics.binary_metrics(yt_mean_b, yp_mean_b, pp_mean_b)

    yt_vote_b, yp_vote_b, pp_vote_b, _ = metrics.aggregate_by_patient(
        patient_ids=p_test,
        y_true=y_te_true,
        y_proba=y_te_proba,
        method="majority_vote",
        threshold=threshold_best,
    )
    patient_best_vote = metrics.binary_metrics(yt_vote_b, yp_vote_b, pp_vote_b)

    utils.save_json(
        {
            "fold": fold,
            "threshold_fixed": threshold_fixed,
            "threshold_train_best": float(threshold_best),
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

    metrics.save_confusion_matrix_png(
        y_true=y_te_true,
        y_pred=pred_te_fixed,
        labels=["Healthy", "Zenker"],
        out_path=os.path.join(fold_dir, "confusion_snippet_fixed.png"),
        title=f"Snippet Confusion (fold {fold}, thr=0.5)",
    )

    metrics.save_roc_pr_curves(
        y_true=y_te_true, y_proba=y_te_proba, out_dir=fold_dir, prefix="snippet"
    )

    return FoldSummary(
        fold=fold,
        threshold_fixed=threshold_fixed,
        threshold_train_best=float(threshold_best),
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
        "--config", default=os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    ap.add_argument("--predefined_folds_dir", default=None)
    args = ap.parse_args()

    cfg = utils.load_yaml(args.config)
    task = str(cfg.get("task", "binary_stage2"))
    if task != "binary_stage2":
        raise ValueError(
            "resnet_mel baseline currently supports only task=binary_stage2"
        )

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
                train_fold(
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
                    logger=logger,
                )
            )
    else:
        if args.data_root is None:
            raise ValueError(
                "No predefined folds found (e.g. data_ast_stage2/) and --data_root was not provided."
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
                train_fold(
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
                    logger=logger,
                )
            )

    rows = []
    for fr in fold_summaries:
        rows.append(
            {
                "fold": fr.fold,
                "snippet_acc_fixed": fr.metrics_snippet_fixed["accuracy"],
                "snippet_bal_acc_fixed": fr.metrics_snippet_fixed["balanced_accuracy"],
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

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    utils.save_json({"rows": rows}, os.path.join(run_dir, "summary.json"))
    logger.info("Done")


if __name__ == "__main__":
    main()
