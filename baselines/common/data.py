import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetIndex:
    filepaths: List[str]
    labels: List[int]
    patient_ids: List[str]
    label_mapping: Dict[str, int]


def parse_patient_id(filepath: str) -> str:
    """Extract patient/subject id deterministically from a filepath.

    Preferred convention (as in this repo README):
        <data_root>/<Class>/<PatientID>/<file>.wav

    If not found, falls back to extracting the first digit-group from filename.
    """

    parts = Path(filepath).parts
    for i, token in enumerate(parts):
        if token in ("Idle", "Healthy", "Zenker") and i + 1 < len(parts):
            return str(parts[i + 1])

    fname = Path(filepath).name
    m = re.search(r"(\d+)", fname)
    if m:
        return m.group(1)
    return "UNKNOWN"


def _index_from_folder(
    data_root: str, *, classes: Sequence[str], label_mapping: Dict[str, int]
) -> DatasetIndex:
    filepaths: List[str] = []
    labels: List[int] = []
    patient_ids: List[str] = []

    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for patient in sorted(os.listdir(cls_dir)):
            patient_dir = os.path.join(cls_dir, patient)
            if not os.path.isdir(patient_dir):
                continue
            for fname in sorted(os.listdir(patient_dir)):
                if not fname.lower().endswith(".wav"):
                    continue
                fp = os.path.join(patient_dir, fname)
                filepaths.append(fp)
                labels.append(int(label_mapping[cls]))
                patient_ids.append(str(patient))

    return DatasetIndex(
        filepaths=filepaths,
        labels=labels,
        patient_ids=patient_ids,
        label_mapping=label_mapping,
    )


def index_dataset(
    *,
    data_root: str,
    label_csv: Optional[str],
    task: str,
) -> DatasetIndex:
    """Create a dataset index.

    task:
        - "binary_stage2": Healthy vs Zenker, labels {Healthy:0, Zenker:1}
        - "multiclass": Idle vs Healthy vs Zenker, labels {Idle:0, Healthy:1, Zenker:2}
    """

    if task not in ("binary_stage2", "multiclass"):
        raise ValueError(f"Unsupported task: {task}")

    if task == "binary_stage2":
        classes = ("Healthy", "Zenker")
        label_mapping = {"Healthy": 0, "Zenker": 1}
    else:
        classes = ("Idle", "Healthy", "Zenker")
        label_mapping = {"Idle": 0, "Healthy": 1, "Zenker": 2}

    if label_csv is None:
        return _index_from_folder(
            data_root, classes=classes, label_mapping=label_mapping
        )

    df = pd.read_csv(label_csv)
    if "file" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "label_csv must contain columns: file,label (and optional patient_id)"
        )

    files = df["file"].astype(str).tolist()
    labels_raw = df["label"].tolist()

    resolved_files: List[str] = []
    for f in files:
        fp = f if os.path.isabs(f) else os.path.join(data_root, f)
        resolved_files.append(fp)

    if df["label"].dtype == object:
        labels = [int(label_mapping[str(x)]) for x in labels_raw]
    else:
        labels = [int(x) for x in labels_raw]

    if "patient_id" in df.columns:
        patient_ids = df["patient_id"].astype(str).tolist()
    else:
        patient_ids = [parse_patient_id(fp) for fp in resolved_files]

    if task == "binary_stage2":
        keep = [i for i, y in enumerate(labels) if y in (0, 1)]
        resolved_files = [resolved_files[i] for i in keep]
        labels = [labels[i] for i in keep]
        patient_ids = [patient_ids[i] for i in keep]

    return DatasetIndex(
        filepaths=resolved_files,
        labels=labels,
        patient_ids=patient_ids,
        label_mapping=label_mapping,
    )


def _stratified_group_folds(
    *,
    labels: Sequence[int],
    patient_ids: Sequence[str],
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create stratified folds at patient level.

    We stratify on a per-patient representative label (majority label).
    """

    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels)
    patient_ids = np.asarray(patient_ids)

    patient_to_indices: Dict[str, List[int]] = {}
    for i, pid in enumerate(patient_ids):
        patient_to_indices.setdefault(str(pid), []).append(i)

    patients = sorted(patient_to_indices.keys())
    patient_labels: List[int] = []
    for pid in patients:
        ys = labels[patient_to_indices[pid]]
        vals, counts = np.unique(ys, return_counts=True)
        patient_labels.append(int(vals[np.argmax(counts)]))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    for train_p_idx, test_p_idx in skf.split(
        np.array(patients), np.array(patient_labels)
    ):
        train_patients = set(np.array(patients)[train_p_idx].tolist())
        test_patients = set(np.array(patients)[test_p_idx].tolist())

        train_idx: List[int] = []
        test_idx: List[int] = []
        for pid, idxs in patient_to_indices.items():
            if pid in train_patients:
                train_idx.extend(idxs)
            elif pid in test_patients:
                test_idx.extend(idxs)

        folds.append((np.array(sorted(train_idx)), np.array(sorted(test_idx))))

    return folds


def make_splits(
    filepaths: Sequence[str],
    labels: Sequence[int],
    patient_ids: Sequence[str],
    *,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, test_idx) arrays."""

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(sgkf.split(np.zeros(len(labels)), labels, groups=patient_ids))
        return [(np.asarray(tr), np.asarray(te)) for tr, te in splits]
    except Exception:
        return _stratified_group_folds(
            labels=labels, patient_ids=patient_ids, n_splits=n_splits, seed=seed
        )


def load_predefined_fold_from_numpy(
    *,
    folds_dir: str,
    fold: int,
    split: str,
) -> Tuple[List[str], List[int]]:
    """Load (filepaths, labels) for a predefined fold.

    Expects numpy arrays named like:
        train_x_fold{fold}.npy, train_y_fold{fold}.npy
        test_x_fold{fold}.npy,  test_y_fold{fold}.npy

    The arrays are expected to contain absolute filepaths.
    """

    if split not in ("train", "test", "val"):
        raise ValueError(f"Unsupported split: {split}")

    x_name = f"{split}_x_fold{fold}.npy"
    y_name = f"{split}_y_fold{fold}.npy"

    x_path = os.path.join(folds_dir, x_name)
    y_path = os.path.join(folds_dir, y_name)

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"Missing predefined fold files: {x_path} / {y_path}")

    xs = np.load(x_path, allow_pickle=True).tolist()
    ys = np.load(y_path, allow_pickle=True).astype(int).tolist()
    return [str(x) for x in xs], [int(y) for y in ys]
