import os
import json
import numpy as np

"""
PrepareTrainingData_AST_cv_2stage
---------------------------------
Generates two-stage (Stage1: Idle vs Swallow, Stage2: Healthy vs Zenker) datasets
from previously created single-stage multiclass CV artifacts produced by
`PrepareTrainingData_AST_cv.py`.

Assumptions:
- You have already run `PrepareTrainingData_AST_cv.py`, which produced:
  data_ast_cv/train_x_foldN.npy, train_y_foldN.npy, test_x_foldN.npy, test_y_foldN.npy
  and a class_mapping.json with Idle=0, Healthy=1, Zenker=2.
- File paths in train_x/test_x point to .wav files on disk.

Outputs:
- data_ast_stage1/ train_x_foldN.npy, train_y_foldN.npy, test_x_foldN.npy, test_y_foldN.npy, CSV label listings
    * Stage1 labels: 0 = Idle, 1 = Swallow (Healthy or Zenker)
- data_ast_stage2/ train_x_foldN.npy, train_y_foldN.npy, test_x_foldN.npy, test_y_foldN.npy, CSV label listings
    * Stage2 labels: 0 = Healthy, 1 = Zenker
- Extended metadata per fold saved into data_ast_stage2/foldN_2stage_meta.json (includes stage1 & stage2 distributions & warnings)

Usage:
    python PrepareTrainingData_AST_cv_2stage.py --cv-dir ../data_ast_cv/ \
        --out-stage1 ../data_ast_stage1/ --out-stage2 ../data_ast_stage2/ --num-folds 5

"""

import argparse
from typing import Tuple, Sequence

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv-dir', default=os.path.join(os.path.dirname(__file__), '../data_ast_cv/'),
                    help='Directory containing single-stage CV artifacts (from PrepareTrainingData_AST_cv.py)')
    ap.add_argument('--out-stage1', default=os.path.join(os.path.dirname(__file__), '../data_ast_stage1/'))
    ap.add_argument('--out-stage2', default=os.path.join(os.path.dirname(__file__), '../data_ast_stage2/'))
    ap.add_argument('--num-folds', type=int, default=5)
    ap.add_argument('--val-ratio', type=float, default=0.0, help='Fraction of train set reserved for validation (0 disables)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for validation split')
    ap.add_argument('--val-mode', choices=['random', 'per-fold'], default='random',
                    help='Validation patient selection strategy: random (seeded) or per-fold rotation')
    return ap.parse_args()


def simple_dist(arr):
    u, c = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def stratified_split(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio <= 0 or len(x) == 0:
        return x, y, np.empty((0,), dtype=object), np.empty((0,), dtype=y.dtype)
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    train_idx = []
    val_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        val_count = int(round(len(cls_idx) * val_ratio))
        if val_count > 0 and val_count < len(cls_idx):
            val_idx.extend(cls_idx[:val_count])
            train_idx.extend(cls_idx[val_count:])
        else:
            train_idx.extend(cls_idx)
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def extract_patient_id(fp: str) -> str:
    parts = str(fp).split('/')
    for i, token in enumerate(parts):
        if token in ("Idle", "Healthy", "Zenker") and i + 1 < len(parts):
            return parts[i+1]
    return "UNKNOWN"


def patient_stratified_split(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
    """Split by patient IDs (derived from path) ensuring no patient leakage."""
    if val_ratio <= 0:
        return x, y, np.empty((0,), dtype=object), np.empty((0,), dtype=y.dtype), set(), set()
    rng = np.random.default_rng(seed)
    patient_to_indices = {}
    for idx, fp in enumerate(x):
        pid = extract_patient_id(fp)
        patient_to_indices.setdefault(pid, []).append(idx)
    # Representative label per patient (majority)
    patient_label = {}
    for pid, indices in patient_to_indices.items():
        labels = y[indices]
        vals, counts = np.unique(labels, return_counts=True)
        patient_label[pid] = int(vals[np.argmax(counts)])
    # Group patients by representative label
    label_to_patients = {}
    for pid, lbl in patient_label.items():
        label_to_patients.setdefault(lbl, []).append(pid)
    val_patients = set()
    train_patients = set()
    for lbl, plist in label_to_patients.items():
        rng.shuffle(plist)
        val_count = int(round(len(plist) * val_ratio))
        # Ensure at least 1 stays in train if all would go to val
        if val_count >= len(plist):
            val_count = max(0, len(plist) - 1)
        val_patients.update(plist[:val_count])
        train_patients.update(plist[val_count:])
    # Build index lists
    val_indices = []
    train_indices = []
    for pid, indices in patient_to_indices.items():
        target = val_indices if pid in val_patients else train_indices
        target.extend(indices)
    train_indices = np.array(sorted(train_indices))
    val_indices = np.array(sorted(val_indices))
    return x[train_indices], y[train_indices], x[val_indices], y[val_indices], train_patients, val_patients


def patient_per_fold_split(x: np.ndarray, y: np.ndarray, val_ratio: float, fold: int):
    """Deterministic per-fold patient rotation; fold is 1-based."""
    if val_ratio <= 0:
        return x, y, np.empty((0,), dtype=object), np.empty((0,), dtype=y.dtype), set(), set()
    patient_to_indices = {}
    for idx, fp in enumerate(x):
        pid = extract_patient_id(fp)
        patient_to_indices.setdefault(pid, []).append(idx)
    # Majority label per patient
    patient_label = {}
    for pid, indices in patient_to_indices.items():
        labels = y[indices]
        vals, counts = np.unique(labels, return_counts=True)
        patient_label[pid] = int(vals[np.argmax(counts)])
    label_to_patients = {}
    for pid, lbl in patient_label.items():
        label_to_patients.setdefault(lbl, []).append(pid)
    val_patients = set()
    train_patients = set()
    for lbl, plist in label_to_patients.items():
        plist_sorted = sorted(plist)
        val_count = int(round(len(plist_sorted) * val_ratio))
        if val_count >= len(plist_sorted):
            val_count = max(0, len(plist_sorted) - 1)
        # Rotate list by (fold-1) to vary which patients selected
        if len(plist_sorted) > 0:
            rot = (fold - 1) % len(plist_sorted)
            rotated = plist_sorted[rot:] + plist_sorted[:rot]
        else:
            rotated = plist_sorted
        val_patients.update(rotated[:val_count])
        train_patients.update(rotated[val_count:])
    train_indices = []
    val_indices = []
    for pid, indices in patient_to_indices.items():
        (val_indices if pid in val_patients else train_indices).extend(indices)
    train_indices = np.array(sorted(train_indices))
    val_indices = np.array(sorted(val_indices))
    return x[train_indices], y[train_indices], x[val_indices], y[val_indices], train_patients, val_patients


def main():
    args = parse_args()
    cv_dir = args.cv_dir
    out_stage1 = args.out_stage1
    out_stage2 = args.out_stage2
    num_folds = args.num_folds
    val_ratio = args.val_ratio
    seed = args.seed
    val_mode = args.val_mode

    os.makedirs(out_stage1, exist_ok=True)
    os.makedirs(out_stage2, exist_ok=True)

    # Load class mapping
    mapping_path = os.path.join(cv_dir, 'class_mapping.json')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"class_mapping.json not found in {cv_dir}; run PrepareTrainingData_AST_cv.py first")
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)

    idle_idx = class_mapping['Idle']
    healthy_idx = class_mapping['Healthy']
    zenker_idx = class_mapping['Zenker']

    stage2_label_mapping = {"Healthy": 0, "Zenker": 1}

    for fold in range(1, num_folds + 1):
        train_x_path = os.path.join(cv_dir, f'train_x_fold{fold}.npy')
        train_y_path = os.path.join(cv_dir, f'train_y_fold{fold}.npy')
        test_x_path  = os.path.join(cv_dir, f'test_x_fold{fold}.npy')
        test_y_path  = os.path.join(cv_dir, f'test_y_fold{fold}.npy')

        if not all(os.path.exists(p) for p in [train_x_path, train_y_path, test_x_path, test_y_path]):
            print(f"[WARN] Missing one or more base files for fold {fold}; skipping.")
            continue

        train_x = np.load(train_x_path, allow_pickle=True)
        train_y = np.load(train_y_path, allow_pickle=True)
        test_x  = np.load(test_x_path, allow_pickle=True)
        test_y  = np.load(test_y_path, allow_pickle=True)

        # Stage 1: Idle vs Swallow
        stage1_full_labels = np.array([  # Idle=0, Healthy=1, Zenker=2 for splitting
            0 if y == idle_idx else (1 if y == healthy_idx else 2)
            for y in train_y
        ], dtype=int)
        train_y_stage1 = np.where(stage1_full_labels == 0, 0, 1)
        test_y_stage1 = np.where(np.array([
            0 if y == idle_idx else (1 if y == healthy_idx else 2)
            for y in test_y
        ], dtype=int) == 0, 0, 1)

        # Patient-level split (Stage1) – stratify by full class so Healthy/Zenker balance val
        train_x_stage1_final = train_x
        train_y_stage1_full = stage1_full_labels
        train_y_stage1_final = train_y_stage1
        val_x_stage1 = np.empty((0,), dtype=object)
        val_y_stage1 = np.empty((0,), dtype=int)
        train_patients_stage1 = set()
        val_patients_stage1 = set()
        if val_ratio > 0:
            if val_mode == 'per-fold':
                (train_x_stage1_final,
                 train_y_stage1_full,
                 val_x_stage1,
                 val_y_stage1_full,
                 train_patients_stage1,
                 val_patients_stage1) = patient_per_fold_split(train_x, stage1_full_labels, val_ratio, fold)
            else:
                (train_x_stage1_final,
                 train_y_stage1_full,
                 val_x_stage1,
                 val_y_stage1_full,
                 train_patients_stage1,
                 val_patients_stage1) = patient_stratified_split(train_x, stage1_full_labels, val_ratio, seed)
            train_y_stage1_final = np.where(train_y_stage1_full == 0, 0, 1)
            val_y_stage1 = np.where(val_y_stage1_full == 0, 0, 1)

        # Stage 2 base (exclude Idle)
        train_swallow_mask = train_y != idle_idx
        test_swallow_mask  = test_y != idle_idx
        train_x_stage2_base = train_x[train_swallow_mask]
        train_y_stage2_base = np.array([0 if y == healthy_idx else 1 for y in train_y[train_swallow_mask]], dtype=int)
        test_x_stage2  = test_x[test_swallow_mask]
        test_y_stage2  = np.array([0 if y == healthy_idx else 1 for y in test_y[test_swallow_mask]], dtype=int)

        # Patient-driven Stage2 split (independent from Stage1)
        train_x_stage2 = train_x_stage2_base
        train_y_stage2 = train_y_stage2_base
        val_x_stage2 = np.empty((0,), dtype=object)
        val_y_stage2 = np.empty((0,), dtype=int)
        train_patients_stage2 = sorted({extract_patient_id(fp) for fp in train_x_stage2})
        val_patients_stage2 = []
        if val_ratio > 0:
            split_fn = patient_per_fold_split if val_mode == 'per-fold' else patient_stratified_split
            (train_x_stage2,
             train_y_stage2,
             val_x_stage2,
             val_y_stage2,
             train_patients_stage2_set,
             val_patients_stage2_set) = split_fn(train_x_stage2_base, train_y_stage2_base, val_ratio, fold if val_mode == 'per-fold' else seed)
            train_patients_stage2 = sorted(list(train_patients_stage2_set))
            val_patients_stage2 = sorted(list(val_patients_stage2_set))

        # Safety warnings
        warnings_stage2 = []
        if len(set(train_y_stage2.tolist())) < 2:
            warnings_stage2.append('Stage2 train missing one of the classes')
        if len(set(test_y_stage2.tolist())) < 2:
            warnings_stage2.append('Stage2 test missing one of the classes')
        if val_ratio > 0 and len(val_x_stage2) and len(set(val_y_stage2.tolist())) < 2:
            warnings_stage2.append('Stage2 val missing one of the classes')
        if warnings_stage2:
            print(f"[WARN][Fold {fold}] {'; '.join(warnings_stage2)}")

        # Specimen IDs (unchanged logic)
        def derive_specimen_ids(file_array: np.ndarray):
            ids = set()
            for fp in file_array:
                parts = str(fp).split('/')
                for i, token in enumerate(parts):
                    if token in ("Idle", "Healthy", "Zenker") and i + 1 < len(parts):
                        ids.add(f"{token}/{parts[i+1]}")
                        break
            return sorted(ids)

        stage1_train_ids = derive_specimen_ids(train_x_stage1_final)
        stage1_val_ids   = derive_specimen_ids(val_x_stage1) if len(val_x_stage1) else []
        stage1_test_ids  = derive_specimen_ids(test_x)
        stage2_train_ids = derive_specimen_ids(train_x_stage2)
        stage2_val_ids   = derive_specimen_ids(val_x_stage2) if len(val_x_stage2) else []
        stage2_test_ids  = derive_specimen_ids(test_x_stage2)

        # Persist Stage1
        np.save(os.path.join(out_stage1, f'train_x_fold{fold}.npy'), train_x_stage1_final)
        np.save(os.path.join(out_stage1, f'train_y_fold{fold}.npy'), train_y_stage1_final)
        if val_ratio > 0:
            np.save(os.path.join(out_stage1, f'val_x_fold{fold}.npy'), val_x_stage1)
            np.save(os.path.join(out_stage1, f'val_y_fold{fold}.npy'), val_y_stage1)
        np.save(os.path.join(out_stage1, f'test_x_fold{fold}.npy'), test_x)
        np.save(os.path.join(out_stage1, f'test_y_fold{fold}.npy'), test_y_stage1)
        # CSVs
        with open(os.path.join(out_stage1, f'train_stage1_labels_fold{fold}.csv'), 'w') as f:
            f.write('file,label_stage1\n')
            for file, lbl in zip(train_x_stage1_final, train_y_stage1_final):
                f.write(f"{file},{lbl}\n")
        if val_ratio > 0:
            with open(os.path.join(out_stage1, f'val_stage1_labels_fold{fold}.csv'), 'w') as f:
                f.write('file,label_stage1\n')
                for file, lbl in zip(val_x_stage1, val_y_stage1):
                    f.write(f"{file},{lbl}\n")
        with open(os.path.join(out_stage1, f'test_stage1_labels_fold{fold}.csv'), 'w') as f:
            f.write('file,label_stage1\n')
            for file, lbl in zip(test_x, test_y_stage1):
                f.write(f"{file},{lbl}\n")
        # IDs
        with open(os.path.join(out_stage1, f'train_ids_fold{fold}.txt'), 'w') as f:
            for sid in stage1_train_ids:
                f.write(sid + '\n')
        if val_ratio > 0:
            with open(os.path.join(out_stage1, f'val_ids_fold{fold}.txt'), 'w') as f:
                for sid in stage1_val_ids:
                    f.write(sid + '\n')
        with open(os.path.join(out_stage1, f'test_ids_fold{fold}.txt'), 'w') as f:
            for sid in stage1_test_ids:
                f.write(sid + '\n')

        # Persist Stage2
        np.save(os.path.join(out_stage2, f'train_x_fold{fold}.npy'), train_x_stage2)
        np.save(os.path.join(out_stage2, f'train_y_fold{fold}.npy'), train_y_stage2)
        if val_ratio > 0:
            np.save(os.path.join(out_stage2, f'val_x_fold{fold}.npy'), val_x_stage2)
            np.save(os.path.join(out_stage2, f'val_y_fold{fold}.npy'), val_y_stage2)
        np.save(os.path.join(out_stage2, f'test_x_fold{fold}.npy'), test_x_stage2)
        np.save(os.path.join(out_stage2, f'test_y_fold{fold}.npy'), test_y_stage2)
        with open(os.path.join(out_stage2, f'train_stage2_labels_fold{fold}.csv'), 'w') as f:
            f.write('file,label_stage2\n')
            for file, lbl in zip(train_x_stage2, train_y_stage2):
                f.write(f"{file},{lbl}\n")
        if val_ratio > 0:
            with open(os.path.join(out_stage2, f'val_stage2_labels_fold{fold}.csv'), 'w') as f:
                f.write('file,label_stage2\n')
                for file, lbl in zip(val_x_stage2, val_y_stage2):
                    f.write(f"{file},{lbl}\n")
        with open(os.path.join(out_stage2, f'test_stage2_labels_fold{fold}.csv'), 'w') as f:
            f.write('file,label_stage2\n')
            for file, lbl in zip(test_x_stage2, test_y_stage2):
                f.write(f"{file},{lbl}\n")
        with open(os.path.join(out_stage2, f'train_ids_fold{fold}.txt'), 'w') as f:
            for sid in stage2_train_ids:
                f.write(sid + '\n')
        if val_ratio > 0:
            with open(os.path.join(out_stage2, f'val_ids_fold{fold}.txt'), 'w') as f:
                for sid in stage2_val_ids:
                    f.write(sid + '\n')
        with open(os.path.join(out_stage2, f'test_ids_fold{fold}.txt'), 'w') as f:
            for sid in stage2_test_ids:
                f.write(sid + '\n')

        # Metadata (add patient sets)
        meta = {
            'fold': fold,
            'stage1_train_distribution': simple_dist(train_y_stage1_final),
            'stage1_test_distribution': simple_dist(test_y_stage1),
            'stage1_val_distribution': simple_dist(val_y_stage1) if len(val_y_stage1) else {},
            'stage1_num_train_files': int(len(train_x_stage1_final)),
            'stage1_num_val_files': int(len(val_x_stage1)),
            'stage1_num_test_files': int(len(test_x)),
            'stage2_num_train_files': int(len(train_x_stage2)),
            'stage2_num_val_files': int(len(val_x_stage2)),
            'stage2_num_test_files': int(len(test_x_stage2)),
            'stage2_train_distribution': simple_dist(train_y_stage2),
            'stage2_val_distribution': simple_dist(val_y_stage2) if len(val_y_stage2) else {},
            'stage2_test_distribution': simple_dist(test_y_stage2),
            'stage2_label_mapping': stage2_label_mapping,
            'stage2_warnings': warnings_stage2,
            'base_class_mapping': class_mapping,
            'val_ratio': val_ratio,
            'seed': seed,
            'val_mode': val_mode,
            'stage1_train_patients': sorted(list(train_patients_stage1)),
            'stage1_val_patients': sorted(list(val_patients_stage1)),
            'stage2_train_patients': sorted(list({extract_patient_id(fp) for fp in train_x_stage2})),
            'stage2_val_patients': sorted(list({extract_patient_id(fp) for fp in val_x_stage2})) if len(val_x_stage2) else [],
        }
        with open(os.path.join(out_stage2, f'fold{fold}_2stage_meta.json'), 'w') as mf:
            json.dump(meta, mf, indent=2)

        print(f"Fold {fold}: Stage1 train={len(train_y_stage1_final)} val={len(val_y_stage1)} test={len(test_y_stage1)} | Stage2 train={len(train_y_stage2)} val={len(val_y_stage2)} test={len(test_y_stage2)}")

    print("Done generating two-stage artifacts.")

if __name__ == '__main__':
    main()
