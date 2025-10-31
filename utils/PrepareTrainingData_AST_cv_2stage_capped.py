#!/usr/bin/env python3
"""
PrepareTrainingData_AST_cv_2stage_capped.py
-------------------------------------------
Creates Stage1 and Stage2 splits from base CV data, with CAPPING ONLY IN STAGE 2.

Key insight: Stage 1 (Idle vs Swallow) benefits from ALL swallow data.
The imbalance problem only affects Stage 2 (Healthy vs Zenker classification).

This script:
1. Reads base CV splits (data_ast_cv/) - UNCAPPED
2. Creates Stage 1 splits - NO CAPPING (want all swallow data)
3. Creates Stage 2 splits - WITH CAPPING (max N files per patient)
4. Uses random sampling (seeded for reproducibility)

Usage:
    python PrepareTrainingData_AST_cv_2stage_capped.py \
        --cv-dir ../data_ast_cv/ \
        --out-stage1 ../data_ast_stage1/ \
        --out-stage2 ../data_ast_stage2_capped/ \
        --max-files-per-patient 30 \
        --val-ratio 0.25
"""

import os
import json
import numpy as np
import argparse
from typing import Tuple
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv-dir', default=os.path.join(os.path.dirname(__file__), '../data_ast_cv/'),
                    help='Directory containing base CV splits (uncapped)')
    ap.add_argument('--out-stage1', default=os.path.join(os.path.dirname(__file__), '../data_ast_stage1/'))
    ap.add_argument('--out-stage2', default=os.path.join(os.path.dirname(__file__), '../data_ast_stage2_capped/'))
    ap.add_argument('--num-folds', type=int, default=5)
    ap.add_argument('--val-ratio', type=float, default=0.25, help='Fraction of train set for validation')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-mode', choices=['random', 'per-fold'], default='random')
    ap.add_argument('--max-files-per-patient', type=int, default=30,
                    help='Maximum files per patient IN STAGE 2 ONLY (default: 30)')
    return ap.parse_args()


def extract_patient_id(fp: str) -> str:
    """Extract patient ID from file path."""
    parts = str(fp).split('/')
    for i, token in enumerate(parts):
        if token in ("Idle", "Healthy", "Zenker") and i + 1 < len(parts):
            return f"{token}/{parts[i+1]}"
    return "UNKNOWN"


def cap_files_per_patient(x: np.ndarray, y: np.ndarray, max_files: int, seed: int):
    """
    Cap the number of files per patient using random sampling.
    
    Args:
        x: Array of file paths
        y: Array of labels
        max_files: Maximum files to keep per patient
        seed: Random seed for reproducibility
    
    Returns:
        (x_capped, y_capped, capping_stats)
    """
    rng = np.random.default_rng(seed)
    
    # Group files by patient
    patient_files = defaultdict(list)
    for idx, (file, label) in enumerate(zip(x, y)):
        pid = extract_patient_id(file)
        patient_files[pid].append((idx, file, label))
    
    # Cap each patient's files with random sampling
    selected_indices = []
    capping_stats = {
        'patients_capped': 0,
        'files_before': len(x),
        'files_after': 0,
        'files_removed': 0,
        'patient_details': {}
    }
    
    for pid, file_list in patient_files.items():
        original_count = len(file_list)
        
        if len(file_list) > max_files:
            # Randomly sample max_files from this patient
            indices_array = np.array([idx for idx, _, _ in file_list])
            sampled_indices = rng.choice(indices_array, size=max_files, replace=False)
            sampled = [(idx, x[idx], y[idx]) for idx in sampled_indices]
            
            capping_stats['patients_capped'] += 1
            capping_stats['patient_details'][pid] = {
                'before': original_count,
                'after': max_files,
                'removed': original_count - max_files
            }
        else:
            sampled = file_list
        
        selected_indices.extend([idx for idx, _, _ in sampled])
    
    # Sort indices to maintain order
    selected_indices.sort()
    capping_stats['files_after'] = len(selected_indices)
    capping_stats['files_removed'] = capping_stats['files_before'] - capping_stats['files_after']
    
    return x[selected_indices], y[selected_indices], capping_stats


def simple_dist(arr):
    u, c = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def patient_stratified_split(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
    """Split by patient IDs ensuring no patient leakage."""
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
    
    # Group patients by label
    label_to_patients = {}
    for pid, lbl in patient_label.items():
        label_to_patients.setdefault(lbl, []).append(pid)
    
    val_patients = set()
    train_patients = set()
    for lbl, plist in label_to_patients.items():
        plist_copy = plist.copy()
        rng.shuffle(plist_copy)
        val_count = int(round(len(plist_copy) * val_ratio))
        if val_count >= len(plist_copy):
            val_count = max(0, len(plist_copy) - 1)
        val_patients.update(plist_copy[:val_count])
        train_patients.update(plist_copy[val_count:])
    
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
    """Deterministic per-fold patient rotation."""
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
        
        # Rotate list by (fold-1)
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
    max_files = args.max_files_per_patient

    os.makedirs(out_stage1, exist_ok=True)
    os.makedirs(out_stage2, exist_ok=True)

    # Load class mapping
    mapping_path = os.path.join(cv_dir, 'class_mapping.json')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"class_mapping.json not found in {cv_dir}")
    
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)

    idle_idx = class_mapping['Idle']
    healthy_idx = class_mapping['Healthy']
    zenker_idx = class_mapping['Zenker']

    # Save capping config to Stage 2 directory
    with open(os.path.join(out_stage2, 'capping_config.json'), 'w') as f:
        json.dump({
            'max_files_per_patient': max_files,
            'capping_applied_to': 'Stage 2 only (Healthy vs Zenker)',
            'stage1_uncapped': True,
            'seed': seed,
            'val_ratio': val_ratio
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"STAGE 2 CAPPING ONLY: max={max_files} files per patient")
    print(f"Stage 1 (Idle vs Swallow): NO CAPPING - using all swallow data")
    print(f"{'='*80}\n")

    all_capping_stats = {}

    for fold in range(1, num_folds + 1):
        print(f"\n{'='*80}")
        print(f"Processing Fold {fold}")
        print(f"{'='*80}")
        
        train_x_path = os.path.join(cv_dir, f'train_x_fold{fold}.npy')
        train_y_path = os.path.join(cv_dir, f'train_y_fold{fold}.npy')
        test_x_path = os.path.join(cv_dir, f'test_x_fold{fold}.npy')
        test_y_path = os.path.join(cv_dir, f'test_y_fold{fold}.npy')

        if not all(os.path.exists(p) for p in [train_x_path, train_y_path, test_x_path, test_y_path]):
            print(f"[WARN] Missing base files for fold {fold}; skipping.")
            continue

        train_x = np.load(train_x_path, allow_pickle=True)
        train_y = np.load(train_y_path, allow_pickle=True)
        test_x = np.load(test_x_path, allow_pickle=True)
        test_y = np.load(test_y_path, allow_pickle=True)

        # ==========================================
        # Stage 1: Idle vs Swallow - NO CAPPING
        # ==========================================
        print(f"\n--- Stage 1: Idle vs Swallow (NO CAPPING) ---")
        
        stage1_full_labels = np.array([
            0 if y == idle_idx else (1 if y == healthy_idx else 2)
            for y in train_y
        ], dtype=int)
        train_y_stage1 = np.where(stage1_full_labels == 0, 0, 1)
        test_y_stage1 = np.where(np.array([
            0 if y == idle_idx else (1 if y == healthy_idx else 2)
            for y in test_y
        ], dtype=int) == 0, 0, 1)

        # Patient-level val split for Stage1 - NO CAPPING
        train_x_stage1_final = train_x
        train_y_stage1_full = stage1_full_labels
        train_y_stage1_final = train_y_stage1
        val_x_stage1 = np.empty((0,), dtype=object)
        val_y_stage1 = np.empty((0,), dtype=int)
        
        if val_ratio > 0:
            if val_mode == 'per-fold':
                (train_x_stage1_final, train_y_stage1_full, val_x_stage1, val_y_stage1_full,
                 _, _) = patient_per_fold_split(train_x, stage1_full_labels, val_ratio, fold)
            else:
                (train_x_stage1_final, train_y_stage1_full, val_x_stage1, val_y_stage1_full,
                 _, _) = patient_stratified_split(train_x, stage1_full_labels, val_ratio, seed)
            
            train_y_stage1_final = np.where(train_y_stage1_full == 0, 0, 1)
            val_y_stage1 = np.where(val_y_stage1_full == 0, 0, 1)

        print(f"Stage1: train={len(train_y_stage1_final)} val={len(val_y_stage1)} test={len(test_y_stage1)}")

        # Save Stage1 data (UNCAPPED)
        np.save(os.path.join(out_stage1, f'train_x_fold{fold}.npy'), train_x_stage1_final)
        np.save(os.path.join(out_stage1, f'train_y_fold{fold}.npy'), train_y_stage1_final)
        if val_ratio > 0:
            np.save(os.path.join(out_stage1, f'val_x_fold{fold}.npy'), val_x_stage1)
            np.save(os.path.join(out_stage1, f'val_y_fold{fold}.npy'), val_y_stage1)
        np.save(os.path.join(out_stage1, f'test_x_fold{fold}.npy'), test_x)
        np.save(os.path.join(out_stage1, f'test_y_fold{fold}.npy'), test_y_stage1)

        # ==========================================
        # Stage 2: Healthy vs Zenker - WITH CAPPING
        # ==========================================
        print(f"\n--- Stage 2: Healthy vs Zenker (WITH CAPPING) ---")
        
        # Filter out Idle samples
        train_swallow_mask = train_y != idle_idx
        test_swallow_mask = test_y != idle_idx
        
        train_x_stage2_uncapped = train_x[train_swallow_mask]
        train_y_stage2_uncapped = np.array([
            0 if y == healthy_idx else 1
            for y in train_y[train_swallow_mask]
        ], dtype=int)
        
        test_x_stage2_uncapped = test_x[test_swallow_mask]
        test_y_stage2_uncapped = np.array([
            0 if y == healthy_idx else 1
            for y in test_y[test_swallow_mask]
        ], dtype=int)

        print(f"\nBefore capping:")
        print(f"  Train: {len(train_y_stage2_uncapped)} files - {simple_dist(train_y_stage2_uncapped)}")
        print(f"  Test:  {len(test_y_stage2_uncapped)} files - {simple_dist(test_y_stage2_uncapped)}")

        # Apply capping to Stage 2 training data
        train_x_stage2_capped, train_y_stage2_capped, train_cap_stats = cap_files_per_patient(
            train_x_stage2_uncapped, train_y_stage2_uncapped, max_files, seed + fold
        )
        
        # Apply capping to Stage 2 test data
        test_x_stage2_capped, test_y_stage2_capped, test_cap_stats = cap_files_per_patient(
            test_x_stage2_uncapped, test_y_stage2_uncapped, max_files, seed + fold + 100
        )

        print(f"\nAfter capping:")
        print(f"  Train: {len(train_y_stage2_capped)} files ({train_cap_stats['files_removed']} removed)")
        print(f"         {simple_dist(train_y_stage2_capped)} - Patients capped: {train_cap_stats['patients_capped']}")
        print(f"  Test:  {len(test_y_stage2_capped)} files ({test_cap_stats['files_removed']} removed)")
        print(f"         {simple_dist(test_y_stage2_capped)} - Patients capped: {test_cap_stats['patients_capped']}")

        # Patient-driven Stage2 val split on CAPPED data
        train_x_stage2 = train_x_stage2_capped
        train_y_stage2 = train_y_stage2_capped
        val_x_stage2 = np.empty((0,), dtype=object)
        val_y_stage2 = np.empty((0,), dtype=int)
        
        if val_ratio > 0:
            split_fn = patient_per_fold_split if val_mode == 'per-fold' else patient_stratified_split
            (train_x_stage2, train_y_stage2, val_x_stage2, val_y_stage2, _, _) = split_fn(
                train_x_stage2_capped, train_y_stage2_capped, val_ratio,
                fold if val_mode == 'per-fold' else seed
            )

        print(f"\nFinal Stage2 splits (after val split):")
        print(f"  Train: {len(train_y_stage2)} - {simple_dist(train_y_stage2)}")
        print(f"  Val:   {len(val_y_stage2)} - {simple_dist(val_y_stage2) if len(val_y_stage2) else 'N/A'}")
        print(f"  Test:  {len(test_y_stage2_capped)} - {simple_dist(test_y_stage2_capped)}")

        # Save Stage2 data (CAPPED)
        np.save(os.path.join(out_stage2, f'train_x_fold{fold}.npy'), train_x_stage2)
        np.save(os.path.join(out_stage2, f'train_y_fold{fold}.npy'), train_y_stage2)
        if val_ratio > 0:
            np.save(os.path.join(out_stage2, f'val_x_fold{fold}.npy'), val_x_stage2)
            np.save(os.path.join(out_stage2, f'val_y_fold{fold}.npy'), val_y_stage2)
        np.save(os.path.join(out_stage2, f'test_x_fold{fold}.npy'), test_x_stage2_capped)
        np.save(os.path.join(out_stage2, f'test_y_fold{fold}.npy'), test_y_stage2_capped)

        # Save metadata
        meta = {
            'fold': fold,
            'stage2_capping': {
                'max_files_per_patient': max_files,
                'train_capping': train_cap_stats,
                'test_capping': test_cap_stats,
            },
            'stage2_train_distribution': simple_dist(train_y_stage2),
            'stage2_val_distribution': simple_dist(val_y_stage2) if len(val_y_stage2) else {},
            'stage2_test_distribution': simple_dist(test_y_stage2_capped),
            'stage2_num_train_files': int(len(train_x_stage2)),
            'stage2_num_val_files': int(len(val_x_stage2)),
            'stage2_num_test_files': int(len(test_x_stage2_capped)),
            'val_ratio': val_ratio,
            'seed': seed,
            'val_mode': val_mode,
        }
        
        with open(os.path.join(out_stage2, f'fold{fold}_meta.json'), 'w') as mf:
            json.dump(meta, mf, indent=2)

        all_capping_stats[f'fold{fold}'] = {
            'train': train_cap_stats,
            'test': test_cap_stats
        }

    # Save aggregate capping stats
    with open(os.path.join(out_stage2, 'capping_stats_all_folds.json'), 'w') as f:
        json.dump(all_capping_stats, f, indent=2)

    print(f"\n{'='*80}")
    print("Data preparation complete!")
    print(f"Stage 1 (uncapped): {out_stage1}")
    print(f"Stage 2 (capped):   {out_stage2}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
