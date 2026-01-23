# Baseline models

These baselines are intended as reviewer-facing comparisons with consistent:
- patient-level splits (no leakage)
- snippet-level and patient-level metrics
- saved artifacts per run

All baselines expect the dataset layout described in the top-level README:

```
<data_root>/
  Healthy/<patient_id>/*.wav
  Zenker/<patient_id>/*.wav
  Idle/<patient_id>/*.wav   (optional)
```

## Output structure

All scripts write to:

- `<output_dir>/<run_name_or_timestamp>/fold{k}/...`

Per fold:
- `metrics_snippet.json`
- `metrics_patient.json`
- confusion matrix PNG(s)
- ROC/PR curves (binary)

## Recommended usage (predefined folds)

If you have the repo-generated fold artifacts available locally, the baselines will prefer them:

- `data_ast_stage2/` for the binary task (Healthy vs Zenker)
- `data_ast_cv/` for the multiclass task (Idle/Healthy/Zenker)

This is recommended for reproducible comparisons because every model is trained/tested on the exact same folds.

In this mode, `--data_root` is optional.

## Folds

- Use `--n_folds 5` to run all predefined folds.
- Use `--fold k` (1-based) to run only a specific fold.

`--n_splits` is accepted as an alias for `--n_folds`.

## Binary vs multiclass

- MFCC+SVM and YAMNet+LR support both `binary_stage2` and `multiclass`.
- You can override the config task via CLI: `--task binary_stage2` or `--task multiclass`.

The ResNet log-mel baseline is currently binary-only.

## Baseline C: MFCC + SVM

```bash
python baselines/mfcc_svm/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5

# run only fold 3
python baselines/mfcc_svm/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5 \
  --fold 3

# force binary task (even if config is multiclass)
python baselines/mfcc_svm/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5 \
  --task binary_stage2
```

## Baseline B: log-mel + ResNet18 (PyTorch, CPU-friendly)

```bash
python baselines/resnet_mel/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5

# run only fold 3
python baselines/resnet_mel/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5 \
  --fold 3
```

## Baseline A: YAMNet embeddings + Logistic Regression

This baseline requires optional dependencies not used elsewhere in the repo:
- `tensorflow`
- `tensorflow_hub`

```bash
python baselines/yamnet_lr/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5

# run only fold 3
python baselines/yamnet_lr/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5 \
  --fold 3

# force multiclass task
python baselines/yamnet_lr/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5 \
  --task multiclass
```

## Baseline D: SE-ResNet on precomputed spectrograms

This baseline reuses the SE-ResNet (Keras) implementation and trains on **precomputed mel-spectrograms saved as `.npy`**.

Requirements:

- `tensorflow` (not included in `requirements.txt`)
- Spectrograms stored under a root directory in the same class/patient structure as the dataset, e.g.
  `/home/ksvoai/source/datasets/spectrograms/New_SwallowSet_Test/<Class>/<Patient>/<file_stem>.npy`

Run (binary stage2 by default using `data_ast_stage2/` when present):

```bash
python baselines/se_resnet/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5 \
  --task binary_stage2 \
  --spectrogram_root /home/ksvoai/source/datasets/spectrograms/New_SwallowSet_Test/
```

Run only fold 3:

```bash
python baselines/se_resnet/train.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5 \
  --fold 3 \
  --task binary_stage2 \
  --spectrogram_root /home/ksvoai/source/datasets/spectrograms/New_SwallowSet_Test/
```

## Run all baselines

```bash
python baselines/run_all.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5

# run only fold 3 for all baselines
python baselines/run_all.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5 \
  --fold 3

# force binary for baselines that support it
python baselines/run_all.py \
  --output_dir baselines/runs \
  --seed 42 \
  --n_folds 5 \
  --task binary_stage2
```

## Reproducibility

- Use `--seed 42` (or any fixed seed)
- Use the same `--n_folds` (or `--n_splits` alias)
- If predefined fold artifacts exist (e.g. `data_ast_stage2/`), the scripts will use them by default.
