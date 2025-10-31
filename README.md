# Zenker Audio Detection

Deep learning pipeline for Zenker's diverticulum detection using cervical auscultation and Audio Spectrogram Transformers (AST).

## Overview

This repository contains the code accompanying the paper:

> **"Deep Learning for Early Detection of Zenker's Diverticulum Based on Swallowing Sound Analysis"**  
> Daniel Ostler-Mildner, Alissa Jell, Matthias Seibold, Hubertus Feussner, Simone Graf, Dirk Wilhelm, Jonas Fuchtmann

The code implements a two-stage deep learning approach for detecting Zenker's diverticulum (ZD) from cervical auscultation recordings:
- **Stage 1**: Distinguishes swallowing sounds from idle/non-swallowing sounds
- **Stage 2**: Classifies swallowing sounds as healthy or pathological (ZD)

## Repository Structure

```
zenker-audio-detection/
├── README.md                                   # This file
├── LICENSE                                     # MIT License
├── requirements.txt                            # Python dependencies
├── .env.example                                # Environment configuration template
│
├── src/                                        # Main source code
│   ├── train_ast_stage1_cross_validation.py   # Stage 1 training (Idle vs Swallow)
│   ├── train_ast_stage2_cross_validation.py   # Stage 2 training (Healthy vs ZD)
│   ├── test_trained_model_stage1_cv.py        # Stage 1 model testing
│   ├── test_trained_model_stage2_cv.py        # Stage 2 model testing
│   ├── test_long_audio_windows_2stage.py      # Patient-level inference
│   ├── test_long_audio_windows_2stage_cache.py # Cached version for efficiency
│   ├── run_batch_simple_2stage.py             # Batch inference script (patient-level)
│   └──run_all_folds_simple_batch.sh          # Run inference on all folds
│
├── utils/                                      # Data preparation and analysis tools
│   ├── config.py                              # Environment configuration loader
│   ├── PrepareTrainingData_AST_cv.py          # Basic CV dataset preparation
│   ├── PrepareTrainingData_AST_cv_2stage.py   # Two-stage dataset preparation
│   ├── PrepareTrainingData_AST_cv_2stage_capped.py # Capped version e.g. cap to 40 snippets per patient
│   ├── PrepareDataset.py                      # General dataset preparation
│   ├── PrepareDatasetLongAudio.py             # Long audio preparation
│   ├── PrepareLongFile.py                     # Single long file preparation
│   ├── PrepareLongFileSet.py                  # Long file set preparation
│   ├── compute_ast_normalization_stats.py     # Compute dataset statistics
│   ├── analyze_ROC_PR_stage1.py               # Stage 1 ROC/PR analysis
│   ├── analyze_ROC_PR_stage2.py               # Stage 2 ROC/PR analysis
│   ├── aggregate_2stage_results.py            # Aggregate patient-level results
│   ├── extract_thresholds_per_fold.py         # Extract optimal thresholds
│   └── plot_confusion_matrices.py             # Confusion matrix visualization
│
└── wandb_sweeps/                               # Weights & Biases sweep configurations
    ├── sweep_stage1_comprehensive.yaml        # Stage 1 hyperparameter sweep
    └── sweep_stage2_comprehensive.yaml        # Stage 2 hyperparameter sweep
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- Ubuntu 22.04 LTS or similar Linux distribution

### Setup

1. Clone this repository:
```bash
git clone https://github.com/NodOzz/zenker-audio-detection.git
cd zenker-audio-detection
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure dataset paths:
```bash
cp .env.example .env
# Edit .env to set your dataset paths
```

## Model Architecture

The system uses the **Audio Spectrogram Transformer (AST)** architecture, specifically the pretrained model:
- `MIT/ast-finetuned-audioset-10-10-0.4593` (pretrained on AudioSet)

The model operates on log-mel spectrograms derived from 1-second audio windows at 16 kHz sampling rate.

## Data Preparation

### Expected Data Structure

The audio dataset should be organized in the following hierarchical structure to ensure proper patient-level cross-validation splits:

#### Short Audio Segments (for training)

```
dataset_root/
├── Idle/
│   ├── 004/
│   │   ├── audio_001.wav
│   │   ├── audio_002.wav
│   │   └── ...
│   ├── 203/
│   │   └── ...
│   └── ...
├── Healthy/
│   ├── 201/
│   │   ├── audio_001.wav
│   │   ├── audio_002.wav
│   │   └── ...
│   ├── 203/
│   │   └── ...
│   └── ...
└── Zenker/
    ├── 004/
    │   ├── audio_001.wav
    │   ├── audio_002.wav
    │   └── ...
    ├── 005/
    │   └── ...
    └── ...
```

#### Long Audio Files (for patient-level inference)

```
long_audio_root/
├── Healthy/
│   ├── 201/
│   │   ├── long_recording_1.wav
|   |   └── long_recording_2.wav
│   ├── 203/
│   │   ├── long_recording_1.wav
|   |   └── long_recording_2.wav
│   └── ...
└── Zenker/
    ├── 001/
    │   ├── long_recording_1.wav
|   |   └── long_recording_2.wav
    ├── 002/
    │   ├── long_recording_1.wav
|   |   └── long_recording_2.wav
    └── ...
```

### Audio Specifications

Audio recordings should be:
- **Short segments**: 1-second segments at 48 kHz (will be resampled to 16 kHz during processing)
- **Long recordings**: Continuous recordings at any sample rate (will be processed with sliding windows)
- **Format**: WAV files
- **Labels**: Organized by directory structure as shown above:
  - `Idle`: Non-swallowing sounds (background, silence, speech, etc.)
  - `Healthy`: Normal swallowing sounds from healthy subjects
  - `Zenker`: Pathological swallowing sounds from subjects with Zenker's diverticulum

### Patient ID Organization

**Critical**: Patient IDs must be consistent across classes. For example:
- If subject `001` appears in both `Idle/` and `Zenker/`, they represent the same patient
- This ensures the cross-validation splits maintain patient-level separation
- Prevents data leakage where the same patient appears in both training and test sets


### Data Preparation Scripts

Before training, you need to prepare the cross-validation datasets. The scripts will automatically load dataset paths from your `.env` file.

1. **Ensure your `.env` file is configured** with the correct dataset paths (see Installation section above).

2. **Create basic CV datasets** from the organized audio files:
   ```bash
   python utils/PrepareTrainingData_AST_cv.py 
   ```
   This creates 5-fold cross-validation splits ensuring patient-level separation between folds.

3. **Generate two-stage datasets** for the sequential classification approach:
   ```bash
   python utils/PrepareTrainingData_AST_cv_2stage.py --val-mode per-fold --val-ratio 0.20
   ```
   This creates:
   - **Stage 1 datasets**: `data_ast_stage1/` (Idle vs Swallow classification)
   - **Stage 2 datasets**: `data_ast_stage2/` (Healthy vs Zenker classification)
   - Validation splits with 20% of patients reserved per fold

4. **Optional: Compute normalization statistics** across the dataset:
   ```bash
   python utils/compute_ast_normalization_stats.py
   ```
   If present, these values will be applied during feature extraction in training.

### Long Audio Preparation (for patient-level inference)

If you have continuous recordings for patient-level evaluation, the scripts will automatically use paths from your `.env` file:

1. **Process long recordings**:
   ```bash
   python utils/PrepareDatasetLongAudio.py
   ```
   This converts and organizes long audio files for patient-level inference. 

## Training

### Default Output Structure

By default, trained models are saved to:
- **Stage 1**: `runs/ast_classifier_stage1/fold{N}/best/`
- **Stage 2**: `runs/ast_classifier_stage2/fold{N}/best/`

You can customize the output directory using the `--output-root` argument.

### Stage 1: Idle vs Swallow Classification

Train the first stage model across all 5 folds:

```bash
python src/train_ast_stage1_cross_validation.py
```

For custom output directory:
```bash
python src/train_ast_stage1_cross_validation.py --output-root models/my_experiment_stage1
```

Other useful options:
- `--no-wandb`: Disable Weights & Biases logging
- `--fold N`: Train only a specific fold (1-5)
- `--disable-early-stopping`: Train for full epochs without early stopping


Key hyperparameters (optimized via random search using wandb sweeps):
- Focal loss gamma: 2.0
- Label smoothing: 0.07
- Learning rate: 3.7e-5
- Weight decay: 0.013
- Warmup ratio: 0.20
- Optimizer: AdamW with β₂=0.97

### Stage 2: Healthy vs Zenker Classification

Train the second stage model across all 5 folds:

```bash
python src/train_ast_stage2_cross_validation.py
```

For custom output directory:
```bash
python src/train_ast_stage2_cross_validation.py --output-root models/my_experiment_stage2
```

Key hyperparameters (optimized via random search):
- Focal loss gamma: 1.0
- Label smoothing: 0.09
- Learning rate: 4e-5
- Weight decay: 0.007
- Warmup ratio: 0.14
- Optimizer: AdamW (fused) with β₂=0.976

### Training Options

Both training scripts support:
- `--output-root PATH`: Custom output directory (default: `runs/ast_classifier_stage{1|2}`)
- `--disable-early-stopping`: Train for full NUM_EPOCHS without early stopping
- `--no-wandb`: Disable Weights & Biases logging
- `--wandb-per-fold`: Create separate W&B runs for each fold
- `--fold N`: Train only a specific fold (1-5)

### Data Augmentation

During training, the following augmentations are applied (probability 0.8):
- Gaussian noise (SNR: 10-20 dB)
- Gain scaling (-6 to +6 dB)
- Time stretching (rate 0.8-1.2)
- Pitch shifting (-4 to +4 semitones)
- Time masking (1-20% of signal)



## Evaluation / Testing

### Test on snippets
To test the best trained model of stage 1, run
``` bash 
python src/test_trained_model_stage1_cv.py --all
``` 
and for stage 2:
``` bash
python src/test_trained_model_stage2_cv.py --all
``` 
This will store the results in the folder of the respective model.

For custom model directories:
```bash
python src/test_trained_model_stage1_cv.py --all --model-root models/my_experiment_stage1
python src/test_trained_model_stage2_cv.py --all --model-root models/my_experiment_stage2
```

## Inference

### Patient-Level Classification

To run the two-stage pipeline on long audio recordings:

```bash
python src/test_long_audio_windows_2stage.py \
    --fold 1 \
    --patient-id <PATIENT_ID> \
    --audio-path /path/to/audio.wav \
    --plot
```

The system uses a sliding window approach:
- 1-second windows with 0.5-second hop size
- Stage 1 threshold: probability ≥ 0.5 for swallowing detection
- Stage 2 threshold: probability ≥ 0.5 for ZD classification
- Final prediction based on Zenker-to-Swallow Ratio (ZSR) ≥ 0.5

There is also a version `test_long_audio_windows_2stage_cache.py` which will caches calculated features of each long audio file, to speed up repeated inference (as long as mean and std values of the dataset are not changed).

### Batch Inference

Process multiple patients:

```bash
python src/run_batch_simple_2stage.py \
    --fold 1 \
    --long-audio-root /path/to/audio/directory \
    --pattern "*.wav" \
    --plot
```

Or run all 5 folds using default model location:

```bash
bash run_all_folds_simple_batch.sh
```

For custom model directories:

```bash
bash run_all_folds_simple_batch.sh models/my_experiment
```

**Note**: The script will automatically load dataset paths from your `.env` file. If `LONG_AUDIO_ROOT` is not set, it will use a fallback path.

## Analysis and Evaluation

### ROC and Precision-Recall Curves

Generate performance curves for each stage:

```bash
python utils/analyze_ROC_PR_stage1.py
python utils/analyze_ROC_PR_stage2.py
```

For custom model directories:
```bash
python utils/analyze_ROC_PR_stage1.py --model-root models/my_experiment
python utils/analyze_ROC_PR_stage2.py --model-root models/my_experiment
```

### Aggregate Results

Combine results from patient-level inference:

```bash
python utils/aggregate_2stage_results.py
```

### Generate Confusion Matrices

Create confusion matrix plots:

```bash
python utils/plot_confusion_matrices.py
```

For custom model directories:
```bash
python utils/plot_confusion_matrices.py \
    --model-root models/my_experiment \
    --output-dir models/my_experiment/plots
```

### Extract Optimal Thresholds

Compute per-fold optimal thresholds:

```bash
python utils/extract_thresholds_per_fold.py \
    --stage2-metrics runs/results/ROC_PR_stage2/validation_metrics.json \
    --output-config optimal_thresholds_per_fold.json
```

For custom model directories:
```bash
python utils/extract_thresholds_per_fold.py \
    --stage2-metrics models/my_experiment/results/ROC_PR_stage2/validation_metrics.json \
    --output-config models/my_experiment/optimal_thresholds_per_fold.json
```
This script optimizes the detection threshold based on validation F1 score per fold. This can then be passed to the `src/run_batch_simple_2stage.py` or `src/run_all_folds_simple_batch.sh`.

## Ethics and Data Availability

**Note**: The audio dataset used for this study cannot be publicly shared due to privacy concerns and ethical restrictions. 

## Citation

If you use this code in your research, please cite:
[BibTeX entry to be added after publication]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.