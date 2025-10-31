#!/usr/bin/env bash
set -euo pipefail

# Simple driver to run two-stage long-audio window inference for folds 1..5
# Adjust LONG_AUDIO_ROOT if your dataset path differs.
#
# Usage:
#   bash run_all_folds_simple_batch.sh [MODEL_DIR] [--no-threshold-config] [--stage1-forward-min-prob <val>] [--stage2-argmax]
#
# Examples:
#   bash run_all_folds_simple_batch.sh runs                    # Use default runs/ directory
#   bash run_all_folds_simple_batch.sh models/experiment_v1   # Use custom model directory
#   bash run_all_folds_simple_batch.sh runs --no-threshold-config  # Force 0.5 threshold
#   bash run_all_folds_simple_batch.sh runs --stage2-argmax        # Use argmax for Stage2
#
# If MODEL_DIR is not specified, defaults to "runs"
#
# Threshold behavior:
#   - If {MODEL_DIR}/optimal_thresholds_per_fold_both_stages.json exists, it will be used
#   - Use --no-threshold-config flag to force 0.5 threshold even if config exists
#   - Otherwise, defaults to 0.5 threshold for both stages

LONG_AUDIO_ROOT="${LONG_AUDIO_ROOT:-}"
PATTERN="*.wav"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON=${PYTHON:-python}

# Try to load dataset paths from .env file
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  # Source the .env file to get environment variables
  set -a  # automatically export all variables
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# If LONG_AUDIO_ROOT is still not set, use fallback
if [[ -z "$LONG_AUDIO_ROOT" ]]; then
  echo "Warning: LONG_AUDIO_ROOT not set. Please set it as environment variable or in .env file"
  echo "Using fallback: datasets/New_SwallowSet/Long"
  LONG_AUDIO_ROOT="datasets/New_SwallowSet/Long"
fi

echo "Long audio directory: ${LONG_AUDIO_ROOT}"

# Parse arguments
MODEL_DIR=${1:-runs}
NO_THRESHOLD_CONFIG=false
STAGE2_ARGMAX=false
DRY_RUN=false

# Optional Stage1 forwarding probability filter
STAGE1_FORWARD_MIN_PROB=""

# Parse arguments (flags can appear in any order)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-threshold-config)
      NO_THRESHOLD_CONFIG=true
      ;;
    --stage2-argmax)
      STAGE2_ARGMAX=true
      ;;
    --dry-run)
      DRY_RUN=true
      ;;
    --stage1-forward-min-prob)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --stage1-forward-min-prob requires a value" >&2
        exit 1
      fi
      STAGE1_FORWARD_MIN_PROB="$2"
      shift
      ;;
    --*)
      echo "Warning: Unknown option $1" >&2
      ;;
    *)
      MODEL_DIR="$1"
      ;;
  esac
  shift
done

PLOT_FLAG="--plot"
# If you do NOT want plots (faster), set: PLOT_FLAG=""

echo "Using models from: ${MODEL_DIR}"

# Setup output directory structure - use PROJECT_ROOT for model directory
OUTPUT_BASE="${PROJECT_ROOT}/${MODEL_DIR}/results/patient_inference"
mkdir -p "$OUTPUT_BASE"
echo "Output directory: ${OUTPUT_BASE}"

# Check for threshold config in MODEL_DIR (unless --no-threshold-config is set) - use PROJECT_ROOT
THRESHOLD_CONFIG="${PROJECT_ROOT}/${MODEL_DIR}/optimal_thresholds_per_fold_both_stages.json"
if [[ "$NO_THRESHOLD_CONFIG" == true ]]; then
  echo "Threshold config disabled (--no-threshold-config), will use default 0.5 threshold"
  USE_THRESHOLD_CONFIG=false
elif [[ -f "$THRESHOLD_CONFIG" ]]; then
  echo "Found threshold config: ${THRESHOLD_CONFIG}"
  USE_THRESHOLD_CONFIG=true
else
  echo "No threshold config found in ${MODEL_DIR}, will use default 0.5 threshold"
  USE_THRESHOLD_CONFIG=false
fi
echo ""

for FOLD in 1 2 3 4 5; do
  echo "================ Fold ${FOLD} ================"
  
  # Construct model paths - use PROJECT_ROOT instead of SCRIPT_DIR for model locations
  STAGE1_MODEL="${PROJECT_ROOT}/${MODEL_DIR}/ast_classifier_stage1/fold${FOLD}/best"
  STAGE2_MODEL="${PROJECT_ROOT}/${MODEL_DIR}/ast_classifier_stage2/fold${FOLD}/best"
  
  # Build command with optional threshold config
  CMD=("$PYTHON" "${SCRIPT_DIR}/run_batch_simple_2stage.py" \
       --fold "$FOLD" \
       --long-audio-root "$LONG_AUDIO_ROOT" \
       --pattern "$PATTERN" \
       --stage1-model-root "$STAGE1_MODEL" \
       --stage2-model-root "$STAGE2_MODEL" \
       --output-dir "$OUTPUT_BASE")
  
  # Add threshold config if it exists
  if [[ "$USE_THRESHOLD_CONFIG" == true ]]; then
    CMD+=(--threshold-config "$THRESHOLD_CONFIG")
  fi
  
  # Add Stage1 forward min prob if set
  if [[ -n "$STAGE1_FORWARD_MIN_PROB" ]]; then
    CMD+=(--stage1-forward-min-prob "$STAGE1_FORWARD_MIN_PROB")
  fi

  # Add Stage2 argmax flag if set
  if [[ "$STAGE2_ARGMAX" == true ]]; then
    CMD+=(--stage2-argmax)
  fi

  # Add dry-run flag if set
  if [[ "$DRY_RUN" == true ]]; then
    CMD+=(--dry-run)
  fi

  # Add plot flag
  if [[ -n "$PLOT_FLAG" ]]; then
    CMD+=($PLOT_FLAG)
  fi
  
  echo "+ ${CMD[*]}"
  # Execute
  "${CMD[@]}"
  echo
  echo "Done fold ${FOLD}"
  echo
done

echo "All folds completed."