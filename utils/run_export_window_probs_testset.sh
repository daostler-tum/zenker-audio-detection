#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-}"
LONG_AUDIO_ROOT="${LONG_AUDIO_ROOT:-}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-paper_cache_v1}"
PIPELINE_TAG="${PIPELINE_TAG:-s1opt_s2opt}"
WINDOW_PROBS_ROOT="${WINDOW_PROBS_ROOT:-caches}"
PATTERN="${PATTERN:-*.wav}"
TEST_IDS_DIR="${TEST_IDS_DIR:-data_ast_stage2}"
FOLDS="${FOLDS:-1 2 3 4 5}"
START_FOLD="${START_FOLD:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

if [[ -z "${MODEL_DIR}" || -z "${LONG_AUDIO_ROOT}" ]]; then
  echo "Usage:"
  echo "  MODEL_DIR=/abs/path/to/runs/<model_dir> LONG_AUDIO_ROOT=/abs/path/to/Long"
  echo "  EXPERIMENT_TAG=... PIPELINE_TAG=... START_FOLD=... SKIP_EXISTING=... bash $0 [extra args forwarded to python]"
  echo ""
  echo "Example:"
  echo "  MODEL_DIR=/home/.../zenker-audio-detection/runs/paper_opt_v3"
  echo "  LONG_AUDIO_ROOT=/home/.../New_SwallowSet/Long"
  echo "  EXPERIMENT_TAG=paper_cache_v1 PIPELINE_TAG=s1opt_s2opt START_FOLD=3 bash $0"
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

if [[ "${WINDOW_PROBS_ROOT}" = /* ]]; then
  WINDOW_PROBS_ROOT_ABS="${WINDOW_PROBS_ROOT}"
else
  WINDOW_PROBS_ROOT_ABS="${PROJECT_ROOT}/${WINDOW_PROBS_ROOT}"
fi

for FOLD in ${FOLDS}; do
  if [[ "${FOLD}" -lt "${START_FOLD}" ]]; then
    continue
  fi
  FOLD_FILE="${PROJECT_ROOT}/${TEST_IDS_DIR}/test_ids_fold${FOLD}.txt"
  if [[ ! -f "${FOLD_FILE}" ]]; then
    echo "Missing fold file: ${FOLD_FILE}"
    exit 1
  fi

  STAGE1_MODEL="${MODEL_DIR}/ast_classifier_stage1/fold${FOLD}/best"
  STAGE2_MODEL="${MODEL_DIR}/ast_classifier_stage2/fold${FOLD}/best"

  if [[ ! -d "${STAGE1_MODEL}" ]]; then
    echo "Missing stage1 model dir: ${STAGE1_MODEL}"
    exit 1
  fi
  if [[ ! -d "${STAGE2_MODEL}" ]]; then
    echo "Missing stage2 model dir: ${STAGE2_MODEL}"
    exit 1
  fi

  echo "================ Fold ${FOLD} ================"

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line//$'\r'/}"
    if [[ -z "${line}" ]]; then
      continue
    fi

    PATIENT_ID="${line##*/}"

    if [[ "${SKIP_EXISTING}" == "true" ]]; then
      OUT_DIR="${WINDOW_PROBS_ROOT_ABS}/${EXPERIMENT_TAG}/${PIPELINE_TAG}/fold${FOLD}/${PATIENT_ID}"
      if compgen -G "${OUT_DIR}/*_window_probs.npz" > /dev/null; then
        echo "[skip] fold=${FOLD} patient_id=${PATIENT_ID} (cache exists)"
        continue
      fi
    fi

    echo "--- fold=${FOLD} patient_id=${PATIENT_ID} ---"

    set +e
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/test_long_audio_windows_2stage_cache.py" \
        --fold "${FOLD}" \
        --patient-id "${PATIENT_ID}" \
        --stage1-model-root "${STAGE1_MODEL}" \
        --stage2-model-root "${STAGE2_MODEL}" \
        --long-audio-root "${LONG_AUDIO_ROOT}" \
        --pattern "${PATTERN}" \
        --export-window-probs \
        --window-probs-root "${WINDOW_PROBS_ROOT_ABS}" \
        --experiment-tag "${EXPERIMENT_TAG}" \
        --pipeline-tag "${PIPELINE_TAG}" \
        --stage2-all-windows \
        --export-logits \
        "$@"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "[WARN] fold=${FOLD} patient_id=${PATIENT_ID} failed with exit_code=${rc}; continuing" >&2
    fi

  done < "${FOLD_FILE}"

done
