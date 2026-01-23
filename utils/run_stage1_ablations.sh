#!/usr/bin/env bash
set -euo pipefail

# Stage-1 ablation suite (A0–A3 + A1b) aligned with the Stage-2 ablation ladder.
# Runs across all folds by default.
#
# A0: tuned optimizer schedule; augmentation OFF; focal OFF; label smoothing OFF
# A1: A0 + augmentation (p=AUG_P_ON); focal OFF; label smoothing OFF
# A1b: A0 + focal only; augmentation OFF; label smoothing OFF
# A2: A1 + focal (augmentation ON + focal ON); label smoothing OFF
# A3: A2 + label smoothing (everything ON)

# ----------------------------
# User-configurable parameters
# ----------------------------
RUN_BASE="${RUN_BASE:-runs/ablations_stage1_A0_A3_$(date +%Y%m%d_%H%M%S)}"
NORM="${NORM:-pretrained}"
FOLDS_STR="${FOLDS:-1 2 3 4 5}"

# Tuned optimizer schedule (kept fixed across all ablations)
LR="${LR:-3.7e-5}"
WD_TUNED="${WD_TUNED:-0.013}"
WARM_TUNED="${WARM_TUNED:-0.20}"
BETA2_TUNED="${BETA2_TUNED:-0.970}"

# Loss / regularization knobs
LS_TUNED="${LS_TUNED:-0.07}"
AUG_P_ON="${AUG_P_ON:-0.8}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"

# W&B toggle (default: disabled for overnight ablations)
WANDB="${WANDB:-0}"  # set to 1 to enable W&B logging

PYTHON="${PYTHON:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/train_ast_stage1_cross_validation.py}"

# ----------------------------
# Helper setup
# ----------------------------
read -r -a FOLDS <<< "${FOLDS_STR}"

COMMON_BASE=(--normalization-source "${NORM}")
if [[ "${WANDB}" -eq 0 ]]; then
  COMMON_BASE+=(--no-wandb)
fi

# Tuned optimizer args (fixed across all ablations)
OPT_TUNED=(--learning-rate "${LR}" --weight-decay "${WD_TUNED}" --warmup-ratio "${WARM_TUNED}" --adam-beta2 "${BETA2_TUNED}")

run () {
  local fold="$1"
  local name="$2"
  shift 2

  local out_dir="${RUN_BASE}/fold${fold}/${name}"
  mkdir -p "${out_dir}"

  # Skip if output already exists (useful when resuming overnight jobs)
  if [[ -f "${out_dir}/summary.json" || -f "${out_dir}/metrics.json" ]]; then
    echo ""
    echo "============================================================"
    echo "SKIP (already exists): fold=${fold} run=${name}"
    echo "  -> ${out_dir}"
    echo "============================================================"
    return 0
  fi

  echo ""
  echo "============================================================"
  echo "RUN: fold=${fold}  variant=${name}"
  echo "OUT: ${out_dir}"
  echo "============================================================"

  "${PYTHON}" "${TRAIN_SCRIPT}" \
    "${COMMON_BASE[@]}" \
    --fold "${fold}" \
    --output-root "${out_dir}" \
    "${OPT_TUNED[@]}" \
    "$@"
}

# ----------------------------
# Ablation runs (Stage 1)
# ----------------------------
for fold in "${FOLDS[@]}"; do
  # A0: Optimizer-only baseline (tuned optimizer schedule; augmentation OFF; focal OFF; label smoothing OFF)
  run "${fold}" "A0_optimizer_only" \
    --no-focal-loss --label-smoothing 0.0 --augmentation-prob 0.0

  # A1: + Augmentation only (p=AUG_P_ON)
  run "${fold}" "A1_plus_augmentation_p${AUG_P_ON}" \
    --no-focal-loss --label-smoothing 0.0 --augmentation-prob "${AUG_P_ON}"

  # A1b: + Focal only (no augmentation)
  run "${fold}" "A1b_plus_focal_only_gamma${FOCAL_GAMMA}" \
    --focal-gamma "${FOCAL_GAMMA}" --label-smoothing 0.0 --augmentation-prob 0.0

  # A2: + Augmentation + Focal
  run "${fold}" "A2_plus_aug_p${AUG_P_ON}_plus_focal_gamma${FOCAL_GAMMA}" \
    --focal-gamma "${FOCAL_GAMMA}" --label-smoothing 0.0 --augmentation-prob "${AUG_P_ON}"

  # A3: everything on (Aug + Focal + Label smoothing)
  run "${fold}" "A3_all_on_aug_p${AUG_P_ON}_focal_gamma${FOCAL_GAMMA}_ls${LS_TUNED}" \
    --focal-gamma "${FOCAL_GAMMA}" --label-smoothing "${LS_TUNED}" --augmentation-prob "${AUG_P_ON}"
done

echo ""
echo "All ablation runs completed."
echo "Outputs under: ${RUN_BASE}/"
