#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/Johnny/challenge/dd/FastWAM_jepa"
LIBERO_ROOT="/manifoldai-training/johnny/challenge/FastWAM_Data/libero_mujoco3.3.2"
TASK="libero_idm_2cam224_1e-4"
VJEPA_REPO="${ROOT}/external/vjepa2"
VJEPA_CKPT="${ROOT}/checkpoints/vjepa2/vitg.pt"
OUTPUT_DIR="${ROOT}/analysis/v3_stage4_eval"
LOG_DIR="${ROOT}/logs/v3_stage4_eval"
TMP_ROOT="${ROOT}/tmp"

EXPORT_SCRIPT="${ROOT}/tools/export_fastwam_jepa_idm_v3_stage4_latents.py"
EVAL_SCRIPT="${ROOT}/tools/evaluate_fastwam_jepa_idm_v3_stage4_repr.py"

STAGE4_CKPT="${STAGE4_CKPT:-}"
GPU_ID="${GPU_ID:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-16}"
FORCE="${FORCE:-0}"

LATENTS="${OUTPUT_DIR}/stage4_latents.npz"
METRICS="${OUTPUT_DIR}/stage4_metrics.csv"
SUMMARY="${OUTPUT_DIR}/stage4_summary.md"
LANGUAGE_PCA="${OUTPUT_DIR}/stage4_language_pca.png"
ACTION_PCA="${OUTPUT_DIR}/stage4_action_pca.png"

error() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -n "$STAGE4_CKPT" ]] || error "STAGE4_CKPT must be provided"
[[ "$FORCE" == "0" || "$FORCE" == "1" ]] || error "FORCE must be 0 or 1"
[[ -f "$STAGE4_CKPT" ]] || error "Stage4 checkpoint not found: $STAGE4_CKPT"
[[ -d "$LIBERO_ROOT" ]] || error "LIBERO root not found: $LIBERO_ROOT"
[[ -d "$VJEPA_REPO" ]] || error "V-JEPA2 repo not found: $VJEPA_REPO"
[[ -f "$VJEPA_CKPT" ]] || error "V-JEPA2 checkpoint not found: $VJEPA_CKPT"
[[ -f "$EXPORT_SCRIPT" ]] || error "Export script not found: $EXPORT_SCRIPT"
[[ -f "$EVAL_SCRIPT" ]] || error "Evaluation script not found: $EVAL_SCRIPT"
[[ -f /opt/utils/tione/etc/profile.d/conda.sh ]] || error "Conda profile not found"
[[ -d /data1/Johnny/challenge/dd/envs/fastwam ]] || error "FastWAM environment not found"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$TMP_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TMPDIR="$TMP_ROOT"
export TEMP="$TMP_ROOT"
export TMP="$TMP_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${ROOT}/tools:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

source /opt/utils/tione/etc/profile.d/conda.sh
conda activate /data1/Johnny/challenge/dd/envs/fastwam
cd "$ROOT"

if [[ "$FORCE" == "0" && -s "$LATENTS" ]]; then
  printf 'SKIP export: %s\n' "$LATENTS"
else
  python "$EXPORT_SCRIPT" \
    --stage4-checkpoint "$STAGE4_CKPT" \
    --output "$LATENTS" \
    --libero-data-root "$LIBERO_ROOT" \
    --task "$TASK" \
    --split train \
    --vjepa-repo "$VJEPA_REPO" \
    --vjepa-checkpoint "$VJEPA_CKPT" \
    --batch-size "$EXPORT_BATCH_SIZE" \
    --num-workers 0 \
    --max-samples "$MAX_SAMPLES" \
    --seed 42 \
    --precision bf16 \
    2>&1 | tee "${LOG_DIR}/01_export_stage4.log"
fi

analysis_complete=0
if [[ -s "$SUMMARY" && -s "$METRICS" && -s "$LANGUAGE_PCA" && -s "$ACTION_PCA" ]]; then
  analysis_complete=1
fi
if [[ "$FORCE" == "0" && "$analysis_complete" == "1" ]]; then
  printf 'SKIP evaluation: Stage4 analysis outputs already exist\n'
else
  python "$EVAL_SCRIPT" \
    --input "$LATENTS" \
    --output-dir "$OUTPUT_DIR" \
    --max-pca-points 2000 \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/02_evaluate_stage4.log"
fi

printf 'outputs:\n'
find "$OUTPUT_DIR" -maxdepth 1 -type f -print | sort
printf 'disk_usage:\n'
du -sh "$OUTPUT_DIR" "$LOG_DIR"
