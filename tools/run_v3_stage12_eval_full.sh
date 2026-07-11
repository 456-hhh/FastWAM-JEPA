#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data1/Johnny/challenge/dd/FastWAM_jepa"
LIBERO_DATA_ROOT="/manifoldai-training/johnny/challenge/FastWAM_Data/libero_mujoco3.3.2"
TASK="libero_idm_2cam224_1e-4"
STAGE1_CHECKPOINT="${PROJECT_ROOT}/runs/v3_stage1_text_action_10k_ddp8_bs32_save500/checkpoints/checkpoint_step_004000.pt"
STAGE2_CHECKPOINT="${PROJECT_ROOT}/runs/v3_stage2_rawjepa_vl_10k_from_stage1_4000_ddp8_bs32/checkpoints/checkpoint_step_002000.pt"
VJEPA_REPO="${PROJECT_ROOT}/external/vjepa2"
VJEPA_CHECKPOINT="${PROJECT_ROOT}/checkpoints/vjepa2/vitg.pt"
OUTPUT_DIR="${PROJECT_ROOT}/analysis/v3_stage12_eval_full"
LOG_DIR="${PROJECT_ROOT}/logs/v3_stage12_eval_full"
TMP_ROOT="${PROJECT_ROOT}/tmp"

EXPORT_SCRIPT="${PROJECT_ROOT}/tools/export_v3_stage12_latents.py"
PROBE_SCRIPT="${PROJECT_ROOT}/tools/eval_v3_linear_probe.py"
VISUALIZE_SCRIPT="${PROJECT_ROOT}/tools/visualize_v3_latents_pca_umap.py"

GPU_ID="${GPU_ID:-0}"
STAGE1_MAX_SAMPLES="${STAGE1_MAX_SAMPLES:-10000}"
STAGE2_MAX_SAMPLES="${STAGE2_MAX_SAMPLES:-10000}"
STAGE1_EXPORT_BATCH_SIZE="${STAGE1_EXPORT_BATCH_SIZE:-128}"
STAGE2_EXPORT_BATCH_SIZE="${STAGE2_EXPORT_BATCH_SIZE:-16}"
PROBE_EPOCHS="${PROBE_EPOCHS:-100}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
PLOT_MAX_POINTS="${PLOT_MAX_POINTS:-5000}"
FORCE="${FORCE:-0}"
RUN_UMAP="${RUN_UMAP:-1}"

STAGE1_NPZ="${OUTPUT_DIR}/stage1_step4000_latents.npz"
STAGE2_NPZ="${OUTPUT_DIR}/stage2_step2000_latents.npz"
STAGE1_LINEAR_CSV="${OUTPUT_DIR}/stage1_linear_probe.csv"
STAGE1_MLP_CSV="${OUTPUT_DIR}/stage1_mlp_probe.csv"
STAGE2_LINEAR_CSV="${OUTPUT_DIR}/stage2_linear_probe.csv"
STAGE2_MLP_CSV="${OUTPUT_DIR}/stage2_mlp_probe.csv"
STAGE1_VIS_DIR="${OUTPUT_DIR}/stage1_visualization"
STAGE2_VIS_DIR="${OUTPUT_DIR}/stage2_visualization"
MANIFEST="${OUTPUT_DIR}/evaluation_manifest.txt"

STAGE1_LATENTS=(z_l q_a_text z_a)
STAGE2_LATENTS=(z_l z_v q_a_vl z_a)
TARGET_KEYS=(
  action_mean action_first action_norm proprio object_pos object_state
  robot_state eef_pos distance_to_goal task_id task_index success
)
COLOR_KEYS=(task_id task_index episode_id timestep action_norm success)
completed_phases=()
skipped_phases=()

error() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || error "required file does not exist: $1"
}

require_dir() {
  [[ -d "$1" ]] || error "required directory does not exist: $1"
}

mark_skip() {
  local phase="$1"
  local output="$2"
  printf 'SKIP: %s output already exists: %s\n' "$phase" "$output"
  skipped_phases+=("$phase")
}

run_logged() {
  local phase="$1"
  local log_path="$2"
  shift 2
  printf 'RUN: %s\n' "$phase"
  "$@" 2>&1 | tee "$log_path"
  completed_phases+=("$phase")
}

visualization_complete() {
  local directory="$1"
  [[ -d "$directory" ]] && compgen -G "${directory}/*.png" >/dev/null
}

check_npz() {
  local npz_path="$1"
  shift
  python - "$npz_path" "$@" <<'PY'
from pathlib import Path
import sys
import numpy as np

path = Path(sys.argv[1])
required = tuple(sys.argv[2:])
if not path.is_file() or path.stat().st_size <= 0:
    raise SystemExit(f"ERROR: NPZ is missing or empty: {path}")

with np.load(path, allow_pickle=False) as data:
    keys = sorted(data.files)
    print(f"npz={path}")
    print(f"keys={','.join(keys)}")
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"ERROR: missing required latents: {','.join(missing)}")
    sample_counts = {}
    for key in keys:
        value = data[key]
        print(f"key={key} shape={value.shape} dtype={value.dtype}")
        if value.ndim > 0:
            sample_counts[key] = int(value.shape[0])
    unique_counts = sorted(set(sample_counts.values()))
    if len(unique_counts) != 1:
        details = ", ".join(f"{key}:{count}" for key, count in sorted(sample_counts.items()))
        raise SystemExit(f"ERROR: inconsistent first dimensions: {details}")
    if not unique_counts or unique_counts[0] <= 0:
        raise SystemExit("ERROR: NPZ contains no sample-aligned arrays")
    print(f"num_samples={unique_counts[0]}")
PY
}

npz_sample_count() {
  python - "$1" "$2" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    print(int(data[sys.argv[2]].shape[0]))
PY
}

join_phases() {
  if [[ "$#" -eq 0 ]]; then
    printf 'none'
    return
  fi
  local IFS=','
  printf '%s' "$*"
}

[[ "$FORCE" == "0" || "$FORCE" == "1" ]] || error "FORCE must be 0 or 1"
[[ "$RUN_UMAP" == "0" || "$RUN_UMAP" == "1" ]] || error "RUN_UMAP must be 0 or 1"

require_file "$STAGE1_CHECKPOINT"
require_file "$STAGE2_CHECKPOINT"
require_file "$VJEPA_CHECKPOINT"
require_dir "$VJEPA_REPO"
require_dir "$LIBERO_DATA_ROOT"
require_file "$EXPORT_SCRIPT"
require_file "$PROBE_SCRIPT"
require_file "$VISUALIZE_SCRIPT"
require_file "/opt/utils/tione/etc/profile.d/conda.sh"
require_dir "/data1/Johnny/challenge/dd/envs/fastwam"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$TMP_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TMPDIR="$TMP_ROOT"
export TEMP="$TMP_ROOT"
export TMP="$TMP_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/tools:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

source /opt/utils/tione/etc/profile.d/conda.sh
conda activate /data1/Johnny/challenge/dd/envs/fastwam
cd "$PROJECT_ROOT"

stage1_export_ran=0
if [[ "$FORCE" == "0" && -s "$STAGE1_NPZ" ]]; then
  mark_skip "01_export_stage1" "$STAGE1_NPZ"
else
  run_logged "01_export_stage1" "${LOG_DIR}/01_export_stage1.log" \
    python "$EXPORT_SCRIPT" \
      --stage stage1 \
      --checkpoint "$STAGE1_CHECKPOINT" \
      --output "$STAGE1_NPZ" \
      --libero-data-root "$LIBERO_DATA_ROOT" \
      --config-name train \
      --task "$TASK" \
      --split train \
      --batch-size "$STAGE1_EXPORT_BATCH_SIZE" \
      --num-workers 0 \
      --max-samples "$STAGE1_MAX_SAMPLES" \
      --seed 42 \
      --context-tokens 128 \
      --action-horizon 32 \
      --precision bf16
  stage1_export_ran=1
fi
if [[ "$stage1_export_ran" == "1" ]]; then
  check_npz "$STAGE1_NPZ" "${STAGE1_LATENTS[@]}" 2>&1 | tee -a "${LOG_DIR}/01_export_stage1.log"
else
  check_npz "$STAGE1_NPZ" "${STAGE1_LATENTS[@]}"
fi
completed_phases+=("03_check_stage1_npz")

stage2_export_ran=0
if [[ "$FORCE" == "0" && -s "$STAGE2_NPZ" ]]; then
  mark_skip "02_export_stage2" "$STAGE2_NPZ"
else
  run_logged "02_export_stage2" "${LOG_DIR}/02_export_stage2.log" \
    python "$EXPORT_SCRIPT" \
      --stage stage2 \
      --checkpoint "$STAGE2_CHECKPOINT" \
      --output "$STAGE2_NPZ" \
      --libero-data-root "$LIBERO_DATA_ROOT" \
      --config-name train \
      --task "$TASK" \
      --split train \
      --batch-size "$STAGE2_EXPORT_BATCH_SIZE" \
      --num-workers 0 \
      --max-samples "$STAGE2_MAX_SAMPLES" \
      --seed 42 \
      --context-tokens 128 \
      --action-horizon 32 \
      --precision bf16 \
      --vjepa-repo "$VJEPA_REPO" \
      --vjepa-checkpoint "$VJEPA_CHECKPOINT" \
      --vjepa-model-name vjepa2_vit_giant \
      --vjepa-img-size 256 \
      --vjepa-input-range=-1_1 \
      --vjepa-tubelet-size 2 \
      --vjepa-dim 1408 \
      --raw-vjepa-tokens 512 \
      --current-frame-count 4
  stage2_export_ran=1
fi
if [[ "$stage2_export_ran" == "1" ]]; then
  check_npz "$STAGE2_NPZ" "${STAGE2_LATENTS[@]}" 2>&1 | tee -a "${LOG_DIR}/02_export_stage2.log"
else
  check_npz "$STAGE2_NPZ" "${STAGE2_LATENTS[@]}"
fi
completed_phases+=("03_check_stage2_npz")

if [[ "$FORCE" == "0" && -s "$STAGE1_LINEAR_CSV" ]]; then
  mark_skip "04_stage1_linear_probe" "$STAGE1_LINEAR_CSV"
else
  run_logged "04_stage1_linear_probe" "${LOG_DIR}/03_stage1_linear_probe.log" \
    python "$PROBE_SCRIPT" \
      --input "$STAGE1_NPZ" \
      --output-csv "$STAGE1_LINEAR_CSV" \
      --latent-keys "${STAGE1_LATENTS[@]}" \
      --target-keys "${TARGET_KEYS[@]}" \
      --probe linear \
      --epochs "$PROBE_EPOCHS" \
      --batch-size "$PROBE_BATCH_SIZE" \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --train-ratio 0.8 \
      --seed 42 \
      --device cuda
fi

if [[ "$FORCE" == "0" && -s "$STAGE1_MLP_CSV" ]]; then
  mark_skip "05_stage1_mlp_probe" "$STAGE1_MLP_CSV"
else
  run_logged "05_stage1_mlp_probe" "${LOG_DIR}/04_stage1_mlp_probe.log" \
    python "$PROBE_SCRIPT" \
      --input "$STAGE1_NPZ" \
      --output-csv "$STAGE1_MLP_CSV" \
      --latent-keys "${STAGE1_LATENTS[@]}" \
      --target-keys "${TARGET_KEYS[@]}" \
      --probe mlp \
      --epochs "$PROBE_EPOCHS" \
      --batch-size "$PROBE_BATCH_SIZE" \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --train-ratio 0.8 \
      --seed 42 \
      --device cuda
fi

if [[ "$FORCE" == "0" && -s "$STAGE2_LINEAR_CSV" ]]; then
  mark_skip "06_stage2_linear_probe" "$STAGE2_LINEAR_CSV"
else
  run_logged "06_stage2_linear_probe" "${LOG_DIR}/05_stage2_linear_probe.log" \
    python "$PROBE_SCRIPT" \
      --input "$STAGE2_NPZ" \
      --output-csv "$STAGE2_LINEAR_CSV" \
      --latent-keys "${STAGE2_LATENTS[@]}" \
      --target-keys "${TARGET_KEYS[@]}" \
      --probe linear \
      --epochs "$PROBE_EPOCHS" \
      --batch-size "$PROBE_BATCH_SIZE" \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --train-ratio 0.8 \
      --seed 42 \
      --device cuda
fi

if [[ "$FORCE" == "0" && -s "$STAGE2_MLP_CSV" ]]; then
  mark_skip "07_stage2_mlp_probe" "$STAGE2_MLP_CSV"
else
  run_logged "07_stage2_mlp_probe" "${LOG_DIR}/06_stage2_mlp_probe.log" \
    python "$PROBE_SCRIPT" \
      --input "$STAGE2_NPZ" \
      --output-csv "$STAGE2_MLP_CSV" \
      --latent-keys "${STAGE2_LATENTS[@]}" \
      --target-keys "${TARGET_KEYS[@]}" \
      --probe mlp \
      --epochs "$PROBE_EPOCHS" \
      --batch-size "$PROBE_BATCH_SIZE" \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --train-ratio 0.8 \
      --seed 42 \
      --device cuda
fi

UMAP_ARGS=()
if [[ "$RUN_UMAP" == "1" ]]; then
  UMAP_ARGS+=(--run-umap)
fi

if [[ "$FORCE" == "0" ]] && visualization_complete "$STAGE1_VIS_DIR"; then
  mark_skip "08_stage1_pca_umap" "$STAGE1_VIS_DIR"
else
  run_logged "08_stage1_pca_umap" "${LOG_DIR}/07_stage1_pca_umap.log" \
    python "$VISUALIZE_SCRIPT" \
      --input "$STAGE1_NPZ" \
      --output-dir "$STAGE1_VIS_DIR" \
      --latent-keys "${STAGE1_LATENTS[@]}" \
      --color-keys "${COLOR_KEYS[@]}" \
      --max-points "$PLOT_MAX_POINTS" \
      --seed 42 \
      "${UMAP_ARGS[@]}"
fi

if [[ "$FORCE" == "0" ]] && visualization_complete "$STAGE2_VIS_DIR"; then
  mark_skip "09_stage2_pca_umap" "$STAGE2_VIS_DIR"
else
  run_logged "09_stage2_pca_umap" "${LOG_DIR}/08_stage2_pca_umap.log" \
    python "$VISUALIZE_SCRIPT" \
      --input "$STAGE2_NPZ" \
      --output-dir "$STAGE2_VIS_DIR" \
      --latent-keys "${STAGE2_LATENTS[@]}" \
      --color-keys "${COLOR_KEYS[@]}" \
      --max-points "$PLOT_MAX_POINTS" \
      --seed 42 \
      "${UMAP_ARGS[@]}"
fi

stage1_samples="$(npz_sample_count "$STAGE1_NPZ" z_l)"
stage2_samples="$(npz_sample_count "$STAGE2_NPZ" z_l)"

{
  printf 'evaluation_time=%s\n' "$(date -Iseconds)"
  printf 'project_root=%s\n' "$PROJECT_ROOT"
  printf 'stage1_checkpoint=%s\n' "$STAGE1_CHECKPOINT"
  printf 'stage2_checkpoint=%s\n' "$STAGE2_CHECKPOINT"
  printf 'vjepa2_checkpoint=%s\n' "$VJEPA_CHECKPOINT"
  printf 'gpu_id=%s\n' "$GPU_ID"
  printf 'stage1_samples=%s\n' "$stage1_samples"
  printf 'stage2_samples=%s\n' "$stage2_samples"
  printf 'probe_epochs=%s\n' "$PROBE_EPOCHS"
  printf 'completed_phases=%s\n' "$(join_phases "${completed_phases[@]}")"
  printf 'skipped_phases=%s\n' "$(join_phases "${skipped_phases[@]}")"
  printf 'output_files:\n'
  find "$OUTPUT_DIR" -maxdepth 2 -type f \
    \( -name '*.npz' -o -name '*.csv' -o -name '*.png' \) -print | sort
} > "$MANIFEST"

printf '\ncompleted_phases=%s\n' "$(join_phases "${completed_phases[@]}")"
printf 'skipped_phases=%s\n' "$(join_phases "${skipped_phases[@]}")"
printf 'output_dir=%s\n' "$OUTPUT_DIR"
printf 'log_dir=%s\n' "$LOG_DIR"
printf 'manifest=%s\n' "$MANIFEST"
printf 'generated_outputs:\n'
find "$OUTPUT_DIR" -maxdepth 2 -type f \
  \( -name '*.npz' -o -name '*.csv' -o -name '*.png' -o -name 'evaluation_manifest.txt' \) \
  -print | sort
printf 'disk_usage:\n'
du -sh "$OUTPUT_DIR" "$LOG_DIR" "$TMP_ROOT"
