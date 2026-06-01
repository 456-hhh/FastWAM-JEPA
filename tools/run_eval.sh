#!/usr/bin/env bash
# Wrapper to invoke FastWAM RoboTwin eval with correct python environment
set -euo pipefail

FASTWAM_ROOT=/ML-vePFS/protected/jinlei/challenge/dd/FastWAM
PYTHON=/ML-vePFS/protected/jinlei/challenge/dd/envs/fastwam_robotwin/bin/python

cd "$FASTWAM_ROOT"

exec "$PYTHON" experiments/robotwin/run_robotwin_manager.py \
    ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
    EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
    task=robotwin_uncond_3cam_384_1e-4 \
    MULTIRUN.enabled=true \
    MULTIRUN.num_gpus=8 \
    MULTIRUN.max_tasks_per_gpu=2 \
    "$@"
