from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

import train_fastwam_jepa_idm_v2_stage1_predictor_mixed as v2


TEMPORAL_METADATA = {
    "current_offset": 0,
    "current_repeat": 4,
    "future_offsets": [1, 2, 3, 4],
    "action_start_offset": 0,
    "action_horizon": 32,
    "proprio_offset": 0,
    "future_stride": 1,
    "causal": True,
}

_ORIGINAL_PARSE_ARGS = v2.parse_args
_ORIGINAL_COMPOSE_CFG = v2.compose_cfg
_ACTIVE_ARGS: argparse.Namespace | None = None


def _flag_present(name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in sys.argv[1:])


def parse_args() -> argparse.Namespace:
    global _ACTIVE_ARGS
    for required in ("--libero-data-root", "--dataset-stats-path", "--vjepa-repo", "--vjepa-checkpoint"):
        if not _flag_present(required):
            raise ValueError(f"v2.1 Stage1 requires explicit {required}")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dataset-stats-path", required=True)
    extra, remaining = pre_parser.parse_known_args(sys.argv[1:])
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv
    args.dataset_stats_path = str(Path(extra.dataset_stats_path).expanduser())
    if not _flag_present("--dataset-mix"):
        args.dataset_mix = "libero:1.0"
    if not _flag_present("--libero-task"):
        args.libero_task = "libero_idm_2cam224_1e-4"
    if not _flag_present("--lambda-cos"):
        args.lambda_cos = 0.1
    if not _flag_present("--future-predictor-hidden-dim"):
        args.future_predictor_hidden_dim = 1024
    if not _flag_present("--adapter-current-tokens"):
        args.adapter_current_tokens = 64
    if not _flag_present("--adapter-future-tokens"):
        args.adapter_future_tokens = 64
    args.current_frame_count = 4
    args.future_frame_count = 4
    args.stage1_drop_action_proprio = True
    args.use_proprio = False
    args.resume_expected_temporal_config = {
        "current_repeat": 4,
        "future_offsets": [1, 2, 3, 4],
        "causal": True,
    }
    _ACTIVE_ARGS = args
    return args


def compose_cfg(config_name: str, task: str):
    cfg = _ORIGINAL_COMPOSE_CFG(config_name, task)
    if _ACTIVE_ARGS is None:
        raise RuntimeError("v2.1 arguments were not initialized")
    OmegaConf.update(
        cfg, "data.train.action_video_freq_ratio", 1, merge=False, force_add=True
    )
    OmegaConf.update(cfg, "data.train.num_frames", 33, merge=False, force_add=True)
    OmegaConf.update(
        cfg,
        "data.train.pretrained_norm_stats",
        str(Path(_ACTIVE_ARGS.dataset_stats_path).expanduser().resolve()),
        merge=False,
        force_add=True,
    )
    return cfg


def canonicalize_v21_sample(sample: dict[str, Any], current_T: int, future_T: int) -> dict[str, Any]:
    if int(current_T) != 4 or int(future_T) != 4:
        raise ValueError("v2.1 requires current_repeat=4 and four future frames")
    video = sample.get("video")
    if not torch.is_tensor(video) or video.ndim != 4 or int(video.shape[0]) != 3:
        raise ValueError("v2.1 Stage1 sample video must be [3,T,H,W]")
    if int(video.shape[1]) < 5:
        raise ValueError(f"v2.1 needs continuous frames t..t+4, got T={video.shape[1]}")
    current_frame = video[:, 0:1]
    current_video = current_frame.repeat(1, 4, 1, 1).contiguous()
    future_video = video[:, 1:5].contiguous()
    if not torch.equal(current_video[:, 0], current_video[:, 3]):
        raise RuntimeError("current-frame repeat contract failed")
    canonical = dict(sample)
    canonical["video"] = current_video
    canonical["current_video"] = current_video
    canonical["future_video"] = future_video
    return canonical


def save_checkpoint(
    *,
    output_dir: Path,
    predictor: torch.nn.Module,
    adapter: torch.nn.Module,
    proprio_encoder: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    load_stats: dict[str, Any] | None,
    loss_dict: dict[str, float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_step_{int(step):06d}.pt"
    temporal_config = {
        **TEMPORAL_METADATA,
        "current_frame_count": 4,
        "future_frame_count": 4,
        "resolved_num_future_tokens": int(args.resolved_num_future_tokens),
        "video_size": int(args.video_size),
        "vjepa_img_size": int(args.vjepa_img_size),
        "future_predictor_layers": int(args.future_predictor_layers),
        "future_predictor_hidden_dim": int(args.future_predictor_hidden_dim),
        "future_predictor_heads": int(args.future_predictor_heads),
    }
    payload = {
        "future_predictor": v2.unwrap_ddp(predictor).state_dict(),
        "jepa_adapter": v2.unwrap_ddp(adapter).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "args": dict(vars(args)),
        "config": {**dict(vars(args)), **temporal_config},
        "stage1_temporal_config": temporal_config,
        "vjepa2ac_load_stats": load_stats,
        "vjepa2ac_block_summary": v2.predictor_block_load_summary(load_stats or {}),
        "loss_dict": dict(loss_dict),
    }
    if proprio_encoder is not None:
        payload["proprio_encoder"] = v2.unwrap_ddp(proprio_encoder).state_dict()
    torch.save(payload, path)
    return path


def main() -> None:
    v2.parse_args = parse_args
    v2.compose_cfg = compose_cfg
    v2._canonicalize_stage1_sample_temporal = canonicalize_v21_sample
    v2.save_checkpoint = save_checkpoint
    v2.main()


if __name__ == "__main__":
    main()
