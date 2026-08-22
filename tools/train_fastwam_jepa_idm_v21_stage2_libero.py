from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import train_fastwam_jepa_idm_v2_stage2_libero as v2


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
_ORIGINAL_VALIDATE_STAGE1 = v2.validate_stage1_config
_ACTIVE_ARGS: argparse.Namespace | None = None


def _flag_present(name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in sys.argv[1:])


def parse_args() -> argparse.Namespace:
    global _ACTIVE_ARGS
    for required in (
        "--libero-data-root",
        "--dataset-stats-path",
        "--stage1-checkpoint",
        "--action-checkpoint",
        "--vjepa-repo",
        "--vjepa-checkpoint",
    ):
        if not _flag_present(required):
            raise ValueError(f"v2.1 Stage2 requires explicit {required}")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dataset-stats-path", required=True)
    pre_parser.add_argument(
        "--action-grad-to-predictor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    pre_parser.add_argument("--unfreeze-action-last-n", type=int, choices=(0, 2, 4), default=0)
    extra, remaining = pre_parser.parse_known_args(sys.argv[1:])
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv
    args.dataset_stats_path = str(Path(extra.dataset_stats_path).expanduser())
    args.action_grad_to_predictor = bool(extra.action_grad_to_predictor)
    args.unfreeze_action_last_n = int(extra.unfreeze_action_last_n)
    args.current_frame_count = 4
    args.future_frame_count = 4
    if not _flag_present("--lambda-cos"):
        args.lambda_cos = 0.1
    if not _flag_present("--lr-action"):
        args.lr_action = 1.0e-6
    _ACTIVE_ARGS = args
    return args


def compose_cfg(config_name: str, task: str):
    cfg = _ORIGINAL_COMPOSE_CFG(config_name, task)
    if _ACTIVE_ARGS is None:
        raise RuntimeError("v2.1 arguments were not initialized")
    OmegaConf.update(cfg, "data.train.action_video_freq_ratio", 1, merge=False)
    OmegaConf.update(cfg, "data.train.num_frames", 33, merge=False)
    OmegaConf.update(
        cfg,
        "data.train.pretrained_norm_stats",
        str(Path(_ACTIVE_ARGS.dataset_stats_path).expanduser().resolve()),
        merge=False,
    )
    return cfg


def canonicalize_v21_libero_batch(
    batch: dict[str, Any],
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    video = batch.get("video")
    if not torch.is_tensor(video) or video.ndim != 5 or int(video.shape[1]) != 3:
        raise ValueError("v2.1 LIBERO video must be [B,3,T,H,W]")
    if int(video.shape[2]) < 5:
        raise ValueError(f"v2.1 needs continuous frames t..t+4, got T={video.shape[2]}")
    current_raw = video[:, :, 0:1]
    current_video = current_raw.repeat(1, 1, 4, 1, 1)
    future_video = video[:, :, 1:5]
    current_video = v2.resize_video(current_video, size=int(args.vjepa_img_size))
    future_video = v2.resize_video(future_video, size=int(args.vjepa_img_size))
    if not torch.equal(current_video[:, :, 0], current_video[:, :, 3]):
        raise RuntimeError("v2.1 current-frame repeat contract failed")

    action = batch.get("action")
    if not torch.is_tensor(action) or action.ndim != 3 or int(action.shape[-1]) != 7:
        raise ValueError("v2.1 action must be [B,T,7]")
    if int(action.shape[1]) < 32:
        raise ValueError(f"v2.1 action target needs a(t:t+31), got T={action.shape[1]}")
    action = action[:, :32]
    context = batch.get("context")
    context_mask = batch.get("context_mask")
    if not torch.is_tensor(context) or not torch.is_tensor(context_mask):
        raise ValueError("v2.1 batch is missing context/context_mask")
    result: dict[str, Any] = {
        "video": current_video.to(device=device, dtype=dtype, non_blocking=True),
        "future_video": future_video.to(device=device, dtype=dtype, non_blocking=True),
        "action": action.to(device=device, dtype=dtype, non_blocking=True),
        "context": context.to(device=device, dtype=dtype, non_blocking=True),
        "context_mask": context_mask.to(device=device, dtype=torch.bool, non_blocking=True),
        "source_name": "libero",
    }
    action_is_pad = batch.get("action_is_pad")
    if torch.is_tensor(action_is_pad):
        if int(action_is_pad.shape[1]) < 32:
            raise ValueError("v2.1 action padding mask is shorter than 32")
        result["action_is_pad"] = action_is_pad[:, :32].to(
            device=device, dtype=torch.bool, non_blocking=True
        )
    if bool(args.use_proprio):
        proprio = batch.get("proprio")
        if not torch.is_tensor(proprio):
            raise ValueError("v2.1 requires proprio s(t)")
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        if proprio.ndim != 2:
            raise ValueError("v2.1 proprio must be [B,T,D] or [B,D]")
        result["proprio"] = proprio.to(device=device, dtype=dtype, non_blocking=True)
    return result


def predicted_action_forward(
    module: torch.nn.Module,
    *,
    current_jepa_tokens: torch.Tensor,
    pred_future_jepa_tokens: torch.Tensor,
    condition_context: torch.Tensor,
    condition_mask: torch.Tensor,
    noisy_action: torch.Tensor,
    timestep_action: torch.Tensor,
    action_grad_to_predictor: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    future_for_action = (
        pred_future_jepa_tokens
        if bool(action_grad_to_predictor)
        else pred_future_jepa_tokens.detach()
    )
    action_context, action_context_mask = module.jepa_adapter(
        current_jepa_tokens=current_jepa_tokens,
        future_jepa_tokens=future_for_action,
        base_context=condition_context,
        base_context_mask=condition_mask,
    )
    pred_action = module.action_expert(
        action_tokens=noisy_action,
        timestep=timestep_action,
        context=action_context,
        context_mask=action_context_mask,
    )
    return pred_action, action_context, action_context_mask


def stage2_forward_loss(
    model: torch.nn.Module,
    sample: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    module = v2.unwrap_ddp(model)
    if module is None:
        raise RuntimeError("DDP unwrap failed")
    video = sample["video"]
    future_video = sample["future_video"]
    action = sample["action"]
    context = sample["context"]
    context_mask = sample["context_mask"]
    proprio = sample.get("proprio")
    condition_context, condition_mask = module._append_proprio_to_context(
        context=context, context_mask=context_mask, proprio=proprio
    )
    current_jepa_tokens = module._encode_jepa_video(video)
    target_future_jepa_tokens = module._encode_jepa_video(future_video).detach()
    future_out = module.future_predictor(
        current_jepa_tokens=current_jepa_tokens,
        condition_context=condition_context,
        condition_mask=condition_mask,
    )
    pred_future_jepa_tokens = future_out["pred_future_tokens"]
    if tuple(pred_future_jepa_tokens.shape) != tuple(target_future_jepa_tokens.shape):
        raise ValueError("v2.1 predicted/teacher future token shapes differ")

    batch_size = int(action.shape[0])
    noise_action = torch.randn_like(action)
    timestep_action = module.train_action_scheduler.sample_training_t(
        batch_size=batch_size, device=action.device, dtype=action.dtype
    )
    noisy_action = module.train_action_scheduler.add_noise(action, noise_action, timestep_action)
    target_action = module.train_action_scheduler.training_target(
        action, noise_action, timestep_action
    )
    pred_action, action_context, action_context_mask = predicted_action_forward(
        module,
        current_jepa_tokens=current_jepa_tokens,
        pred_future_jepa_tokens=pred_future_jepa_tokens,
        condition_context=condition_context,
        condition_mask=condition_mask,
        noisy_action=noisy_action,
        timestep_action=timestep_action,
        action_grad_to_predictor=bool(args.action_grad_to_predictor),
    )
    action_loss_token = F.mse_loss(
        pred_action.float(), target_action.float(), reduction="none"
    ).mean(dim=2)
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is not None:
        valid = (~action_is_pad).to(action_loss_token.dtype)
        action_loss_per_sample = (action_loss_token * valid).sum(1) / valid.sum(1).clamp(min=1.0)
    else:
        action_loss_per_sample = action_loss_token.mean(1)
    action_weight = module.train_action_scheduler.training_weight(timestep_action).to(
        action_loss_per_sample.dtype
    )
    loss_action = (action_loss_per_sample * action_weight).mean()
    loss_future_l1 = F.l1_loss(
        pred_future_jepa_tokens.float(), target_future_jepa_tokens.float()
    )
    loss_future_cos = v2.future_cosine_loss(
        pred_future_jepa_tokens, target_future_jepa_tokens
    )
    loss_future = loss_future_l1 + float(args.lambda_cos) * loss_future_cos
    loss_total = loss_action + float(args.lambda_future) * loss_future
    module.last_forward_shapes = {
        "current_jepa_tokens": tuple(current_jepa_tokens.shape),
        "target_future_jepa_tokens": tuple(target_future_jepa_tokens.shape),
        "pred_future_jepa_tokens": tuple(pred_future_jepa_tokens.shape),
        "action_context": tuple(action_context.shape),
        "action_context_mask": tuple(action_context_mask.shape),
        "pred_action": tuple(pred_action.shape),
        "action_grad_to_predictor": str(bool(args.action_grad_to_predictor)).lower(),
    }
    return loss_total, {
        "loss_total": loss_total.detach(),
        "loss_action": loss_action.detach(),
        "loss_future_jepa": loss_future.detach(),
        "loss_future_l1": loss_future_l1.detach(),
        "loss_future_cos": loss_future_cos.detach(),
    }


def configure_trainability(model: torch.nn.Module) -> list[dict[str, Any]]:
    if _ACTIVE_ARGS is None:
        raise RuntimeError("v2.1 arguments were not initialized")
    model.vjepa_encoder.requires_grad_(False)
    model.future_predictor.requires_grad_(True)
    model.jepa_adapter.requires_grad_(True)
    model.action_expert.requires_grad_(False)
    if model.proprio_encoder is not None:
        model.proprio_encoder.requires_grad_(True)
    last_n = int(_ACTIVE_ARGS.unfreeze_action_last_n)
    if last_n:
        for block in model.action_expert.blocks[-last_n:]:
            block.requires_grad_(True)
        model.action_expert.head.requires_grad_(True)
    groups = [
        {"name": "adapter", "params": [p for p in model.jepa_adapter.parameters() if p.requires_grad]},
        {"name": "future_predictor", "params": [p for p in model.future_predictor.parameters() if p.requires_grad]},
        {"name": "action_dit", "params": [p for p in model.action_expert.parameters() if p.requires_grad]},
    ]
    if model.proprio_encoder is not None:
        groups.append(
            {"name": "proprio", "params": [p for p in model.proprio_encoder.parameters() if p.requires_grad]}
        )
    rank = int(getattr(_ACTIVE_ARGS, "_rank", 0))
    v2.rank0_print(rank, f"action_unfreeze_last_n={last_n}", flush=True)
    v2.rank0_print(rank, f"action_trainable_parameters={v2.trainable_count(model.action_expert)}", flush=True)
    v2.rank0_print(rank, f"predictor_trainable_parameters={v2.trainable_count(model.future_predictor)}", flush=True)
    v2.rank0_print(rank, f"adapter_trainable_parameters={v2.trainable_count(model.jepa_adapter)}", flush=True)
    return groups


def _validate_temporal_metadata(config: dict[str, Any], *, name: str) -> None:
    for key, expected in TEMPORAL_METADATA.items():
        got = config.get(key)
        if got != expected:
            raise ValueError(f"{name} temporal metadata mismatch for {key}: {got!r} != {expected!r}")


def validate_stage1_config(
    stage1_payload: dict[str, Any], args: argparse.Namespace, *, rank: int
) -> None:
    _ORIGINAL_VALIDATE_STAGE1(stage1_payload, args, rank=rank)
    _validate_temporal_metadata(v2.checkpoint_config(stage1_payload), name="Stage1 checkpoint")


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    loss_dict: dict[str, float],
    load_stats: dict[str, Any],
) -> Path:
    module = v2.unwrap_ddp(model)
    if module is None:
        raise RuntimeError("Cannot save an empty model")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_step_{int(step):06d}.pt"
    config = {
        **dict(vars(args)),
        **TEMPORAL_METADATA,
        "current_frame_count": 4,
        "future_frame_count": 4,
        "resolved_num_future_tokens": int(args.resolved_num_future_tokens),
        "action_dim": 7,
    }
    torch.save(
        {
            "model": module.state_dict(),
            "future_predictor": module.future_predictor.state_dict(),
            "jepa_adapter": module.jepa_adapter.state_dict(),
            "proprio_encoder": None if module.proprio_encoder is None else module.proprio_encoder.state_dict(),
            "action_expert": module.action_expert.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "args": dict(vars(args)),
            "config": config,
            "temporal_metadata": dict(TEMPORAL_METADATA),
            "loss_dict": dict(loss_dict),
            "load_stats": load_stats,
        },
        path,
    )
    return path


def main() -> None:
    v2.parse_args = parse_args
    v2.compose_cfg = compose_cfg
    v2.canonicalize_libero_batch = canonicalize_v21_libero_batch
    v2.stage2_forward_loss = stage2_forward_loss
    v2.configure_trainability = configure_trainability
    v2.validate_stage1_config = validate_stage1_config
    v2.save_checkpoint = save_checkpoint
    v2.main()


if __name__ == "__main__":
    main()
