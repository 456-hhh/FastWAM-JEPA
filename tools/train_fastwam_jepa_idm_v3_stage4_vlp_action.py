from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (SRC_ROOT, PROJECT_ROOT, TOOLS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import train_fastwam_jepa_idm_v2_stage2_libero as stage2_libero
from fastwam.models.wan22.pairwise_stage4 import PairwiseStage4Model


DEFAULT_VJEPA_REPO = stage2_libero.DEFAULT_VJEPA_REPO
DEFAULT_VJEPA_CHECKPOINT = stage2_libero.DEFAULT_VJEPA_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastWAM-JEPA-IDM v3 Stage4 VLP-to-action latent training."
    )
    parser.add_argument("--stage2-checkpoint", default=None)
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--vjepa-repo", default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=["-1_1", "0_1"])
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--raw-vjepa-tokens", type=int, default=512)
    parser.add_argument("--current-frame-count", type=int, default=4)
    parser.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda-va-to-l", type=float, default=1.0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", default="runs/fastwam_jepa_idm_v3_stage4_vlp_action")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _fit_sequence(
    value: torch.Tensor,
    length: int,
    *,
    dim: int,
    pad_value: bool | float = 0.0,
) -> torch.Tensor:
    current = int(value.shape[dim])
    if current == int(length):
        return value
    if current > int(length):
        return value.narrow(dim, 0, int(length))
    pad_shape = list(value.shape)
    pad_shape[dim] = int(length) - current
    if value.dtype == torch.bool:
        pad = torch.full(pad_shape, bool(pad_value), dtype=value.dtype, device=value.device)
    else:
        pad = torch.full(pad_shape, float(pad_value), dtype=value.dtype, device=value.device)
    return torch.cat([value, pad], dim=dim)


def canonicalize_stage4_batch(
    batch: dict[str, Any],
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | None]:
    if not isinstance(batch, dict):
        raise ValueError(f"Expected LIBERO dataloader batch dict, got {type(batch)}.")

    video = batch["video"]
    if not torch.is_tensor(video) or video.ndim != 5 or int(video.shape[1]) != 3:
        raise ValueError(f"LIBERO batch video must be [B, 3, T, H, W], got {tuple(video.shape)}.")
    if int(video.shape[2]) < int(args.current_frame_count):
        raise ValueError(
            f"LIBERO video has T={video.shape[2]}, but Stage4 requires at least {args.current_frame_count} frames."
        )
    current_video = video[:, :, : int(args.current_frame_count)]
    current_video = stage2_libero.resize_video(current_video, size=int(args.vjepa_img_size))

    action = batch["action"]
    if not torch.is_tensor(action) or action.ndim != 3 or int(action.shape[-1]) != 7:
        raise ValueError(f"LIBERO action labels must be [B, T_a, 7], got {tuple(action.shape)}.")
    action = _fit_sequence(action, int(args.action_horizon), dim=1, pad_value=0.0)

    context = batch["context"]
    context_mask = batch["context_mask"]
    if not torch.is_tensor(context) or not torch.is_tensor(context_mask):
        raise ValueError("LIBERO batch must contain tensor context/context_mask.")
    if context.ndim != 3 or context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
        raise ValueError(
            f"context/context_mask must be [B, L, D]/[B, L], got {tuple(context.shape)} and {tuple(context_mask.shape)}."
        )
    if int(context.shape[-1]) != 4096:
        raise ValueError(f"context last dim must be 4096, got {context.shape[-1]}.")
    context = _fit_sequence(context, int(args.context_tokens), dim=1, pad_value=0.0)
    context_mask = _fit_sequence(context_mask.to(dtype=torch.bool), int(args.context_tokens), dim=1, pad_value=False)

    proprio_out: torch.Tensor | None = None
    if bool(args.use_proprio):
        proprio = batch.get("proprio")
        if not torch.is_tensor(proprio):
            raise ValueError("--use-proprio=True but batch has no proprio tensor.")
        if proprio.ndim == 3:
            proprio_out = proprio[:, 0, :]
        elif proprio.ndim == 2:
            proprio_out = proprio
        else:
            raise ValueError(f"proprio must be [B, T, D] or [B, D], got {tuple(proprio.shape)}.")

    return {
        "video": current_video.to(device=device, dtype=dtype, non_blocking=True),
        "action": action.to(device=device, dtype=dtype, non_blocking=True),
        "context": context.to(device=device, dtype=dtype, non_blocking=True),
        "context_mask": context_mask.to(device=device, dtype=torch.bool, non_blocking=True),
        "proprio": None if proprio_out is None else proprio_out.to(device=device, dtype=dtype, non_blocking=True),
    }


def _filter_shape_matching_state(
    module: torch.nn.Module,
    state_dict: dict[str, Any] | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if not state_dict:
        return {}, []
    own_state = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    mismatches: list[dict[str, Any]] = []
    for key, value in stage2_libero.strip_prefixes(state_dict).items():
        if not torch.is_tensor(value) or key not in own_state:
            continue
        if tuple(value.shape) != tuple(own_state[key].shape):
            mismatches.append(
                {
                    "key": str(key),
                    "source_shape": tuple(value.shape),
                    "target_shape": tuple(own_state[key].shape),
                }
            )
            continue
        matched[str(key)] = value.to(dtype=own_state[key].dtype)
    return matched, mismatches


def load_shape_matching_relaxed(
    module: torch.nn.Module,
    state_dict: dict[str, Any] | None,
    *,
    name: str,
    rank: int,
) -> dict[str, Any]:
    matched, mismatches = _filter_shape_matching_state(module, state_dict)
    if not matched:
        stage2_libero.rank0_print(rank, f"{name}_load skipped: no compatible keys", flush=True)
        return {
            "loaded_keys_count": 0,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatch_count": len(mismatches),
            "shape_mismatch": mismatches[:20],
        }
    missing, unexpected = module.load_state_dict(matched, strict=False)
    stats = {
        "loaded_keys_count": len(matched),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "shape_mismatch_count": len(mismatches),
        "shape_mismatch": mismatches[:20],
    }
    stage2_libero.rank0_print(
        rank,
        f"{name}_load loaded_keys={len(matched)} missing={len(missing)} "
        f"unexpected={len(unexpected)} shape_mismatch={len(mismatches)}",
        flush=True,
    )
    return stats


def load_optional_stage1_stage2_checkpoint(
    model: PairwiseStage4Model,
    stage1_checkpoint: str | Path | None,
    stage2_checkpoint: str | Path | None,
    rank: int,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "stage1_checkpoint": None,
        "stage2_checkpoint": None,
    }

    if stage1_checkpoint is None and stage2_checkpoint is None:
        stage2_libero.rank0_print(
            rank,
            "WARNING no Stage1/Stage2 checkpoint provided; Stage4 training will start from random initialization.",
            flush=True,
        )
        return stats

    if stage1_checkpoint is not None:
        stage1_path = resolve_path(stage1_checkpoint)
        if stage1_path is None or not stage1_path.exists():
            raise FileNotFoundError(f"Stage1 checkpoint does not exist: {stage1_path}")
        payload = torch.load(stage1_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Stage1 checkpoint must be dict, got {type(payload)}.")
        stats["stage1_checkpoint"] = {
            "checkpoint": str(stage1_path),
            "language_projector": load_shape_matching_relaxed(
                model.language_projector,
                payload.get("language_projector"),
                name="stage1_language_projector",
                rank=rank,
            ),
            "action_encoder": load_shape_matching_relaxed(
                model.action_encoder,
                payload.get("action_encoder"),
                name="stage1_action_encoder",
                rank=rank,
            ),
        }

    if stage2_checkpoint is not None:
        stage2_path = resolve_path(stage2_checkpoint)
        if stage2_path is None or not stage2_path.exists():
            raise FileNotFoundError(f"Stage2 checkpoint does not exist: {stage2_path}")
        payload = torch.load(stage2_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Stage2 checkpoint must be dict, got {type(payload)}.")
        stats["stage2_checkpoint"] = {
            "checkpoint": str(stage2_path),
            "language_projector": load_shape_matching_relaxed(
                model.language_projector,
                stage2_libero.extract_nested_state(
                    payload,
                    direct_keys=("language_projector",),
                    model_prefixes=("language_projector.", "module.language_projector."),
                ),
                name="stage2_language_projector",
                rank=rank,
            ),
            "action_encoder": load_shape_matching_relaxed(
                model.action_encoder,
                stage2_libero.extract_nested_state(
                    payload,
                    direct_keys=("action_encoder",),
                    model_prefixes=("action_encoder.", "module.action_encoder."),
                ),
                name="stage2_action_encoder",
                rank=rank,
            ),
            "vision_projector": load_shape_matching_relaxed(
                model.vision_projector,
                stage2_libero.extract_nested_state(
                    payload,
                    direct_keys=("vision_projector",),
                    model_prefixes=("vision_projector.", "module.vision_projector."),
                ),
                name="stage2_vision_projector",
                rank=rank,
            ),
            "fusion_vl": load_shape_matching_relaxed(
                model.fusion_vl,
                stage2_libero.extract_nested_state(
                    payload,
                    direct_keys=("fusion_vl",),
                    model_prefixes=("fusion_vl.", "module.fusion_vl."),
                ),
                name="stage2_fusion_vl",
                rank=rank,
            ),
        }

    return stats


def reduce_metrics(
    metrics: dict[str, torch.Tensor],
    *,
    ddp_enabled: bool,
    world_size: int,
) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for name, value in metrics.items():
        tensor = value.detach().float()
        if ddp_enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / float(world_size)
        reduced[name] = float(tensor.item())
    return reduced


def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    world_size: int,
    last_metrics: dict[str, float],
    load_stats: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    module = stage2_libero.unwrap_ddp(model)
    if not isinstance(module, PairwiseStage4Model):
        raise TypeError(f"Unexpected model type: {type(module)}")
    path = output_dir / f"checkpoint_step_{int(step):06d}.pt"
    torch.save(
        {
            "model": module.state_dict(),
            "language_projector": module.language_projector.state_dict(),
            "action_encoder": module.action_encoder.state_dict(),
            "vision_projector": module.vision_projector.state_dict(),
            "proprio_projector": module.proprio_projector.state_dict(),
            "fusion_vl": module.fusion_vl.state_dict(),
            "fusion_vlp": module.fusion_vlp.state_dict(),
            "fusion_va": module.fusion_va.state_dict(),
            "vlp_to_action_head": module.vlp_to_action_head.state_dict(),
            "va_to_language_head": module.va_to_language_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "args": vars(args),
            "world_size": int(world_size),
            "stage": "stage4_vlp_action",
            "losses": dict(last_metrics),
            "load_stats": load_stats,
        },
        path,
    )
    return path


def main() -> None:
    args = parse_args()
    if int(args.context_tokens) != 128:
        raise ValueError("--context-tokens must remain 128 for Stage4.")
    if int(args.action_horizon) != 32:
        raise ValueError("--action-horizon must remain 32 for Stage4.")
    if int(args.current_frame_count) != 4:
        raise ValueError("--current-frame-count must remain 4 for Stage4.")
    if int(args.raw_vjepa_tokens) != 512:
        raise ValueError("--raw-vjepa-tokens must remain 512 for Stage4.")
    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive.")
    if int(args.grad_accum_steps) <= 0:
        raise ValueError("--grad-accum-steps must be positive.")
    if float(args.tau) <= 0.0:
        raise ValueError("--tau must be positive.")
    if float(args.lambda_va_to_l) < 0.0:
        raise ValueError("--lambda-va-to-l must be non-negative.")

    stage2_libero.seed_everything(int(args.seed))
    ddp_enabled, world_size, rank, local_rank, device = stage2_libero.init_distributed_from_env()
    param_dtype, autocast_dtype = stage2_libero.precision_to_dtype(str(args.precision), device)
    cfg = stage2_libero.compose_cfg(str(args.config_name), str(args.task))
    loader, sampler = stage2_libero.build_libero_loader(
        cfg,
        args=args,
        ddp_enabled=ddp_enabled,
        rank=rank,
        world_size=world_size,
    )
    data_iter = iter(loader)

    vjepa_encoder = stage2_libero.build_vjepa_encoder(args, device=device, dtype=param_dtype)
    model: nn.Module = PairwiseStage4Model().to(device=device, dtype=param_dtype)
    load_stats = load_optional_stage1_stage2_checkpoint(
        stage2_libero.unwrap_ddp(model),
        args.stage1_checkpoint,
        args.stage2_checkpoint,
        rank,
    )
    vjepa_encoder.eval()
    vjepa_encoder.requires_grad_(False)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )

    output_dir = resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir is required.")
    checkpoint_dir = output_dir / "checkpoints"

    stage2_libero.rank0_print(rank, f"ddp_enabled={ddp_enabled}", flush=True)
    stage2_libero.rank0_print(rank, f"world_size={world_size}", flush=True)
    stage2_libero.rank0_print(rank, f"rank={rank}", flush=True)
    stage2_libero.rank0_print(rank, f"local_rank={local_rank}", flush=True)
    stage2_libero.rank0_print(rank, f"local_batch_size_per_gpu={args.batch_size}", flush=True)
    stage2_libero.rank0_print(rank, f"grad_accum_steps={args.grad_accum_steps}", flush=True)
    stage2_libero.rank0_print(
        rank,
        f"effective_global_batch_size={int(args.batch_size) * int(world_size) * int(args.grad_accum_steps)}",
        flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    last_metrics: dict[str, float] = {}
    start_time = time.time()
    accum_start = time.time()

    while update_step < int(args.steps):
        if sampler is not None and micro_step % max(len(loader), 1) == 0:
            sampler.set_epoch(micro_step // max(len(loader), 1))
        try:
            raw_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            raw_batch = next(data_iter)

        batch = canonicalize_stage4_batch(
            raw_batch,
            args=args,
            device=device,
            dtype=param_dtype,
        )
        with torch.no_grad():
            current_jepa_tokens = vjepa_encoder(batch["video"])
        if current_jepa_tokens.ndim != 3 or int(current_jepa_tokens.shape[1]) != int(args.raw_vjepa_tokens):
            raise ValueError(
                f"current_jepa_tokens must be [B, {args.raw_vjepa_tokens}, {args.vjepa_dim}], got {tuple(current_jepa_tokens.shape)}."
            )

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            out = model(
                world_tokens=current_jepa_tokens,
                text_tokens=batch["context"],
                text_mask=batch["context_mask"],
                proprio=batch["proprio"],
                action_chunk=batch["action"],
                tau=float(args.tau),
            )
            loss = (
                out["loss_vlp_to_a"] + float(args.lambda_va_to_l) * out["loss_va_to_l"]
            ) / float(args.grad_accum_steps)
        loss.backward()
        micro_step += 1
        if micro_step % int(args.grad_accum_steps) != 0:
            continue

        if float(args.max_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        z_task = out["z_task"]
        z_task_token = out["z_task_token"]
        metrics = {
            "loss": out["loss_vlp_to_a"] + float(args.lambda_va_to_l) * out["loss_va_to_l"],
            "loss_vlp_to_a": out["loss_vlp_to_a"],
            "loss_va_to_l": out["loss_va_to_l"],
            "retrieval_acc_vlp_to_a": out["retrieval_acc_vlp_to_a"],
            "retrieval_acc_va_to_l": out["retrieval_acc_va_to_l"],
            "z_task_norm": z_task.detach().float().norm(dim=-1).mean(),
            "z_task_token_norm": z_task_token.detach().float().norm(dim=-1).mean(),
        }
        reduced = reduce_metrics(metrics, ddp_enabled=ddp_enabled, world_size=world_size)
        last_metrics = dict(reduced)
        if rank == 0 and (update_step == 1 or update_step % int(args.log_every) == 0):
            elapsed = max(time.time() - accum_start, 1.0e-6)
            print(
                " ".join(
                    [
                        f"step={update_step}",
                        f"loss={reduced['loss']:.6f}",
                        f"loss_vlp_to_a={reduced['loss_vlp_to_a']:.6f}",
                        f"loss_va_to_l={reduced['loss_va_to_l']:.6f}",
                        f"retrieval_acc_vlp_to_a={reduced['retrieval_acc_vlp_to_a']:.6f}",
                        f"retrieval_acc_va_to_l={reduced['retrieval_acc_va_to_l']:.6f}",
                        f"z_task_norm={reduced['z_task_norm']:.6f}",
                        f"z_task_token_norm={reduced['z_task_token_norm']:.6f}",
                        f"lr={optimizer.param_groups[0]['lr']:.6g}",
                        f"iter_time_sec={elapsed:.3f}",
                    ]
                ),
                flush=True,
            )
            accum_start = time.time()

        if rank == 0 and int(args.save_every) > 0 and update_step % int(args.save_every) == 0:
            path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=checkpoint_dir,
                step=update_step,
                args=args,
                world_size=world_size,
                last_metrics=last_metrics,
                load_stats=load_stats,
            )
            print(f"saved_checkpoint={path}", flush=True)

    if rank == 0:
        path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=checkpoint_dir,
            step=update_step,
            args=args,
            world_size=world_size,
            last_metrics=last_metrics,
            load_stats=load_stats,
        )
        print(
            " ".join(
                [
                    f"saved_final_checkpoint={path}",
                    f"step={update_step}",
                    f"loss={last_metrics.get('loss', float('nan')):.6f}",
                    f"elapsed_sec={time.time() - start_time:.3f}",
                ]
            ),
            flush=True,
        )

    if ddp_enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
