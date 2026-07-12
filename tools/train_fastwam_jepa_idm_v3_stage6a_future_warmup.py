from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (SRC_ROOT, PROJECT_ROOT, TOOLS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fastwam_jepa_runtime_guard import configure_runtime_stability
import train_fastwam_jepa_idm_v2_stage2_libero as stage2_libero
import train_fastwam_jepa_idm_v3_stage5_ztask_action as stage5_base


DEFAULT_VJEPA_REPO = stage5_base.DEFAULT_VJEPA_REPO
DEFAULT_VJEPA_CHECKPOINT = stage5_base.DEFAULT_VJEPA_CHECKPOINT
DEFAULT_STAGE1_DIR = stage5_base.DEFAULT_STAGE1_DIR
DEFAULT_ACTION_CHECKPOINT = stage5_base.DEFAULT_ACTION_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastWAM-JEPA-IDM v3 Stage6a future predictor warmup."
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--stage1-checkpoint", default="auto")
    parser.add_argument("--action-checkpoint", default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument("--vjepa-repo", default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=["-1_1", "0_1"])
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--current-frame-count", type=int, default=4)
    parser.add_argument("--future-frame-count", type=int, default=4)
    parser.add_argument("--num-future-tokens", default="auto")
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--future-predictor-layers", type=int, default=24)
    parser.add_argument("--future-predictor-hidden-dim", type=int, default=1024)
    parser.add_argument("--future-predictor-heads", type=int, default=16)
    parser.add_argument("--adapter-current-tokens", type=int, default=64)
    parser.add_argument("--adapter-future-tokens", type=int, default=64)
    parser.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-proprio-in-warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lambda-future", type=float, default=1.0)
    parser.add_argument("--lambda-cos", type=float, default=0.0)
    parser.add_argument("--lr-predictor", type=float, default=2.0e-5)
    parser.add_argument("--lr-proprio", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--freeze-vjepa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output-dir", default="runs/fastwam_jepa_idm_v3_stage6a_future_warmup")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-rank0-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-wsl-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--runtime-log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--runtime-log-path", default=None)
    parser.add_argument("--runtime-log-max-mb", type=int, default=100)
    return parser.parse_args()


def configure_trainability(model: torch.nn.Module, *, args: argparse.Namespace) -> list[dict[str, Any]]:
    model.vjepa_encoder.requires_grad_(False)
    model.action_expert.requires_grad_(False)
    model.jepa_adapter.requires_grad_(False)
    model.future_predictor.requires_grad_(True)
    if model.proprio_encoder is not None:
        model.proprio_encoder.requires_grad_(bool(args.train_proprio_in_warmup))

    groups = [
        {"name": "future_predictor", "params": [p for p in model.future_predictor.parameters() if p.requires_grad]},
    ]
    if model.proprio_encoder is not None and bool(args.train_proprio_in_warmup):
        groups.append({"name": "proprio", "params": [p for p in model.proprio_encoder.parameters() if p.requires_grad]})
    return groups


def future_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()


def stage6a_forward_loss(
    model: torch.nn.Module,
    sample: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    module = stage5_base.unwrap_ddp(model)
    if module is None:
        raise RuntimeError("DDP unwrap failed.")

    video = sample["video"]
    future_video = sample["future_video"]
    context = sample["context"]
    context_mask = sample["context_mask"]
    proprio = sample.get("proprio")

    condition_context, condition_mask = module._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=proprio,
    )
    current_jepa_tokens = module._encode_jepa_video(video)
    target_future_jepa_tokens = module._encode_jepa_video(future_video).detach()
    future_out = module.future_predictor(
        current_jepa_tokens=current_jepa_tokens,
        condition_context=condition_context,
        condition_mask=condition_mask,
    )
    pred_future_jepa_tokens = future_out["pred_future_tokens"]

    loss_future_l1 = F.l1_loss(pred_future_jepa_tokens.float(), target_future_jepa_tokens.float())
    loss_future_cos = future_cosine_loss(pred_future_jepa_tokens, target_future_jepa_tokens)
    loss_future_jepa = loss_future_l1 + float(args.lambda_cos) * loss_future_cos
    loss_total = float(args.lambda_future) * loss_future_jepa

    module.last_forward_shapes = {
        "current_jepa_tokens": tuple(current_jepa_tokens.shape),
        "target_future_jepa_tokens": tuple(target_future_jepa_tokens.shape),
        "pred_future_jepa_tokens": tuple(pred_future_jepa_tokens.shape),
        "proprio": None if proprio is None else tuple(proprio.shape),
    }
    return loss_total, {
        "loss_total": loss_total.detach(),
        "loss_future_jepa": loss_future_jepa.detach(),
        "loss_future_l1": loss_future_l1.detach(),
        "loss_future_cos": loss_future_cos.detach(),
    }


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    world_size: int,
    loss_dict: dict[str, float],
    load_stats: dict[str, Any],
) -> Path:
    module = stage5_base.unwrap_ddp(model)
    if module is None:
        raise RuntimeError("Cannot save an empty model.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_step_{int(step):06d}.pt"
    payload = {
        "future_predictor": module.future_predictor.state_dict(),
        "proprio_encoder": None if module.proprio_encoder is None else module.proprio_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "args": vars(args),
        "world_size": int(world_size),
        "stage": "stage6a_future_warmup",
        "loss_dict": dict(loss_dict),
        "load_stats": load_stats,
    }
    torch.save(payload, path)
    return path


def main() -> None:
    args = parse_args()
    args.use_z_task_token = False
    args.stage1_text_action_checkpoint = None
    args.freeze_pairwise_latent = True
    args.lambda_vlp_to_a = 0.0
    args.lambda_va_to_l = 0.0

    ddp_enabled, world_size, rank, local_rank, device = stage5_base.init_distributed_from_env()
    args._rank = rank
    args._world_size = world_size
    args._local_rank = local_rank
    args._rank_seed = int(args.seed) + rank * 100003
    configure_runtime_stability(
        disable_wsl_fallback=bool(args.disable_wsl_fallback),
        log_level=str(args.runtime_log_level),
        log_path=args.runtime_log_path,
        max_log_mb=int(args.runtime_log_max_mb),
    )
    if not bool(args.freeze_vjepa):
        raise ValueError("Stage6a must keep V-JEPA2 frozen.")
    stage5_base.seed_everything(int(args._rank_seed))
    param_dtype, autocast_dtype = stage5_base.precision_to_dtype(str(args.precision), device)

    cfg = stage5_base.compose_cfg(str(args.config_name), str(args.task))
    loader, sampler = stage5_base.build_libero_loader(
        cfg,
        args=args,
        ddp_enabled=ddp_enabled,
        rank=rank,
        world_size=world_size,
    )
    data_iter = iter(loader)
    raw_first = next(data_iter)
    first_batch = stage5_base.canonicalize_libero_batch(raw_first, args=args, device=device, dtype=param_dtype)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise ValueError("cfg.model must resolve to a dict.")
    action_cfg = dict(model_cfg["action_dit_config"])
    action_checkpoint = stage5_base.require_file(args.action_checkpoint, name="--action-checkpoint")
    stage1_checkpoint = stage5_base.resolve_stage1_checkpoint(args.stage1_checkpoint)

    vjepa_encoder = stage5_base.build_vjepa_encoder(args, device=device, dtype=param_dtype)
    stage5_base.infer_token_counts(
        vjepa_encoder=vjepa_encoder,
        first_batch=first_batch,
        args=args,
        rank=rank,
    )
    action_expert, action_load_stats = stage5_base.build_action_expert(
        action_cfg=action_cfg,
        checkpoint_path=action_checkpoint,
        device=device,
        dtype=param_dtype,
        rank=rank,
    )
    model = stage5_base.build_model(
        cfg=cfg,
        args=args,
        vjepa_encoder=vjepa_encoder,
        action_expert=action_expert,
        device=device,
        dtype=param_dtype,
    )
    stage1_load_stats = stage5_base.load_stage1_checkpoint(model, stage1_checkpoint, args=args, rank=rank)
    groups = configure_trainability(model, args=args)

    param_groups = []
    lr_by_name = {
        "future_predictor": float(args.lr_predictor),
        "proprio": float(args.lr_proprio),
    }
    for group in groups:
        params = list(group["params"])
        if params:
            param_groups.append(
                {
                    "params": params,
                    "lr": lr_by_name[str(group["name"])],
                    "weight_decay": float(args.weight_decay),
                    "name": str(group["name"]),
                }
            )
    if not param_groups:
        raise RuntimeError("No trainable parameters were configured.")
    optimizer = torch.optim.AdamW(param_groups)

    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )
    module = stage5_base.unwrap_ddp(model)
    if module is None:
        raise RuntimeError("DDP unwrap failed after wrapping.")
    output_dir = stage5_base.resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir is required.")
    checkpoint_dir = output_dir / "checkpoints"
    load_stats = {
        "action_checkpoint": action_load_stats,
        "stage1_checkpoint": stage1_load_stats,
    }

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

        batch = stage5_base.canonicalize_libero_batch(
            raw_batch,
            args=args,
            device=device,
            dtype=param_dtype,
        )
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            loss_total, loss_dict = stage6a_forward_loss(model, batch, args=args)
            loss = loss_total / float(args.grad_accum_steps)
        loss.backward()
        micro_step += 1
        if micro_step % int(args.grad_accum_steps) != 0:
            continue

        if float(args.max_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        reduced = stage5_base.reduce_loss_dict(loss_dict, ddp_enabled=ddp_enabled, world_size=world_size)
        last_metrics = dict(reduced)
        if rank == 0 and (update_step == 1 or update_step % int(args.log_every) == 0):
            elapsed = max(time.time() - accum_start, 1.0e-6)
            samples_per_sec = (int(args.batch_size) * int(world_size) * int(args.grad_accum_steps)) / elapsed
            print(
                " ".join(
                    [
                        f"step={update_step}",
                        f"loss_total={reduced['loss_total']:.6f}",
                        f"loss_future_jepa={reduced['loss_future_jepa']:.6f}",
                        f"loss_future_l1={reduced['loss_future_l1']:.6f}",
                        f"loss_future_cos={reduced['loss_future_cos']:.6f}",
                        f"lr={optimizer.param_groups[0]['lr']:.6g}",
                        f"iter_time_sec={elapsed:.3f}",
                        f"samples_per_sec={samples_per_sec:.3f}",
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
                loss_dict=last_metrics,
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
            loss_dict=last_metrics,
            load_stats=load_stats,
        )
        print(
            " ".join(
                [
                    f"saved_final_checkpoint={path}",
                    f"step={update_step}",
                    f"loss_total={last_metrics.get('loss_total', float('nan')):.6f}",
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
