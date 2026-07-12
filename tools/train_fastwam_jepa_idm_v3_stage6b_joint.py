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
        sys.path.insert(0, str(path_str))

from fastwam_jepa_runtime_guard import configure_runtime_stability
from fastwam.models.wan22.pairwise_stage4 import PairwiseStage4Model, load_stage4_checkpoint
from fastwam.models.wan22.z_task_adapter import ZTaskContextAdapter, append_z_task_to_context
from fastwam.training.pairwise_joint_loss import PairwiseJointLossWeights, combine_stage6_losses
import train_fastwam_jepa_idm_v3_stage5_ztask_action as stage5_base


DEFAULT_VJEPA_REPO = stage5_base.DEFAULT_VJEPA_REPO
DEFAULT_VJEPA_CHECKPOINT = stage5_base.DEFAULT_VJEPA_CHECKPOINT
DEFAULT_STAGE1_DIR = stage5_base.DEFAULT_STAGE1_DIR
DEFAULT_ACTION_CHECKPOINT = stage5_base.DEFAULT_ACTION_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastWAM-JEPA-IDM v3 Stage6b full joint training."
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--stage1-checkpoint", default="auto")
    parser.add_argument("--stage4-checkpoint", required=True)
    parser.add_argument("--stage6a-checkpoint", default=None)
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
    parser.add_argument("--freeze-pairwise-latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda-future", type=float, default=0.1)
    parser.add_argument("--lambda-cos", type=float, default=0.0)
    parser.add_argument("--lambda-vlp-to-a", type=float, default=0.05)
    parser.add_argument("--lambda-va-to-l", type=float, default=0.05)
    parser.add_argument("--pairwise-tau", type=float, default=0.07)
    parser.add_argument("--z-task-dim", type=int, default=1024)
    parser.add_argument("--z-task-gate-init", type=float, default=-4.0)
    parser.add_argument("--lr-adapter", type=float, default=1.0e-4)
    parser.add_argument("--lr-proprio", type=float, default=1.0e-4)
    parser.add_argument("--lr-predictor", type=float, default=2.0e-5)
    parser.add_argument("--lr-action", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--freeze-vjepa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output-dir", default="runs/fastwam_jepa_idm_v3_stage6b_joint")
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


def build_model(
    *,
    cfg,
    args: argparse.Namespace,
    vjepa_encoder: torch.nn.Module,
    action_expert: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    args.use_z_task_token = False
    args.stage1_text_action_checkpoint = None
    model = stage5_base.build_model(
        cfg=cfg,
        args=args,
        vjepa_encoder=vjepa_encoder,
        action_expert=action_expert,
        device=device,
        dtype=dtype,
    )
    text_dim = int(model.text_dim)
    pairwise_latent = PairwiseStage4Model()
    pairwise_latent_load_stats = load_stage4_checkpoint(
        pairwise_latent,
        args.stage4_checkpoint,
        strict=False,
    )
    z_task_adapter = ZTaskContextAdapter(
        z_task_dim=int(args.z_task_dim),
        context_dim=text_dim,
        gate_init=float(args.z_task_gate_init),
        pool_tokens=False,
    )
    setattr(model, "pairwise_latent", pairwise_latent)
    setattr(model, "z_task_adapter", z_task_adapter)
    setattr(model, "pairwise_latent_load_stats", pairwise_latent_load_stats)
    return model.to(device=device, dtype=dtype)


def load_optional_stage6a_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path | None,
    *,
    rank: int,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"checkpoint": None, "loaded": False}
    path = stage5_base.resolve_path(checkpoint_path)
    if path is None or not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Stage6a checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Stage6a checkpoint payload must be dict, got {type(payload)}.")
    stats = {
        "checkpoint": str(path),
        "loaded": True,
        "future_predictor": stage5_base.load_shape_matching(
            model.future_predictor,
            payload.get("future_predictor"),
            name="stage6a_future_predictor",
            rank=rank,
            required=False,
        ),
    }
    if model.proprio_encoder is not None:
        stats["proprio_encoder"] = stage5_base.load_shape_matching(
            model.proprio_encoder,
            payload.get("proprio_encoder"),
            name="stage6a_proprio_encoder",
            rank=rank,
            required=False,
        )
    return stats


def configure_trainability(model: torch.nn.Module, *, args: argparse.Namespace) -> list[dict[str, Any]]:
    model.vjepa_encoder.requires_grad_(False)
    model.future_predictor.requires_grad_(True)
    model.jepa_adapter.requires_grad_(True)
    model.action_expert.requires_grad_(True)
    if model.proprio_encoder is not None:
        model.proprio_encoder.requires_grad_(True)

    pairwise_latent = getattr(model, "pairwise_latent", None)
    if pairwise_latent is not None:
        pairwise_latent.requires_grad_(not bool(args.freeze_pairwise_latent))
    z_task_adapter = getattr(model, "z_task_adapter", None)
    if z_task_adapter is not None:
        z_task_adapter.requires_grad_(True)

    groups = [
        {"name": "adapter", "params": [p for p in model.jepa_adapter.parameters() if p.requires_grad]},
        {"name": "future_predictor", "params": [p for p in model.future_predictor.parameters() if p.requires_grad]},
        {"name": "action_dit", "params": [p for p in model.action_expert.parameters() if p.requires_grad]},
    ]
    if model.proprio_encoder is not None:
        groups.append({"name": "proprio", "params": [p for p in model.proprio_encoder.parameters() if p.requires_grad]})
    if z_task_adapter is not None:
        groups.append({"name": "z_task_adapter", "params": [p for p in z_task_adapter.parameters() if p.requires_grad]})
    if pairwise_latent is not None and not bool(args.freeze_pairwise_latent):
        groups.append({"name": "pairwise_latent", "params": [p for p in pairwise_latent.parameters() if p.requires_grad]})
    return groups


def future_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()


def stage6b_forward_loss(
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
    action = sample["action"]
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
    action_context, action_context_mask = module.jepa_adapter(
        current_jepa_tokens=current_jepa_tokens,
        future_jepa_tokens=pred_future_jepa_tokens,
        base_context=condition_context,
        base_context_mask=condition_mask,
    )

    pairwise_out = module.pairwise_latent.forward_train(
        world_tokens=current_jepa_tokens,
        text_tokens=context,
        text_mask=context_mask,
        proprio=proprio,
        action_chunk=action,
        tau=float(args.pairwise_tau),
    )
    z_task_token = pairwise_out["z_task_token"]
    if tuple(z_task_token.shape[1:]) != (4, int(args.z_task_dim)):
        raise ValueError(
            f"Stage6b expects z_task_token [B,4,{args.z_task_dim}], got {tuple(z_task_token.shape)}."
        )
    z_task_context_token = module.z_task_adapter(z_task_token)
    action_context, action_context_mask = append_z_task_to_context(
        context=action_context,
        context_mask=action_context_mask,
        z_task_context_token=z_task_context_token,
    )
    z_task_gate_value = module.z_task_adapter.gate().detach()

    batch_size = int(action.shape[0])
    noise_action = torch.randn_like(action)
    timestep_action = module.train_action_scheduler.sample_training_t(
        batch_size=batch_size,
        device=action.device,
        dtype=action.dtype,
    )
    noisy_action = module.train_action_scheduler.add_noise(action, noise_action, timestep_action)
    target_action = module.train_action_scheduler.training_target(action, noise_action, timestep_action)
    pred_action = module.action_expert(
        action_tokens=noisy_action,
        timestep=timestep_action,
        context=action_context,
        context_mask=action_context_mask,
    )
    action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is not None:
        valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    else:
        action_loss_per_sample = action_loss_token.mean(dim=1)
    action_weight = module.train_action_scheduler.training_weight(timestep_action).to(
        device=action_loss_per_sample.device,
        dtype=action_loss_per_sample.dtype,
    )
    loss_action = (action_loss_per_sample * action_weight).mean()
    loss_future_l1 = F.l1_loss(pred_future_jepa_tokens.float(), target_future_jepa_tokens.float())
    loss_future_cos = future_cosine_loss(pred_future_jepa_tokens, target_future_jepa_tokens)
    loss_future_jepa = loss_future_l1 + float(args.lambda_cos) * loss_future_cos

    loss_total, joint_loss_items = combine_stage6_losses(
        loss_action=loss_action,
        loss_future_jepa=loss_future_jepa,
        loss_vlp_to_a=pairwise_out["loss_vlp_to_a"],
        loss_va_to_l=pairwise_out["loss_va_to_l"],
        weights=PairwiseJointLossWeights(
            lambda_future=float(args.lambda_future),
            lambda_vlp_to_a=float(args.lambda_vlp_to_a),
            lambda_va_to_l=float(args.lambda_va_to_l),
        ),
    )

    module.last_forward_shapes = {
        "current_jepa_tokens": tuple(current_jepa_tokens.shape),
        "target_future_jepa_tokens": tuple(target_future_jepa_tokens.shape),
        "pred_future_jepa_tokens": tuple(pred_future_jepa_tokens.shape),
        "z_task_token": tuple(z_task_token.shape),
        "z_task_context_token": tuple(z_task_context_token.shape),
        "z_task_context_tokens_appended": int(z_task_context_token.shape[1]),
        "action_context_after_z_task": tuple(action_context.shape),
        "pred_action": tuple(pred_action.shape),
    }
    loss_dict = {
        "loss_total": loss_total.detach(),
        "loss_action": loss_action.detach(),
        "loss_future_jepa": loss_future_jepa.detach(),
        "loss_future_l1": loss_future_l1.detach(),
        "loss_future_cos": loss_future_cos.detach(),
        "loss_vlp_to_a": joint_loss_items["loss_vlp_to_a"],
        "loss_va_to_l": joint_loss_items["loss_va_to_l"],
        "retrieval_acc_vlp_to_a": pairwise_out["retrieval_acc_vlp_to_a"].detach(),
        "retrieval_acc_va_to_l": pairwise_out["retrieval_acc_va_to_l"].detach(),
        "z_task_gate": z_task_gate_value.detach(),
        "z_task_context_tokens": z_task_gate_value.new_tensor(float(z_task_context_token.shape[1])),
    }
    return loss_total, loss_dict


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
        "action_expert": module.action_expert.state_dict(),
        "future_predictor": module.future_predictor.state_dict(),
        "jepa_adapter": module.jepa_adapter.state_dict(),
        "proprio_encoder": None if module.proprio_encoder is None else module.proprio_encoder.state_dict(),
        "pairwise_latent": module.pairwise_latent.state_dict(),
        "z_task_adapter": module.z_task_adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "args": vars(args),
        "world_size": int(world_size),
        "stage": "stage6b_joint",
        "loss_dict": dict(loss_dict),
        "load_stats": load_stats,
        "pairwise_latent_load_stats": getattr(module, "pairwise_latent_load_stats", {}),
    }
    torch.save(payload, path)
    return path


def main() -> None:
    args = parse_args()
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
        raise ValueError("Stage6b must keep V-JEPA2 frozen.")
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
    model = build_model(
        cfg=cfg,
        args=args,
        vjepa_encoder=vjepa_encoder,
        action_expert=action_expert,
        device=device,
        dtype=param_dtype,
    )
    stage1_load_stats = stage5_base.load_stage1_checkpoint(model, stage1_checkpoint, args=args, rank=rank)
    stage6a_load_stats = load_optional_stage6a_checkpoint(model, args.stage6a_checkpoint, rank=rank)
    groups = configure_trainability(model, args=args)

    lr_by_name = {
        "adapter": float(args.lr_adapter),
        "future_predictor": float(args.lr_predictor),
        "action_dit": float(args.lr_action),
        "proprio": float(args.lr_proprio),
        "z_task_adapter": float(args.lr_adapter),
        "pairwise_latent": float(args.lr_adapter),
    }
    param_groups = []
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
        "stage6a_checkpoint": stage6a_load_stats,
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
            loss_total, loss_dict = stage6b_forward_loss(model, batch, args=args)
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
                        f"loss_action={reduced['loss_action']:.6f}",
                        f"loss_future_jepa={reduced['loss_future_jepa']:.6f}",
                        f"loss_future_l1={reduced['loss_future_l1']:.6f}",
                        f"loss_future_cos={reduced['loss_future_cos']:.6f}",
                        f"loss_vlp_to_a={reduced['loss_vlp_to_a']:.6f}",
                        f"loss_va_to_l={reduced['loss_va_to_l']:.6f}",
                        f"retrieval_acc_vlp_to_a={reduced['retrieval_acc_vlp_to_a']:.6f}",
                        f"retrieval_acc_va_to_l={reduced['retrieval_acc_va_to_l']:.6f}",
                        f"z_task_gate={reduced['z_task_gate']:.6f}",
                        f"z_task_context_tokens={reduced['z_task_context_tokens']:.1f}",
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
