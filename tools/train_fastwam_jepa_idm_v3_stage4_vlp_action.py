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
from torch.nn.parameter import UninitializedParameter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (SRC_ROOT, PROJECT_ROOT, TOOLS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import train_fastwam_jepa_idm_v2_stage2_libero as stage2_libero
import train_fastwam_jepa_idm_v3_stage3_vlp_action as stage3_train
from fastwam.models.wan22.pairwise_stage4 import Stage4VLPVAActionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastWAM-JEPA-IDM v3 Stage4: preserve VLP-to-action and add VA-to-language."
    )
    parser.add_argument("--stage3-checkpoint", required=True)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--vjepa-repo", default=stage2_libero.DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=stage2_libero.DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=["-1_1", "0_1"])
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--raw-vjepa-tokens", type=int, default=512)
    parser.add_argument("--current-frame-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda-va-to-l", type=float, default=0.1)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", default="runs/fastwam_jepa_idm_v3_stage4_vlp_va")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _strip_uniform_module_prefix(state: dict[str, Any], *, name: str) -> dict[str, Any]:
    keys = list(state)
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError(f"{name} state_dict mixes module.-prefixed and unprefixed keys.")
    if prefixed and all(prefixed):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


def load_stage3_checkpoint_strict(
    model: Stage4VLPVAActionModel,
    checkpoint_path: str | Path,
    *,
    rank: int,
) -> None:
    path = stage2_libero.require_file(checkpoint_path, name="--stage3-checkpoint")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stage3 checkpoint must be a dict, got {type(payload)}.")
    checkpoint_stage = payload.get("stage")
    if checkpoint_stage is not None and checkpoint_stage != "stage3_vlp_action":
        raise RuntimeError(
            f"Expected stage='stage3_vlp_action', got {checkpoint_stage!r} in {path}."
        )

    required_modules = (
        "language_projector",
        "action_encoder",
        "vision_projector",
        "proprio_projector",
        "fusion_vlp",
    )
    for name in required_modules:
        state = payload.get(name)
        if not isinstance(state, dict):
            raise RuntimeError(f"Stage3 checkpoint missing required {name} state_dict: {path}")
        normalized = _strip_uniform_module_prefix(state, name=name)
        try:
            getattr(model, name).load_state_dict(normalized, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Strict Stage3 load failed for {name}: {exc}") from exc

    stage2_libero.rank0_print(
        rank,
        "loaded_stage3_modules=" + ",".join(required_modules),
        flush=True,
    )
    stage2_libero.rank0_print(rank, f"stage3_checkpoint={path}", flush=True)


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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    module = stage2_libero.unwrap_ddp(model)
    if not isinstance(module, Stage4VLPVAActionModel):
        raise TypeError(f"Unexpected model type: {type(module)}")
    path = output_dir / f"checkpoint_step_{int(step):06d}.pt"
    torch.save(
        {
            "language_projector": module.language_projector.state_dict(),
            "action_encoder": module.action_encoder.state_dict(),
            "vision_projector": module.vision_projector.state_dict(),
            "proprio_projector": module.proprio_projector.state_dict(),
            "fusion_vlp": module.fusion_vlp.state_dict(),
            "fusion_va": module.fusion_va.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "args": vars(args),
            "world_size": int(world_size),
            "stage": "stage4_vlp_va",
        },
        path,
    )
    return path


def _latent_norm(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().norm(dim=-1).mean()


def main() -> None:
    args = parse_args()
    if int(args.context_tokens) != 128:
        raise ValueError("--context-tokens must remain 128 for LanguageProjector.")
    if int(args.action_horizon) != 32:
        raise ValueError("--action-horizon must remain 32 for ActionEncoder.")
    if int(args.proprio_dim) != 8:
        raise ValueError("--proprio-dim must remain 8 for ProprioProjector.")
    if int(args.current_frame_count) != 4 or int(args.vjepa_img_size) != 256:
        raise ValueError("Stage4 expects current_video [B, 3, 4, 256, 256].")
    if int(args.raw_vjepa_tokens) != 512 or int(args.vjepa_dim) != 1408:
        raise ValueError("Stage4 expects raw V-JEPA tokens [B, 512, 1408].")
    if int(args.steps) <= 0 or int(args.grad_accum_steps) <= 0:
        raise ValueError("--steps and --grad-accum-steps must be positive.")
    if int(args.log_every) <= 0:
        raise ValueError("--log-every must be positive.")
    if float(args.tau) <= 0.0:
        raise ValueError("--tau must be positive.")
    if float(args.lambda_va_to_l) <= 0.0:
        raise ValueError("--lambda-va-to-l must be positive.")

    ddp_enabled, world_size, rank, local_rank, device = stage2_libero.init_distributed_from_env()
    stage2_libero.seed_everything(int(args.seed))
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

    model = Stage4VLPVAActionModel(
        raw_vjepa_tokens=int(args.raw_vjepa_tokens),
        vjepa_dim=int(args.vjepa_dim),
        context_tokens=int(args.context_tokens),
        action_horizon=int(args.action_horizon),
        proprio_dim=int(args.proprio_dim),
    )
    load_stage3_checkpoint_strict(model, args.stage3_checkpoint, rank=rank)
    model.requires_grad_(True)
    if any(isinstance(param, UninitializedParameter) for param in model.parameters()):
        raise RuntimeError("Stage4 model contains an uninitialized parameter.")
    model = model.to(device=device, dtype=param_dtype)
    vjepa_encoder = stage3_train.build_frozen_vjepa_encoder(
        args,
        device=device,
        dtype=param_dtype,
        rank=rank,
    )
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

    output_dir = stage3_train.resolve_path(args.output_dir)
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
    stage2_libero.rank0_print(rank, "vision_source=raw_jepa", flush=True)
    stage2_libero.rank0_print(rank, "stage=stage4_vlp_va", flush=True)

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

        batch = stage3_train.canonicalize_stage3_vlp_batch(
            raw_batch,
            args=args,
            device=device,
            dtype=param_dtype,
        )
        expected_video = (
            int(batch["action"].shape[0]),
            3,
            4,
            256,
            256,
        )
        if tuple(batch["current_video"].shape) != expected_video:
            raise ValueError(
                f"current_video must be {expected_video}, got {tuple(batch['current_video'].shape)}."
            )
        current_jepa_tokens = stage3_train.encode_current_jepa_tokens(
            vjepa_encoder=vjepa_encoder,
            current_video=batch["current_video"],
            args=args,
        )

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            out = model(
                current_jepa_tokens=current_jepa_tokens,
                context=batch["context"],
                context_mask=batch["context_mask"],
                proprio=batch["proprio"],
                action=batch["action"],
                tau=float(args.tau),
            )
            weighted_loss_va_l = float(args.lambda_va_to_l) * out["loss_va_l"]
            loss_total = out["loss_vlp_a"] + weighted_loss_va_l
            scaled_loss = loss_total / float(args.grad_accum_steps)
        scaled_loss.backward()
        micro_step += 1
        if micro_step % int(args.grad_accum_steps) != 0:
            continue

        if float(args.max_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        metrics = {
            "loss_total": loss_total,
            "loss_vlp_a": out["loss_vlp_a"],
            "loss_va_l": out["loss_va_l"],
            "weighted_loss_va_l": weighted_loss_va_l,
            "retrieval_vlp_a": out["retrieval_vlp_a"],
            "retrieval_va_l": out["retrieval_va_l"],
            "z_v_norm": _latent_norm(out["z_v"]),
            "z_l_norm": _latent_norm(out["z_l"]),
            "z_p_norm": _latent_norm(out["z_p"]),
            "z_task_norm": _latent_norm(out["z_task"]),
            "z_a_norm": _latent_norm(out["z_a"]),
            "q_l_norm": _latent_norm(out["q_l"]),
        }
        reduced = reduce_metrics(metrics, ddp_enabled=ddp_enabled, world_size=world_size)
        last_metrics = dict(reduced)
        if rank == 0 and (update_step == 1 or update_step % int(args.log_every) == 0):
            elapsed = max(time.time() - accum_start, 1.0e-6)
            print(
                " ".join(
                    [
                        f"step={update_step}",
                        f"loss_total={reduced['loss_total']:.6f}",
                        f"loss_vlp_a={reduced['loss_vlp_a']:.6f}",
                        f"loss_va_l={reduced['loss_va_l']:.6f}",
                        f"weighted_loss_va_l={reduced['weighted_loss_va_l']:.6f}",
                        f"retrieval_vlp_a={reduced['retrieval_vlp_a']:.6f}",
                        f"retrieval_va_l={reduced['retrieval_va_l']:.6f}",
                        f"z_v_norm={reduced['z_v_norm']:.6f}",
                        f"z_l_norm={reduced['z_l_norm']:.6f}",
                        f"z_p_norm={reduced['z_p_norm']:.6f}",
                        f"z_task_norm={reduced['z_task_norm']:.6f}",
                        f"z_a_norm={reduced['z_a_norm']:.6f}",
                        f"q_l_norm={reduced['q_l_norm']:.6f}",
                        f"lambda_va_to_l={float(args.lambda_va_to_l):.6g}",
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
