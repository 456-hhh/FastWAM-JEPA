from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import nullcontext, redirect_stderr, redirect_stdout
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
from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper
from fastwam.models.wan22.pairwise_conditional_latent_v3 import (
    ActionEncoder,
    FusionVL,
    LanguageProjector,
    VisionProjector,
    contrastive_loss,
    latent_norms,
)


class Stage2VLActionModel(nn.Module):
    def __init__(self, *, raw_vjepa_tokens: int, vjepa_dim: int) -> None:
        super().__init__()
        self.language_projector = LanguageProjector()
        self.action_encoder = ActionEncoder()
        self.vision_projector = VisionProjector(input_dim=int(vjepa_dim), token_count=int(raw_vjepa_tokens))
        self.fusion_vl = FusionVL()

    def forward(
        self,
        *,
        current_jepa_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
        tau: float,
    ) -> dict[str, torch.Tensor]:
        z_v = self.vision_projector(current_jepa_tokens)
        z_l = self.language_projector(context, text_mask=context_mask)
        z_a = self.action_encoder(action)
        q_a_vl = self.fusion_vl(z_v, z_l)
        loss, retrieval_acc = contrastive_loss(q_a_vl, z_a, tau=tau)
        norms = latent_norms(z_v=z_v, z_l=z_l, z_a=z_a, q_a_vl=q_a_vl)
        return {
            "loss": loss,
            "retrieval_acc": retrieval_acc,
            "z_v_norm": loss.new_tensor(norms["z_v"]),
            "z_l_norm": loss.new_tensor(norms["z_l"]),
            "z_a_norm": loss.new_tensor(norms["z_a"]),
            "q_a_vl_norm": loss.new_tensor(norms["q_a_vl"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-IDM v3 Stage2 raw V-JEPA vision-language to action latent training.")
    parser.add_argument("--stage1-checkpoint", required=True)
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
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", default="runs/fastwam_jepa_idm_v3_stage2_vl_action")
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


def _fit_sequence(value: torch.Tensor, length: int, *, dim: int, pad_value: bool | float = 0.0) -> torch.Tensor:
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


def canonicalize_stage2_vl_batch(
    batch: dict[str, Any],
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if not isinstance(batch, dict):
        raise ValueError(f"Expected LIBERO dataloader batch dict, got {type(batch)}.")
    video = batch["video"]
    if not torch.is_tensor(video) or video.ndim != 5 or int(video.shape[1]) != 3:
        raise ValueError(f"LIBERO video must be [B, 3, T, H, W], got {tuple(video.shape)}.")
    current_t = int(args.current_frame_count)
    if int(video.shape[2]) < current_t:
        raise ValueError(f"LIBERO video has T={video.shape[2]}, but Stage2 requires at least {current_t} frames.")
    current_video = stage2_libero.resize_video(video[:, :, :current_t], size=int(args.vjepa_img_size))
    expected_video_tail = (3, current_t, int(args.vjepa_img_size), int(args.vjepa_img_size))
    if tuple(current_video.shape[1:]) != expected_video_tail:
        raise ValueError(f"current_video must be [B, {expected_video_tail}], got {tuple(current_video.shape)}.")

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

    return {
        "current_video": current_video.to(device=device, dtype=dtype, non_blocking=True),
        "action": action.to(device=device, dtype=dtype, non_blocking=True),
        "context": context.to(device=device, dtype=dtype, non_blocking=True),
        "context_mask": context_mask.to(device=device, dtype=torch.bool, non_blocking=True),
    }


def build_frozen_vjepa_encoder(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
) -> nn.Module:
    repo = stage2_libero.require_dir(args.vjepa_repo, name="--vjepa-repo")
    checkpoint = stage2_libero.require_file(args.vjepa_checkpoint, name="--vjepa-checkpoint")
    stdout_cm = redirect_stdout(io.StringIO())
    stderr_cm = redirect_stderr(io.StringIO())
    with stdout_cm, stderr_cm:
        encoder = VJepaEncoderWrapper(
            dummy=False,
            model_name=str(args.vjepa_model_name),
            external_repo_path=str(repo),
            checkpoint_path=str(checkpoint),
            pretrained=False,
            vjepa_dim=int(args.vjepa_dim),
            num_tokens=int(args.raw_vjepa_tokens),
            freeze=True,
            normalize_tokens=False,
            img_size=int(args.vjepa_img_size),
            input_range=str(args.vjepa_input_range),
            tubelet_size=int(args.vjepa_tubelet_size),
            frame_encoding_mode="clip_or_repeat",
        ).to(device=device, dtype=dtype)
    encoder.eval()
    encoder.requires_grad_(False)
    if any(param.requires_grad for param in encoder.parameters()):
        raise RuntimeError("V-JEPA2 encoder must be frozen for Stage2 v3.")
    stage2_libero.rank0_print(rank, f"vjepa_checkpoint={checkpoint}", flush=True)
    return encoder


@torch.no_grad()
def encode_current_jepa_tokens(
    *,
    vjepa_encoder: nn.Module,
    current_video: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    vjepa_encoder.eval()
    tokens = vjepa_encoder(current_video)
    if not torch.is_tensor(tokens) or tokens.ndim != 3:
        raise ValueError(f"V-JEPA2 encoder must return [B, N, D], got {type(tokens)} {getattr(tokens, 'shape', None)}.")
    expected = (int(current_video.shape[0]), int(args.raw_vjepa_tokens), int(args.vjepa_dim))
    if tuple(tokens.shape) != expected:
        raise ValueError(f"raw current_jepa_tokens must be {expected}, got {tuple(tokens.shape)}.")
    return tokens.detach()


def load_stage1_checkpoint(model: Stage2VLActionModel, checkpoint_path: str | Path, *, rank: int) -> dict[str, Any]:
    path = stage2_libero.require_file(checkpoint_path, name="--stage1-checkpoint")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Stage1 checkpoint payload must be dict, got {type(payload)}.")
    stats: dict[str, Any] = {}
    for key, module in (
        ("language_projector", model.language_projector),
        ("action_encoder", model.action_encoder),
    ):
        state = payload.get(key)
        if not isinstance(state, dict):
            raise ValueError(f"Stage1 checkpoint missing {key} state_dict.")
        module.load_state_dict(state, strict=True)
        stats[key] = {"loaded": True, "keys": len(state)}
    stage2_libero.rank0_print(rank, f"loaded_stage1_checkpoint={path}", flush=True)
    return stats


def reduce_metrics(metrics: dict[str, torch.Tensor], *, ddp_enabled: bool, world_size: int) -> dict[str, float]:
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
    if not isinstance(module, Stage2VLActionModel):
        raise TypeError(f"Unexpected model type: {type(module)}")
    path = output_dir / f"checkpoint_step_{int(step):06d}.pt"
    torch.save(
        {
            "language_projector": module.language_projector.state_dict(),
            "action_encoder": module.action_encoder.state_dict(),
            "vision_projector": module.vision_projector.state_dict(),
            "fusion_vl": module.fusion_vl.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "args": vars(args),
            "world_size": int(world_size),
            "vision_source": "raw_jepa",
        },
        path,
    )
    return path


def main() -> None:
    args = parse_args()
    if int(args.context_tokens) != 128:
        raise ValueError("--context-tokens must remain 128 for LanguageProjector.")
    if int(args.action_horizon) != 32:
        raise ValueError("--action-horizon must remain 32 for ActionEncoder.")
    if int(args.current_frame_count) != 4:
        raise ValueError("Stage2 v3 expects --current-frame-count 4.")
    if int(args.vjepa_img_size) != 256:
        raise ValueError("Stage2 v3 expects --vjepa-img-size 256.")
    if int(args.raw_vjepa_tokens) != 512 or int(args.vjepa_dim) != 1408:
        raise ValueError("Stage2 v3 expects raw V-JEPA tokens [B, 512, 1408].")
    if int(args.steps) <= 0 or int(args.grad_accum_steps) <= 0:
        raise ValueError("--steps and --grad-accum-steps must be positive.")
    if float(args.tau) <= 0.0:
        raise ValueError("--tau must be positive.")

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

    model = Stage2VLActionModel(raw_vjepa_tokens=int(args.raw_vjepa_tokens), vjepa_dim=int(args.vjepa_dim))
    load_stage1_checkpoint(model, args.stage1_checkpoint, rank=rank)
    model = model.to(device=device, dtype=param_dtype)
    vjepa_encoder = build_frozen_vjepa_encoder(args, device=device, dtype=param_dtype, rank=rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
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
    stage2_libero.rank0_print(rank, "vision_source=raw_jepa", flush=True)

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
        batch = canonicalize_stage2_vl_batch(raw_batch, args=args, device=device, dtype=param_dtype)
        current_jepa_tokens = encode_current_jepa_tokens(
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
                action=batch["action"],
                tau=float(args.tau),
            )
            loss = out["loss"] / float(args.grad_accum_steps)
        loss.backward()
        micro_step += 1
        if micro_step % int(args.grad_accum_steps) != 0:
            continue

        if float(args.max_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        reduced = reduce_metrics(out, ddp_enabled=ddp_enabled, world_size=world_size)
        last_metrics = dict(reduced)
        if rank == 0 and (update_step == 1 or update_step % int(args.log_every) == 0):
            elapsed = max(time.time() - accum_start, 1.0e-6)
            print(
                " ".join(
                    [
                        f"step={update_step}",
                        f"loss={reduced['loss']:.6f}",
                        f"retrieval_acc={reduced['retrieval_acc']:.6f}",
                        f"z_v_norm={reduced['z_v_norm']:.6f}",
                        f"z_l_norm={reduced['z_l_norm']:.6f}",
                        f"z_a_norm={reduced['z_a_norm']:.6f}",
                        f"q_a_vl_norm={reduced['q_a_vl_norm']:.6f}",
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