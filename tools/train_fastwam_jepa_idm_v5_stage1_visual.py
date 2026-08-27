from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.wan22.fastwam_jepa_idm_v5 import compute_visual_flow_loss  # noqa: E402
from fastwam.models.wan22.jepa_visual_dit_v5 import JEPAVisualDiTV5  # noqa: E402
from fastwam.models.wan22.schedulers.scheduler_continuous import (  # noqa: E402
    WanContinuousFlowMatchScheduler,
)
from fastwam.models.wan22.v5_contract import (  # noqa: E402
    CURRENT_TOKEN_COUNT_PER_CAMERA,
    FUTURE_TOKEN_COUNT_PER_CAMERA,
    VJEPA_DIM,
    build_vjepa_clips,
    canonicalize_v5_batch,
    pool_dual_camera_vjepa_tokens,
)
from fastwam_jepa_v5_data import (  # noqa: E402
    autocast_context,
    build_v5_loader,
    build_vjepa_encoder,
    checkpoint_metadata,
    compose_cfg,
    cosine_with_warmup,
    init_distributed,
    load_release_modules,
    make_loader_iterator,
    next_batch,
    precision_dtypes,
    provenance_paths,
    rank0_print,
    resume_training_state,
    save_compact_checkpoint,
    seed_everything,
    unwrap,
    verify_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-IDM V5 Stage1 visual world model.")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--release-checkpoint", required=True)
    parser.add_argument("--vjepa-repo", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint-key", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--visual-layers", type=int, default=30)
    parser.add_argument("--visual-hidden-dim", type=int, default=768)
    parser.add_argument("--visual-ffn-dim", type=int, default=3072)
    parser.add_argument("--visual-heads", type=int, default=24)
    parser.add_argument("--visual-head-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4, help="Local batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr-visual", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


@torch.no_grad()
def encode_v5_latents(vjepa, video: torch.Tensor):
    clips = build_vjepa_clips(video)
    agent_current = vjepa(clips["agentview_current"])
    wrist_current = vjepa(clips["wrist_current"])
    agent_future = vjepa(clips["agentview_future"])
    wrist_future = vjepa(clips["wrist_future"])
    if tuple(agent_current.shape[1:]) != (CURRENT_TOKEN_COUNT_PER_CAMERA, VJEPA_DIM):
        raise ValueError(f"Strict current V-JEPA contract failed: {tuple(agent_current.shape)}.")
    if tuple(wrist_current.shape[1:]) != (CURRENT_TOKEN_COUNT_PER_CAMERA, VJEPA_DIM):
        raise ValueError(f"Strict wrist V-JEPA contract failed: {tuple(wrist_current.shape)}.")
    if tuple(agent_future.shape[1:]) != (FUTURE_TOKEN_COUNT_PER_CAMERA, VJEPA_DIM):
        raise ValueError(f"Strict future V-JEPA contract failed: {tuple(agent_future.shape)}.")
    if tuple(wrist_future.shape[1:]) != (FUTURE_TOKEN_COUNT_PER_CAMERA, VJEPA_DIM):
        raise ValueError(f"Strict wrist future V-JEPA contract failed: {tuple(wrist_future.shape)}.")
    current = pool_dual_camera_vjepa_tokens(agent_current, wrist_current, temporal_groups=1)
    future = pool_dual_camera_vjepa_tokens(agent_future, wrist_future, temporal_groups=2)
    return current[:, 0], future[:, 0], future[:, 1]


def build_base_context(
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor,
    proprio_encoder: torch.nn.Linear,
):
    real_mask = context_mask.to(dtype=torch.bool)
    if bool((real_mask.sum(dim=1) == 0).any()):
        raise ValueError("Every Stage1 context must contain a valid token.")
    context = context.masked_fill(~real_mask.unsqueeze(-1), 0.0)
    proprio_token = proprio_encoder(proprio).unsqueeze(1)
    return torch.cat((context, proprio_token), dim=1), torch.cat(
        (
            real_mask,
            torch.ones((context.shape[0], 1), dtype=torch.bool, device=context.device),
        ),
        dim=1,
    )


def main() -> None:
    args = parse_args()
    ddp_enabled, world_size, rank, local_rank, device = init_distributed()
    try:
        if min(args.steps, args.batch_size, args.log_every, args.save_every) <= 0:
            raise ValueError("steps, batch-size, log-every, and save-every must be positive.")
        seed_everything(int(args.seed) + rank * 100003)
        parameter_dtype, autocast_dtype = precision_dtypes(args.precision)
        paths = provenance_paths(args, rank=rank)
        cfg = compose_cfg(args.config_name, args.task)
        loader, sampler = build_v5_loader(
            cfg,
            libero_data_root=args.libero_data_root,
            dataset_stats_path=args.dataset_stats_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            ddp_enabled=ddp_enabled,
            world_size=world_size,
            rank=rank,
        )
        vjepa = build_vjepa_encoder(args, device=device, dtype=parameter_dtype)
        _, proprio_encoder = load_release_modules(
            cfg,
            release_checkpoint=paths["release_path"],
            device=device,
            dtype=parameter_dtype,
            instantiate_action=False,
        )
        proprio_encoder.requires_grad_(False)
        proprio_encoder.eval()
        visual = JEPAVisualDiTV5(
            num_layers=args.visual_layers,
            hidden_dim=args.visual_hidden_dim,
            ffn_dim=args.visual_ffn_dim,
            num_heads=args.visual_heads,
            attn_head_dim=args.visual_head_dim,
            use_gradient_checkpointing=True,
        ).to(device=device, dtype=parameter_dtype)
        train_model: torch.nn.Module = visual
        if ddp_enabled:
            train_model = DDP(visual, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(
            train_model.parameters(), lr=args.lr_visual, weight_decay=args.weight_decay
        )
        lr_scheduler = cosine_with_warmup(optimizer, total_steps=args.steps)
        flow_scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
        metadata = checkpoint_metadata(
            stage="stage1",
            args=args,
            vjepa_sha256=paths["vjepa_sha"],
            release_sha256=paths["release_sha"],
            dataset_stats_sha256=paths["stats_sha"],
            parameter_counts={
                "visual_dit": visual.parameter_count,
                "proprio_encoder": sum(p.numel() for p in proprio_encoder.parameters()),
            },
        )
        metadata["visual_config"] = {
            "num_layers": args.visual_layers,
            "hidden_dim": args.visual_hidden_dim,
            "ffn_dim": args.visual_ffn_dim,
            "num_heads": args.visual_heads,
            "attn_head_dim": args.visual_head_dim,
            "vjepa_dim": 1408,
            "text_dim": 4096,
            "spatial_pool_size": 6,
        }
        metadata["hydra_config"] = OmegaConf.to_container(cfg, resolve=True)
        global_step = 0
        epoch = 0
        batches_in_epoch = 0
        resume_rng_state = None
        if args.resume_checkpoint:
            resume_path = Path(args.resume_checkpoint).expanduser().resolve()
            global_step, epoch, batches_in_epoch, resume_metadata = resume_training_state(
                checkpoint_path=resume_path,
                expected_stage="stage1",
                modules={"visual_dit": unwrap(train_model), "proprio_encoder": proprio_encoder},
                optimizer=optimizer,
                scheduler=lr_scheduler,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            verify_provenance(
                resume_metadata,
                expected_stage="stage1",
                vjepa_sha256=paths["vjepa_sha"],
                release_sha256=paths["release_sha"],
                dataset_stats_sha256=paths["stats_sha"],
            )
            resume_rng_state = resume_metadata.pop("_resume_rng_state")
            rank0_print(rank, f"resumed_checkpoint={resume_path} resumed_step={global_step}")
        rank0_print(rank, f"visual_parameter_count={visual.parameter_count}")
        if args.steps <= global_step:
            rank0_print(rank, f"target_steps={args.steps} already_reached_at={global_step}")
            return
        iterator = make_loader_iterator(
            loader,
            sampler,
            epoch=epoch,
            batches_in_epoch=batches_in_epoch,
            resume_rng_state=resume_rng_state,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        launch_step = global_step
        start_time = time.perf_counter()
        while global_step < args.steps:
            raw_batch, iterator, epoch, batches_in_epoch = next_batch(
                loader, iterator, sampler, epoch, batches_in_epoch
            )
            batch = canonicalize_v5_batch(raw_batch, device=device, dtype=parameter_dtype)
            z0, z1, z2 = encode_v5_latents(vjepa, batch["video"])
            context, context_mask = build_base_context(
                batch["context"], batch["context_mask"], batch["proprio"], proprio_encoder
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(autocast_dtype):
                loss, metrics = compute_visual_flow_loss(
                    visual_dit=train_model,
                    scheduler=flow_scheduler,
                    z0=z0,
                    z1=z1,
                    z2=z2,
                    context=context,
                    context_mask=context_mask,
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_model.parameters(), args.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            if global_step % args.log_every == 0:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                completed = global_step - launch_step
                rank0_print(
                    rank,
                    f"step={global_step} loss_visual={float(metrics['loss_visual']):.6f} "
                    f"grad_norm={float(grad_norm):.4f} lr={lr_scheduler.get_last_lr()[0]:.3e} "
                    f"samples_per_sec={completed * args.batch_size * world_size / elapsed:.2f}",
                )
            if global_step % args.save_every == 0:
                path = save_compact_checkpoint(
                    output_dir=output_dir,
                    step=global_step,
                    epoch=epoch,
                    batches_in_epoch=batches_in_epoch,
                    weights={
                        "visual_dit": unwrap(train_model).state_dict(),
                        "proprio_encoder": proprio_encoder.state_dict(),
                    },
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    metadata=metadata,
                    rank=rank,
                    world_size=world_size,
                )
                if path is not None:
                    rank0_print(rank, f"saved_checkpoint={path}")
        if global_step % args.save_every != 0:
            path = save_compact_checkpoint(
                output_dir=output_dir,
                step=global_step,
                epoch=epoch,
                batches_in_epoch=batches_in_epoch,
                weights={
                    "visual_dit": unwrap(train_model).state_dict(),
                    "proprio_encoder": proprio_encoder.state_dict(),
                },
                optimizer=optimizer,
                scheduler=lr_scheduler,
                metadata=metadata,
                rank=rank,
                world_size=world_size,
            )
            if path is not None:
                rank0_print(rank, f"saved_checkpoint={path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
