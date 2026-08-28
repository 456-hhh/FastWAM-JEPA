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

from fastwam.models.wan22.fastwam_jepa_idm_v5 import FastWAMJEPAIDMV5  # noqa: E402
from fastwam.models.wan22.jepa_visual_dit_v5 import JEPAVisualDiTV5  # noqa: E402
from fastwam.models.wan22.v5_contract import canonicalize_v5_batch  # noqa: E402
from fastwam_jepa_v5_data import (  # noqa: E402
    autocast_context,
    build_v5_loader,
    build_vjepa_encoder,
    checkpoint_metadata,
    compose_cfg,
    cosine_with_warmup,
    distributed_sha256,
    init_distributed,
    load_release_modules,
    make_loader_iterator,
    next_batch,
    precision_dtypes,
    provenance_paths,
    rank0_print,
    require_file,
    resume_training_state,
    save_compact_checkpoint,
    seed_everything,
    verify_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-IDM V5 Stage3 joint training.")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--release-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--vjepa-repo", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint-key", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="Local batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr-visual", type=float, default=2e-5)
    parser.add_argument("--lr-action", type=float, default=1e-6)
    parser.add_argument("--lr-proprio", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--video-cond-noise-prob", type=float, default=0.5)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_stage2_model_parts(
    *,
    checkpoint_path: Path,
    cfg,
    release_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("Stage2 checkpoint is missing V5 metadata.")
    metadata = payload["metadata"]
    config = metadata.get("visual_config")
    if not isinstance(config, dict):
        raise ValueError("Stage2 metadata is missing visual_config.")
    visual = JEPAVisualDiTV5(
        num_layers=int(config["num_layers"]),
        hidden_dim=int(config["hidden_dim"]),
        ffn_dim=int(config["ffn_dim"]),
        num_heads=int(config["num_heads"]),
        attn_head_dim=int(config["attn_head_dim"]),
        vjepa_dim=int(config["vjepa_dim"]),
        text_dim=int(config["text_dim"]),
        spatial_pool_size=int(config["spatial_pool_size"]),
        use_gradient_checkpointing=True,
    ).to(device=device, dtype=dtype)
    action, proprio = load_release_modules(
        cfg,
        release_checkpoint=release_path,
        device=device,
        dtype=dtype,
        instantiate_action=True,
    )
    if action is None:
        raise RuntimeError("Stage3 release ActionDiT was not instantiated.")
    for key, module in (
        ("visual_dit", visual),
        ("action_expert", action),
        ("proprio_encoder", proprio),
    ):
        state = payload.get(key)
        if not isinstance(state, dict):
            raise ValueError(f"Stage2 checkpoint is missing {key} weights.")
        module.load_state_dict(state, strict=True)
    return visual, action, proprio, metadata


def main() -> None:
    args = parse_args()
    ddp_enabled, world_size, rank, local_rank, device = init_distributed()
    try:
        if min(args.steps, args.batch_size, args.log_every, args.save_every) <= 0:
            raise ValueError("steps, batch-size, log-every, and save-every must be positive.")
        if float(args.video_cond_noise_prob) != 0.5:
            raise ValueError("V5 Stage3 first version fixes --video-cond-noise-prob at 0.5.")
        seed_everything(int(args.seed) + rank * 100003)
        parameter_dtype, autocast_dtype = precision_dtypes(args.precision)
        paths = provenance_paths(args, rank=rank)
        stage2_path = require_file(args.stage2_checkpoint, name="--stage2-checkpoint")
        stage2_sha = distributed_sha256(stage2_path, rank=rank)
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
        visual, action, proprio, stage2_metadata = _load_stage2_model_parts(
            checkpoint_path=stage2_path,
            cfg=cfg,
            release_path=paths["release_path"],
            device=device,
            dtype=parameter_dtype,
        )
        verify_provenance(
            stage2_metadata,
            expected_stage="stage2",
            vjepa_sha256=paths["vjepa_sha"],
            release_sha256=paths["release_sha"],
            dataset_stats_sha256=paths["stats_sha"],
        )
        vjepa = build_vjepa_encoder(args, device=device, dtype=parameter_dtype)
        model = FastWAMJEPAIDMV5(
            vjepa_encoder=vjepa,
            visual_dit=visual,
            action_expert=action,
            proprio_encoder=proprio,
            video_cond_noise_prob=args.video_cond_noise_prob,
        ).to(device=device, dtype=parameter_dtype)
        model.set_stage3_trainability()
        action_trainable = sum(p.numel() for p in model.action_expert.parameters() if p.requires_grad)
        action_total = sum(p.numel() for p in model.action_expert.parameters())
        if action_trainable != action_total:
            raise RuntimeError("Stage3 ActionDiT full-unfreeze assertion failed.")
        parameter_groups = [
            {"params": model.visual_dit.parameters(), "lr": args.lr_visual},
            {"params": model.action_expert.parameters(), "lr": args.lr_action},
            {"params": model.proprio_encoder.parameters(), "lr": args.lr_proprio},
        ]
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
        lr_scheduler = cosine_with_warmup(optimizer, total_steps=args.steps)
        train_model: torch.nn.Module = model
        if ddp_enabled:
            train_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        metadata = checkpoint_metadata(
            stage="stage3",
            args=args,
            vjepa_sha256=paths["vjepa_sha"],
            release_sha256=paths["release_sha"],
            dataset_stats_sha256=paths["stats_sha"],
            parameter_counts={
                "visual_dit": model.visual_dit.parameter_count,
                "action_expert": action_total,
                "proprio_encoder": sum(p.numel() for p in model.proprio_encoder.parameters()),
            },
            parent_checkpoint=stage2_path,
            parent_sha256=stage2_sha,
        )
        metadata["visual_config"] = dict(stage2_metadata["visual_config"])
        metadata["hydra_config"] = OmegaConf.to_container(cfg, resolve=True)
        global_step = 0
        epoch = 0
        batches_in_epoch = 0
        resume_rng_state = None
        if args.resume_checkpoint:
            resume_path = require_file(args.resume_checkpoint, name="--resume-checkpoint")
            global_step, epoch, batches_in_epoch, resume_metadata = resume_training_state(
                checkpoint_path=resume_path,
                expected_stage="stage3",
                modules={
                    "visual_dit": model.visual_dit,
                    "action_expert": model.action_expert,
                    "proprio_encoder": model.proprio_encoder,
                },
                optimizer=optimizer,
                scheduler=lr_scheduler,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            verify_provenance(
                resume_metadata,
                expected_stage="stage3",
                vjepa_sha256=paths["vjepa_sha"],
                release_sha256=paths["release_sha"],
                dataset_stats_sha256=paths["stats_sha"],
            )
            if resume_metadata.get("parent_checkpoint_sha256") != stage2_sha:
                raise ValueError("Stage3 resume parent Stage2 SHA256 mismatch.")
            resume_rng_state = resume_metadata.pop("_resume_rng_state")
            rank0_print(rank, f"resumed_checkpoint={resume_path} resumed_step={global_step}")
        rank0_print(
            rank,
            f"visual_parameter_count={model.visual_dit.parameter_count} "
            f"action_parameter_count={action_total} action_trainable={action_trainable}",
        )
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
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        while global_step < args.steps:
            raw_batch, iterator, epoch, batches_in_epoch = next_batch(
                loader, iterator, sampler, epoch, batches_in_epoch
            )
            batch = canonicalize_v5_batch(raw_batch, device=device, dtype=parameter_dtype)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(autocast_dtype):
                loss, metrics = train_model(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, args.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            if global_step % args.log_every == 0:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                completed = global_step - launch_step
                lrs = lr_scheduler.get_last_lr()
                rank0_print(
                    rank,
                    f"step={global_step} loss_total={metrics['loss_total']:.6f} "
                    f"loss_visual={metrics['loss_visual']:.6f} loss_action={metrics['loss_action']:.6f} "
                    f"grad_norm={float(grad_norm):.4f} lr_visual={lrs[0]:.3e} "
                    f"lr_action={lrs[1]:.3e} lr_proprio={lrs[2]:.3e} "
                    f"samples_per_sec={completed * args.batch_size * world_size / elapsed:.2f}",
                )
            if global_step % args.save_every == 0:
                path = save_compact_checkpoint(
                    output_dir=output_dir,
                    step=global_step,
                    epoch=epoch,
                    batches_in_epoch=batches_in_epoch,
                    weights={
                        "visual_dit": model.visual_dit.state_dict(),
                        "action_expert": model.action_expert.state_dict(),
                        "proprio_encoder": model.proprio_encoder.state_dict(),
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
                    "visual_dit": model.visual_dit.state_dict(),
                    "action_expert": model.action_expert.state_dict(),
                    "proprio_encoder": model.proprio_encoder.state_dict(),
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
