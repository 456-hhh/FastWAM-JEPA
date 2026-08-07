from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.vjepa.jepa_kv_cache_generator import JepaKVCacheGenerator  # noqa: E402
from fastwam.models.wan22.action_dit import ActionDiT  # noqa: E402
from fastwam.models.wan22.fastwam_jepa_kv_v4 import (  # noqa: E402
    FastWAMJEPAKVV4,
    sha256_file,
    validate_checkpoint_context_mask_mode,
)
from train_fastwam_jepa_kv_v4_stage1_distill import (  # noqa: E402
    autocast_context,
    build_loader,
    build_teacher,
    build_vjepa_encoder,
    camera_order_from_cfg,
    canonicalize_batch,
    compose_cfg,
    init_distributed,
    next_batch,
    precision_dtypes,
    prepare_teacher_context,
    rank0_print,
    require_file,
    seed_everything,
    teacher_current_frame_cache,
    unwrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-KV v4 Stage2 action adaptation.")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--vjepa-repo", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint-key", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--batch-size", type=int, default=8, help="Local batch size per rank.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-kv", type=float, default=0.1)
    parser.add_argument("--kv-lambda-cos", type=float, default=0.1)
    parser.add_argument(
        "--context-mask-mode",
        choices=("baseline_all_true", "cached_real_mask"),
        default="baseline_all_true",
    )
    parser.add_argument(
        "--allow-context-mask-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow-action-teacher-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--unfreeze-action-last-n", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _state_dict_candidates(payload: dict[str, Any]):
    yield payload
    for key in ("action_expert", "mot", "model", "state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            yield value


def extract_action_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    prefixes = (
        "mixtures.action.",
        "mot.mixtures.action.",
        "dit.mixtures.action.",
        "module.mot.mixtures.action.",
        "action_expert.",
        "model.action_expert.",
        "module.action_expert.",
    )
    expected_roots = ("action_encoder.", "text_embedding.", "time_embedding.", "blocks.", "head.")
    for state in _state_dict_candidates(payload):
        direct = {
            str(key): value
            for key, value in state.items()
            if isinstance(key, str)
            and torch.is_tensor(value)
            and key.startswith(expected_roots)
        }
        if direct:
            return direct
        for prefix in prefixes:
            extracted = {
                str(key)[len(prefix) :]: value
                for key, value in state.items()
                if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
            }
            if extracted:
                return extracted
    raise ValueError("Could not find a complete ActionDiT state_dict in the checkpoint.")


def strict_load_action(
    action_expert: ActionDiT,
    state: dict[str, torch.Tensor],
    *,
    rank: int,
) -> None:
    expected = action_expert.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = sorted(
        key for key in set(expected) & set(state) if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    rank0_print(
        rank,
        f"action_load matched={len(set(expected) & set(state))}/{len(expected)} "
        f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}",
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Strict ActionDiT checkpoint validation failed: "
            f"missing={missing[:8]} unexpected={unexpected[:8]} shape_mismatch={shape_mismatch[:8]}."
        )
    action_expert.load_state_dict(state, strict=True)


def build_action_expert(
    cfg,
    *,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
) -> tuple[ActionDiT, dict[str, torch.Tensor]]:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise ValueError("cfg.model must resolve to a dict.")
    action_cfg = model_cfg.get("action_dit_config")
    if not isinstance(action_cfg, dict):
        raise ValueError("cfg.model.action_dit_config must resolve to a dict.")
    action_expert = ActionDiT(**action_cfg)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Action checkpoint payload must be a dict.")
    strict_load_action(action_expert, extract_action_state_dict(payload), rank=rank)
    proprio_state = payload.get("proprio_encoder")
    if not isinstance(proprio_state, dict):
        raise ValueError("Original FastWAM checkpoint is missing proprio_encoder state_dict.")
    return action_expert.to(device=device, dtype=dtype), proprio_state


def load_proprio_strict(
    model: FastWAMJEPAKVV4, state: dict[str, torch.Tensor]
) -> None:
    if model.proprio_encoder is None:
        return
    model.proprio_encoder.load_state_dict(state, strict=True)


def load_stage1_generator(
    checkpoint_path: Path,
    *,
    action_expert: ActionDiT,
    camera_order: tuple[str, str],
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
) -> tuple[JepaKVCacheGenerator, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Stage1 checkpoint payload must be a dict.")
    config = payload.get("model_configuration")
    metadata = payload.get("metadata")
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("Stage1 checkpoint is missing model_configuration or metadata.")
    if metadata.get("selected_frame_index") != 0:
        raise ValueError("Stage1 checkpoint was not trained with selected_frame_index=0.")
    if metadata.get("input_policy") != "single_current_frame_duplicated_to_2":
        raise ValueError("Stage1 checkpoint has an incompatible input policy.")
    if tuple(metadata.get("camera_order", ())) != tuple(camera_order):
        raise ValueError(
            f"Stage1 camera order {metadata.get('camera_order')} does not match {camera_order}."
        )
    generator = JepaKVCacheGenerator(
        input_dim=int(config["input_dim"]),
        context_dim=int(config["context_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        attn_head_dim=int(config["head_dim"]),
        video_seq_len=int(config["video_seq_len"]),
        layer_rank=int(config["layer_rank"]),
        num_cameras=2,
    )
    if int(config["num_layers"]) != len(action_expert.blocks):
        raise ValueError("Stage1 num_layers does not match ActionDiT.")
    if int(config["num_heads"]) != int(action_expert.num_heads):
        raise ValueError("Stage1 num_heads does not match ActionDiT.")
    if int(config["head_dim"]) != int(action_expert.attn_head_dim):
        raise ValueError("Stage1 head_dim does not match ActionDiT.")
    generator.load_state_dict(payload["kv_generator"], strict=True)
    rank0_print(rank, "stage1_kv_generator_load strict=true missing=0 unexpected=0")
    return generator.to(device=device, dtype=dtype), metadata


def distributed_rank0_sha256(path: Path, *, rank: int) -> str:
    value = sha256_file(path) if rank == 0 else ""
    if dist.is_initialized():
        values = [value]
        dist.broadcast_object_list(values, src=0)
        value = str(values[0])
    return value


def verify_stage1_metadata(
    metadata: dict[str, Any],
    *,
    vjepa_checkpoint: Path,
    dataset_stats_path: Path,
    action_checkpoint: Path,
    context_mask_mode: str,
    allow_context_mask_mismatch: bool,
    allow_action_teacher_mismatch: bool,
    rank: int,
) -> tuple[str, str, str]:
    vjepa_sha = distributed_rank0_sha256(vjepa_checkpoint, rank=rank)
    stats_sha = distributed_rank0_sha256(dataset_stats_path, rank=rank)
    action_sha = distributed_rank0_sha256(action_checkpoint, rank=rank)
    expected_vjepa = metadata.get("vjepa_checkpoint_sha256")
    expected_stats = metadata.get("dataset_stats_sha256")
    if expected_vjepa is None:
        raise ValueError("Stage1 checkpoint metadata lacks vjepa_checkpoint_sha256.")
    if expected_stats is None:
        raise ValueError("Stage1 checkpoint metadata lacks dataset_stats_sha256.")
    if str(expected_vjepa) != vjepa_sha:
        raise ValueError("Stage1 and Stage2 V-JEPA checkpoint SHA256 values differ.")
    if str(expected_stats) != stats_sha:
        raise ValueError("Stage1 and Stage2 dataset stats SHA256 values differ.")
    validate_checkpoint_context_mask_mode(
        metadata,
        context_mask_mode,
        checkpoint_name="Stage1 checkpoint",
        allow_mismatch=allow_context_mask_mismatch,
    )
    expected_teacher = metadata.get("teacher_fastwam_checkpoint_sha256")
    if expected_teacher is None:
        if not allow_action_teacher_mismatch:
            raise ValueError(
                "Stage1 checkpoint lacks teacher_fastwam_checkpoint_sha256; "
                "use --allow-action-teacher-mismatch only for an intentional override."
            )
    elif str(expected_teacher) != action_sha and not allow_action_teacher_mismatch:
        raise ValueError(
            "Stage1 teacher checkpoint SHA256 does not match the Stage2 action checkpoint."
        )
    return vjepa_sha, stats_sha, action_sha


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
    module = unwrap(model)
    payload = {
        "kv_generator": module.kv_generator.state_dict(),
        "action_expert": module.action_expert.state_dict(),
        "proprio_encoder": (
            None if module.proprio_encoder is None else module.proprio_encoder.state_dict()
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": int(step),
        "args": vars(args),
        "model_configuration": {
            "input_dim": module.kv_generator.input_dim,
            "context_dim": module.kv_generator.context_dim,
            "hidden_dim": module.kv_generator.hidden_dim,
            "layer_rank": module.kv_generator.layer_rank,
            "video_seq_len": module.video_seq_len,
            "num_layers": len(module.action_expert.blocks),
            "num_heads": int(module.action_expert.num_heads),
            "head_dim": int(module.action_expert.attn_head_dim),
            "cache_dim": module.kv_generator.cache_dim,
            "action_horizon": module.action_horizon,
        },
        "metadata": metadata,
    }
    torch.save(payload, path)
    return path


def main() -> None:
    args = parse_args()
    ddp_enabled, world_size, rank, local_rank, device = init_distributed()
    try:
        if args.seed <= 0 or args.steps <= 0:
            raise ValueError("--seed and --steps must be positive.")
        if args.lambda_kv < 0 or args.kv_lambda_cos < 0:
            raise ValueError("--lambda-kv and --kv-lambda-cos must be non-negative.")
        if not 0 <= int(args.unfreeze_action_last_n) <= 4:
            raise ValueError("--unfreeze-action-last-n must be between 0 and 4.")
        if args.lambda_kv > 0 and not args.teacher_checkpoint:
            raise ValueError("--lambda-kv > 0 requires an explicit --teacher-checkpoint.")
        seed_everything(int(args.seed) + rank * 100003)
        param_dtype, autocast_dtype = precision_dtypes(str(args.precision), device)
        cfg = compose_cfg(str(args.config_name), str(args.task))
        camera_order = camera_order_from_cfg(cfg)
        loader, sampler = build_loader(
            cfg,
            args=args,
            ddp_enabled=ddp_enabled,
            world_size=world_size,
            rank=rank,
        )
        if sampler is not None:
            sampler.set_epoch(0)

        action_path = require_file(args.action_checkpoint, name="--action-checkpoint")
        stage1_path = require_file(args.stage1_checkpoint, name="--stage1-checkpoint")
        stats_path = require_file(args.dataset_stats_path, name="--dataset-stats-path")
        vjepa_path = require_file(args.vjepa_checkpoint, name="--vjepa-checkpoint")
        action_expert, proprio_state = build_action_expert(
            cfg,
            checkpoint_path=action_path,
            device=device,
            dtype=param_dtype,
            rank=rank,
        )
        generator, stage1_metadata = load_stage1_generator(
            stage1_path,
            action_expert=action_expert,
            camera_order=camera_order,
            device=device,
            dtype=param_dtype,
            rank=rank,
        )
        vjepa_sha, stats_sha, action_sha = verify_stage1_metadata(
            stage1_metadata,
            vjepa_checkpoint=vjepa_path,
            dataset_stats_path=stats_path,
            action_checkpoint=action_path,
            context_mask_mode=str(args.context_mask_mode),
            allow_context_mask_mismatch=bool(args.allow_context_mask_mismatch),
            allow_action_teacher_mismatch=bool(args.allow_action_teacher_mismatch),
            rank=rank,
        )
        vjepa, vjepa_report = build_vjepa_encoder(
            args, device=device, dtype=param_dtype, rank=rank
        )
        proprio_dim = int(OmegaConf.select(cfg, "model.proprio_dim"))
        model = FastWAMJEPAKVV4(
            action_expert=action_expert,
            vjepa_encoder=vjepa,
            kv_generator=generator,
            camera_order=camera_order,
            proprio_dim=proprio_dim,
            action_horizon=32,
            action_train_shift=float(cfg.model.action_scheduler.train_shift),
            action_infer_shift=float(cfg.model.action_scheduler.infer_shift),
            action_num_train_timesteps=int(cfg.model.action_scheduler.num_train_timesteps),
            freeze_vjepa=True,
            freeze_action=True,
            freeze_proprio=True,
            context_mask_mode=str(args.context_mask_mode),
            device=device,
            torch_dtype=param_dtype,
        )
        load_proprio_strict(model, proprio_state)
        model.requires_grad_(False)
        model.kv_generator.requires_grad_(True)
        model.freeze_action = int(args.unfreeze_action_last_n) == 0
        if int(args.unfreeze_action_last_n) > 0:
            for block in model.action_expert.blocks[-int(args.unfreeze_action_last_n) :]:
                block.requires_grad_(True)
        model.train()
        rank0_print(
            rank,
            f"generator_parameters={model.kv_generator.parameter_count} "
            f"trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)} "
            f"action_unfrozen_last_n={args.unfreeze_action_last_n}",
        )

        teacher = None
        teacher_sha = None
        if float(args.lambda_kv) > 0:
            teacher_path = require_file(args.teacher_checkpoint, name="--teacher-checkpoint")
            teacher_sha = distributed_rank0_sha256(teacher_path, rank=rank)
            if teacher_sha != action_sha and not bool(args.allow_action_teacher_mismatch):
                raise ValueError(
                    "Stage2 teacher checkpoint SHA256 does not match --action-checkpoint."
                )
            teacher = build_teacher(
                cfg,
                checkpoint_path=teacher_path,
                device=device,
                dtype=param_dtype,
                rank=rank,
            )
        train_model: torch.nn.Module = model
        if ddp_enabled:
            train_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        trainable = [parameter for parameter in train_model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(args.steps), 1)
        )

        metadata = {
            "input_policy": "single_current_frame_duplicated_to_2",
            "selected_frame_index": 0,
            "camera_order": list(camera_order),
            "video_seq_len": model.video_seq_len,
            "num_layers": len(model.action_expert.blocks),
            "num_heads": int(model.action_expert.num_heads),
            "head_dim": int(model.action_expert.attn_head_dim),
            "cache_dim": model.kv_generator.cache_dim,
            "context_mask_mode": str(args.context_mask_mode),
            "context_preprocessing": "zero_real_padding_then_apply_context_mask_mode",
            "stage1_context_mask_mode": stage1_metadata.get("context_mask_mode"),
            "dataset_stats_path": str(stats_path),
            "dataset_stats_sha256": stats_sha,
            "vjepa_checkpoint_path": str(vjepa_path),
            "vjepa_checkpoint_sha256": vjepa_sha,
            "vjepa_load_report": vjepa_report,
            "stage1_checkpoint_path": str(stage1_path),
            "action_checkpoint_path": str(action_path),
            "action_checkpoint_sha256": action_sha,
            "lambda_kv": float(args.lambda_kv),
            "kv_lambda_cos": float(args.kv_lambda_cos),
            "allow_context_mask_mismatch": bool(args.allow_context_mask_mismatch),
            "allow_action_teacher_mismatch": bool(args.allow_action_teacher_mismatch),
            "world_size": world_size,
            "action_frozen_by_default": int(args.unfreeze_action_last_n) == 0,
        }
        if rank == 0:
            metadata["stage1_checkpoint_sha256"] = sha256_file(stage1_path)
            if teacher is not None:
                metadata["teacher_fastwam_checkpoint_path"] = str(
                    require_file(args.teacher_checkpoint, name="--teacher-checkpoint")
                )
                metadata["teacher_fastwam_checkpoint_sha256"] = teacher_sha

        iterator = iter(loader)
        epoch = 0
        start_time = time.perf_counter()
        output_dir = Path(args.output_dir).expanduser().resolve()
        last_step = 0
        for step in range(1, int(args.steps) + 1):
            raw_batch, iterator, epoch = next_batch(loader, iterator, sampler, epoch)
            batch = canonicalize_batch(raw_batch, device=device, dtype=param_dtype)
            teacher_cache = None
            if teacher is not None:
                teacher_context, teacher_mask = prepare_teacher_context(
                    teacher,
                    batch,
                    context_mask_mode=str(args.context_mask_mode),
                )
                teacher_cache, grid_size = teacher_current_frame_cache(
                    teacher, batch["video"], teacher_context, teacher_mask
                )
                if grid_size != (1, 7, 14):
                    raise ValueError(f"Teacher single-frame grid must be (1,7,14), got {grid_size}.")
            with autocast_context(device, autocast_dtype):
                loss, metrics = train_model(
                    batch,
                    teacher_video_kv_cache=teacher_cache,
                    lambda_kv=float(args.lambda_kv),
                    kv_lambda_cos=float(args.kv_lambda_cos),
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_step = step

            if step % int(args.log_every) == 0:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                samples_per_sec = step * int(args.batch_size) * world_size / elapsed
                rank0_print(
                    rank,
                    f"step={step} loss_total={metrics['loss_total']:.6f} "
                    f"loss_action={metrics['loss_action']:.6f} loss_kv={metrics['loss_kv']:.6f} "
                    f"loss_k={metrics['loss_k']:.6f} loss_v={metrics['loss_v']:.6f} "
                    f"loss_cos={metrics['loss_cos']:.6f} "
                    f"cos_first={metrics['cos_first']:.4f} "
                    f"cos_middle={metrics['cos_middle']:.4f} "
                    f"cos_last={metrics['cos_last']:.4f} "
                    f"samples_per_sec={samples_per_sec:.2f} lr={scheduler.get_last_lr()[0]:.3e}",
                )
            if rank == 0 and step % int(args.save_every) == 0:
                path = save_checkpoint(
                    model=train_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    step=step,
                    args=args,
                    metadata=metadata,
                )
                rank0_print(rank, f"saved_checkpoint={path}")

        if rank == 0 and last_step % int(args.save_every) != 0:
            path = save_checkpoint(
                model=train_model,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=output_dir,
                step=last_step,
                args=args,
                metadata=metadata,
            )
            rank0_print(rank, f"saved_checkpoint={path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
