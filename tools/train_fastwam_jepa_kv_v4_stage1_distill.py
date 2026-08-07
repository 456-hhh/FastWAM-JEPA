from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.vjepa.jepa_kv_cache_generator import (  # noqa: E402
    JepaKVCacheGenerator,
    kv_cache_distillation_loss,
)
from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper  # noqa: E402
from fastwam.models.wan22.fastwam_jepa_kv_v4 import (  # noqa: E402
    encode_causal_dual_camera_tokens,
    extract_causal_current_frame,
    prepare_v4_context,
    sha256_file,
    validate_teacher_cache_row_major,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-KV v4 Stage1 cache distillation.")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--vjepa-repo", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint-key", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--layer-rank", type=int, default=16)
    parser.add_argument("--video-seq-len", type=int, default=98)
    parser.add_argument("--batch-size", type=int, default=8, help="Local batch size per rank.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-cos", type=float, default=0.1)
    parser.add_argument(
        "--context-mask-mode",
        choices=("baseline_all_true", "cached_real_mask"),
        default="baseline_all_true",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--val-seed", type=int, default=2026)
    parser.add_argument("--val-parity-seed", type=int, default=12345)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("v4 DDP requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, world_size, rank, local_rank, torch.device(f"cuda:{local_rank}")
    return False, world_size, rank, local_rank, torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def rank0_print(rank: int, *values: Any) -> None:
    if rank == 0:
        print(*values, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_dtypes(name: str, device: torch.device) -> tuple[torch.dtype, Optional[torch.dtype]]:
    if name == "fp32" or device.type != "cuda":
        return torch.float32, None
    if name == "fp16":
        return torch.float16, torch.float16
    return torch.bfloat16, torch.bfloat16


def autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def compose_cfg(config_name: str, task: str) -> DictConfig:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(config_dir=str((PROJECT_ROOT / "configs").resolve()), version_base="1.3"):
        return compose(config_name=config_name, overrides=[f"task={task}"])


def require_file(path: str, *, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def require_dir(path: str, *, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def resolve_dataset_dirs(cfg: DictConfig, *, root: str) -> None:
    resolved = require_dir(root, name="--libero-data-root")
    if (resolved / "meta").exists() or (resolved / "data").exists():
        dataset_dirs = [resolved]
    else:
        dataset_dirs = sorted(
            child for child in resolved.iterdir() if child.is_dir() and child.name.endswith("_lerobot")
        )
    if not dataset_dirs:
        raise FileNotFoundError(f"No LIBERO *_lerobot directories found under {resolved}.")
    cfg.data.train.dataset_dirs = [str(path) for path in dataset_dirs]


def camera_order_from_cfg(cfg: DictConfig) -> tuple[str, str]:
    images = OmegaConf.to_container(cfg.data.train.shape_meta.images, resolve=True)
    if not isinstance(images, list):
        raise ValueError("data.train.shape_meta.images must be a list.")
    order = tuple(str(item["key"]) for item in images if isinstance(item, dict) and "key" in item)
    if len(order) != 2:
        raise ValueError(f"v4 requires exactly two configured camera keys, got {order}.")
    if str(cfg.data.train.get("concat_multi_camera")) != "horizontal":
        raise ValueError("v4 currently requires horizontal camera concatenation.")
    return order  # type: ignore[return-value]


class DistributedEvalSampler(Sampler[int]):
    """Deterministic rank striding without DistributedSampler padding duplicates."""

    def __init__(self, dataset, *, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"Invalid validation shard rank={self.rank}, replicas={self.num_replicas}."
            )

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, _epoch: int) -> None:
        return None


def build_loader(
    cfg: DictConfig,
    *,
    args: argparse.Namespace,
    ddp_enabled: bool,
    world_size: int,
    rank: int,
    validation: bool = False,
) -> tuple[DataLoader, Optional[Sampler[int]]]:
    resolve_dataset_dirs(cfg, root=str(args.libero_data_root))
    stats_path = require_file(str(args.dataset_stats_path), name="--dataset-stats-path")
    OmegaConf.update(
        cfg, "data.train.pretrained_norm_stats", str(stats_path), force_add=True
    )
    OmegaConf.update(cfg, "data.train.preserve_context_mask", True, force_add=True)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    if hasattr(args, "val_fraction"):
        OmegaConf.update(
            dataset_cfg,
            "val_set_proportion",
            float(args.val_fraction),
            force_add=True,
        )
        OmegaConf.update(
            dataset_cfg,
            "split_seed",
            int(args.val_seed),
            force_add=True,
        )
    OmegaConf.update(
        dataset_cfg,
        "is_training_set",
        not bool(validation),
        force_add=True,
    )
    dataset = instantiate(dataset_cfg)
    sampler: Optional[Sampler[int]] = None
    if ddp_enabled:
        if validation:
            sampler = DistributedEvalSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(args.seed),
                drop_last=True,
            )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=sampler is None and not validation,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=not validation,
    )
    return loader, sampler


def canonicalize_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    action_horizon: int = 32,
) -> dict[str, torch.Tensor]:
    required = ("video", "action", "context", "context_mask", "proprio")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"LIBERO batch is missing keys: {missing}.")
    video = batch["video"]
    action = batch["action"]
    context = batch["context"]
    context_mask = batch["context_mask"]
    proprio = batch["proprio"]
    if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] < 1:
        raise ValueError(f"Video must be [B,3,T,H,W] with T>=1, got {tuple(video.shape)}.")
    if action.ndim != 3 or action.shape[-1] != 7 or action.shape[1] < action_horizon:
        raise ValueError(f"Action must include [B,{action_horizon},7], got {tuple(action.shape)}.")
    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError("Context and context_mask must be [B,L,D] and [B,L].")
    if tuple(context_mask.shape) != tuple(context.shape[:2]):
        raise ValueError("context_mask shape must match context tokens.")
    if context_mask.dtype != torch.bool:
        context_mask = context_mask.bool()
    if bool((context_mask.sum(dim=1) == 0).any()):
        raise ValueError("Real context mask contains a sample with no valid token.")
    result = {
        "video": video.to(device=device, dtype=dtype, non_blocking=True),
        "action": action[:, :action_horizon].to(device=device, dtype=dtype, non_blocking=True),
        "context": context.to(device=device, dtype=dtype, non_blocking=True),
        "context_mask": context_mask.to(device=device, dtype=torch.bool, non_blocking=True),
        "proprio": proprio.to(device=device, dtype=dtype, non_blocking=True),
    }
    action_is_pad = batch.get("action_is_pad")
    if action_is_pad is not None:
        result["action_is_pad"] = action_is_pad[:, :action_horizon].to(
            device=device, dtype=torch.bool, non_blocking=True
        )
    return result


def load_vjepa_checkpoint_with_report(
    wrapper: VJepaEncoderWrapper,
    *,
    checkpoint_path: Path,
    checkpoint_key: Optional[str],
    rank: int,
) -> dict[str, Any]:
    if wrapper.encoder is None:
        raise RuntimeError("Real V-JEPA encoder is unavailable.")
    payload = torch.load(checkpoint_path, map_location="cpu")
    wrapper.checkpoint_key = checkpoint_key
    state, source = wrapper._select_checkpoint_state_dict(payload)
    state = wrapper._strip_state_dict_prefixes(state)
    expected = set(wrapper.encoder.state_dict())
    result = wrapper.encoder.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    matched = len(expected) - len(missing)
    coverage = matched / max(len(expected), 1)
    rank0_print(
        rank,
        f"vjepa_load source={source} matched={matched}/{len(expected)} "
        f"coverage={coverage:.4f} missing={len(missing)} unexpected={len(unexpected)}",
    )
    if matched <= 0:
        raise RuntimeError("V-JEPA checkpoint did not match any encoder parameters.")
    return {
        "source": source,
        "matched": matched,
        "total": len(expected),
        "coverage": coverage,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def build_vjepa_encoder(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
) -> tuple[VJepaEncoderWrapper, dict[str, Any]]:
    checkpoint = require_file(str(args.vjepa_checkpoint), name="--vjepa-checkpoint")
    wrapper = VJepaEncoderWrapper(
        dummy=False,
        num_tokens=256,
        vjepa_dim=int(args.vjepa_dim),
        freeze=True,
        normalize_tokens=False,
        model_name=str(args.vjepa_model_name),
        external_repo_path=str(require_dir(str(args.vjepa_repo), name="--vjepa-repo")),
        checkpoint_path=None,
        pretrained=False,
        img_size=int(args.vjepa_img_size),
        input_range=str(args.vjepa_input_range),
        tubelet_size=int(args.vjepa_tubelet_size),
        frame_encoding_mode="clip_or_repeat",
    ).to(device=device, dtype=dtype)
    report = load_vjepa_checkpoint_with_report(
        wrapper,
        checkpoint_path=checkpoint,
        checkpoint_key=args.vjepa_checkpoint_key,
        rank=rank,
    )
    wrapper.eval()
    wrapper.requires_grad_(False)
    return wrapper, report


def build_teacher(
    cfg: DictConfig,
    *,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    checkpoint_info: Optional[dict[str, Any]] = None,
) -> torch.nn.Module:
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.action_dit_pretrained_path = None
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.mot_checkpoint_mixed_attn = False
    teacher = instantiate(model_cfg, device=str(device), model_dtype=dtype)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "mot" not in payload:
        raise ValueError("Teacher checkpoint must contain the original FastWAM `mot` state_dict.")
    if checkpoint_info is not None:
        checkpoint_info["top_level_keys"] = tuple(sorted(str(key) for key in payload))
        for key in (
            "metadata",
            "args",
            "config",
            "cfg",
            "task",
            "model_kind",
            "model_type",
            "teacher_kind",
            "model_class",
        ):
            if key in payload:
                checkpoint_info[key] = payload[key]
    teacher.mot.load_state_dict(payload["mot"], strict=True)
    if teacher.proprio_encoder is not None:
        if payload.get("proprio_encoder") is None:
            raise ValueError("Teacher checkpoint is missing proprio_encoder weights.")
        teacher.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    teacher.eval()
    teacher.requires_grad_(False)
    rank0_print(rank, "teacher_load strict=true missing=0 unexpected=0")
    return teacher


def prepare_teacher_context(
    teacher: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    context_mask_mode: str = "baseline_all_true",
) -> tuple[torch.Tensor, torch.Tensor]:
    context = batch["context"].to(device=teacher.device, dtype=teacher.torch_dtype)
    mask = batch["context_mask"].to(device=teacher.device, dtype=torch.bool)
    context, mask = prepare_v4_context(context, mask, mode=context_mask_mode)
    proprio = batch["proprio"]
    if proprio.ndim == 3:
        proprio = proprio[:, 0]
    return teacher._append_proprio_to_context(
        context=context,
        context_mask=mask,
        proprio=proprio.to(device=teacher.device, dtype=teacher.torch_dtype),
    )


@torch.no_grad()
def teacher_current_frame_cache(
    teacher: torch.nn.Module,
    video: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
) -> tuple[list[dict[str, torch.Tensor]], tuple[int, int, int]]:
    current = extract_causal_current_frame(video).to(
        device=teacher.device, dtype=teacher.torch_dtype
    )
    latents = teacher._encode_video_latents(current.unsqueeze(2), tiled=False)
    timestep = torch.zeros(
        (current.shape[0],), device=teacher.device, dtype=teacher.torch_dtype
    )
    video_pre = teacher.video_expert.pre_dit(
        x=latents,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=bool(teacher.video_expert.fuse_vae_embedding_in_latents),
    )
    grid_size = tuple(int(value) for value in video_pre["meta"]["grid_size"])
    tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
    video_mask = teacher.video_expert.build_video_to_video_mask(
        video_seq_len=int(video_pre["tokens"].shape[1]),
        video_tokens_per_frame=tokens_per_frame,
        device=video_pre["tokens"].device,
    )
    cache = teacher.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
        video_attention_mask=video_mask,
    )
    return validate_teacher_cache_row_major(cache, grid_size=grid_size), grid_size


@torch.no_grad()
def run_validation(
    *,
    generator: torch.nn.Module,
    loader: DataLoader,
    teacher: torch.nn.Module,
    vjepa: torch.nn.Module,
    camera_order: tuple[str, str],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    autocast_dtype: Optional[torch.dtype],
    rank: int,
) -> dict[str, float]:
    module = unwrap(generator)
    was_training = module.training
    module.eval()
    metric_names = (
        "val_loss_total",
        "val_loss_k",
        "val_loss_v",
        "val_loss_cos",
        "val_cos_first",
        "val_cos_middle",
        "val_cos_last",
        "val_action_parity_mse",
        "val_action_parity_cosine",
    )
    sums = {name: 0.0 for name in metric_names}
    sample_count = 0
    parity_generator = torch.Generator(device="cpu").manual_seed(
        int(args.val_parity_seed) + int(rank) * 100003
    )
    try:
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= int(args.val_batches):
                break
            batch = canonicalize_batch(raw_batch, device=device, dtype=dtype)
            context, context_mask = prepare_teacher_context(
                teacher,
                batch,
                context_mask_mode=str(args.context_mask_mode),
            )
            teacher_cache, grid_size = teacher_current_frame_cache(
                teacher,
                batch["video"],
                context,
                context_mask,
            )
            if grid_size != (1, 7, 14):
                raise ValueError(f"Validation teacher grid must be (1,7,14), got {grid_size}.")
            current = extract_causal_current_frame(batch["video"])
            with autocast_context(device, autocast_dtype):
                visual_tokens, _ = encode_causal_dual_camera_tokens(
                    vjepa,
                    current,
                    camera_order=camera_order,
                )
                student_cache = module(visual_tokens, context, context_mask)
                loss, metrics = kv_cache_distillation_loss(
                    student_cache,
                    teacher_cache,
                    lambda_cos=float(args.lambda_cos),
                )

                batch_size = int(batch["action"].shape[0])
                action_horizon = int(batch["action"].shape[1])
                noisy_action = torch.randn(
                    (batch_size, action_horizon, int(teacher.action_expert.action_dim)),
                    generator=parity_generator,
                    dtype=torch.float32,
                ).to(device=device, dtype=dtype)
                action_timestep = torch.full(
                    (batch_size,),
                    0.5,
                    device=device,
                    dtype=dtype,
                )
                attention_mask = teacher._build_mot_attention_mask(
                    video_seq_len=int(module.video_seq_len),
                    action_seq_len=action_horizon,
                    video_tokens_per_frame=int(module.video_seq_len),
                    device=device,
                )
                pred_teacher = teacher._predict_action_noise_with_cache(
                    latents_action=noisy_action,
                    timestep_action=action_timestep,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=teacher_cache,
                    attention_mask=attention_mask,
                    video_seq_len=int(module.video_seq_len),
                )
                pred_student = teacher._predict_action_noise_with_cache(
                    latents_action=noisy_action,
                    timestep_action=action_timestep,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=student_cache,
                    attention_mask=attention_mask,
                    video_seq_len=int(module.video_seq_len),
                )
                parity_mse = torch.nn.functional.mse_loss(
                    pred_student.float(), pred_teacher.float()
                )
                parity_cosine = torch.nn.functional.cosine_similarity(
                    pred_student.float().flatten(1),
                    pred_teacher.float().flatten(1),
                    dim=1,
                ).mean()

            values = {
                "val_loss_total": loss,
                "val_loss_k": metrics["loss_k"],
                "val_loss_v": metrics["loss_v"],
                "val_loss_cos": metrics["loss_cos"],
                "val_cos_first": metrics["cos_first"],
                "val_cos_middle": metrics["cos_middle"],
                "val_cos_last": metrics["cos_last"],
                "val_action_parity_mse": parity_mse,
                "val_action_parity_cosine": parity_cosine,
            }
            for name, value in values.items():
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"Validation metric {name} is not finite.")
                sums[name] += float(value.detach()) * batch_size
            sample_count += batch_size
    finally:
        module.train(was_training)

    packed = torch.tensor(
        [sums[name] for name in metric_names] + [float(sample_count)],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    global_count = float(packed[-1].item())
    if global_count <= 0:
        raise RuntimeError("Stage1 validation produced zero samples across all ranks.")
    return {
        name: float(packed[index].item() / global_count)
        for index, name in enumerate(metric_names)
    }


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def next_batch(
    loader: DataLoader,
    iterator,
    sampler: Optional[DistributedSampler],
    epoch: int,
):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def save_checkpoint(
    *,
    generator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    teacher: torch.nn.Module,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
    module = unwrap(generator)
    payload = {
        "kv_generator": module.state_dict(),
        "proprio_encoder": (
            None if teacher.proprio_encoder is None else teacher.proprio_encoder.state_dict()
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": int(step),
        "args": vars(args),
        "model_configuration": {
            "input_dim": module.input_dim,
            "context_dim": module.context_dim,
            "hidden_dim": module.hidden_dim,
            "layer_rank": module.layer_rank,
            "video_seq_len": module.video_seq_len,
            "num_layers": module.num_layers,
            "num_heads": module.num_heads,
            "head_dim": module.attn_head_dim,
            "cache_dim": module.cache_dim,
        },
        "metadata": metadata,
    }
    torch.save(payload, path)
    return path


def main() -> None:
    args = parse_args()
    ddp_enabled, world_size, rank, local_rank, device = init_distributed()
    try:
        if args.seed <= 0:
            raise ValueError("--seed must be positive.")
        if args.steps <= 0 or args.lambda_cos < 0:
            raise ValueError("--steps must be positive and --lambda-cos non-negative.")
        if not 0.0 < float(args.val_fraction) < 1.0:
            raise ValueError("--val-fraction must be strictly between 0 and 1.")
        if int(args.val_every) <= 0 or int(args.val_batches) <= 0:
            raise ValueError("--val-every and --val-batches must be positive.")
        if int(args.val_seed) < 0 or int(args.val_parity_seed) < 0:
            raise ValueError("--val-seed and --val-parity-seed must be non-negative.")
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
        val_loader, val_sampler = build_loader(
            cfg,
            args=args,
            ddp_enabled=ddp_enabled,
            world_size=world_size,
            rank=rank,
            validation=True,
        )
        if val_sampler is not None:
            val_sampler.set_epoch(0)
        rank0_print(
            rank,
            f"episode_split train_samples={len(loader.dataset)} "
            f"val_samples={len(val_loader.dataset)} val_fraction={float(args.val_fraction):.4f} "
            f"val_seed={int(args.val_seed)}",
        )

        teacher_path = require_file(str(args.teacher_checkpoint), name="--teacher-checkpoint")
        teacher = build_teacher(
            cfg,
            checkpoint_path=teacher_path,
            device=device,
            dtype=param_dtype,
            rank=rank,
        )
        vjepa, vjepa_report = build_vjepa_encoder(
            args, device=device, dtype=param_dtype, rank=rank
        )
        action = teacher.action_expert
        generator = JepaKVCacheGenerator(
            input_dim=int(args.vjepa_dim),
            context_dim=int(action.text_dim),
            hidden_dim=int(args.hidden_dim),
            num_layers=len(action.blocks),
            num_heads=int(action.num_heads),
            attn_head_dim=int(action.attn_head_dim),
            video_seq_len=int(args.video_seq_len),
            layer_rank=int(args.layer_rank),
            num_cameras=len(camera_order),
        ).to(device=device, dtype=param_dtype)
        rank0_print(
            rank,
            f"generator_parameters={generator.parameter_count} "
            f"({generator.parameter_count / 1e6:.2f}M) camera_order={camera_order}",
        )
        train_module: torch.nn.Module = generator
        if ddp_enabled:
            train_module = DDP(generator, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(
            train_module.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(args.steps), 1)
        )

        metadata = {
            "input_policy": "single_current_frame_duplicated_to_2",
            "selected_frame_index": 0,
            "camera_order": list(camera_order),
            "teacher_cache_order": "horizontal_7x14_row_major",
            "video_seq_len": int(args.video_seq_len),
            "num_layers": len(action.blocks),
            "num_heads": int(action.num_heads),
            "head_dim": int(action.attn_head_dim),
            "cache_dim": int(action.num_heads) * int(action.attn_head_dim),
            "context_mask_mode": str(args.context_mask_mode),
            "context_preprocessing": "zero_real_padding_then_apply_context_mask_mode",
            "validation_split": "episode_level",
            "validation_fraction": float(args.val_fraction),
            "validation_seed": int(args.val_seed),
            "validation_batches_per_rank": int(args.val_batches),
            "validation_parity_seed": int(args.val_parity_seed),
            "vjepa_checkpoint_path": str(require_file(args.vjepa_checkpoint, name="--vjepa-checkpoint")),
            "teacher_fastwam_checkpoint_path": str(teacher_path),
            "dataset_stats_path": str(require_file(args.dataset_stats_path, name="--dataset-stats-path")),
            "vjepa_load_report": vjepa_report,
            "world_size": world_size,
        }
        if rank == 0:
            metadata["vjepa_checkpoint_sha256"] = sha256_file(metadata["vjepa_checkpoint_path"])
            metadata["teacher_fastwam_checkpoint_sha256"] = sha256_file(teacher_path)
            metadata["dataset_stats_sha256"] = sha256_file(metadata["dataset_stats_path"])

        iterator = iter(loader)
        epoch = 0
        start_time = time.perf_counter()
        last_step = 0
        best_val_action_parity_mse = math.inf
        output_dir = Path(args.output_dir).expanduser().resolve()
        for step in range(1, int(args.steps) + 1):
            raw_batch, iterator, epoch = next_batch(loader, iterator, sampler, epoch)
            batch = canonicalize_batch(raw_batch, device=device, dtype=param_dtype)
            teacher_context, teacher_mask = prepare_teacher_context(
                teacher,
                batch,
                context_mask_mode=str(args.context_mask_mode),
            )
            teacher_cache, grid_size = teacher_current_frame_cache(
                teacher, batch["video"], teacher_context, teacher_mask
            )
            if grid_size != (1, 7, 14):
                raise ValueError(
                    f"Teacher single-frame cache must have grid (1,7,14), got {grid_size}."
                )
            current = extract_causal_current_frame(batch["video"])
            with autocast_context(device, autocast_dtype):
                visual_tokens, debug = encode_causal_dual_camera_tokens(
                    vjepa, current, camera_order=camera_order
                )
                if debug["selected_frame_index"] != 0:
                    raise RuntimeError("v4 selected a non-causal frame.")
                student_cache = train_module(visual_tokens, teacher_context, teacher_mask)
                loss, metrics = kv_cache_distillation_loss(
                    student_cache, teacher_cache, lambda_cos=float(args.lambda_cos)
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
                    f"step={step} loss_total={float(loss.detach()):.6f} "
                    f"loss_k={float(metrics['loss_k'].detach()):.6f} "
                    f"loss_v={float(metrics['loss_v'].detach()):.6f} "
                    f"loss_cos={float(metrics['loss_cos'].detach()):.6f} "
                    f"cos_first={float(metrics['cos_first'].detach()):.4f} "
                    f"cos_middle={float(metrics['cos_middle'].detach()):.4f} "
                    f"cos_last={float(metrics['cos_last'].detach()):.4f} "
                    f"samples_per_sec={samples_per_sec:.2f} lr={scheduler.get_last_lr()[0]:.3e}",
                )
            if step % int(args.val_every) == 0:
                val_metrics = run_validation(
                    generator=train_module,
                    loader=val_loader,
                    teacher=teacher,
                    vjepa=vjepa,
                    camera_order=camera_order,
                    args=args,
                    device=device,
                    dtype=param_dtype,
                    autocast_dtype=autocast_dtype,
                    rank=rank,
                )
                metadata["last_validation"] = {"step": step, **val_metrics}
                if val_metrics["val_action_parity_mse"] < best_val_action_parity_mse:
                    best_val_action_parity_mse = val_metrics["val_action_parity_mse"]
                    metadata["best_validation"] = {"step": step, **val_metrics}
                rank0_print(
                    rank,
                    f"validation step={step} "
                    f"loss_total={val_metrics['val_loss_total']:.6f} "
                    f"loss_k={val_metrics['val_loss_k']:.6f} "
                    f"loss_v={val_metrics['val_loss_v']:.6f} "
                    f"loss_cos={val_metrics['val_loss_cos']:.6f} "
                    f"cos_first={val_metrics['val_cos_first']:.4f} "
                    f"cos_middle={val_metrics['val_cos_middle']:.4f} "
                    f"cos_last={val_metrics['val_cos_last']:.4f} "
                    f"action_parity_mse={val_metrics['val_action_parity_mse']:.6f} "
                    f"action_parity_cosine={val_metrics['val_action_parity_cosine']:.4f}",
                )
            if rank == 0 and step % int(args.save_every) == 0:
                path = save_checkpoint(
                    generator=train_module,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    teacher=teacher,
                    output_dir=output_dir,
                    step=step,
                    args=args,
                    metadata=metadata,
                )
                rank0_print(rank, f"saved_checkpoint={path}")

        if rank == 0 and last_step % int(args.save_every) != 0:
            path = save_checkpoint(
                generator=train_module,
                optimizer=optimizer,
                scheduler=scheduler,
                teacher=teacher,
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
