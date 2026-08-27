from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rank0_print(rank: int, *values: Any) -> None:
    if rank == 0:
        print(*values, flush=True)


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


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distributed_sha256(path: Path, *, rank: int) -> str:
    value = sha256_file(path) if rank == 0 else ""
    if dist.is_initialized():
        payload = [value]
        dist.broadcast_object_list(payload, src=0)
        value = str(payload[0])
    return value


def git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("V5 production training requires CUDA; CPU fallback is disabled.")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return world_size > 1, world_size, rank, local_rank, torch.device(f"cuda:{local_rank}")


def seed_everything(seed: int) -> None:
    if seed <= 0:
        raise ValueError("V5 seed must be > 0.")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def precision_dtypes(name: str) -> tuple[torch.dtype, Optional[torch.dtype]]:
    if name == "fp32":
        return torch.float32, None
    if name == "fp16":
        return torch.float16, torch.float16
    if name == "bf16":
        return torch.bfloat16, torch.bfloat16
    raise ValueError(f"Unsupported precision {name!r}.")


def autocast_context(dtype: Optional[torch.dtype]):
    return nullcontext() if dtype is None else torch.autocast("cuda", dtype=dtype)


def compose_cfg(config_name: str, task: str) -> DictConfig:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(
        config_dir=str((PROJECT_ROOT / "configs").resolve()), version_base="1.3"
    ):
        return compose(config_name=config_name, overrides=[f"task={task}"])


def _resolve_dataset_dirs(cfg: DictConfig, *, root: str) -> None:
    resolved = require_dir(root, name="--libero-data-root")
    if (resolved / "meta").is_dir() or (resolved / "data").is_dir():
        dataset_dirs = [resolved]
    else:
        dataset_dirs = sorted(
            child for child in resolved.iterdir() if child.is_dir() and child.name.endswith("_lerobot")
        )
    if not dataset_dirs:
        raise FileNotFoundError(f"No LIBERO *_lerobot directories found under {resolved}.")
    OmegaConf.update(
        cfg, "data.train.dataset_dirs", [str(path) for path in dataset_dirs], force_add=True
    )


def validate_camera_config(cfg: DictConfig) -> None:
    images = OmegaConf.to_container(cfg.data.train.shape_meta.images, resolve=True)
    if not isinstance(images, list):
        raise ValueError("data.train.shape_meta.images must be a list.")
    keys = tuple(str(item["key"]) for item in images if isinstance(item, dict) and "key" in item)
    if keys != ("image", "wrist_image"):
        raise ValueError(f"V5 requires dataset camera keys ('image','wrist_image'), got {keys}.")
    if str(cfg.data.train.concat_multi_camera) != "horizontal":
        raise ValueError("V5 requires horizontal camera concatenation.")
    if list(cfg.data.train.video_size) != [224, 448]:
        raise ValueError(f"V5 requires data.train.video_size=[224,448], got {cfg.data.train.video_size}.")


def build_v5_loader(
    cfg: DictConfig,
    *,
    libero_data_root: str,
    dataset_stats_path: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    ddp_enabled: bool,
    world_size: int,
    rank: int,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    _resolve_dataset_dirs(cfg, root=libero_data_root)
    stats_path = require_file(dataset_stats_path, name="--dataset-stats-path")
    for key, value in (
        ("data.train.action_video_freq_ratio", 4),
        ("data.train.num_frames", 33),
        ("data.train.pretrained_norm_stats", str(stats_path)),
        ("data.train.preserve_context_mask", True),
        ("data.train.strict_errors", True),
        ("data.train.skip_padding_as_possible", False),
        ("data.train.is_training_set", True),
    ):
        OmegaConf.update(cfg, key, value, force_add=True)
    validate_camera_config(cfg)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    dataset = instantiate(dataset_cfg)
    if not bool(getattr(dataset, "strict_errors", False)):
        raise RuntimeError("V5 dataset was not constructed with strict_errors=True.")
    sampler = None
    if ddp_enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def build_vjepa_encoder(
    args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype
):
    from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper

    repo = require_dir(args.vjepa_repo, name="--vjepa-repo")
    checkpoint = require_file(args.vjepa_checkpoint, name="--vjepa-checkpoint")
    encoder = VJepaEncoderWrapper(
        dummy=False,
        num_tokens=256,
        vjepa_dim=1408,
        freeze=True,
        normalize_tokens=False,
        model_name=args.vjepa_model_name,
        external_repo_path=str(repo),
        checkpoint_path=str(checkpoint),
        checkpoint_key=args.vjepa_checkpoint_key,
        pretrained=False,
        img_size=args.vjepa_img_size,
        input_range=args.vjepa_input_range,
        tubelet_size=args.vjepa_tubelet_size,
        frame_encoding_mode="clip_or_repeat",
        strict_checkpoint_load=True,
    ).to(device=device, dtype=dtype)
    report = encoder.checkpoint_load_report
    if not isinstance(report, dict):
        raise RuntimeError("Strict V-JEPA checkpoint load did not produce a report.")
    if report["missing_keys"] or report["unexpected_keys"] or not report["strict"]:
        raise RuntimeError(f"V5 V-JEPA strict load verification failed: {report}.")
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def _release_action_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    mot_state = payload.get("mot")
    if not isinstance(mot_state, dict):
        raise ValueError("Release checkpoint must contain an exact `mot` state_dict.")
    prefix = "mixtures.action."
    state = {
        str(key)[len(prefix) :]: value
        for key, value in mot_state.items()
        if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
    }
    if not state:
        raise ValueError("Release checkpoint `mot` has no `mixtures.action.*` parameters.")
    return state


def load_release_modules(
    cfg: DictConfig,
    *,
    release_checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
    instantiate_action: bool,
):
    from fastwam.models.wan22.action_dit import ActionDiT

    payload = torch.load(release_checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Release checkpoint payload must be a dict.")
    proprio_state = payload.get("proprio_encoder")
    if not isinstance(proprio_state, dict):
        raise ValueError("Release checkpoint is missing exact `proprio_encoder` weights.")
    proprio_encoder = torch.nn.Linear(8, 4096)
    proprio_encoder.load_state_dict(proprio_state, strict=True)
    proprio_encoder = proprio_encoder.to(device=device, dtype=dtype)
    if not instantiate_action:
        return None, proprio_encoder
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict) or not isinstance(model_cfg.get("action_dit_config"), dict):
        raise ValueError("cfg.model.action_dit_config is required for release ActionDiT.")
    action_expert = ActionDiT(**dict(model_cfg["action_dit_config"]))
    state = _release_action_state(payload)
    expected = action_expert.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(state) if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Strict release ActionDiT load failed: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, shape={mismatched[:8]}."
        )
    action_expert.load_state_dict(state, strict=True)
    return action_expert.to(device=device, dtype=dtype), proprio_encoder


def load_v5_model_checkpoint(
    args: argparse.Namespace,
    *,
    cfg: DictConfig,
    checkpoint_path: Path,
    expected_stage: str,
    device: torch.device,
    dtype: torch.dtype,
    provenance: dict[str, Any],
):
    from fastwam.models.wan22.fastwam_jepa_idm_v5 import FastWAMJEPAIDMV5
    from fastwam.models.wan22.jepa_visual_dit_v5 import JEPAVisualDiTV5

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("V5 model checkpoint is missing metadata.")
    metadata = payload["metadata"]
    verify_provenance(
        metadata,
        expected_stage=expected_stage,
        vjepa_sha256=provenance["vjepa_sha"],
        release_sha256=provenance["release_sha"],
        dataset_stats_sha256=provenance["stats_sha"],
    )
    visual_config = metadata.get("visual_config")
    if not isinstance(visual_config, dict):
        raise ValueError("V5 model checkpoint metadata is missing visual_config.")
    visual = JEPAVisualDiTV5(
        num_layers=int(visual_config["num_layers"]),
        hidden_dim=int(visual_config["hidden_dim"]),
        ffn_dim=int(visual_config["ffn_dim"]),
        num_heads=int(visual_config["num_heads"]),
        attn_head_dim=int(visual_config["attn_head_dim"]),
        vjepa_dim=int(visual_config["vjepa_dim"]),
        text_dim=int(visual_config["text_dim"]),
        spatial_pool_size=int(visual_config["spatial_pool_size"]),
        use_gradient_checkpointing=False,
    ).to(device=device, dtype=dtype)
    action, proprio = load_release_modules(
        cfg,
        release_checkpoint=provenance["release_path"],
        device=device,
        dtype=dtype,
        instantiate_action=True,
    )
    if action is None:
        raise RuntimeError("V5 checkpoint load did not instantiate ActionDiT.")
    for key, module in (
        ("visual_dit", visual),
        ("action_expert", action),
        ("proprio_encoder", proprio),
    ):
        state = payload.get(key)
        if not isinstance(state, dict):
            raise ValueError(f"V5 checkpoint is missing {key} weights.")
        module.load_state_dict(state, strict=True)
    vjepa = build_vjepa_encoder(args, device=device, dtype=dtype)
    model = FastWAMJEPAIDMV5(
        vjepa_encoder=vjepa,
        visual_dit=visual,
        action_expert=action,
        proprio_encoder=proprio,
    ).to(device=device, dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    return model, metadata, payload


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = ("python", "numpy", "torch", "cuda")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Resume RNG state is missing {missing}.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def checkpoint_metadata(
    *,
    stage: str,
    args: argparse.Namespace,
    vjepa_sha256: str,
    release_sha256: str,
    dataset_stats_sha256: str,
    parameter_counts: dict[str, int],
    parent_checkpoint: Optional[Path] = None,
    parent_sha256: Optional[str] = None,
) -> dict[str, Any]:
    from fastwam.models.wan22.v5_contract import temporal_metadata

    metadata = {
        **temporal_metadata(),
        "stage": stage,
        "git_commit": git_commit_hash(),
        "resolved_config": dict(vars(args)),
        "vjepa_checkpoint_sha256": vjepa_sha256,
        "release_checkpoint_sha256": release_sha256,
        "dataset_stats_sha256": dataset_stats_sha256,
        "parameter_counts": dict(parameter_counts),
        "vjepa_contract": {
            "model_name": args.vjepa_model_name,
            "img_size": args.vjepa_img_size,
            "tubelet_size": args.vjepa_tubelet_size,
            "feature_dim": 1408,
            "current_tokens_per_camera": 256,
            "future_tokens_per_camera": 512,
        },
    }
    if parent_checkpoint is not None:
        if parent_sha256 is None:
            raise ValueError("parent_sha256 is required with parent_checkpoint.")
        metadata["parent_checkpoint"] = str(parent_checkpoint)
        metadata["parent_checkpoint_sha256"] = parent_sha256
    return metadata


def verify_provenance(
    metadata: dict[str, Any],
    *,
    expected_stage: str,
    vjepa_sha256: str,
    release_sha256: str,
    dataset_stats_sha256: str,
) -> None:
    expected = {
        "version": "v5",
        "stage": expected_stage,
        "vjepa_checkpoint_sha256": vjepa_sha256,
        "release_checkpoint_sha256": release_sha256,
        "dataset_stats_sha256": dataset_stats_sha256,
        "action_horizon": 16,
        "exec_horizon_default": 4,
        "dataset_video_indices": [0, 1, 2, 3, 4],
        "raw_observation_offsets": [0, 4, 8, 12, 16],
        "visual_stride": 4,
        "camera_order": ["agentview", "wrist"],
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V5 checkpoint provenance mismatch: {mismatches}.")


def save_compact_checkpoint(
    *,
    output_dir: Path,
    step: int,
    epoch: int,
    batches_in_epoch: int,
    weights: dict[str, dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metadata: dict[str, Any],
    rank: int,
    world_size: int,
) -> Optional[Path]:
    local_rng = capture_rng_state()
    rng_by_rank: Optional[list[Any]] = None
    if dist.is_initialized():
        if rank == 0:
            rng_by_rank = [None for _ in range(world_size)]
        dist.gather_object(local_rng, rng_by_rank, dst=0)
    else:
        rng_by_rank = [local_rng]
    output_path = None
    save_error = None
    if rank == 0:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_path = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
        payload = {
            **weights,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": int(step),
            "epoch": int(epoch),
            "batches_in_epoch": int(batches_in_epoch),
            "rng_state_by_rank": rng_by_rank,
            "metadata": metadata,
        }
        try:
            torch.save(payload, output_path)
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise IOError(f"V5 checkpoint save did not produce a non-empty file: {output_path}")
        except Exception as exc:
            if not dist.is_initialized():
                raise
            save_error = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        status = [save_error]
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(f"Rank0 V5 checkpoint save failed: {status[0]}")
        dist.barrier()
    return output_path


def resume_training_state(
    *,
    checkpoint_path: Path,
    expected_stage: str,
    modules: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[int, int, int, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("V5 resume checkpoint is missing metadata.")
    metadata = payload["metadata"]
    if metadata.get("version") != "v5" or metadata.get("stage") != expected_stage:
        raise ValueError(
            f"Cannot resume {expected_stage} from version={metadata.get('version')} stage={metadata.get('stage')}."
        )
    for name, module in modules.items():
        state = payload.get(name)
        if not isinstance(state, dict):
            raise ValueError(f"V5 resume checkpoint is missing {name} weights.")
        module.load_state_dict(state, strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    optimizer_to(optimizer, device)
    scheduler.load_state_dict(payload["scheduler"])
    rng_by_rank = payload.get("rng_state_by_rank")
    if not isinstance(rng_by_rank, list) or len(rng_by_rank) != world_size:
        raise ValueError(
            f"Resume RNG world_size mismatch: checkpoint={len(rng_by_rank) if isinstance(rng_by_rank, list) else None}, current={world_size}."
        )
    restore_rng_state(rng_by_rank[rank])
    metadata = dict(metadata)
    metadata["_resume_rng_state"] = rng_by_rank[rank]
    for key in ("global_step", "epoch", "batches_in_epoch"):
        if key not in payload:
            raise ValueError(f"V5 resume checkpoint is missing {key}.")
    return (
        int(payload["global_step"]),
        int(payload["epoch"]),
        int(payload["batches_in_epoch"]),
        metadata,
    )


def make_loader_iterator(
    loader: DataLoader,
    sampler: Optional[DistributedSampler],
    *,
    epoch: int,
    batches_in_epoch: int,
    resume_rng_state: Optional[dict[str, Any]] = None,
):
    if sampler is not None:
        sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(batches_in_epoch):
        try:
            next(iterator)
        except StopIteration as exc:
            raise ValueError(
                "Resume batches_in_epoch exceeds the current dataloader length."
            ) from exc
    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state)
    return iterator


def next_batch(loader, iterator, sampler, epoch: int, batches_in_epoch: int):
    try:
        batch = next(iterator)
        return batch, iterator, epoch, batches_in_epoch + 1
    except StopIteration:
        epoch += 1
        batches_in_epoch = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        batch = next(iterator)
        return batch, iterator, epoch, 1


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_fraction: float = 0.05
):
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def provenance_paths(args: argparse.Namespace, *, rank: int) -> dict[str, Any]:
    vjepa = require_file(args.vjepa_checkpoint, name="--vjepa-checkpoint")
    release = require_file(args.release_checkpoint, name="--release-checkpoint")
    stats = require_file(args.dataset_stats_path, name="--dataset-stats-path")
    require_dir(args.vjepa_repo, name="--vjepa-repo")
    require_dir(args.libero_data_root, name="--libero-data-root")
    return {
        "vjepa_path": vjepa,
        "release_path": release,
        "stats_path": stats,
        "vjepa_sha": distributed_sha256(vjepa, rank=rank),
        "release_sha": distributed_sha256(release, rank=rank),
        "stats_sha": distributed_sha256(stats, rank=rank),
    }


def json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
