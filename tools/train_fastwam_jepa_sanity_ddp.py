from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_VJEPA_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM_jepa/checkpoints/vjepa2/vitg.pt"
)
DEFAULT_ACTION_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM/runs/libero_joint_2cam224_1e-4/"
    "libero_joint_4gpu_20k_20260524_171732/checkpoints/weights/step_020000.pt"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "fastwam_jepa_sanity_ddp"
    / "checkpoint_step_100.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DDP sanity training for FastWAM-JEPA-Joint. Launch with "
            "`torchrun --nproc_per_node=N tools/train_fastwam_jepa_sanity_ddp.py`."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-rank batch size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--action-checkpoint", default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument(
        "--external-vjepa-repo",
        default=str(PROJECT_ROOT / "external" / "vjepa2"),
    )
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--num-future-tokens", type=int, default=256)
    parser.add_argument("--current-frame-count", type=int, default=2)
    parser.add_argument("--future-frame-count", type=int, default=2)
    parser.add_argument("--lambda-future", type=float, default=0.1)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP sanity training requires CUDA.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch this script with torchrun so RANK/WORLD_SIZE are set.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def is_rank0(rank: int) -> bool:
    return rank == 0


def rank0_print(rank: int, *values: Any) -> None:
    if is_rank0(rank):
        print(*values, flush=True)


def compose_cfg(config_name: str, task: str) -> DictConfig:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = (PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=[f"task={task}"])


def resolve_dataset_dirs(cfg: DictConfig, *, rank: int) -> None:
    dataset_dirs = cfg.data.train.get("dataset_dirs")
    if dataset_dirs is None:
        raise ValueError("`cfg.data.train.dataset_dirs` is required.")

    resolved_dirs: list[str] = []
    if is_rank0(rank):
        print("Resolved dataset_dirs:", flush=True)
    for dataset_dir in dataset_dirs:
        path = Path(str(dataset_dir))
        abs_path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if is_rank0(rank):
            print(f"  {abs_path}", flush=True)
        if not abs_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {abs_path}")
        if not abs_path.is_dir():
            raise FileNotFoundError(f"Dataset path is not a directory: {abs_path}")
        resolved_dirs.append(str(abs_path))

    cfg.data.train.dataset_dirs = resolved_dirs


def build_loader(
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DistributedSampler]:
    resolve_dataset_dirs(cfg, rank=rank)
    dataset = instantiate(cfg.data.train)
    rank0_print(rank, f"Dataset: {type(dataset).__name__}, len={len(dataset)}")
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
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def assert_video_layout(sample: dict[str, Any]) -> None:
    video = sample.get("video")
    if not torch.is_tensor(video):
        raise ValueError("Batch is missing tensor `sample['video']`.")
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(
            "`sample['video']` must be [B, 3, T, H, W]. "
            f"Got {tuple(video.shape)}. This script does not silently permute."
        )


def extract_action_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload["mot"] if "mot" in payload and isinstance(payload["mot"], dict) else payload
    prefixes = (
        "mixtures.action.",
        "dit.mixtures.action.",
        "mot.mixtures.action.",
        "module.mixtures.action.",
        "module.dit.mixtures.action.",
        "module.mot.mixtures.action.",
        "action_expert.",
        "model.action_expert.",
        "module.action_expert.",
    )
    for prefix in prefixes:
        action_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if action_state:
            return action_state

    raise ValueError(
        "Could not find ActionDiT weights in checkpoint. Expected keys with "
        "prefix like `mixtures.action.*` under payload['mot']."
    )


def build_action_expert(
    *,
    action_cfg: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from fastwam.models.wan22.action_dit import ActionDiT

    action_expert = ActionDiT(**action_cfg)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Action checkpoint payload must be a dict, got {type(payload)}.")
    action_expert.load_state_dict(extract_action_state_dict(payload), strict=True)
    return action_expert.to(device=device, dtype=dtype)


def build_model(
    *,
    cfg: DictConfig,
    action_checkpoint: str,
    vjepa_checkpoint: str,
    external_vjepa_repo: str,
    vjepa_model_name: str,
    vjepa_dim: int,
    num_future_tokens: int,
    current_frame_count: int,
    future_frame_count: int,
    lambda_future: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from fastwam.models.vjepa import VJepaEncoderWrapper
    from fastwam.models.wan22.fastwam_jepa_joint import FastWAMJEPAJoint

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise ValueError(f"`cfg.model` must resolve to dict, got {type(model_cfg)}.")

    action_cfg = dict(model_cfg["action_dit_config"])
    action_expert = build_action_expert(
        action_cfg=action_cfg,
        checkpoint_path=action_checkpoint,
        device=device,
        dtype=dtype,
    )

    vjepa_encoder = VJepaEncoderWrapper(
        dummy=False,
        model_name=vjepa_model_name,
        external_repo_path=external_vjepa_repo,
        checkpoint_path=vjepa_checkpoint,
        pretrained=False,
        vjepa_dim=vjepa_dim,
        num_tokens=num_future_tokens,
        freeze=True,
        normalize_tokens=True,
    ).to(device=device, dtype=dtype)

    action_scheduler_cfg = model_cfg.get("action_scheduler", {})
    proprio_dim = model_cfg.get("proprio_dim")
    model = FastWAMJEPAJoint(
        action_expert=action_expert,
        vjepa_encoder=vjepa_encoder,
        action_dim=int(action_cfg["action_dim"]),
        hidden_dim=int(action_cfg["hidden_dim"]),
        vjepa_dim=int(vjepa_dim),
        num_future_tokens=int(num_future_tokens),
        text_dim=int(action_cfg["text_dim"]),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=None,
        torch_dtype=dtype,
        action_train_shift=float(action_scheduler_cfg.get("train_shift", 5.0)),
        action_num_train_timesteps=int(action_scheduler_cfg.get("num_train_timesteps", 1000)),
        lambda_future=float(lambda_future),
        current_frame_count=int(current_frame_count),
        future_frame_count=int(future_frame_count),
    )
    return model.to(device=device, dtype=dtype)


def configure_trainable_modules(model: torch.nn.Module, *, rank: int) -> list[torch.nn.Parameter]:
    model.eval()
    model.requires_grad_(False)

    model.vjepa_encoder.eval()
    model.action_expert.eval()

    model.joint_predictor.train()
    model.joint_predictor.requires_grad_(True)

    if model.proprio_encoder is not None:
        model.proprio_encoder.train()
        model.proprio_encoder.requires_grad_(True)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for DDP sanity training.")

    if is_rank0(rank):
        print("Trainable parameter groups:", flush=True)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {tuple(param.shape)}", flush=True)
        print(f"Total trainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    return trainable_params


def reduce_mean_scalar(value: float, *, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / float(world_size)).item())


def check_finite_across_ranks(loss_total: torch.Tensor, *, device: torch.device) -> bool:
    finite_flag = torch.tensor(
        [1 if torch.isfinite(loss_total).item() else 0],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
    return bool(finite_flag.item())


def gpu_memory_text(device: torch.device) -> str:
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    return f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB"


def save_sanity_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if isinstance(model, DDP) else model
    payload = {
        "step": int(step),
        "joint_predictor": module.joint_predictor.state_dict(),
        "proprio_encoder": (
            None if module.proprio_encoder is None else module.proprio_encoder.state_dict()
        ),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, output_path)


def next_batch(
    *,
    loader: DataLoader,
    sampler: DistributedSampler,
    iterator,
    epoch: int,
) -> tuple[dict[str, Any], Any, int]:
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    rank, world_size, local_rank, device = init_distributed()
    dtype = resolve_dtype(args.dtype)
    torch.manual_seed(int(args.seed) + rank)

    rank0_print(rank, f"DDP world_size={world_size}")
    rank0_print(rank, f"Using dtype={dtype}")
    rank0_print(
        rank,
        "Frame settings: "
        f"current_frame_count={args.current_frame_count}, "
        f"future_frame_count={args.future_frame_count}, "
        f"num_future_tokens={args.num_future_tokens}",
    )

    try:
        cfg = compose_cfg(config_name=args.config_name, task=args.task)
        loader, sampler = build_loader(
            cfg,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        sampler.set_epoch(0)

        model = build_model(
            cfg=cfg,
            action_checkpoint=args.action_checkpoint,
            vjepa_checkpoint=args.vjepa_checkpoint,
            external_vjepa_repo=args.external_vjepa_repo,
            vjepa_model_name=args.vjepa_model_name,
            vjepa_dim=args.vjepa_dim,
            num_future_tokens=args.num_future_tokens,
            current_frame_count=args.current_frame_count,
            future_frame_count=args.future_frame_count,
            lambda_future=args.lambda_future,
            device=device,
            dtype=dtype,
        )
        trainable_params = configure_trainable_modules(model, rank=rank)

        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            betas=(0.9, 0.95),
        )

        torch.cuda.reset_peak_memory_stats(device)
        iterator = iter(loader)
        epoch = 0
        for step in range(1, int(args.steps) + 1):
            sample, iterator, epoch = next_batch(
                loader=loader,
                sampler=sampler,
                iterator=iterator,
                epoch=epoch,
            )
            assert_video_layout(sample)

            optimizer.zero_grad(set_to_none=True)
            loss_total, loss_dict = ddp_model(sample)

            if not check_finite_across_ranks(loss_total, device=device):
                raise FloatingPointError(
                    f"Non-finite loss detected at step {step}; stopping all ranks."
                )

            loss_total.backward()
            optimizer.step()

            if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
                mean_total = reduce_mean_scalar(
                    loss_dict["loss_total"], device=device, world_size=world_size
                )
                mean_action = reduce_mean_scalar(
                    loss_dict["loss_action"], device=device, world_size=world_size
                )
                mean_future = reduce_mean_scalar(
                    loss_dict["loss_future_vjepa"], device=device, world_size=world_size
                )
                if is_rank0(rank):
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"step={step} "
                        f"loss_total={mean_total:.6f} "
                        f"loss_action={mean_action:.6f} "
                        f"loss_future_vjepa={mean_future:.6f} "
                        f"lr={lr:.6e} "
                        f"gpu_memory={gpu_memory_text(device)}",
                        flush=True,
                    )

        if is_rank0(rank):
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = (PROJECT_ROOT / output_path).resolve()
            save_sanity_checkpoint(
                model=ddp_model,
                optimizer=optimizer,
                step=int(args.steps),
                args=args,
                output_path=output_path,
            )
            print(f"Saved sanity checkpoint to {output_path}", flush=True)

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
