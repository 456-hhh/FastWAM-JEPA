from __future__ import annotations

import argparse
import os
import sys
import time
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

DEFAULT_VJEPA_REPO = "/data1/Johnny/challenge/dd/FastWAM_jepa/external/vjepa2"
DEFAULT_VJEPA_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM_jepa/checkpoints/vjepa2/vitg.pt"
)
DEFAULT_ACTION_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM/runs/libero_joint_2cam224_1e-4/"
    "libero_joint_4gpu_20k_20260524_171732/checkpoints/weights/step_020000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Formal DDP training for FastWAM-JEPA-Joint v1. Launch with "
            "`torchrun --nproc_per_node=N tools/train_fastwam_jepa_joint.py`."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-rank batch size.")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/fastwam_jepa_joint_v1")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--freeze-action-dit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze ActionDiT by default. Use --no-freeze-action-dit to train all used/unused ActionDiT params.",
    )
    parser.add_argument("--train-action-head", action="store_true", default=False)
    parser.add_argument("--train-action-encoder", action="store_true", default=False)
    parser.add_argument("--lambda-future", type=float, default=0.1)
    parser.add_argument("--current-frame-count", type=int, default=2)
    parser.add_argument("--future-frame-count", type=int, default=2)
    parser.add_argument("--num-future-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--vjepa-repo", default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--action-checkpoint", default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA.")
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


def resolve_path(path_value: str | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


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
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def assert_batch_video(sample: dict[str, Any], *, current_frames: int, future_frames: int) -> None:
    video = sample.get("video")
    if not torch.is_tensor(video):
        raise ValueError("Batch is missing tensor `sample['video']`.")
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(
            "`sample['video']` must be [B, 3, T, H, W]. "
            f"Got {tuple(video.shape)}. This script does not silently permute."
        )
    required_frames = int(current_frames) + int(future_frames)
    if int(video.shape[2]) < required_frames:
        raise ValueError(
            "`sample['video']` does not contain enough frames: "
            f"T={int(video.shape[2])}, required at least {required_frames} "
            f"({current_frames} current + {future_frames} future)."
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
    args: argparse.Namespace,
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
        checkpoint_path=str(resolve_path(args.action_checkpoint)),
        device=device,
        dtype=dtype,
    )

    vjepa_encoder = VJepaEncoderWrapper(
        dummy=False,
        model_name=args.vjepa_model_name,
        external_repo_path=str(resolve_path(args.vjepa_repo)),
        checkpoint_path=str(resolve_path(args.vjepa_checkpoint)),
        pretrained=False,
        vjepa_dim=int(args.vjepa_dim),
        num_tokens=int(args.num_future_tokens),
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
        vjepa_dim=int(args.vjepa_dim),
        num_future_tokens=int(args.num_future_tokens),
        text_dim=int(action_cfg["text_dim"]),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=None,
        torch_dtype=dtype,
        action_train_shift=float(action_scheduler_cfg.get("train_shift", 5.0)),
        action_num_train_timesteps=int(action_scheduler_cfg.get("num_train_timesteps", 1000)),
        lambda_future=float(args.lambda_future),
        current_frame_count=int(args.current_frame_count),
        future_frame_count=int(args.future_frame_count),
    )
    return model.to(device=device, dtype=dtype)


def set_module_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    module.train(trainable)
    module.requires_grad_(trainable)


def configure_trainable_modules(model: torch.nn.Module, args: argparse.Namespace) -> None:
    model.eval()
    model.requires_grad_(False)

    model.vjepa_encoder.eval()
    model.vjepa_encoder.requires_grad_(False)

    model.action_expert.eval()
    model.action_expert.requires_grad_(False)

    set_module_trainable(model.joint_predictor, True)
    set_module_trainable(model.proprio_encoder, True)

    if not bool(args.freeze_action_dit):
        set_module_trainable(model.action_expert, True)
    else:
        if bool(args.train_action_head):
            set_module_trainable(getattr(model.action_expert, "head", None), True)
        if bool(args.train_action_encoder):
            set_module_trainable(getattr(model.action_expert, "action_encoder", None), True)


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found.")
    return params


def parameter_counts(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def print_trainable_summary(model: torch.nn.Module, *, rank: int) -> None:
    if not is_rank0(rank):
        return
    total, trainable, frozen = parameter_counts(model)
    print(f"Total params: {total:,}", flush=True)
    print(f"Trainable params: {trainable:,}", flush=True)
    print(f"Frozen params: {frozen:,}", flush=True)
    print("requires_grad=True modules/parameters:", flush=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {tuple(param.shape)}", flush=True)


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


def checkpoint_path_for_step(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"checkpoint_step_{step:06d}.pt"


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    loss_dict: dict[str, float],
    output_dir: Path,
) -> Path:
    output_path = checkpoint_path_for_step(output_dir, step)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if isinstance(model, DDP) else model
    payload = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "args": vars(args),
        "loss_dict": dict(loss_dict),
    }
    torch.save(payload, output_path)
    return output_path


def load_resume_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: Path,
    device: torch.device,
) -> int:
    payload = torch.load(resume_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Resume checkpoint must be a dict, got {type(payload)}.")
    if "model" not in payload or "optimizer" not in payload or "step" not in payload:
        raise ValueError(
            "Resume checkpoint must contain `model`, `optimizer`, and `step` keys."
        )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"])


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

    output_dir = resolve_path(args.output_dir)
    assert output_dir is not None
    resume_path = resolve_path(args.resume) if args.resume else None

    rank0_print(rank, f"DDP world_size={world_size}")
    rank0_print(rank, f"Using dtype={dtype}")
    rank0_print(rank, f"Output dir: {output_dir}")
    rank0_print(
        rank,
        "Frame settings: "
        f"current_frame_count={args.current_frame_count}, "
        f"future_frame_count={args.future_frame_count}, "
        f"num_future_tokens={args.num_future_tokens}",
    )
    rank0_print(
        rank,
        "Train settings: "
        f"steps={args.steps}, per_rank_batch_size={args.batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}",
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

        model = build_model(cfg=cfg, args=args, device=device, dtype=dtype)
        configure_trainable_modules(model, args)
        print_trainable_summary(model, rank=rank)

        params = trainable_parameters(model)
        optimizer = torch.optim.AdamW(
            params,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            betas=(0.9, 0.95),
        )

        start_step = 0
        if resume_path is not None:
            start_step = load_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                resume_path=resume_path,
                device=device,
            )
            rank0_print(rank, f"Resumed from {resume_path} at step {start_step}")

        find_unused = not bool(args.freeze_action_dit)
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )

        torch.cuda.reset_peak_memory_stats(device)
        iterator = iter(loader)
        epoch = 0
        last_log_time = time.perf_counter()
        last_log_step = start_step
        last_loss_dict = {
            "loss_total": float("nan"),
            "loss_action": float("nan"),
            "loss_future_vjepa": float("nan"),
        }

        for step in range(start_step + 1, int(args.steps) + 1):
            sample, iterator, epoch = next_batch(
                loader=loader,
                sampler=sampler,
                iterator=iterator,
                epoch=epoch,
            )
            assert_batch_video(
                sample,
                current_frames=int(args.current_frame_count),
                future_frames=int(args.future_frame_count),
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total, loss_dict = ddp_model(sample)

            if not check_finite_across_ranks(loss_total, device=device):
                raise FloatingPointError(
                    f"Non-finite loss detected at step {step}; stopping all ranks."
                )

            loss_total.backward()
            optimizer.step()
            last_loss_dict = dict(loss_dict)

            should_log = (
                step == start_step + 1
                or step % int(args.log_every) == 0
                or step == int(args.steps)
            )
            if should_log:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-9)
                step_delta = max(step - last_log_step, 1)
                iter_time = elapsed / float(step_delta)
                samples_per_sec = (
                    float(step_delta)
                    * float(args.batch_size)
                    * float(world_size)
                    / elapsed
                )
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
                        f"gpu_memory={gpu_memory_text(device)} "
                        f"iter_time={iter_time:.3f}s "
                        f"samples_per_sec={samples_per_sec:.2f}",
                        flush=True,
                    )
                last_log_time = now
                last_log_step = step

            should_save = (
                int(args.save_every) > 0
                and step % int(args.save_every) == 0
                and is_rank0(rank)
            )
            if should_save:
                ckpt_path = save_checkpoint(
                    model=ddp_model,
                    optimizer=optimizer,
                    step=step,
                    args=args,
                    loss_dict=last_loss_dict,
                    output_dir=output_dir,
                )
                print(f"Saved checkpoint to {ckpt_path}", flush=True)

        if is_rank0(rank):
            final_path = save_checkpoint(
                model=ddp_model,
                optimizer=optimizer,
                step=int(args.steps),
                args=args,
                loss_dict=last_loss_dict,
                output_dir=output_dir,
            )
            print(f"Saved final checkpoint to {final_path}", flush=True)

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
