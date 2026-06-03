from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real LIBERO batch through real V-JEPA2 encoder, real "
            "ActionDiT weights, and FastWAMJEPAJoint. This is a sanity check "
            "only: no optimizer step and no checkpoint save."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--future-frame-count", type=int, default=1)
    parser.add_argument("--lambda-future", type=float, default=0.1)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def compose_cfg(config_name: str, task: str) -> DictConfig:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = (PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=[f"task={task}"])


def resolve_dataset_dirs(cfg: DictConfig) -> None:
    dataset_dirs = cfg.data.train.get("dataset_dirs")
    if dataset_dirs is None:
        raise ValueError("`cfg.data.train.dataset_dirs` is required for this sanity check.")

    resolved_dirs: list[str] = []
    print("Resolved dataset_dirs:")
    for dataset_dir in dataset_dirs:
        path = Path(str(dataset_dir))
        if path.is_absolute():
            abs_path = path.resolve()
        else:
            abs_path = (PROJECT_ROOT / path).resolve()

        print(f"  {abs_path}")
        if not abs_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {abs_path}")
        if not abs_path.is_dir():
            raise FileNotFoundError(f"Dataset path is not a directory: {abs_path}")
        resolved_dirs.append(str(abs_path))

    cfg.data.train.dataset_dirs = resolved_dirs


def build_one_batch_loader(cfg: DictConfig, *, batch_size: int, num_workers: int) -> DataLoader:
    resolve_dataset_dirs(cfg)
    dataset = instantiate(cfg.data.train)
    print(f"Dataset: {type(dataset).__name__}, len={len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def tensor_shape(value: Any) -> str:
    if torch.is_tensor(value):
        return f"{tuple(value.shape)} dtype={value.dtype} device={value.device}"
    return f"{type(value).__name__}"


def print_sample_shapes(sample: dict[str, Any]) -> None:
    print("Sample keys:")
    for key in sorted(sample.keys()):
        print(f"  {key}: {tensor_shape(sample[key])}")

    print("Required FastWAMJEPAJoint sample fields:")
    for key in ("video", "action", "context", "context_mask", "proprio", "action_is_pad"):
        value = sample.get(key)
        if value is None:
            print(f"  {key}: <missing>")
        else:
            print(f"  {key}: {tensor_shape(value)}")


def assert_video_layout(sample: dict[str, Any]) -> None:
    video = sample.get("video")
    if not torch.is_tensor(video):
        raise ValueError("Batch is missing tensor `sample['video']`.")
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(
            "`sample['video']` must be [B, 3, T, H, W] for this sanity check; "
            f"got shape {tuple(video.shape)}. This script intentionally does not "
            "silently permute video layout."
        )


def extract_action_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "mot" in payload and isinstance(payload["mot"], dict):
        state = payload["mot"]
    else:
        state = payload

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
        raise ValueError(
            f"Action checkpoint payload must be a dict, got {type(payload)}."
        )

    action_state = extract_action_state_dict(payload)
    missing, unexpected = action_expert.load_state_dict(action_state, strict=True)
    if missing or unexpected:
        raise ValueError(
            "Unexpected ActionDiT load_state_dict result with strict=True: "
            f"missing={missing}, unexpected={unexpected}."
        )
    action_expert.to(device=device, dtype=dtype)
    return action_expert


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
    )
    vjepa_encoder.to(device=device, dtype=dtype)

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
        action_num_train_timesteps=int(
            action_scheduler_cfg.get("num_train_timesteps", 1000)
        ),
        lambda_future=float(lambda_future),
        current_frame_count=int(current_frame_count),
        future_frame_count=int(future_frame_count),
    )
    model.to(device=device, dtype=dtype)
    return model


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"Using device={device}, dtype={dtype}")

    cfg = compose_cfg(config_name=args.config_name, task=args.task)
    loader = build_one_batch_loader(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    sample = next(iter(loader))
    print_sample_shapes(sample)
    assert_video_layout(sample)

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
    model.eval()

    with torch.no_grad():
        loss_total, loss_dict = model.training_loss(sample)

    print(f"loss_total: {float(loss_total.detach().item())}")
    print(f"loss_dict: {loss_dict}")


if __name__ == "__main__":
    main()
