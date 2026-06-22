from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_ACTION_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM/runs/libero_joint_2cam224_1e-4/"
    "libero_joint_4gpu_20k_20260524_171732/checkpoints/weights/step_020000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check FastWAM-JEPA-IDM v2 with dummy V-JEPA tokens and a real "
            "ActionDiT loaded from a FastWAM/FastWAM-IDM checkpoint."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--action-checkpoint", default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument("--future-source", default="oracle", choices=["oracle", "predicted", "no_future"])
    parser.add_argument("--lambda-future", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--current-frame-count", type=int, default=2)
    parser.add_argument("--future-frame-count", type=int, default=2)
    parser.add_argument("--num-future-tokens", type=int, default=256)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--adapter-current-tokens", type=int, default=16)
    parser.add_argument("--adapter-future-tokens", type=int, default=16)
    parser.add_argument("--future-predictor-layers", type=int, default=2)
    parser.add_argument("--future-predictor-heads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def _state_dict_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in ("mot", "model", "model_state_dict", "state_dict", "module"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    candidates.append(payload)
    return candidates


def extract_action_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
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
    for state in _state_dict_candidates(payload):
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
        "prefix like `mixtures.action.*` under payload['mot'], or `action_expert.*`."
    )


def build_action_expert(
    *,
    action_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from fastwam.models.wan22.action_dit import ActionDiT

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Action checkpoint does not exist: {checkpoint_path}")
    action_expert = ActionDiT(**action_cfg)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Action checkpoint payload must be a dict, got {type(payload)}.")

    action_state = extract_action_state_dict(payload)
    missing, unexpected = action_expert.load_state_dict(action_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "Unexpected ActionDiT load_state_dict result with strict=True: "
            f"missing={missing}, unexpected={unexpected}."
        )
    return action_expert.to(device=device, dtype=dtype)


def build_model(
    *,
    action_expert: torch.nn.Module,
    args: argparse.Namespace,
    cfg: DictConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper
    from fastwam.models.wan22.fastwam_jepa_idm import FastWAMJEPAIDM

    proprio_dim = int(OmegaConf.select(cfg, "model.proprio_dim"))
    vjepa_encoder = VJepaEncoderWrapper(
        dummy=True,
        num_tokens=args.num_future_tokens,
        vjepa_dim=args.vjepa_dim,
        freeze=True,
    )
    model = FastWAMJEPAIDM(
        action_expert=action_expert,
        vjepa_encoder=vjepa_encoder,
        action_dim=int(action_expert.action_dim),
        hidden_dim=int(action_expert.hidden_dim),
        vjepa_dim=args.vjepa_dim,
        num_future_tokens=args.num_future_tokens,
        text_dim=int(action_expert.text_dim),
        proprio_dim=proprio_dim,
        torch_dtype=dtype,
        lambda_future=args.lambda_future,
        current_frame_count=args.current_frame_count,
        future_frame_count=args.future_frame_count,
        adapter_current_tokens=args.adapter_current_tokens,
        adapter_future_tokens=args.adapter_future_tokens,
        future_predictor_layers=args.future_predictor_layers,
        future_predictor_heads=args.future_predictor_heads,
        future_source=args.future_source,
    )
    return model.to(device=device, dtype=dtype)


def build_sample(
    *,
    batch_size: int,
    current_frame_count: int,
    future_frame_count: int,
    action_horizon: int,
    action_dim: int,
    text_dim: int,
    context_len: int,
    image_size: int,
    proprio_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    total_frames = int(current_frame_count) + int(future_frame_count)
    return {
        "video": torch.randn(batch_size, 3, total_frames, image_size, image_size, device=device, dtype=dtype),
        "action": torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=dtype),
        "context": torch.randn(batch_size, context_len, text_dim, device=device, dtype=dtype),
        "context_mask": torch.ones(batch_size, context_len, device=device, dtype=torch.bool),
        "proprio": torch.randn(batch_size, total_frames, proprio_dim, device=device, dtype=dtype),
        "action_is_pad": torch.zeros(batch_size, action_horizon, device=device, dtype=torch.bool),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    checkpoint_path = resolve_path(args.action_checkpoint)
    if checkpoint_path is None:
        raise ValueError("`--action-checkpoint` is required.")

    cfg = compose_cfg(args.config_name, args.task)
    action_cfg = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
    if not isinstance(action_cfg, dict):
        raise ValueError(f"`cfg.model.action_dit_config` must resolve to dict, got {type(action_cfg)}.")

    action_expert = build_action_expert(
        action_cfg=action_cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
    )
    action_expert.eval()

    block_calls = {"count": 0}

    def count_block_call(module, inputs, output) -> None:
        del module, inputs, output
        block_calls["count"] += 1

    handles = [block.register_forward_hook(count_block_call) for block in action_expert.blocks]
    try:
        model = build_model(
            action_expert=action_expert,
            args=args,
            cfg=cfg,
            device=device,
            dtype=dtype,
        )
        model.eval()
        sample = build_sample(
            batch_size=args.batch_size,
            current_frame_count=args.current_frame_count,
            future_frame_count=args.future_frame_count,
            action_horizon=args.action_horizon,
            action_dim=int(action_expert.action_dim),
            text_dim=int(action_expert.text_dim),
            context_len=args.context_len,
            image_size=args.image_size,
            proprio_dim=int(OmegaConf.select(cfg, "model.proprio_dim")),
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            loss_total, loss_dict = model.training_loss(sample)

        if not torch.isfinite(loss_total):
            raise RuntimeError("loss_total is not finite.")
        expected_block_calls = int(len(action_expert.blocks))
        if block_calls["count"] != expected_block_calls:
            raise RuntimeError(
                "ActionDiT.blocks call count mismatch: "
                f"expected {expected_block_calls}, got {block_calls['count']}."
            )

        shapes = model.last_forward_shapes
        print(f"checkpoint_path={checkpoint_path}")
        print(f"future_source={args.future_source}")
        print(f"loss_total={float(loss_total.detach().item()):.6f}")
        print(f"loss_action={loss_dict['loss_action']:.6f}")
        print(f"loss_future_jepa={loss_dict['loss_future_jepa']:.6f}")
        print(f"current_jepa_tokens shape={shapes['current_jepa_tokens']}")
        print(f"target_future_jepa_tokens shape={shapes['target_future_jepa_tokens']}")
        print(f"action_context shape={shapes['action_context']}")
        print(f"pred_action shape={shapes['pred_action']}")
        print(f"action_dit_block_calls={block_calls['count']}")
        print(f"action_dit_hidden_dim={int(action_expert.hidden_dim)}")
        print(f"action_dit_text_dim={int(action_expert.text_dim)}")
        print(f"action_dit_action_dim={int(action_expert.action_dim)}")
        print(f"action_dit_num_blocks={expected_block_calls}")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
