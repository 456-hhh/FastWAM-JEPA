from __future__ import annotations

import argparse
import math
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

from fastwam_jepa_runtime_guard import configure_runtime_stability

DEFAULT_ACTION_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM/runs/libero_joint_2cam224_1e-4/"
    "libero_joint_4gpu_20k_20260524_171732/checkpoints/weights/step_020000.pt"
)
DEFAULT_VJEPA_REPO = "/data1/Johnny/challenge/dd/FastWAM_jepa/external/vjepa2"
DEFAULT_VJEPA_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM_jepa/checkpoints/vjepa2/vitg.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FastWAM-JEPA-IDM v2 JEPA future predictor sanity/training loop. "
            "V-JEPA and ActionDiT stay frozen; --train-predictor trains only "
            "JepaFuturePredictor with L1 V-JEPA latent supervision."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--action-checkpoint", default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument("--vjepa-repo", default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=["-1_1", "0_1"])
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--freeze-vjepa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--future-source", default="oracle", choices=["oracle", "predicted", "no_future"])
    parser.add_argument("--predictor-checkpoint", default=None, help="V-JEPA2-AC predictor checkpoint for partial pretrained init.")
    parser.add_argument(
        "--allow-random-predictor",
        action="store_true",
        default=False,
        help="Explicit debug escape hatch for random predictor init. Never use for eval.",
    )
    parser.add_argument("--train-predictor", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output-dir", default="runs/fastwam_jepa_idm_v2_predictor")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--lambda-action", type=float, default=None)
    parser.add_argument("--lambda-future", type=float, default=None)
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
    parser.add_argument("--disable-wsl-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-log-level", default="INFO", choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--runtime-log-path", default=None)
    parser.add_argument("--runtime-log-max-mb", type=int, default=100)
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


def seed_everything(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


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


def effective_lambdas(args: argparse.Namespace) -> tuple[float, float]:
    lambda_action = args.lambda_action
    lambda_future = args.lambda_future
    if lambda_action is None:
        lambda_action = 0.0 if args.train_predictor else 1.0
    if lambda_future is None:
        lambda_future = 1.0 if args.train_predictor else 0.0
    return float(lambda_action), float(lambda_future)


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
    vjepa_repo = resolve_path(args.vjepa_repo)
    vjepa_checkpoint = resolve_path(args.vjepa_checkpoint)
    if vjepa_repo is None:
        raise ValueError("`--vjepa-repo` is required.")
    if vjepa_checkpoint is None:
        raise ValueError("`--vjepa-checkpoint` is required.")

    lambda_action, lambda_future = effective_lambdas(args)
    vjepa_encoder = VJepaEncoderWrapper(
        dummy=False,
        num_tokens=args.num_future_tokens,
        vjepa_dim=args.vjepa_dim,
        freeze=args.freeze_vjepa,
        model_name=args.vjepa_model_name,
        external_repo_path=str(vjepa_repo),
        checkpoint_path=str(vjepa_checkpoint),
        pretrained=False,
        img_size=args.vjepa_img_size,
        input_range=args.vjepa_input_range,
        tubelet_size=args.vjepa_tubelet_size,
        frame_encoding_mode="clip_or_repeat",
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
        lambda_action=lambda_action,
        lambda_future=lambda_future,
        current_frame_count=args.current_frame_count,
        future_frame_count=args.future_frame_count,
        adapter_current_tokens=args.adapter_current_tokens,
        adapter_future_tokens=args.adapter_future_tokens,
        future_predictor_layers=args.future_predictor_layers,
        future_predictor_heads=args.future_predictor_heads,
        future_source=args.future_source,
    )
    return model.to(device=device, dtype=dtype)


def load_predictor_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path | None,
    future_source: str,
    allow_random_predictor: bool,
    device: torch.device,
) -> tuple[str, dict[str, Any] | None]:
    del device
    if future_source in {"oracle", "no_future"}:
        return f"unused_{future_source}_gt_future", None

    if checkpoint_path is None:
        if allow_random_predictor:
            return "random_init_debug_explicit", None
        raise ValueError(
            "future_source='predicted' requires --predictor-checkpoint for V-JEPA2-AC partial init. "
            "For debug-only random init, pass --allow-random-predictor explicitly."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"V-JEPA2-AC predictor checkpoint does not exist: {checkpoint_path}")

    # Training and eval ablations must share the exact same V-JEPA2-AC predictor
    # mapping, skip rules, strict=False partial loading, and shape-mismatch logging.
    # Do not torch.load a predictor state_dict here and assign it directly.
    stats = model.future_predictor.load_vjepa2ac_predictor_weights(checkpoint_path)
    if int(stats.get("loaded_keys_count", 0)) <= 0:
        raise RuntimeError(
            "V-JEPA2-AC predictor checkpoint did not load any compatible keys. "
            f"stats={stats}"
        )
    return (
        "vjepa2ac_pretrained_init:"
        f"{checkpoint_path}:loaded_keys={stats.get('loaded_keys_count')}:"
        f"skipped_keys={stats.get('skipped_keys_count')}:"
        f"shape_mismatch={stats.get('shape_mismatch_count')}",
        {"vjepa2ac_load_stats": stats},
    )


def save_predictor_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    step: int,
    loss_dict: dict[str, float],
) -> Path:
    output_dir = resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("`--output-dir` is required to save predictor checkpoints.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"predictor_step_{step:06d}.pt"
    payload = {
        "future_predictor": model.future_predictor.state_dict(),
        "step": int(step),
        "args": vars(args),
        "loss_dict": dict(loss_dict),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def configure_trainability(model: torch.nn.Module, *, train_predictor: bool) -> list[torch.nn.Parameter]:
    model.eval()
    model.requires_grad_(False)
    model.vjepa_encoder.eval()
    model.vjepa_encoder.requires_grad_(False)
    model.action_expert.eval()
    model.action_expert.requires_grad_(False)

    if not train_predictor:
        return []

    model.future_predictor.train()
    model.future_predictor.requires_grad_(True)
    return [param for param in model.future_predictor.parameters() if param.requires_grad]


def count_trainable_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


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


def evaluate_oracle_predicted_gap(
    *,
    model: torch.nn.Module,
    sample: dict[str, torch.Tensor],
    predictor_available: bool,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    original_source = model.future_source
    was_training = model.training
    model.eval()
    with torch.no_grad():
        seed_everything(seed, device)
        model.future_source = "oracle"
        _, oracle_loss = model.training_loss(sample)
        oracle_action = float(oracle_loss["loss_action"])
        if predictor_available:
            seed_everything(seed, device)
            model.future_source = "predicted"
            _, predicted_loss = model.training_loss(sample)
            predicted_action = float(predicted_loss["loss_action"])
            predictor_gap = float(predicted_loss["oracle_vs_predicted_gap"])
        else:
            predicted_action = math.nan
            predictor_gap = math.nan
    model.future_source = original_source
    model.train(was_training)
    return {
        "oracle_loss_action": oracle_action,
        "predicted_loss_action": predicted_action,
        "action_loss_invariance_delta": abs(predicted_action - oracle_action) if not math.isnan(predicted_action) else math.nan,
        "oracle_vs_predicted_gap": predictor_gap,
    }


def validate_main_forward(
    *,
    shapes: dict[str, Any],
    vjepa_outputs: list[tuple[tuple[int, ...], bool]],
    args: argparse.Namespace,
) -> None:
    if len(vjepa_outputs) != 2:
        raise RuntimeError(
            "Expected exactly two V-JEPA encoder forwards in the main loss "
            f"(current and future), got {len(vjepa_outputs)}."
        )
    if args.freeze_vjepa and any(requires_grad for _, requires_grad in vjepa_outputs):
        raise RuntimeError(
            "freeze_vjepa=True but at least one V-JEPA encoder output still requires grad: "
            f"{vjepa_outputs}."
        )
    expected_jepa_shape = (args.batch_size, args.num_future_tokens, args.vjepa_dim)
    for key in ("current_jepa_tokens", "target_future_jepa_tokens", "pred_future_jepa_tokens"):
        if tuple(shapes[key]) != expected_jepa_shape:
            raise RuntimeError(
                f"{key} shape must match dummy-compatible shape {expected_jepa_shape}, "
                f"got {shapes[key]}."
            )


def main() -> None:
    args = parse_args()
    runtime_status = configure_runtime_stability(
        disable_wsl_fallback=args.disable_wsl_fallback,
        log_level=args.runtime_log_level,
        log_path=args.runtime_log_path,
        max_log_mb=args.runtime_log_max_mb,
    )
    print(f"runtime_safe_mode={runtime_status['safe_mode']}", flush=True)
    print(f"runtime_disable_wsl_fallback={runtime_status['disable_wsl_fallback']}", flush=True)
    print(f"runtime_sandbox_failure_seen={runtime_status['sandbox_failure_seen']}", flush=True)
    print(f"runtime_log_level={runtime_status['log_level']}", flush=True)
    print(f"runtime_disk_log_enabled={runtime_status['disk_log_enabled']}", flush=True)
    print(f"runtime_log_path={runtime_status['log_path']}", flush=True)
    print(f"runtime_log_rotation_max_mb={runtime_status['log_rotation_max_mb']}", flush=True)
    if args.train_predictor and args.future_source != "predicted":
        raise ValueError("--train-predictor requires --future-source predicted; no silent mode switch is allowed.")
    if args.train_predictor and args.predictor_checkpoint is None and not args.allow_random_predictor:
        raise ValueError(
            "Stage B training needs a reproducible predictor source. Pass --predictor-checkpoint "
            "for V-JEPA2-AC partial pretrained init, or --allow-random-predictor for explicit debug random init."
        )
    if (not args.train_predictor) and args.future_source == "predicted" and args.predictor_checkpoint is None:
        raise ValueError(
            "Predicted/eval mode requires --predictor-checkpoint for V-JEPA2-AC partial init. Random predictor eval is forbidden."
        )
    if args.steps <= 0:
        raise ValueError(f"`--steps` must be positive, got {args.steps}.")

    device = resolve_device(args.device)
    seed_everything(args.seed, device)
    dtype = resolve_dtype(args.dtype, device)
    checkpoint_path = resolve_path(args.action_checkpoint)
    predictor_checkpoint = resolve_path(args.predictor_checkpoint)
    vjepa_repo = resolve_path(args.vjepa_repo)
    vjepa_checkpoint = resolve_path(args.vjepa_checkpoint)
    if checkpoint_path is None:
        raise ValueError("`--action-checkpoint` is required.")
    if vjepa_repo is None or vjepa_checkpoint is None:
        raise ValueError("`--vjepa-repo` and `--vjepa-checkpoint` are required.")

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

    model = build_model(
        action_expert=action_expert,
        args=args,
        cfg=cfg,
        device=device,
        dtype=dtype,
    )
    predictor_weight_source, predictor_payload = load_predictor_checkpoint(
        model=model,
        checkpoint_path=predictor_checkpoint,
        future_source=args.future_source,
        allow_random_predictor=args.allow_random_predictor,
        device=device,
    )
    predictor_available = args.future_source == "predicted"

    trainable_params = configure_trainability(model, train_predictor=args.train_predictor)
    optimizer = None
    start_step = 0
    if args.train_predictor:
        if not trainable_params:
            raise RuntimeError("--train-predictor was set but no predictor parameters are trainable.")
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        if predictor_payload is not None:
            opt_state = predictor_payload.get("optimizer")
            if isinstance(opt_state, dict):
                optimizer.load_state_dict(opt_state)
            if isinstance(predictor_payload.get("step"), int):
                start_step = int(predictor_payload["step"])

    block_calls = {"count": 0}
    vjepa_forward_outputs: list[tuple[tuple[int, ...], bool]] = []

    def count_block_call(module, inputs, output) -> None:
        del module, inputs, output
        block_calls["count"] += 1

    def record_vjepa_output(module, inputs, output) -> None:
        del module, inputs
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"V-JEPA encoder output must be a tensor, got {type(output)}.")
        vjepa_forward_outputs.append((tuple(output.shape), bool(output.requires_grad)))

    handles = [block.register_forward_hook(count_block_call) for block in action_expert.blocks]
    handles.append(model.vjepa_encoder.register_forward_hook(record_vjepa_output))
    try:
        lambda_action, lambda_future = effective_lambdas(args)
        if args.train_predictor:
            stage = "B_train_predictor_only"
        elif args.future_source == "oracle":
            stage = "C_frozen_oracle_gt_future"
        elif args.future_source == "predicted":
            stage = "eval_loaded_predictor"
        else:
            stage = "no_future_sanity"
        print(f"stage={stage}")
        print(f"checkpoint_path={checkpoint_path}")
        print(f"vjepa_repo={vjepa_repo}")
        print(f"vjepa_checkpoint={vjepa_checkpoint}")
        print(f"predictor_checkpoint={predictor_checkpoint}")
        print(f"predictor_weight_source={predictor_weight_source}")
        print(f"freeze_vjepa={args.freeze_vjepa}")
        print(f"future_source={args.future_source}")
        print(f"lambda_action={lambda_action}")
        print(f"lambda_future={lambda_future}")
        print(f"start_step={start_step}")
        print(f"trainable_predictor_params={count_trainable_params(model.future_predictor)}")
        print(f"trainable_action_dit_params={count_trainable_params(model.action_expert)}")
        print(f"action_dit_hidden_dim={int(action_expert.hidden_dim)}")
        print(f"action_dit_text_dim={int(action_expert.text_dim)}")
        print(f"action_dit_action_dim={int(action_expert.action_dim)}")
        print(f"action_dit_num_blocks={int(len(action_expert.blocks))}")

        last_shapes = None
        last_vjepa_outputs = None
        last_loss_dict: dict[str, float] | None = None
        for local_step in range(1, args.steps + 1):
            step = start_step + local_step
            seed_everything(args.seed + step, device)
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

            vjepa_forward_outputs.clear()
            block_before = block_calls["count"]
            if optimizer is None:
                with torch.no_grad():
                    loss_total, loss_dict = model.training_loss(sample)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss_total, loss_dict = model.training_loss(sample)
                loss_total.backward()
                optimizer.step()

            if not torch.isfinite(loss_total):
                raise RuntimeError(f"loss_total is not finite at step {step}.")
            main_block_calls = block_calls["count"] - block_before
            expected_block_calls = int(len(action_expert.blocks))
            if main_block_calls != expected_block_calls:
                raise RuntimeError(
                    "ActionDiT.blocks call count mismatch in main loss: "
                    f"expected {expected_block_calls}, got {main_block_calls}."
                )
            last_shapes = dict(model.last_forward_shapes)
            last_vjepa_outputs = list(vjepa_forward_outputs)
            last_loss_dict = dict(loss_dict)
            validate_main_forward(shapes=last_shapes, vjepa_outputs=last_vjepa_outputs, args=args)

            gap = evaluate_oracle_predicted_gap(
                model=model,
                sample=sample,
                predictor_available=predictor_available,
                seed=args.seed + 100_000 + step,
                device=device,
            )
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"loss_total={float(loss_total.detach().item()):.6f}",
                        f"loss_action={loss_dict['loss_action']:.6f}",
                        f"loss_future_jepa={loss_dict['loss_future_jepa']:.6f}",
                        f"loss_predictor={loss_dict['loss_predictor']:.6f}",
                        f"oracle_vs_predicted_gap={gap['oracle_vs_predicted_gap']:.6f}",
                        f"oracle_loss_action={gap['oracle_loss_action']:.6f}",
                        f"predicted_loss_action={gap['predicted_loss_action']:.6f}",
                        f"action_loss_invariance_delta={gap['action_loss_invariance_delta']:.6f}",
                        f"predictor_used={last_shapes['predictor_used']}",
                        f"action_dit_block_calls={main_block_calls}",
                    ]
                ),
                flush=True,
            )

            if args.train_predictor and args.save_every > 0 and local_step % args.save_every == 0:
                saved = save_predictor_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    step=step,
                    loss_dict=last_loss_dict,
                )
                print(f"saved_predictor_checkpoint={saved}", flush=True)

        if args.train_predictor and last_loss_dict is not None:
            saved = save_predictor_checkpoint(
                model=model,
                optimizer=optimizer,
                args=args,
                step=start_step + args.steps,
                loss_dict=last_loss_dict,
            )
            print(f"saved_predictor_checkpoint={saved}", flush=True)

        if last_shapes is not None and last_vjepa_outputs is not None:
            print(f"current_jepa_tokens shape={last_shapes['current_jepa_tokens']}")
            print(f"target_future_jepa_tokens shape={last_shapes['target_future_jepa_tokens']}")
            print(f"pred_future_jepa_tokens shape={last_shapes['pred_future_jepa_tokens']}")
            print(f"action_context shape={last_shapes['action_context']}")
            print(f"pred_action shape={last_shapes['pred_action']}")
            print(f"predictor_used={last_shapes['predictor_used']}")
            print(f"vjepa_forward_outputs={last_vjepa_outputs}")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
