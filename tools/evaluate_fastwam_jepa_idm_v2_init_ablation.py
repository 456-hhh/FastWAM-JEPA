from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (SRC_ROOT, PROJECT_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam_jepa_runtime_guard import configure_runtime_stability
from evaluate_fastwam_jepa_idm_v2_predictor_value import (
    DEFAULT_ACTION_CHECKPOINT,
    DEFAULT_VJEPA_CHECKPOINT,
    DEFAULT_VJEPA_REPO,
    assert_batch_video,
    build_action_expert,
    build_loader,
    compose_cfg,
    resolve_device,
    resolve_dtype,
    resolve_path,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forward-only ablation for FastWAM-JEPA-IDM v2 predictor initialization. "
            "Compares random init vs V-JEPA2-AC pretrained init on the same batch."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--action-checkpoint", default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument(
        "--predictor-checkpoint",
        required=True,
        help="V-JEPA2-AC predictor checkpoint used for pretrained initialization.",
    )
    parser.add_argument("--allow-random-predictor", action="store_true", default=False)
    parser.add_argument("--vjepa-repo", default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=["-1_1", "0_1"])
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--dummy-vjepa", action="store_true", default=False)
    parser.add_argument("--dummy-batch", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--output-dir", default="evaluate_results/fastwam_jepa_idm_v2_init_ablation")
    parser.add_argument("--current-frame-count", type=int, default=2)
    parser.add_argument("--future-frame-count", type=int, default=2)
    parser.add_argument("--num-future-tokens", type=int, default=256)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--adapter-current-tokens", type=int, default=16)
    parser.add_argument("--adapter-future-tokens", type=int, default=16)
    parser.add_argument("--future-predictor-layers", type=int, default=2)
    parser.add_argument("--future-predictor-heads", type=int, default=8)
    parser.add_argument("--dummy-image-size", type=int, default=256)
    parser.add_argument("--dummy-action-horizon", type=int, default=4)
    parser.add_argument("--dummy-context-len", type=int, default=5)
    parser.add_argument("--disable-wsl-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--runtime-log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--runtime-log-path", default=None)
    parser.add_argument("--runtime-log-max-mb", type=int, default=100)
    return parser.parse_args()


def _require_path(path_value: str | None, name: str) -> Path:
    path = resolve_path(path_value)
    if path is None:
        raise ValueError(f"`{name}` is required.")
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def build_model(
    *,
    cfg: DictConfig,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper
    from fastwam.models.wan22.fastwam_jepa_idm import FastWAMJEPAIDM

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise ValueError(f"`cfg.model` must resolve to dict, got {type(model_cfg)}.")
    action_cfg = dict(model_cfg["action_dit_config"])
    action_scheduler_cfg = model_cfg.get("action_scheduler", {})
    proprio_dim = model_cfg.get("proprio_dim")

    action_expert = build_action_expert(
        action_cfg=action_cfg,
        checkpoint_path=_require_path(args.action_checkpoint, "--action-checkpoint"),
        device=device,
        dtype=dtype,
    )
    action_expert.eval()
    action_expert.requires_grad_(False)

    if args.dummy_vjepa:
        vjepa_encoder = VJepaEncoderWrapper(
            dummy=True,
            num_tokens=int(args.num_future_tokens),
            vjepa_dim=int(args.vjepa_dim),
            freeze=True,
            normalize_tokens=False,
        )
    else:
        vjepa_encoder = VJepaEncoderWrapper(
            dummy=False,
            model_name=str(args.vjepa_model_name),
            external_repo_path=str(_require_path(args.vjepa_repo, "--vjepa-repo")),
            checkpoint_path=str(_require_path(args.vjepa_checkpoint, "--vjepa-checkpoint")),
            pretrained=False,
            vjepa_dim=int(args.vjepa_dim),
            num_tokens=int(args.num_future_tokens),
            freeze=True,
            normalize_tokens=False,
            img_size=int(args.vjepa_img_size),
            input_range=str(args.vjepa_input_range),
            tubelet_size=int(args.vjepa_tubelet_size),
            frame_encoding_mode="clip_or_repeat",
        )

    model = FastWAMJEPAIDM(
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
        action_infer_shift=float(action_scheduler_cfg.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(action_scheduler_cfg.get("num_train_timesteps", 1000)),
        lambda_action=1.0,
        lambda_future=0.0,
        current_frame_count=int(args.current_frame_count),
        future_frame_count=int(args.future_frame_count),
        adapter_current_tokens=int(args.adapter_current_tokens),
        adapter_future_tokens=int(args.adapter_future_tokens),
        future_predictor_layers=int(args.future_predictor_layers),
        future_predictor_heads=int(args.future_predictor_heads),
        future_source="oracle",
    ).to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    model.future_predictor.eval()
    model.future_predictor.requires_grad_(False)
    return model


def build_dummy_sample(
    *,
    cfg: DictConfig,
    args: argparse.Namespace,
    action_dim: int,
    text_dim: int,
) -> dict[str, torch.Tensor]:
    proprio_dim = OmegaConf.select(cfg, "model.proprio_dim")
    total_frames = int(args.current_frame_count) + int(args.future_frame_count)
    sample: dict[str, torch.Tensor] = {
        "video": torch.randn(
            int(args.batch_size),
            3,
            total_frames,
            int(args.dummy_image_size),
            int(args.dummy_image_size),
            dtype=torch.float32,
        ),
        "action": torch.randn(int(args.batch_size), int(args.dummy_action_horizon), int(action_dim)),
        "context": torch.randn(int(args.batch_size), int(args.dummy_context_len), int(text_dim)),
        "context_mask": torch.ones(int(args.batch_size), int(args.dummy_context_len), dtype=torch.bool),
        "action_is_pad": torch.zeros(int(args.batch_size), int(args.dummy_action_horizon), dtype=torch.bool),
    }
    if proprio_dim is not None:
        sample["proprio"] = torch.randn(int(args.batch_size), total_frames, int(proprio_dim))
    return sample


def load_eval_batch(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.dummy_batch:
        sample = build_dummy_sample(
            cfg=cfg,
            args=args,
            action_dim=int(model.action_dim),
            text_dim=int(model.text_dim),
        )
    else:
        loader = build_loader(cfg, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
        sample = next(iter(loader))
        if not isinstance(sample, dict):
            raise ValueError(f"Expected dataloader batch dict, got {type(sample)}.")
    assert_batch_video(
        sample,
        current_frames=int(args.current_frame_count),
        future_frames=int(args.future_frame_count),
    )
    return sample


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_single_mode(
    *,
    model: torch.nn.Module,
    sample: dict[str, Any],
    mode: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.future_source = mode
    seed_everything(seed, device)
    _sync_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        loss_total, loss_dict = model.training_loss(sample)
    _sync_if_cuda(device)
    latency = time.perf_counter() - start
    if not torch.isfinite(loss_total):
        raise RuntimeError(f"loss_total is not finite for mode={mode}.")
    return {
        "mode": mode,
        "loss_total": float(loss_total.detach().item()),
        "loss_action": float(loss_dict["loss_action"]),
        "loss_future_jepa": float(loss_dict["loss_future_jepa"]),
        "loss_predictor": float(loss_dict.get("loss_predictor", loss_dict["loss_future_jepa"])),
        "oracle_vs_predicted_gap": float(loss_dict["oracle_vs_predicted_gap"]),
        "forward_latency_sec": float(latency),
        "predictor_used": model.last_forward_shapes.get("predictor_used"),
        "shapes": dict(model.last_forward_shapes),
    }


def run_init_arm(
    *,
    model: torch.nn.Module,
    sample: dict[str, Any],
    init_name: str,
    init_source: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    modes = ("oracle", "predicted", "no_future")
    results_by_mode = {
        mode: run_single_mode(
            model=model,
            sample=sample,
            mode=mode,
            seed=int(seed) + {"oracle": 11, "predicted": 22, "no_future": 33}[mode],
            device=device,
        )
        for mode in modes
    }
    predicted = results_by_mode["predicted"]
    summary = {
        "init_name": init_name,
        "init_source": init_source,
        "loss_predictor": float(predicted["loss_predictor"]),
        "loss_future_jepa": float(predicted["loss_future_jepa"]),
        "loss_action_oracle": float(results_by_mode["oracle"]["loss_action"]),
        "loss_action_predicted": float(predicted["loss_action"]),
        "loss_action_no_future": float(results_by_mode["no_future"]["loss_action"]),
        "oracle_vs_predicted_gap": float(predicted["oracle_vs_predicted_gap"]),
        "forward_latency_sec": float(sum(row["forward_latency_sec"] for row in results_by_mode.values())),
    }
    return {"summary": summary, "modes": results_by_mode}


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = (
        "init_name",
        "init_source",
        "loss_predictor",
        "loss_future_jepa",
        "loss_action_oracle",
        "loss_action_predicted",
        "loss_action_no_future",
        "oracle_vs_predicted_gap",
        "forward_latency_sec",
    )
    print("INIT_COMPARISON_TABLE", flush=True)
    print("\t".join(columns), flush=True)
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        print("\t".join(values), flush=True)


def conclusion_hint(init_effectiveness_score: float, policy_effectiveness_score: float) -> str:
    eps = 1.0e-8
    predictor_better = init_effectiveness_score > eps
    policy_better = policy_effectiveness_score > eps
    if predictor_better and policy_better:
        return "pretrained useful"
    if predictor_better or policy_better:
        return "partially useful"
    return "not useful"


def print_sample_shapes(sample: dict[str, Any]) -> None:
    print("sample_shapes:", flush=True)
    for key in ("video", "action", "context", "context_mask", "proprio", "action_is_pad"):
        value = sample.get(key)
        if torch.is_tensor(value):
            print(f"  {key}={tuple(value.shape)} dtype={value.dtype}", flush=True)
        else:
            print(f"  {key}=None", flush=True)


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
    print(f"runtime_log_level={runtime_status['log_level']}", flush=True)

    if not args.allow_random_predictor:
        raise ValueError(
            "This ablation intentionally evaluates a random predictor arm. "
            "Pass --allow-random-predictor to make that explicit."
        )
    predictor_checkpoint = _require_path(args.predictor_checkpoint, "--predictor-checkpoint")
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(int(args.seed), device)
    cfg = compose_cfg(str(args.config_name), str(args.task))

    model = build_model(cfg=cfg, args=args, device=device, dtype=dtype)
    sample = load_eval_batch(cfg=cfg, model=model, args=args)
    print_sample_shapes(sample)

    print("evaluation_protocol=forward_only_predictor_init_ablation", flush=True)
    print(f"seed={args.seed}", flush=True)
    print(f"action_checkpoint={resolve_path(args.action_checkpoint)}", flush=True)
    print(f"vjepa_checkpoint={resolve_path(args.vjepa_checkpoint)}", flush=True)
    print(f"vjepa2ac_predictor_checkpoint={predictor_checkpoint}", flush=True)
    print(f"dummy_batch={args.dummy_batch}", flush=True)
    print(f"dummy_vjepa={args.dummy_vjepa}", flush=True)

    random_result = run_init_arm(
        model=model,
        sample=sample,
        init_name="random_init",
        init_source=str(getattr(model.future_predictor, "init_source", "random_init")),
        seed=int(args.seed) + 1000,
        device=device,
    )

    load_stats = model.future_predictor.load_vjepa2ac_predictor_weights(predictor_checkpoint)
    if int(load_stats.get("loaded_keys_count", 0)) <= 0:
        raise RuntimeError(
            "V-JEPA2-AC checkpoint did not load any compatible predictor keys. "
            f"stats={load_stats}"
        )
    pretrained_result = run_init_arm(
        model=model,
        sample=sample,
        init_name="pretrained_init",
        init_source=str(load_stats.get("init_source", getattr(model.future_predictor, "init_source", "unknown"))),
        seed=int(args.seed) + 1000,
        device=device,
    )

    random_summary = random_result["summary"]
    pretrained_summary = pretrained_result["summary"]
    init_effectiveness_score = float(random_summary["loss_predictor"] - pretrained_summary["loss_predictor"])
    policy_effectiveness_score = float(
        random_summary["loss_action_predicted"] - pretrained_summary["loss_action_predicted"]
    )
    hint = conclusion_hint(init_effectiveness_score, policy_effectiveness_score)

    rows = [random_summary, pretrained_summary]
    print_table(rows)
    print("KEY_METRICS_DIFF", flush=True)
    print(f"init_effectiveness_score={init_effectiveness_score:.6f}", flush=True)
    print(f"policy_effectiveness_score={policy_effectiveness_score:.6f}", flush=True)
    print(
        "loss_action_oracle_delta="
        f"{float(random_summary['loss_action_oracle'] - pretrained_summary['loss_action_oracle']):.6f}",
        flush=True,
    )
    print(
        "loss_action_no_future_delta="
        f"{float(random_summary['loss_action_no_future'] - pretrained_summary['loss_action_no_future']):.6f}",
        flush=True,
    )
    print(f"conclusion_hint={hint}", flush=True)
    print(
        "vjepa2ac_load_stats="
        f"loaded_keys_count={load_stats.get('loaded_keys_count')} "
        f"skipped_keys_count={load_stats.get('skipped_keys_count')} "
        f"shape_mismatch_count={load_stats.get('shape_mismatch_count')} "
        f"init_source={load_stats.get('init_source')}",
        flush=True,
    )

    output_dir = resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "args": vars(args),
        "random_init": random_result,
        "pretrained_init": pretrained_result,
        "vjepa2ac_load_stats": load_stats,
        "init_effectiveness_score": init_effectiveness_score,
        "policy_effectiveness_score": policy_effectiveness_score,
        "conclusion_hint": hint,
    }
    output_path = output_dir / "predictor_init_ablation_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved_results={output_path}", flush=True)


if __name__ == "__main__":
    main()