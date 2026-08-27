from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.wan22.v5_contract import canonicalize_v5_batch  # noqa: E402
from fastwam_jepa_v5_data import (  # noqa: E402
    autocast_context,
    build_v5_loader,
    compose_cfg,
    load_v5_model_checkpoint,
    precision_dtypes,
    provenance_paths,
    require_file,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-IDM V5 real-weight smoke test.")
    parser.add_argument("--checkpoint", required=True, help="A strict V5 Stage2 checkpoint.")
    parser.add_argument("--release-checkpoint", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--libero-data-root", required=True)
    parser.add_argument("--vjepa-repo", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint-key", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().item())
    return total**0.5


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V5 real smoke requires CUDA; dummy/CPU fallback is disabled.")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    dtype, autocast_dtype = precision_dtypes(args.precision)
    cfg = compose_cfg(args.config_name, args.task)
    loader, _ = build_v5_loader(
        cfg,
        libero_data_root=args.libero_data_root,
        dataset_stats_path=args.dataset_stats_path,
        batch_size=1,
        num_workers=0,
        seed=args.seed,
        ddp_enabled=False,
        world_size=1,
        rank=0,
    )
    provenance = provenance_paths(args, rank=0)
    checkpoint = require_file(args.checkpoint, name="--checkpoint")
    model, _, _ = load_v5_model_checkpoint(
        args,
        cfg=cfg,
        checkpoint_path=checkpoint,
        expected_stage="stage2",
        device=device,
        dtype=dtype,
        provenance=provenance,
    )
    model.set_stage2_trainability()
    model.train()
    batch = canonicalize_v5_batch(next(iter(loader)), device=device, dtype=dtype)
    z0 = model.encode_current(batch["video"])
    z1, z2 = model.encode_future_gt(batch["video"])
    context, context_mask = model.build_base_context(
        batch["context"], batch["context_mask"], batch["proprio"]
    )
    with autocast_context(autocast_dtype):
        loss_visual, _ = model.visual_training_loss(
            z0=z0, z1=z1, z2=z2, context=context, context_mask=context_mask
        )
        loss_action, _ = model.action_training_loss_teacher_forcing(
            z0=z0,
            z1=z1,
            z2=z2,
            action=batch["action"],
            context=context,
            context_mask=context_mask,
            action_is_pad=batch.get("action_is_pad"),
        )
    model.zero_grad(set_to_none=True)
    loss_action.backward()
    visual_grad = module_grad_norm(model.visual_dit)
    action_grad = module_grad_norm(model.action_expert)
    proprio_grad = module_grad_norm(model.proprio_encoder)
    if visual_grad <= 0 or action_grad != 0 or proprio_grad != 0:
        raise AssertionError(
            f"Stage2 gradient routing failed: visual={visual_grad}, action={action_grad}, proprio={proprio_grad}."
        )
    if any(parameter.grad is not None for parameter in model.vjepa_encoder.parameters()):
        raise AssertionError("Frozen V-JEPA received gradients.")

    model.eval()
    future = model.infer_future_jepa(z0, context, context_mask, 2, args.seed)
    visual_tokens = torch.cat((z0, future["z1"], future["z2"]), dim=1)
    pred_action = model.infer_action(visual_tokens, context, context_mask, 2, args.seed + 1)
    if tuple(pred_action.shape) != (1, 16, 7):
        raise AssertionError(f"Real V5 action shape failed: {tuple(pred_action.shape)}.")
    print(f"current_vjepa_shape={(1, 256, 1408)}", flush=True)
    print(f"future_vjepa_shape={(1, 512, 1408)}", flush=True)
    print(f"z0_shape={tuple(z0.shape)} z1_shape={tuple(z1.shape)} z2_shape={tuple(z2.shape)}", flush=True)
    print(f"visual_parameter_count={model.visual_dit.parameter_count}", flush=True)
    print(f"action_parameter_count={sum(p.numel() for p in model.action_expert.parameters())}", flush=True)
    print(f"visual_grad_norm={visual_grad:.6f}", flush=True)
    print(f"action_grad_status={action_grad:.6f} proprio_grad_status={proprio_grad:.6f}", flush=True)
    print(f"loss_visual={float(loss_visual):.6f} loss_action={float(loss_action):.6f}", flush=True)
    print(f"pred_action_shape={tuple(pred_action.shape)} PASS", flush=True)


if __name__ == "__main__":
    main()
