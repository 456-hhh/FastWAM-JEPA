from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.models.wan22.pairwise_conditional_latent_v3 import (
    ActionEncoder,
    FusionVLP,
    LanguageProjector,
    ProprioProjector,
    VisionProjector,
    contrastive_loss,
)


def _check_shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != tuple(expected):
        raise AssertionError(f"{name} shape mismatch: expected {expected}, got {tuple(value.shape)}")


def _check_has_grad(name: str, module: torch.nn.Module) -> None:
    has_grad = any(param.requires_grad and param.grad is not None for param in module.parameters())
    if not has_grad:
        raise AssertionError(f"{name} has no parameter gradients")


def main() -> None:
    torch.manual_seed(7)
    batch_size = 2
    current_jepa_tokens = torch.randn(batch_size, 512, 1408)
    context = torch.randn(batch_size, 128, 4096)
    context_mask = torch.ones(batch_size, 128, dtype=torch.bool)
    context_mask[1, 100:] = False
    proprio = torch.randn(batch_size, 8)
    action = torch.randn(batch_size, 32, 7)

    language_projector = LanguageProjector()
    action_encoder = ActionEncoder()
    vision_projector = VisionProjector(input_dim=1408)
    proprio_projector = ProprioProjector()
    fusion_vlp = FusionVLP()

    z_v = vision_projector(current_jepa_tokens)
    z_l = language_projector(context, text_mask=context_mask)
    z_p = proprio_projector(proprio)
    z_task = fusion_vlp(z_v, z_l, z_p)
    z_a = action_encoder(action)

    _check_shape("z_v", z_v, (batch_size, 4, 1024))
    _check_shape("z_l", z_l, (batch_size, 4, 1024))
    _check_shape("z_p", z_p, (batch_size, 1, 1024))
    _check_shape("z_task", z_task, (batch_size, 4, 1024))
    _check_shape("z_a", z_a, (batch_size, 4, 1024))

    loss, retrieval_acc = contrastive_loss(z_task, z_a)
    if not torch.isfinite(loss):
        raise AssertionError("contrastive loss is not finite")
    if not torch.isfinite(retrieval_acc):
        raise AssertionError("retrieval accuracy is not finite")
    loss.backward()

    _check_has_grad("VisionProjector", vision_projector)
    _check_has_grad("LanguageProjector", language_projector)
    _check_has_grad("ProprioProjector", proprio_projector)
    _check_has_grad("FusionVLP", fusion_vlp)
    _check_has_grad("ActionEncoder", action_encoder)
    print("PASS")


if __name__ == "__main__":
    main()