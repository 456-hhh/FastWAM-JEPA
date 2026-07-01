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
    LanguageProjector,
    TextToActionHead,
    contrastive_loss,
)


def _check_shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != tuple(expected):
        raise AssertionError(f"{name} shape mismatch: expected {expected}, got {tuple(value.shape)}")


def main() -> None:
    torch.manual_seed(7)
    batch_size = 2
    text = torch.randn(batch_size, 128, 4096)
    context_mask = torch.ones(batch_size, 128, dtype=torch.bool)
    context_mask[1, 100:] = False
    action = torch.randn(batch_size, 32, 7)

    language_projector = LanguageProjector()
    action_encoder = ActionEncoder()
    text_to_action_head = TextToActionHead()

    z_l = language_projector(text, text_mask=context_mask)
    z_a = action_encoder(action)
    q_a_text = text_to_action_head(z_l)
    _check_shape("text", text, (batch_size, 128, 4096))
    _check_shape("context_mask", context_mask, (batch_size, 128))
    _check_shape("action", action, (batch_size, 32, 7))
    _check_shape("z_l", z_l, (batch_size, 4, 1024))
    _check_shape("z_a", z_a, (batch_size, 4, 1024))
    _check_shape("q_a_text", q_a_text, (batch_size, 4, 1024))

    loss, retrieval_acc = contrastive_loss(q_a_text, z_a)
    if not torch.isfinite(loss):
        raise AssertionError("contrastive loss is not finite")
    if not torch.isfinite(retrieval_acc):
        raise AssertionError("retrieval accuracy is not finite")
    loss.backward()
    for name, module in (
        ("LanguageProjector", language_projector),
        ("ActionEncoder", action_encoder),
        ("TextToActionHead", text_to_action_head),
    ):
        has_grad = any(param.requires_grad and param.grad is not None for param in module.parameters())
        if not has_grad:
            raise AssertionError(f"{name} has no parameter gradients")
    print("PASS")


if __name__ == "__main__":
    main()
