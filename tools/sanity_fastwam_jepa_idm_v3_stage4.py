from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.models.wan22.pairwise_stage4 import Stage4VLPVAActionModel


def _assert_module_has_grad(module: torch.nn.Module, *, name: str) -> None:
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable:
        raise AssertionError(f"{name} has no trainable parameters.")
    if not any(parameter.grad is not None for parameter in trainable):
        raise AssertionError(f"{name} received no gradients.")


def main() -> None:
    torch.manual_seed(7)
    batch_size = 2
    raw_jepa_tokens = torch.randn(batch_size, 512, 1408)
    context = torch.randn(batch_size, 128, 4096)
    context_mask = torch.ones(batch_size, 128, dtype=torch.bool)
    context_mask[1, 100:] = False
    proprio = torch.randn(batch_size, 8)
    action = torch.randn(batch_size, 32, 7)

    model = Stage4VLPVAActionModel()
    out = model(
        current_jepa_tokens=raw_jepa_tokens,
        context=context,
        context_mask=context_mask,
        proprio=proprio,
        action=action,
        tau=0.07,
    )

    expected_shapes = {
        "z_v": (batch_size, 4, 1024),
        "z_l": (batch_size, 4, 1024),
        "z_p": (batch_size, 1, 1024),
        "z_task": (batch_size, 4, 1024),
        "z_a": (batch_size, 4, 1024),
        "q_l": (batch_size, 4, 1024),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(out[name].shape)
        if actual != expected:
            raise AssertionError(f"{name} shape must be {expected}, got {actual}.")

    loss_vlp_a = out["loss_vlp_a"]
    loss_va_l = out["loss_va_l"]
    loss_total = loss_vlp_a + 0.1 * loss_va_l
    for name, value in (
        ("loss_vlp_a", loss_vlp_a),
        ("loss_va_l", loss_va_l),
        ("loss_total", loss_total),
    ):
        if value.ndim != 0 or not torch.isfinite(value).item():
            raise AssertionError(f"{name} must be a finite scalar.")

    loss_total.backward()
    for name in (
        "language_projector",
        "vision_projector",
        "action_encoder",
        "proprio_projector",
        "fusion_vlp",
        "fusion_va",
    ):
        _assert_module_has_grad(getattr(model, name), name=name)

    print("PASS")


if __name__ == "__main__":
    main()
