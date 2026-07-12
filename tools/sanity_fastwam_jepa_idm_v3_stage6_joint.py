from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    from fastwam.models.wan22.pairwise_stage4 import PairwiseStage4Model
    from fastwam.models.wan22.z_task_adapter import ZTaskContextAdapter, append_z_task_to_context
    from fastwam.training.pairwise_joint_loss import PairwiseJointLossWeights, combine_stage6_losses

    torch.manual_seed(0)

    batch_size = 2
    world_tokens = torch.randn(batch_size, 512, 1408)
    text_tokens = torch.randn(batch_size, 128, 4096)
    text_mask = torch.ones(batch_size, 128, dtype=torch.bool)
    proprio = torch.randn(batch_size, 8)
    action_chunk = torch.randn(batch_size, 32, 7)
    base_context = torch.randn(batch_size, 10, 4096)
    base_mask = torch.ones(batch_size, 10, dtype=torch.bool)

    pairwise = PairwiseStage4Model()
    adapter = ZTaskContextAdapter(z_task_dim=1024, context_dim=4096, pool_tokens=False)

    out = pairwise.forward_train(
        world_tokens=world_tokens,
        text_tokens=text_tokens,
        text_mask=text_mask,
        proprio=proprio,
        action_chunk=action_chunk,
    )
    assert tuple(out["z_task_token"].shape) == (batch_size, 4, 1024)
    z_task_context = adapter(out["z_task_token"])
    assert tuple(z_task_context.shape) == (batch_size, 4, 4096)

    new_context, new_mask = append_z_task_to_context(
        context=base_context,
        context_mask=base_mask,
        z_task_context_token=z_task_context,
    )
    assert tuple(new_context.shape) == (batch_size, 14, 4096)
    assert new_mask is not None
    assert tuple(new_mask.shape) == (batch_size, 14)
    assert bool(new_mask[:, -4:].all().item())

    dummy_loss_action = out["loss_vlp_to_a"] * 0.5 + out["loss_va_to_l"] * 0.25
    dummy_loss_future_jepa = out["loss_vlp_to_a"] * 0.125 + out["loss_va_to_l"] * 0.125
    loss_total, loss_items = combine_stage6_losses(
        loss_action=dummy_loss_action,
        loss_future_jepa=dummy_loss_future_jepa,
        loss_vlp_to_a=out["loss_vlp_to_a"],
        loss_va_to_l=out["loss_va_to_l"],
        weights=PairwiseJointLossWeights(
            lambda_future=0.1,
            lambda_vlp_to_a=0.05,
            lambda_va_to_l=0.05,
        ),
    )
    assert loss_total.ndim == 0
    assert loss_items["loss_total"].ndim == 0

    total_loss = loss_total + z_task_context.float().pow(2).mean()
    total_loss.backward()
    grad_count = sum(
        1
        for param in list(pairwise.parameters()) + list(adapter.parameters())
        if param.requires_grad and param.grad is not None
    )
    assert grad_count > 0

    print(
        "sanity_fastwam_jepa_idm_v3_stage6_joint passed "
        f"z_task_token={tuple(out['z_task_token'].shape)} "
        f"z_task_context={tuple(z_task_context.shape)} "
        f"new_context={tuple(new_context.shape)} "
        f"loss_total={float(loss_total.detach().item()):.6f}"
    )


if __name__ == "__main__":
    main()
