from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    from fastwam.models.wan22.pairwise_stage1_compat import PairwiseStage1TextActionCompatWrapper
    from fastwam.models.wan22.z_task_adapter import ZTaskContextAdapter, append_z_task_to_context

    torch.manual_seed(0)

    batch_size = 2
    text_tokens = torch.randn(batch_size, 128, 4096)
    text_mask = torch.ones(batch_size, 128, dtype=torch.bool)
    text_mask[1, 100:] = False
    action_chunk = torch.randn(batch_size, 32, 7)
    base_context = torch.randn(batch_size, 10, 4096)
    base_mask = torch.ones(batch_size, 10, dtype=torch.bool)

    wrapper = PairwiseStage1TextActionCompatWrapper()
    out = wrapper.forward_train(
        text_tokens=text_tokens,
        text_mask=text_mask,
        action_chunk=action_chunk,
    )

    assert tuple(out["z_task"].shape) == (batch_size, 1024)
    assert tuple(out["z_task_token"].shape) == (batch_size, 4, 1024)

    adapter = ZTaskContextAdapter(z_task_dim=1024, context_dim=4096)
    z_task_context_token = adapter(out["z_task_token"])
    assert tuple(z_task_context_token.shape) == (batch_size, 4, 4096)

    pooled_z_task_context_token = adapter(out["z_task"])
    assert tuple(pooled_z_task_context_token.shape) == (batch_size, 1, 4096)

    new_context, new_mask = append_z_task_to_context(
        context=base_context,
        context_mask=base_mask,
        z_task_context_token=z_task_context_token,
    )
    assert tuple(new_context.shape) == (batch_size, 14, 4096)
    assert new_mask is not None
    assert tuple(new_mask.shape) == (batch_size, 14)
    assert bool(new_mask[:, -4:].all().item())

    total_loss = out["loss_vlp_to_a"] + z_task_context_token.float().pow(2).mean()
    total_loss.backward()

    grad_count = sum(
        1
        for param in list(wrapper.parameters()) + list(adapter.parameters())
        if param.requires_grad and param.grad is not None
    )
    assert grad_count > 0

    print(
        "sanity_fastwam_jepa_idm_v3_stage5_ztask_context passed "
        f"z_task={tuple(out['z_task'].shape)} "
        f"z_task_token={tuple(out['z_task_token'].shape)} "
        f"z_task_context_token={tuple(z_task_context_token.shape)} "
        f"pooled_z_task_context_token={tuple(pooled_z_task_context_token.shape)} "
        f"new_context={tuple(new_context.shape)} "
        f"gate={float(adapter.gate().detach().item()):.6f}"
    )


if __name__ == "__main__":
    main()
