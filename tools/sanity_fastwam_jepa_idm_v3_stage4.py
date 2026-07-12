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

    torch.manual_seed(0)

    batch_size = 2
    world_tokens = torch.randn(batch_size, 512, 1408)
    text_tokens = torch.randn(batch_size, 128, 4096)
    text_mask = torch.ones(batch_size, 128, dtype=torch.bool)
    proprio = torch.randn(batch_size, 8)
    proprio_tokens = torch.randn(batch_size, 4, 1024)
    action_chunk = torch.randn(batch_size, 32, 7)

    model = PairwiseStage4Model()

    train_out = model.forward_train(
        world_tokens=world_tokens,
        text_tokens=text_tokens,
        action_chunk=action_chunk,
        text_mask=text_mask,
        proprio=proprio,
    )
    required_train_keys = (
        "z_v",
        "z_l",
        "z_p",
        "z_a",
        "q_a_vlp",
        "q_l_va",
        "z_task",
        "z_task_token",
        "loss_vlp_to_a",
        "loss_va_to_l",
        "retrieval_acc_vlp_to_a",
        "retrieval_acc_va_to_l",
    )
    for key in required_train_keys:
        assert key in train_out, f"Missing train key: {key}"
    assert tuple(train_out["z_task"].shape) == (batch_size, 1024)
    assert tuple(train_out["z_task_token"].shape) == (batch_size, 4, 1024)
    assert train_out["loss_vlp_to_a"].ndim == 0
    assert train_out["loss_va_to_l"].ndim == 0
    assert torch.isfinite(train_out["loss_vlp_to_a"]).item()
    assert torch.isfinite(train_out["loss_va_to_l"]).item()

    infer_out = model.forward_infer(
        world_tokens=world_tokens,
        text_tokens=text_tokens,
        text_mask=text_mask,
        proprio=proprio,
    )
    required_infer_keys = (
        "z_v",
        "z_l",
        "z_p",
        "q_a_vlp",
        "z_task",
        "z_task_token",
    )
    for key in required_infer_keys:
        assert key in infer_out, f"Missing infer key: {key}"
    assert tuple(infer_out["z_task"].shape) == (batch_size, 1024)
    assert tuple(infer_out["z_task_token"].shape) == (batch_size, 4, 1024)

    train_out_with_tokens = model.forward_train(
        world_tokens=world_tokens,
        text_tokens=text_tokens,
        action_chunk=action_chunk,
        text_mask=text_mask,
        proprio_tokens=proprio_tokens,
    )
    assert tuple(train_out_with_tokens["z_p"].shape) == (batch_size, 4, 1024)

    total_loss = train_out["loss_vlp_to_a"] + train_out["loss_va_to_l"]
    total_loss.backward()
    grad_count = sum(
        1 for param in model.parameters() if param.requires_grad and param.grad is not None
    )
    assert grad_count > 0

    print(
        "sanity_fastwam_jepa_idm_v3_stage4 passed "
        f"z_v={tuple(train_out['z_v'].shape)} "
        f"z_l={tuple(train_out['z_l'].shape)} "
        f"z_p={tuple(train_out['z_p'].shape)} "
        f"z_a={tuple(train_out['z_a'].shape)} "
        f"z_task={tuple(train_out['z_task'].shape)} "
        f"z_task_token={tuple(train_out['z_task_token'].shape)}"
    )


if __name__ == "__main__":
    main()
