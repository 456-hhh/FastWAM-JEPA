from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam_jepa_idm import FastWAMJEPAIDM


def build_action_dit(*, action_dim: int, hidden_dim: int, text_dim: int) -> ActionDiT:
    return ActionDiT(
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        ffn_dim=hidden_dim * 4,
        text_dim=text_dim,
        freq_dim=32,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=16,
        num_layers=2,
        use_gradient_checkpointing=False,
    )


def build_sample() -> dict[str, torch.Tensor]:
    batch_size = 2
    video_frames = 4
    action_horizon = 4
    action_dim = 7
    image_size = 16
    text_len = 5
    text_dim = 32
    proprio_dim = 9

    return {
        "video": torch.randn(batch_size, 3, video_frames, image_size, image_size),
        "action": torch.randn(batch_size, action_horizon, action_dim),
        "context": torch.randn(batch_size, text_len, text_dim),
        "context_mask": torch.ones(batch_size, text_len, dtype=torch.bool),
        "proprio": torch.randn(batch_size, video_frames, proprio_dim),
        "action_is_pad": torch.zeros(batch_size, action_horizon, dtype=torch.bool),
    }


def run_one(future_source: str) -> None:
    torch.manual_seed(7)

    action_dim = 7
    hidden_dim = 64
    text_dim = 32
    vjepa_dim = 48
    num_future_tokens = 8
    proprio_dim = 9

    action_expert = build_action_dit(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        text_dim=text_dim,
    )
    block_calls = {"count": 0}

    def count_block_call(module, inputs, output) -> None:
        del module, inputs, output
        block_calls["count"] += 1

    handles = [block.register_forward_hook(count_block_call) for block in action_expert.blocks]
    try:
        model = FastWAMJEPAIDM(
            action_expert=action_expert,
            vjepa_encoder=VJepaEncoderWrapper(
                dummy=True,
                num_tokens=num_future_tokens,
                vjepa_dim=vjepa_dim,
                freeze=True,
            ),
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            vjepa_dim=vjepa_dim,
            num_future_tokens=num_future_tokens,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            lambda_future=0.1,
            current_frame_count=2,
            future_frame_count=2,
            adapter_current_tokens=4,
            adapter_future_tokens=4,
            future_predictor_layers=2,
            future_predictor_heads=4,
            future_source=future_source,
        )
        model.train()
        sample = build_sample()

        loss_total, loss_dict = model.training_loss(sample)
        if not torch.isfinite(loss_total):
            raise RuntimeError(f"loss_total is not finite for future_source={future_source}.")
        if block_calls["count"] <= 0:
            raise RuntimeError("ActionDiT.blocks were not called.")

        shapes = model.last_forward_shapes
        print(f"future_source={future_source}")
        print(f"  loss_total={float(loss_total.detach().item()):.6f}")
        print(f"  loss_action={loss_dict['loss_action']:.6f}")
        print(f"  loss_future_jepa={loss_dict['loss_future_jepa']:.6f}")
        print(f"  current_jepa_tokens shape={shapes['current_jepa_tokens']}")
        print(f"  target_future_jepa_tokens shape={shapes['target_future_jepa_tokens']}")
        print(f"  action_context shape={shapes['action_context']}")
        print(f"  pred_action shape={shapes['pred_action']}")
        print(f"  action_dit_block_calls={block_calls['count']}")
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    run_one("oracle")
    run_one("predicted")


if __name__ == "__main__":
    main()
