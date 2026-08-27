from __future__ import annotations

import argparse
import inspect
import io
import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.wan22.action_dit import ActionDiT  # noqa: E402
from fastwam.models.wan22.fastwam_jepa_idm_v5 import FastWAMJEPAIDMV5  # noqa: E402
from fastwam.models.wan22.jepa_visual_dit_v5 import (  # noqa: E402
    JEPAVisualDiTV5,
    build_v5_joint_attention_mask,
    build_v5_visual_temporal_mask,
)
from fastwam.models.wan22.v5_contract import (  # noqa: E402
    ACTION_HORIZON,
    DATASET_VIDEO_INDICES,
    RAW_OBSERVATION_OFFSETS,
    TOKENS_PER_TEMPORAL_GROUP,
    VISUAL_TOKEN_COUNT,
    VJEPA_DIM,
    build_vjepa_clips,
    pool_dual_camera_vjepa_tokens,
    split_dual_camera_video,
)


class DummyVJepa(nn.Module):
    dummy = True
    freeze = True

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or int(video.shape[2]) not in (2, 4):
            raise ValueError("Dummy V-JEPA requires a 2-frame or 4-frame clip.")
        tokens = 256 if int(video.shape[2]) == 2 else 512
        scalar = video.float().mean(dim=(1, 2, 3, 4), keepdim=False)
        return scalar[:, None, None].expand(-1, tokens, VJEPA_DIM).to(dtype=video.dtype)


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().item())
    return total**0.5


def assert_no_grad(module: nn.Module, name: str) -> None:
    if any(parameter.grad is not None and bool(parameter.grad.detach().abs().sum() > 0) for parameter in module.parameters()):
        raise AssertionError(f"{name} unexpectedly received gradients.")


def build_tiny_model() -> FastWAMJEPAIDMV5:
    visual = JEPAVisualDiTV5(
        num_layers=2,
        hidden_dim=128,
        ffn_dim=256,
        num_heads=4,
        attn_head_dim=16,
        freq_dim=64,
        use_gradient_checkpointing=False,
    )
    action = ActionDiT(
        hidden_dim=128,
        action_dim=7,
        ffn_dim=256,
        text_dim=4096,
        freq_dim=64,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=16,
        num_layers=2,
        use_gradient_checkpointing=False,
    )
    return FastWAMJEPAIDMV5(
        vjepa_encoder=DummyVJepa(),
        visual_dit=visual,
        action_expert=action,
        proprio_encoder=nn.Linear(8, 4096),
        allow_dummy_vjepa=True,
    )


def random_latents(batch_size: int = 2):
    return (
        torch.randn(batch_size, TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM),
        torch.randn(batch_size, TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM),
        torch.randn(batch_size, TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-IDM V5 contract sanity.")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    if not args.tiny:
        raise ValueError("Dummy sanity requires explicit --tiny; use sanity_fastwam_jepa_idm_v5_real.py for production.")
    torch.manual_seed(42)
    batch_size = 2

    if list(DATASET_VIDEO_INDICES) != [0, 1, 2, 3, 4]:
        raise AssertionError("Dataset video index contract failed.")
    if list(RAW_OBSERVATION_OFFSETS) != [0, 4, 8, 12, 16]:
        raise AssertionError("Raw observation offset contract failed.")
    video = torch.randn(batch_size, 3, 5, 224, 448)
    agent, wrist = split_dual_camera_video(video)
    assert tuple(agent.shape) == (batch_size, 3, 5, 224, 224)
    assert tuple(wrist.shape) == (batch_size, 3, 5, 224, 224)
    clips = build_vjepa_clips(video)
    assert tuple(clips["agentview_current"].shape) == (batch_size, 3, 2, 224, 224)
    assert torch.equal(clips["agentview_current"][:, :, 0], clips["agentview_current"][:, :, 1])
    assert tuple(clips["agentview_future"].shape) == (batch_size, 3, 4, 224, 224)
    assert torch.equal(clips["agentview_future"], agent[:, :, 1:5])

    current_tokens = torch.randn(batch_size, 256, VJEPA_DIM)
    future_tokens = torch.randn(batch_size, 512, VJEPA_DIM)
    current_pooled = pool_dual_camera_vjepa_tokens(
        current_tokens, current_tokens.clone(), temporal_groups=1
    )
    future_pooled = pool_dual_camera_vjepa_tokens(
        future_tokens, future_tokens.clone(), temporal_groups=2
    )
    assert tuple(current_pooled.shape) == (batch_size, 1, 72, VJEPA_DIM)
    assert tuple(future_pooled.shape) == (batch_size, 2, 72, VJEPA_DIM)

    visual_mask = build_v5_visual_temporal_mask()
    assert tuple(visual_mask.shape) == (216, 216)
    for query_group in range(3):
        for key_group in range(3):
            block = visual_mask[
                query_group * 72 : (query_group + 1) * 72,
                key_group * 72 : (key_group + 1) * 72,
            ]
            assert bool(block.all()) == (key_group <= query_group)
    joint_mask = build_v5_joint_attention_mask()
    assert tuple(joint_mask.shape) == (232, 232)
    assert bool(joint_mask[216:, :].all())
    assert not bool(joint_mask[:216, 216:].any())

    model = build_tiny_model()
    context = torch.randn(batch_size, 12, 4096)
    context_mask = torch.ones(batch_size, 12, dtype=torch.bool)
    context_mask[1, 9:] = False
    proprio = torch.randn(batch_size, 8)
    base_context, base_mask = model.build_base_context(context, context_mask, proprio)
    z0, z1, z2 = random_latents(batch_size)
    timestep = torch.zeros(batch_size, VISUAL_TOKEN_COUNT)
    hidden = model.visual_dit.forward_hidden(
        torch.cat((z0, z1, z2), dim=1), timestep, base_context, base_mask
    )
    assert tuple(hidden.shape) == (batch_size, VISUAL_TOKEN_COUNT, 128)
    future_output = model.visual_dit(
        torch.cat((z0, z1, z2), dim=1), timestep, base_context, base_mask
    )
    assert tuple(future_output.shape) == (batch_size, 144, VJEPA_DIM)
    qkv_shape = model.visual_dit.blocks[0].self_attn.q(hidden).shape
    assert tuple(qkv_shape) == (batch_size, VISUAL_TOKEN_COUNT, 64)

    model.visual_dit.eval()
    hidden_a = model.visual_dit.forward_hidden(
        torch.cat((z0, z1, z2), dim=1), timestep, base_context, base_mask
    )[:, :72]
    hidden_b = model.visual_dit.forward_hidden(
        torch.cat((z0, torch.randn_like(z1), torch.randn_like(z2)), dim=1),
        timestep,
        base_context,
        base_mask,
    )[:, :72]
    if not torch.allclose(hidden_a, hidden_b, atol=1e-6, rtol=0.0):
        raise AssertionError("Visual causal mask leaked future information into z0 hidden states.")

    model.visual_dit.train()
    loss_visual, _ = model.visual_training_loss(
        z0=z0, z1=z1, z2=z2, context=base_context, context_mask=base_mask
    )
    assert bool(torch.isfinite(loss_visual))
    model.eval()
    future_a = model.infer_future_jepa(z0, base_context, base_mask, 2, 42)
    future_b = model.infer_future_jepa(z0, base_context, base_mask, 2, 42)
    assert torch.equal(future_a["future_tokens"], future_b["future_tokens"])
    assert future_a["debug"]["z0_bitwise_fixed"]

    action = torch.randn(batch_size, ACTION_HORIZON, 7)
    model.train()
    model.set_stage2_trainability()
    model.zero_grad(set_to_none=True)
    stage2_context, stage2_mask = model.build_base_context(context, context_mask, proprio)
    loss_action, _ = model.action_training_loss_teacher_forcing(
        z0=z0,
        z1=z1,
        z2=z2,
        action=action,
        context=stage2_context,
        context_mask=stage2_mask,
    )
    loss_action.backward()
    if not grad_norm(model.visual_dit) > 0:
        raise AssertionError("Stage2 L_action did not reach Visual DiT.")
    assert_no_grad(model.action_expert, "Stage2 ActionDiT")
    assert_no_grad(model.proprio_encoder, "Stage2 proprio encoder")
    assert_no_grad(model.vjepa_encoder, "Stage2 V-JEPA")

    model.zero_grad(set_to_none=True)
    model.set_stage3_trainability()
    stage3_context, stage3_mask = model.build_base_context(context, context_mask, proprio)
    loss_action_stage3, _ = model.action_training_loss_teacher_forcing(
        z0=z0,
        z1=z1,
        z2=z2,
        action=action,
        context=stage3_context,
        context_mask=stage3_mask,
    )
    loss_action_stage3.backward()
    if min(
        grad_norm(model.visual_dit),
        grad_norm(model.action_expert),
        grad_norm(model.proprio_encoder),
    ) <= 0:
        raise AssertionError("Stage3 L_action gradient routing failed.")
    assert_no_grad(model.vjepa_encoder, "Stage3 V-JEPA")
    action_trainable = sum(p.numel() for p in model.action_expert.parameters() if p.requires_grad)
    action_total = sum(p.numel() for p in model.action_expert.parameters())
    assert action_trainable == action_total

    model.zero_grad(set_to_none=True)
    loss_visual_stage3, _ = model.visual_training_loss(
        z0=z0, z1=z1, z2=z2, context=stage3_context, context_mask=stage3_mask
    )
    loss_visual_stage3.backward()
    if grad_norm(model.visual_dit) <= 0:
        raise AssertionError("Stage3 L_visual did not reach Visual DiT.")
    assert_no_grad(model.vjepa_encoder, "Stage3 visual-loss V-JEPA")

    model.eval()
    visual_tokens = torch.cat((z0, future_a["z1"], future_a["z2"]), dim=1)
    action_a = model.infer_action(visual_tokens, base_context, base_mask, 2, 43)
    action_b = model.infer_action(visual_tokens, base_context, base_mask, 2, 43)
    assert tuple(action_a.shape) == (batch_size, ACTION_HORIZON, 7)
    assert torch.equal(action_a, action_b)

    signature = inspect.signature(model.infer_future_jepa)
    forbidden = {"oracle", "future_source", "no_future", "gt_future"}
    if forbidden.intersection(signature.parameters):
        raise AssertionError("V5 inference accepts a forbidden future input.")

    state_buffer = io.BytesIO()
    torch.save(model.visual_dit.state_dict(), state_buffer)
    state_buffer.seek(0)
    exact_state = torch.load(state_buffer, map_location="cpu")
    model.visual_dit.load_state_dict(exact_state, strict=True)
    incomplete = dict(exact_state)
    incomplete.pop(next(iter(incomplete)))
    try:
        model.visual_dit.load_state_dict(incomplete, strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Strict checkpoint loading accepted a missing key.")

    try:
        model.action_training_loss_teacher_forcing(
            z0=z0,
            z1=z1,
            z2=z2,
            action=torch.full_like(action, float("nan")),
            context=base_context,
            context_mask=base_mask,
        )
    except FloatingPointError:
        pass
    else:
        raise AssertionError("V5 did not fail on nonfinite action input.")

    print(
        f"visual_parameter_count={model.visual_dit.parameter_count} "
        f"action_parameter_count={action_total} PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
