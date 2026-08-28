from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.wan22.jepa_visual_dit_v5 import (
    JEPAVisualDiTV5,
    build_v5_joint_attention_mask,
    build_v5_visual_temporal_mask,
)
from fastwam.models.wan22 import jepa_visual_dit_v5 as jepa_visual_dit_v5_module
from fastwam.models.wan22.wan_video_dit import rope_apply
from fastwam.models.wan22.v5_contract import (
    ACTION_HORIZON,
    TOKENS_PER_TEMPORAL_GROUP,
    VJEPA_DIM,
    build_vjepa_clips,
    pool_dual_camera_vjepa_tokens,
    split_dual_camera_video,
)
from fastwam_jepa_v5_data import resume_training_state, save_compact_checkpoint
from sanity_fastwam_jepa_idm_v5 import build_tiny_model, grad_norm, random_latents


def _conditioning(batch_size: int = 1):
    model = build_tiny_model()
    context = torch.randn(batch_size, 8, 4096)
    mask = torch.ones(batch_size, 8, dtype=torch.bool)
    proprio = torch.randn(batch_size, 8)
    base_context, base_mask = model.build_base_context(context, mask, proprio)
    return model, context, mask, proprio, base_context, base_mask


def test_v5_temporal_contract():
    video = torch.randn(1, 3, 5, 224, 448)
    clips = build_vjepa_clips(video)
    assert torch.equal(clips["agentview_current"][:, :, 0], clips["agentview_current"][:, :, 1])
    assert tuple(clips["agentview_future"].shape) == (1, 3, 4, 224, 224)


def test_v5_camera_split():
    agent, wrist = split_dual_camera_video(torch.randn(2, 3, 5, 224, 448))
    assert tuple(agent.shape) == tuple(wrist.shape) == (2, 3, 5, 224, 224)
    with pytest.raises(ValueError):
        split_dual_camera_video(torch.randn(2, 3, 5, 224, 447))


def test_v5_visual_mask():
    mask = build_v5_visual_temporal_mask()
    assert tuple(mask.shape) == (216, 216)
    assert bool(mask[:72, :72].all())
    assert not bool(mask[:72, 72:].any())
    joint = build_v5_joint_attention_mask()
    assert bool(joint[216:, :].all())
    assert not bool(joint[:216, 216:].any())


def test_v5_pooling_shapes():
    current = pool_dual_camera_vjepa_tokens(
        torch.randn(2, 256, VJEPA_DIM), torch.randn(2, 256, VJEPA_DIM), temporal_groups=1
    )
    future = pool_dual_camera_vjepa_tokens(
        torch.randn(2, 512, VJEPA_DIM), torch.randn(2, 512, VJEPA_DIM), temporal_groups=2
    )
    assert tuple(current.shape) == (2, 1, 72, VJEPA_DIM)
    assert tuple(future.shape) == (2, 2, 72, VJEPA_DIM)


@pytest.mark.parametrize("variant", ("tiny", "production_head_dim"))
def test_v5_visual_rope_stays_complex_after_real_dtype_conversion(variant):
    if variant == "tiny":
        visual_dit = build_tiny_model().visual_dit
    else:
        visual_dit = JEPAVisualDiTV5(
            num_layers=1,
            hidden_dim=32,
            ffn_dim=64,
            num_heads=1,
            attn_head_dim=128,
            freq_dim=32,
            use_gradient_checkpointing=False,
        )

    assert visual_dit.freqs.is_complex()
    original_freqs = visual_dit.freqs.clone()
    assert float(original_freqs.imag.abs().max()) > 0

    visual_dit.to(dtype=torch.bfloat16)
    assert visual_dit.freqs.is_complex()
    assert float(visual_dit.freqs.imag.abs().max()) > 0
    torch.testing.assert_close(visual_dit.freqs, original_freqs, rtol=0.0, atol=0.0)

    visual_tokens = torch.randn(1, 216, VJEPA_DIM, dtype=torch.bfloat16)
    context = torch.randn(1, 8, 4096, dtype=torch.bfloat16)
    context_mask = torch.ones(1, 8, dtype=torch.bool)
    pre_state = visual_dit.pre_dit(
        visual_tokens,
        torch.zeros(1, dtype=torch.bfloat16),
        context,
        context_mask,
    )
    assert pre_state["freqs"].is_complex()
    assert float(pre_state["freqs"].imag.abs().max()) > 0

    rope_input = torch.randn(
        1,
        216,
        visual_dit.num_heads * visual_dit.attn_head_dim,
        dtype=torch.bfloat16,
    )
    rope_output = rope_apply(rope_input, pre_state["freqs"], visual_dit.num_heads)
    assert tuple(rope_output.shape) == tuple(rope_input.shape)
    assert bool(torch.isfinite(rope_output).all())


def test_v5_training_checkpoint_loads_numpy_rng_state(tmp_path):
    source = torch.nn.Linear(2, 2)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    source_scheduler = torch.optim.lr_scheduler.LambdaLR(
        source_optimizer, lr_lambda=lambda _: 1.0
    )
    np.random.seed(42)
    np.random.normal()
    checkpoint_path = save_compact_checkpoint(
        output_dir=tmp_path,
        step=7,
        epoch=2,
        batches_in_epoch=3,
        weights={"probe": source.state_dict()},
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        metadata={"version": "v5", "stage": "stage1"},
        rank=0,
        world_size=1,
    )
    assert checkpoint_path is not None

    destination = torch.nn.Linear(2, 2)
    destination_optimizer = torch.optim.AdamW(destination.parameters(), lr=1e-3)
    destination_scheduler = torch.optim.lr_scheduler.LambdaLR(
        destination_optimizer, lr_lambda=lambda _: 1.0
    )
    step, epoch, batches_in_epoch, metadata = resume_training_state(
        checkpoint_path=checkpoint_path,
        expected_stage="stage1",
        modules={"probe": destination},
        optimizer=destination_optimizer,
        scheduler=destination_scheduler,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
    )
    assert (step, epoch, batches_in_epoch) == (7, 2, 3)
    assert isinstance(metadata["_resume_rng_state"]["numpy"], tuple)
    for source_parameter, destination_parameter in zip(
        source.parameters(), destination.parameters()
    ):
        assert torch.equal(source_parameter, destination_parameter)


def test_v5_visual_dit_scalar_timestep_shape():
    model, _, _, _, context, mask = _conditioning()
    z0, z1, z2 = random_latents(1)
    visual_tokens = torch.cat((z0, z1, z2), dim=1)
    timestep = torch.zeros(1)
    pre_state = model.visual_dit.pre_dit(visual_tokens, timestep, context, mask)
    assert tuple(pre_state["t_mod"].shape) == (1, 6, model.visual_dit.hidden_dim)
    output = model.visual_dit(visual_tokens, timestep, context, mask)
    assert tuple(output.shape) == (1, 144, VJEPA_DIM)


def test_v5_visual_dit_tokenwise_timestep_shape():
    model, _, _, _, context, mask = _conditioning()
    z0, z1, z2 = random_latents(1)
    visual_tokens = torch.cat((z0, z1, z2), dim=1)
    timestep = torch.zeros(1, 216)
    pre_state = model.visual_dit.pre_dit(visual_tokens, timestep, context, mask)
    assert tuple(pre_state["t_mod"].shape) == (1, 216, 6, model.visual_dit.hidden_dim)
    output = model.visual_dit(visual_tokens, timestep, context, mask)
    assert tuple(output.shape) == (1, 144, VJEPA_DIM)


def test_v5_stage1_mixed_timestep_forward():
    model, _, _, _, context, mask = _conditioning()
    z0, z1, z2 = random_latents(1)
    timestep = torch.cat(
        (
            torch.zeros(1, TOKENS_PER_TEMPORAL_GROUP),
            torch.full((1, 2 * TOKENS_PER_TEMPORAL_GROUP), 0.5),
        ),
        dim=1,
    )
    assert bool((timestep[:, :TOKENS_PER_TEMPORAL_GROUP] == 0).all())
    assert bool((timestep[:, TOKENS_PER_TEMPORAL_GROUP:] == 0.5).all())
    visual_tokens = torch.cat((z0, z1, z2), dim=1)
    pre_state = model.visual_dit.pre_dit(visual_tokens, timestep, context, mask)
    t_mod = pre_state["t_mod"]
    assert tuple(t_mod.shape) == (1, 216, 6, model.visual_dit.hidden_dim)
    actual = t_mod[:, TOKENS_PER_TEMPORAL_GROUP:]
    future_mod = model.visual_dit._timestep_modulation(torch.full((1,), 0.5), batch_size=1)
    expected_scalar = future_mod.unsqueeze(1).expand(
        -1, 2 * TOKENS_PER_TEMPORAL_GROUP, -1, -1
    )
    assert bool(torch.isfinite(actual).all())
    assert bool(torch.isfinite(expected_scalar).all())
    scalar_diff = (actual - expected_scalar).abs()
    max_abs_diff = float(scalar_diff.max())
    mean_abs_diff = float(scalar_diff.mean())
    reference = actual[:, :1]
    future_internal_diff = float((actual - reference).abs().max())
    print(
        "stage1_timestep_diagnostics "
        f"max_abs_diff={max_abs_diff:.12g} "
        f"mean_abs_diff={mean_abs_diff:.12g} "
        f"future_internal_diff={future_internal_diff:.12g}"
    )
    assert max_abs_diff < 1e-4, (
        "Scalar/token-wise timestep modulation differs beyond floating-point tolerance: "
        f"max_abs_diff={max_abs_diff:.12g}, mean_abs_diff={mean_abs_diff:.12g}."
    )
    assert future_internal_diff < 1e-6, (
        "Equal future timesteps produced inconsistent token modulation: "
        f"max_abs_diff={future_internal_diff:.12g}."
    )

    all_zero_mod = model.visual_dit._timestep_modulation(
        torch.zeros(1, 216), batch_size=1
    )
    all_future_mod = model.visual_dit._timestep_modulation(
        torch.full((1, 216), 0.5), batch_size=1
    )
    torch.testing.assert_close(
        t_mod[:, :TOKENS_PER_TEMPORAL_GROUP],
        all_zero_mod[:, :TOKENS_PER_TEMPORAL_GROUP],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        t_mod[:, TOKENS_PER_TEMPORAL_GROUP:],
        all_future_mod[:, TOKENS_PER_TEMPORAL_GROUP:],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        all_future_mod[:, 0],
        future_mod,
        rtol=1e-5,
        atol=1e-6,
    )
    output = model.visual_dit(visual_tokens, timestep, context, mask)
    assert tuple(output.shape) == (1, 144, VJEPA_DIM)


def test_v5_tokenwise_modulation_splits_component_axis(monkeypatch):
    model, _, _, _, _, _ = _conditioning()
    block = model.visual_dit.blocks[0]
    batch_size = 1
    sequence_length = 216
    hidden_dim = block.hidden_dim
    x = torch.zeros(batch_size, sequence_length, hidden_dim)
    t_mod = torch.stack(
        [
            torch.full((batch_size, sequence_length, hidden_dim), float(component))
            for component in range(6)
        ],
        dim=2,
    )
    with torch.no_grad():
        block.modulation.zero_()

    modulations = []
    gates = []

    def capture_modulate(value, shift, scale):
        modulations.append((shift.detach().clone(), scale.detach().clone()))
        return value

    class ZeroSelfAttention(torch.nn.Module):
        def forward(self, value, freqs, mask):
            return torch.zeros_like(value)

    class ZeroCrossAttention(torch.nn.Module):
        def forward(self, value, context, ctx_mask=None):
            return torch.zeros_like(value)

    class CaptureGate(torch.nn.Module):
        def forward(self, value, gate, residual):
            gates.append(gate.detach().clone())
            return value

    monkeypatch.setattr(jepa_visual_dit_v5_module, "modulate", capture_modulate)
    block.self_attn = ZeroSelfAttention()
    block.cross_attn = ZeroCrossAttention()
    block.ffn = torch.nn.Identity()
    block.gate = CaptureGate()

    output = block(x, torch.zeros(1, 1, 4096), t_mod, torch.empty(0))
    assert tuple(output.shape) == tuple(x.shape)
    assert len(modulations) == 2
    assert len(gates) == 2
    for tensor, component in (
        (modulations[0][0], 0),
        (modulations[0][1], 1),
        (gates[0], 2),
        (modulations[1][0], 3),
        (modulations[1][1], 4),
        (gates[1], 5),
    ):
        assert tuple(tensor.shape) == (batch_size, sequence_length, hidden_dim)
        assert torch.equal(tensor, torch.full_like(tensor, float(component)))


def test_v5_no_current_leakage():
    model, _, _, _, context, mask = _conditioning()
    model.eval()
    z0, z1, z2 = random_latents(1)
    timestep = torch.zeros(1, 216)
    first = model.visual_dit.forward_hidden(
        torch.cat((z0, z1, z2), dim=1), timestep, context, mask
    )[:, :72]
    second = model.visual_dit.forward_hidden(
        torch.cat((z0, torch.randn_like(z1), torch.randn_like(z2)), dim=1),
        timestep,
        context,
        mask,
    )[:, :72]
    assert torch.allclose(first, second, atol=1e-6, rtol=0.0)


def test_v5_stage1_current_fixed():
    model, _, _, _, context, mask = _conditioning()
    model.eval()
    z0, _, _ = random_latents(1)
    original = z0.clone()
    result = model.infer_future_jepa(z0, context, mask, 2, 42)
    assert torch.equal(z0, original)
    assert result["debug"]["z0_bitwise_fixed"]


def test_v5_stage2_gradient_routing():
    model, context, context_mask, proprio, _, _ = _conditioning()
    model.set_stage2_trainability()
    base_context, base_mask = model.build_base_context(context, context_mask, proprio)
    z0, z1, z2 = random_latents(1)
    loss, _ = model.action_training_loss_teacher_forcing(
        z0=z0,
        z1=z1,
        z2=z2,
        action=torch.randn(1, 16, 7),
        context=base_context,
        context_mask=base_mask,
    )
    loss.backward()
    assert grad_norm(model.visual_dit) > 0
    assert grad_norm(model.action_expert) == 0
    assert grad_norm(model.proprio_encoder) == 0


def test_v5_stage3_gradient_routing():
    model, context, context_mask, proprio, _, _ = _conditioning()
    model.set_stage3_trainability()
    base_context, base_mask = model.build_base_context(context, context_mask, proprio)
    z0, z1, z2 = random_latents(1)
    loss, _ = model.action_training_loss_teacher_forcing(
        z0=z0,
        z1=z1,
        z2=z2,
        action=torch.randn(1, 16, 7),
        context=base_context,
        context_mask=base_mask,
    )
    loss.backward()
    assert grad_norm(model.visual_dit) > 0
    assert grad_norm(model.action_expert) > 0
    assert grad_norm(model.proprio_encoder) > 0


def test_v5_strict_checkpoint_load():
    model = build_tiny_model()
    state = model.visual_dit.state_dict()
    model.visual_dit.load_state_dict(state, strict=True)
    incomplete = dict(state)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(RuntimeError):
        model.visual_dit.load_state_dict(incomplete, strict=True)


def test_v5_inference_no_gt_future():
    parameters = inspect.signature(build_tiny_model().infer_future_jepa).parameters
    assert not {"oracle", "future_source", "no_future", "gt_future"}.intersection(parameters)


def test_v5_action_horizon_16():
    model, _, _, _, context, mask = _conditioning()
    model.eval()
    visual = torch.randn(1, 3 * TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM)
    action = model.infer_action(visual, context, mask, 2, 43)
    assert tuple(action.shape) == (1, ACTION_HORIZON, 7)
