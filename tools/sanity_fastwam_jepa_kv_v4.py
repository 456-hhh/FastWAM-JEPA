from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.vjepa.jepa_kv_cache_generator import JepaKVCacheGenerator  # noqa: E402
from fastwam.models.wan22.action_dit import ActionDiT  # noqa: E402
from fastwam.models.wan22.fastwam_jepa_kv_v4 import (  # noqa: E402
    FastWAMJEPAKVV4,
    build_duplicated_vjepa_clip,
    extract_causal_current_frame,
    flatten_horizontal_camera_grids,
    split_dual_camera_current_frame,
)
from fastwam.models.wan22.mot import MoT  # noqa: E402


class InputAwareDummyVJepa(nn.Module):
    def __init__(self, *, image_size: int = 32, vjepa_dim: int = 16) -> None:
        super().__init__()
        self.dummy = True
        self.freeze = True
        self.img_size = int(image_size)
        self.tubelet_size = 2
        self.num_tokens = 256
        self.vjepa_dim = int(vjepa_dim)
        self.projection = nn.Linear(3, self.vjepa_dim, bias=False)
        nn.init.normal_(self.projection.weight, std=0.2)
        self.forward_calls = 0
        self.requires_grad_(False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        if video.ndim != 5 or tuple(video.shape[1:3]) != (3, 2):
            raise ValueError(f"Dummy V-JEPA expects [B,3,2,H,W], got {tuple(video.shape)}.")
        if not torch.equal(video[:, :, 0], video[:, :, 1]):
            raise AssertionError("Dummy V-JEPA received non-identical temporal frames.")
        frame = video.mean(dim=2)
        patches = F.adaptive_avg_pool2d(frame, (16, 16))
        patches = patches.permute(0, 2, 3, 1).reshape(video.shape[0], 256, 3)
        return self.projection(patches)


def build_action_expert() -> ActionDiT:
    return ActionDiT(
        hidden_dim=32,
        action_dim=7,
        ffn_dim=64,
        text_dim=24,
        freq_dim=16,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=3,
        use_gradient_checkpointing=False,
    )


def assert_cache_close(
    left: list[dict[str, torch.Tensor]],
    right: list[dict[str, torch.Tensor]],
    *,
    atol: float = 1e-6,
) -> None:
    if len(left) != len(right):
        raise AssertionError(f"Cache lengths differ: {len(left)} vs {len(right)}.")
    for layer_idx, (lhs, rhs) in enumerate(zip(left, right)):
        for key in ("k", "v"):
            if not torch.allclose(lhs[key], rhs[key], atol=atol, rtol=0.0):
                raise AssertionError(f"Cache mismatch at layer={layer_idx} key={key}.")


def cache_max_difference(
    left: list[dict[str, torch.Tensor]],
    right: list[dict[str, torch.Tensor]],
) -> float:
    differences = [
        float((lhs[key] - rhs[key]).abs().max())
        for lhs, rhs in zip(left, right)
        for key in ("k", "v")
    ]
    return max(differences)


def run_dummy_sanity() -> None:
    torch.manual_seed(7)
    batch_size = 2
    camera_order = ("image", "wrist_image")

    primary_grid = torch.arange(49).reshape(1, 7, 7, 1)
    wrist_grid = (100 + torch.arange(49)).reshape(1, 7, 7, 1)
    flattened = flatten_horizontal_camera_grids([primary_grid, wrist_grid])
    expected = torch.cat([primary_grid, wrist_grid], dim=2).reshape(1, 98, 1)
    if not torch.equal(flattened, expected):
        raise AssertionError("Camera tokens are not horizontal 7x14 row-major.")
    flat_ids = flattened[0, :, 0]
    if not torch.equal(flat_ids[0:7], torch.arange(7)):
        raise AssertionError("Indices 0..6 must be primary row 0.")
    if not torch.equal(flat_ids[7:14], 100 + torch.arange(7)):
        raise AssertionError("Indices 7..13 must be wrist row 0.")
    if not torch.equal(flat_ids[14:21], torch.arange(7, 14)):
        raise AssertionError("Indices 14..20 must be primary row 1.")

    action_expert = build_action_expert()
    vjepa = InputAwareDummyVJepa()
    generator = JepaKVCacheGenerator(
        input_dim=16,
        context_dim=24,
        hidden_dim=32,
        num_layers=3,
        num_heads=2,
        attn_head_dim=8,
        video_seq_len=98,
        layer_rank=4,
        num_cameras=2,
    )
    model = FastWAMJEPAKVV4(
        action_expert=action_expert,
        vjepa_encoder=vjepa,  # type: ignore[arg-type]
        kv_generator=generator,
        camera_order=camera_order,
        proprio_dim=8,
        action_horizon=32,
        freeze_vjepa=True,
        freeze_action=True,
        freeze_proprio=True,
        device="cpu",
        torch_dtype=torch.float32,
    )

    attention_mask = model._action_attention_mask(
        video_seq_len=98,
        action_seq_len=32,
        device=torch.device("cpu"),
    )
    if tuple(attention_mask.shape) != (130, 130):
        raise AssertionError(f"Unexpected joint attention mask shape {tuple(attention_mask.shape)}.")
    action_rows = attention_mask[98:130, :130]
    if tuple(action_rows.shape) != (32, 130):
        raise AssertionError(f"Unexpected action attention rows shape {tuple(action_rows.shape)}.")
    if not bool(action_rows[:, :98].all()):
        raise AssertionError("Action queries cannot read all 98 video K/V tokens.")
    if not bool(action_rows[:, 98:].all()):
        raise AssertionError("Action queries cannot read all 32 action K/V tokens.")

    video_a = torch.randn(batch_size, 3, 9, 8, 16)
    video_b = video_a.clone()
    video_b[:, :, 1:] = torch.randn_like(video_b[:, :, 1:]) * 9.0
    if not torch.equal(video_a[:, :, 0], video_b[:, :, 0]):
        raise AssertionError("Future-leakage test requires identical frame zero.")
    if torch.equal(video_a[:, :, 1:], video_b[:, :, 1:]):
        raise AssertionError("Future-leakage test requires different future frames.")

    current = extract_causal_current_frame(video_a)
    if not torch.equal(current, video_a[:, :, 0]):
        raise AssertionError("extract_causal_current_frame did not select index zero.")
    single_camera = torch.randn(batch_size, 3, 32, 32)
    duplicated = build_duplicated_vjepa_clip(single_camera)
    if not torch.equal(duplicated[:, :, 0], duplicated[:, :, 1]):
        raise AssertionError("Duplicated V-JEPA temporal frames differ.")
    camera_frames = split_dual_camera_current_frame(
        current,
        camera_order=camera_order,
        image_size=32,
    )
    if any(tuple(frame.shape[-2:]) != (32, 32) for frame in camera_frames):
        raise AssertionError("Camera split did not preserve separate square views.")

    context = torch.randn(batch_size, 6, 24)
    context_mask = torch.ones(batch_size, 6, dtype=torch.bool)
    context_mask[1, 3:] = False
    proprio = torch.randn(batch_size, 9, 8)
    prepared_context, prepared_mask = model._prepare_context(context, context_mask, proprio)

    tokens_a = model.encode_current_frame(video_a)
    tokens_b = model.encode_current_frame(video_b)
    if not torch.allclose(tokens_a, tokens_b, atol=1e-6, rtol=0.0):
        raise AssertionError("Future frames changed causal V-JEPA tokens.")
    cache_a = model.generate_video_kv_cache(video_a, prepared_context, prepared_mask)
    cache_b = model.generate_video_kv_cache(video_b, prepared_context, prepared_mask)
    assert_cache_close(cache_a, cache_b)
    if len(cache_a) != len(action_expert.blocks):
        raise AssertionError("Cache list length does not match ActionDiT layer count.")
    expected_cache_shape = (batch_size, 98, 16)
    for layer in cache_a:
        if tuple(layer["k"].shape) != expected_cache_shape:
            raise AssertionError(f"Unexpected K shape: {tuple(layer['k'].shape)}.")
        if tuple(layer["v"].shape) != expected_cache_shape:
            raise AssertionError(f"Unexpected V shape: {tuple(layer['v'].shape)}.")

    fixed_noise = torch.randn(batch_size, 32, 7)
    fixed_timestep = torch.full((batch_size,), 0.5)
    pred_a = model.predict_action_noise_with_cache(
        fixed_noise, fixed_timestep, prepared_context, prepared_mask, cache_a
    )
    pred_b = model.predict_action_noise_with_cache(
        fixed_noise, fixed_timestep, prepared_context, prepared_mask, cache_b
    )
    if not torch.allclose(pred_a, pred_b, atol=1e-6, rtol=0.0):
        raise AssertionError("Future frames changed fixed-noise action output.")

    video_c = video_a.clone()
    video_c[:, :, 0] = video_c[:, :, 0] + 0.5
    cache_c = model.generate_video_kv_cache(video_c, prepared_context, prepared_mask)
    if cache_max_difference(cache_a, cache_c) <= 1e-6:
        raise AssertionError("Changing current frame did not change the generated cache.")

    padded_context = context.clone()
    padded_context[~context_mask] = torch.randn_like(padded_context[~context_mask]) * 1000.0
    prepared_padded, padded_mask = model._prepare_context(
        padded_context, context_mask, proprio
    )
    cache_padded = model.generate_video_kv_cache(video_a, prepared_padded, padded_mask)
    assert_cache_close(cache_a, cache_padded)
    pred_padded = model.predict_action_noise_with_cache(
        fixed_noise, fixed_timestep, prepared_padded, padded_mask, cache_padded
    )
    if not torch.allclose(pred_a, pred_padded, atol=1e-5, rtol=0.0):
        raise AssertionError("Masked padding changed ActionDiT output.")

    sample = {
        "video": video_a,
        "action": torch.randn(batch_size, 32, 7),
        "context": context,
        "context_mask": context_mask,
        "proprio": proprio,
        "action_is_pad": torch.zeros(batch_size, 32, dtype=torch.bool),
    }
    model.zero_grad(set_to_none=True)
    loss, _ = model.training_loss(sample)
    if not torch.isfinite(loss):
        raise AssertionError("Dummy training loss is not finite.")
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.kv_generator.parameters()):
        raise AssertionError("KV generator received no gradients.")
    if any(parameter.grad is not None for parameter in model.vjepa_encoder.parameters()):
        raise AssertionError("Frozen V-JEPA received gradients.")
    if any(parameter.grad is not None for parameter in model.action_expert.parameters()):
        raise AssertionError("Default-frozen ActionDiT received gradients.")

    state_keys = tuple(model.state_dict())
    forbidden = ("vae", "video_expert", "future_predictor", "jepa_adapter")
    if any(term in key.lower() for key in state_keys for term in forbidden):
        raise AssertionError("Student state_dict contains a forbidden v1/v2 module.")

    inference_context = context[:1]
    inference_mask = context_mask[:1]
    inference_proprio = proprio[:1, 0]
    vjepa.forward_calls = 0
    action_first = model.infer_action(
        input_image=video_a[:1],
        context=inference_context,
        context_mask=inference_mask,
        proprio=inference_proprio,
        num_inference_steps=4,
        seed=19,
    )
    if tuple(action_first.shape) != (32, 7):
        raise AssertionError(f"infer_action returned {tuple(action_first.shape)}.")
    if model.debug_counts != {
        "vjepa_forward": 1,
        "kv_generator_forward": 1,
        "action_forward": 4,
    }:
        raise AssertionError(f"Unexpected infer counters: {model.debug_counts}.")
    if vjepa.forward_calls != 1:
        raise AssertionError(f"V-JEPA was called {vjepa.forward_calls} times during infer.")
    if len(set(model._infer_cache_ids)) != 1:
        raise AssertionError("Denoising steps did not reuse one cache object.")
    if model.last_debug.get("selected_frame_index") != 0:
        raise AssertionError("Debug metadata selected a future frame.")
    if model.last_debug.get("duplicated_frames_equal") is not True:
        raise AssertionError("Debug metadata did not confirm duplicated frames.")
    if any(shape[-2:] != (32, 32) for shape in model.last_debug["camera_frame_shapes"]):
        raise AssertionError("Debug camera shapes are not square.")

    action_second = model.infer_action(
        input_image=video_a[:1],
        context=inference_context,
        context_mask=inference_mask,
        proprio=inference_proprio,
        num_inference_steps=4,
        seed=19,
    )
    if not torch.equal(action_first, action_second):
        raise AssertionError("Fixed-seed infer_action is not repeatable.")

    legacy_video = build_action_expert()
    legacy_action = build_action_expert()
    legacy_mot = MoT(
        mixtures={"video": legacy_video, "action": legacy_action},
        mot_checkpoint_mixed_attn=False,
    )
    if tuple(legacy_mot.expert_order) != ("video", "action"):
        raise AssertionError("Original dual-expert MoT initialization was broken.")

    print(
        "PASS selected_frame_index=0 duplicated_frames_equal=true "
        f"generator_parameters={generator.parameter_count}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-KV v4 sanity checks.")
    parser.add_argument("--dummy", action="store_true", help="Run without real checkpoints.")
    args = parser.parse_args()
    if not args.dummy:
        raise ValueError("Only --dummy is supported locally; real-weight validation belongs on the server.")
    run_dummy_sanity()


if __name__ == "__main__":
    main()
