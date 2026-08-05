from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise AssertionError(f"{name} is not finite.")


def _teacher_kind_from_identity(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if ("fastwam" in normalized and "idm" in normalized) or normalized.startswith(
        "libero_idm"
    ):
        return "fastwam_idm"
    if (
        normalized.endswith("create_fastwam")
        or normalized.endswith("create_fastwam_joint")
        or normalized.endswith(".fastwam")
        or normalized.endswith(".fastwamjoint")
    ):
        return "fastwam"
    return "unknown"


def _scalar_hints(value: Any, *, depth: int = 0) -> Iterator[str]:
    if depth > 4:
        return
    if isinstance(value, argparse.Namespace):
        value = vars(value)
    if isinstance(value, Mapping):
        container_keys = {"metadata", "args", "config", "cfg", "model"}
        identity_keys = {
            "_target_",
            "target",
            "task",
            "model_kind",
            "model_type",
            "teacher_kind",
            "model_class",
            "teacher_class",
            "architecture",
            "arch",
        }
        for index, (key, item) in enumerate(value.items()):
            if index >= 128:
                break
            normalized_key = str(key).strip().lower()
            if normalized_key in container_keys or normalized_key in identity_keys:
                yield from _scalar_hints(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value[:128]:
            yield from _scalar_hints(item, depth=depth + 1)
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        yield str(value)


def _checkpoint_kind(checkpoint_info: Mapping[str, Any]) -> str:
    kinds = {
        kind
        for hint in _scalar_hints(checkpoint_info)
        if (kind := _teacher_kind_from_identity(hint)) != "unknown"
    }
    if len(kinds) > 1:
        return "conflict"
    return next(iter(kinds), "unknown")


def _compact_metadata(value: Any, *, depth: int = 0) -> Any:
    if depth >= 2:
        return f"<{type(value).__name__}>"
    if isinstance(value, argparse.Namespace):
        value = vars(value)
    if isinstance(value, Mapping):
        items = list(value.items())[:16]
        compact = {
            str(key): _compact_metadata(item, depth=depth + 1)
            for key, item in items
            if not torch.is_tensor(item)
        }
        if len(value) > len(items):
            compact["..."] = f"{len(value) - len(items)} more keys"
        return compact
    if isinstance(value, (list, tuple)):
        return [_compact_metadata(item, depth=depth + 1) for item in value[:8]]
    if torch.is_tensor(value):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    text = str(value)
    return text if len(text) <= 160 else text[:157] + "..."


@contextmanager
def _suppress_dataset_stats_write():
    from fastwam.datasets.lerobot import robot_video_dataset

    original = robot_video_dataset.save_dataset_stats_to_json
    robot_video_dataset.save_dataset_stats_to_json = lambda *_args, **_kwargs: None
    try:
        yield
    finally:
        robot_video_dataset.save_dataset_stats_to_json = original


def _vjepa_patch_size(vjepa: nn.Module) -> Any:
    core = getattr(vjepa, "encoder", None)
    patch_embed = getattr(core, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(core, "patch_size", None)
    if isinstance(patch_size, (list, tuple)):
        return tuple(int(value) for value in patch_size)
    return patch_size


def _validate_teacher_identity(
    *,
    teacher: nn.Module,
    config_target: str,
    checkpoint_info: Mapping[str, Any],
    expected_kind: str | None,
) -> tuple[str, str]:
    config_kind = _teacher_kind_from_identity(config_target)
    class_name = f"{teacher.__class__.__module__}.{teacher.__class__.__name__}"
    class_kind = _teacher_kind_from_identity(class_name)
    if config_kind == "unknown" or class_kind == "unknown":
        raise ValueError(
            f"Could not classify teacher config/class: target={config_target}, class={class_name}."
        )
    if config_kind != class_kind:
        raise ValueError(
            f"Teacher config/class mismatch: config={config_kind}, class={class_kind}."
        )
    checkpoint_kind = _checkpoint_kind(checkpoint_info)
    if checkpoint_kind == "conflict":
        raise ValueError("Teacher checkpoint metadata contains conflicting FastWAM kinds.")
    if checkpoint_kind != "unknown" and checkpoint_kind != class_kind:
        raise ValueError(
            "Teacher checkpoint/runtime mismatch: "
            f"checkpoint={checkpoint_kind}, runtime={class_kind}."
        )
    if expected_kind is not None and class_kind != expected_kind:
        raise ValueError(
            f"Expected teacher kind {expected_kind}, but runtime kind is {class_kind}."
        )
    if (
        expected_kind is not None
        and checkpoint_kind != "unknown"
        and checkpoint_kind != expected_kind
    ):
        raise ValueError(
            f"Expected checkpoint kind {expected_kind}, detected {checkpoint_kind}."
        )
    return class_name, checkpoint_kind


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


@torch.no_grad()
def run_real_smoke(args: argparse.Namespace) -> None:
    from tools.train_fastwam_jepa_kv_v4_stage1_distill import (
        build_loader,
        build_teacher,
        build_vjepa_encoder,
        camera_order_from_cfg,
        canonicalize_batch,
        compose_cfg,
        kv_cache_distillation_loss,
        precision_dtypes,
        prepare_teacher_context,
        require_file,
        seed_everything,
        teacher_current_frame_cache,
    )

    required_paths = (
        "libero_data_root",
        "dataset_stats_path",
        "teacher_checkpoint",
        "vjepa_repo",
        "vjepa_checkpoint",
    )
    missing = [f"--{name.replace('_', '-')}" for name in required_paths if not getattr(args, name)]
    if missing:
        raise ValueError(f"--real-smoke requires: {', '.join(missing)}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1 or torch.distributed.is_initialized():
        raise RuntimeError("--real-smoke is single-process only; do not use torchrun or DDP.")
    if int(args.seed) <= 0:
        raise ValueError("--seed must be positive.")

    seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_dtype, _ = precision_dtypes(str(args.precision), device)
    cfg = compose_cfg(str(args.config_name), str(args.task))
    config_target = str(cfg.model.get("_target_", "unknown"))
    config_kind = _teacher_kind_from_identity(config_target)
    if args.expected_teacher_kind is not None and config_kind != args.expected_teacher_kind:
        raise ValueError(
            f"Configured teacher kind is {config_kind}, expected {args.expected_teacher_kind}."
        )
    camera_order = camera_order_from_cfg(cfg)

    args.batch_size = 1
    args.num_workers = 0
    with _suppress_dataset_stats_write():
        loader, sampler = build_loader(
            cfg,
            args=args,
            ddp_enabled=False,
            world_size=1,
            rank=0,
        )
    if sampler is not None:
        raise AssertionError("Real smoke unexpectedly created a DistributedSampler.")
    raw_batch = next(iter(loader))
    batch = canonicalize_batch(raw_batch, device=device, dtype=param_dtype)
    raw_context_mask = batch["context_mask"]
    valid_token_count = [int(value) for value in raw_context_mask.sum(dim=1).tolist()]
    mask_all_true = bool(raw_context_mask.all().item())
    print(f"context_shape={tuple(batch['context'].shape)}", flush=True)
    print(f"context_mask_shape={tuple(raw_context_mask.shape)}", flush=True)
    print(f"valid_token_count={valid_token_count}", flush=True)
    print(f"mask_all_true={str(mask_all_true).lower()}", flush=True)
    if mask_all_true:
        raise AssertionError(
            "Real LIBERO context_mask is all True; refusing legacy all-true mask fallback."
        )

    teacher_path = require_file(str(args.teacher_checkpoint), name="--teacher-checkpoint")
    checkpoint_info: dict[str, Any] = {}
    teacher = build_teacher(
        cfg,
        checkpoint_path=teacher_path,
        device=device,
        dtype=param_dtype,
        rank=0,
        checkpoint_info=checkpoint_info,
    )
    teacher_python_class, teacher_checkpoint_kind = _validate_teacher_identity(
        teacher=teacher,
        config_target=config_target,
        checkpoint_info=checkpoint_info,
        expected_kind=args.expected_teacher_kind,
    )
    teacher_metadata = checkpoint_info.get("metadata")
    print(f"teacher_python_class={teacher_python_class}", flush=True)
    print(f"teacher_config_target={config_target}", flush=True)
    print(f"teacher_task={args.task}", flush=True)
    print(f"teacher_checkpoint_path={teacher_path}", flush=True)
    print(
        "teacher_checkpoint_metadata="
        + ("absent" if teacher_metadata is None else repr(_compact_metadata(teacher_metadata))),
        flush=True,
    )
    print(f"teacher_checkpoint_kind={teacher_checkpoint_kind}", flush=True)
    print("teacher_cache_source=current_frame_only", flush=True)

    required_teacher_api = (
        "video_expert",
        "action_expert",
        "mot",
        "_encode_video_latents",
        "_append_proprio_to_context",
    )
    missing_teacher_api = [name for name in required_teacher_api if not hasattr(teacher, name)]
    if missing_teacher_api:
        raise AssertionError(f"Teacher is missing required shared FastWAM API: {missing_teacher_api}.")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise AssertionError("Teacher parameters are not fully frozen.")

    vjepa, _ = build_vjepa_encoder(args, device=device, dtype=param_dtype, rank=0)
    if any(parameter.requires_grad for parameter in vjepa.parameters()):
        raise AssertionError("V-JEPA parameters are not fully frozen.")
    action_expert = teacher.action_expert
    generator = JepaKVCacheGenerator(
        input_dim=int(args.vjepa_dim),
        context_dim=int(action_expert.text_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=len(action_expert.blocks),
        num_heads=int(action_expert.num_heads),
        attn_head_dim=int(action_expert.attn_head_dim),
        video_seq_len=int(args.video_seq_len),
        layer_rank=int(args.layer_rank),
        num_cameras=len(camera_order),
    ).to(device=device, dtype=param_dtype)
    if generator.parameter_count <= 8440:
        raise AssertionError(
            f"Generator parameter count looks like a dummy configuration: {generator.parameter_count}."
        )
    action_horizon = int(batch["action"].shape[1])
    student = FastWAMJEPAKVV4(
        action_expert=action_expert,
        vjepa_encoder=vjepa,
        kv_generator=generator,
        camera_order=camera_order,
        proprio_dim=None,
        action_horizon=action_horizon,
        freeze_vjepa=True,
        freeze_action=True,
        freeze_proprio=True,
        torch_dtype=param_dtype,
    ).eval()
    forbidden_student_modules = ("vae", "video_expert", "future_predictor")
    student_keys = tuple(student.state_dict())
    if any(term in key.lower() for key in student_keys for term in forbidden_student_modules):
        raise AssertionError("v4 student contains a forbidden Wan/Future module.")

    teacher_context, teacher_mask = prepare_teacher_context(teacher, batch)
    video_a = batch["video"][:1]
    if video_a.shape[2] < 2:
        raise AssertionError("Future-leakage smoke requires at least two video frames.")
    video_b = video_a.clone()
    video_b[:, :, 1:] = video_b[:, :, 1:] + 0.125
    if not torch.equal(video_a[:, :, 0], video_b[:, :, 0]):
        raise AssertionError("Future-leakage inputs do not share frame zero.")
    if torch.equal(video_a[:, :, 1:], video_b[:, :, 1:]):
        raise AssertionError("Future-leakage inputs do not differ after frame zero.")

    current = extract_causal_current_frame(video_a)
    camera_frames = split_dual_camera_current_frame(
        current,
        camera_order=camera_order,
        image_size=int(args.vjepa_img_size),
    )
    camera_clips = [build_duplicated_vjepa_clip(frame) for frame in camera_frames]
    duplicated_frames_equal = all(
        torch.equal(clip[:, :, 0], clip[:, :, 1]) for clip in camera_clips
    )
    if not duplicated_frames_equal:
        raise AssertionError("A real V-JEPA clip contains unequal duplicated frames.")
    expected_clip_shape = (1, 3, 2, int(args.vjepa_img_size), int(args.vjepa_img_size))
    if any(tuple(clip.shape) != expected_clip_shape for clip in camera_clips):
        raise AssertionError(
            f"Unexpected real V-JEPA camera clip shapes: {[tuple(clip.shape) for clip in camera_clips]}."
        )

    captured_vjepa: dict[str, tuple[int, ...]] = {}

    def capture_vjepa_shape(_module, _inputs, output) -> None:
        if not torch.is_tensor(output):
            raise AssertionError("Real V-JEPA wrapper returned a non-tensor output.")
        captured_vjepa["shape"] = tuple(int(value) for value in output.shape)

    hook = vjepa.register_forward_hook(capture_vjepa_shape)
    try:
        visual_tokens_a = student.encode_current_frame(video_a)
    finally:
        hook.remove()
    visual_debug = dict(student.last_debug)
    visual_tokens_b = student.encode_current_frame(video_b)
    if not torch.equal(visual_tokens_a, visual_tokens_b):
        raise AssertionError("Future frames changed real V-JEPA current-frame tokens.")
    raw_shape = captured_vjepa.get("shape")
    if raw_shape is None or len(raw_shape) != 3:
        raise AssertionError(f"Could not capture real V-JEPA output shape: {raw_shape}.")
    if raw_shape[0] != len(camera_order):
        raise AssertionError(f"Real V-JEPA camera batch mismatch: {raw_shape}.")
    dense_grid = tuple(int(value) for value in visual_debug["vjepa_dense_grid"])
    extra_tokens_detected = raw_shape[1] != dense_grid[0] * dense_grid[1]
    if extra_tokens_detected:
        raise AssertionError(
            f"Real V-JEPA produced extra CLS/register tokens: raw={raw_shape}, grid={dense_grid}."
        )
    expected_visual_shape = (1, int(args.video_seq_len), int(args.vjepa_dim))
    if tuple(visual_tokens_a.shape) != expected_visual_shape:
        raise AssertionError(
            f"Visual tokens must be {expected_visual_shape}, got {tuple(visual_tokens_a.shape)}."
        )

    student_cache_a = generator(visual_tokens_a, teacher_context, teacher_mask)
    student_cache_b = generator(visual_tokens_b, teacher_context, teacher_mask)
    assert_cache_close(student_cache_a, student_cache_b, atol=0.0)
    teacher_cache, teacher_grid = teacher_current_frame_cache(
        teacher, video_a, teacher_context, teacher_mask
    )
    if teacher_grid != (1, 7, 14):
        raise AssertionError(f"Teacher grid must be (1,7,14), got {teacher_grid}.")
    if len(teacher_cache) != len(action_expert.blocks):
        raise AssertionError(
            f"Teacher cache layers {len(teacher_cache)} != ActionDiT layers {len(action_expert.blocks)}."
        )
    cache_dim = int(action_expert.num_heads) * int(action_expert.attn_head_dim)
    expected_cache_shape = (1, int(args.video_seq_len), cache_dim)
    for layer_index, (teacher_layer, student_layer) in enumerate(
        zip(teacher_cache, student_cache_a)
    ):
        for key in ("k", "v"):
            if tuple(teacher_layer[key].shape) != expected_cache_shape:
                raise AssertionError(
                    f"Teacher cache layer {layer_index} {key} has {tuple(teacher_layer[key].shape)}."
                )
            if student_layer[key].shape != teacher_layer[key].shape:
                raise AssertionError(
                    f"Student/teacher cache mismatch at layer {layer_index} {key}."
                )

    loss_total, loss_metrics = kv_cache_distillation_loss(
        student_cache_a, teacher_cache, lambda_cos=0.1
    )
    _require_finite("loss_total", loss_total)
    for name in ("loss_k", "loss_v", "loss_cos", "cos_first", "cos_middle", "cos_last"):
        _require_finite(name, loss_metrics[name])

    noise_generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    fixed_noisy_action = torch.randn(
        (1, action_horizon, int(action_expert.action_dim)),
        generator=noise_generator,
        dtype=torch.float32,
    ).to(device=device, dtype=param_dtype)
    fixed_timestep = torch.full((1,), 0.5, device=device, dtype=param_dtype)
    teacher_action = student.predict_action_noise_with_cache(
        fixed_noisy_action,
        fixed_timestep,
        teacher_context,
        teacher_mask,
        teacher_cache,
    )
    student_action = student.predict_action_noise_with_cache(
        fixed_noisy_action,
        fixed_timestep,
        teacher_context,
        teacher_mask,
        student_cache_a,
    )
    teacher_student_action_mse = F.mse_loss(
        teacher_action.float(), student_action.float()
    )
    teacher_student_action_cosine = F.cosine_similarity(
        teacher_action.float().flatten(1), student_action.float().flatten(1), dim=1
    ).mean()
    teacher_action_norm = teacher_action.float().norm()
    student_action_norm = student_action.float().norm()
    for name, value in (
        ("teacher_student_action_mse", teacher_student_action_mse),
        ("teacher_student_action_cosine", teacher_student_action_cosine),
        ("teacher_action_norm", teacher_action_norm),
        ("student_action_norm", student_action_norm),
    ):
        _require_finite(name, value)

    print("REAL_SMOKE_PASS", flush=True)
    print("selected_frame_index=0", flush=True)
    print("duplicated_frames_equal=true", flush=True)
    print("future_leakage=false", flush=True)
    print(f"teacher_class={teacher_python_class}", flush=True)
    print(f"teacher_grid={teacher_grid}", flush=True)
    print(f"camera_order={tuple(camera_order)}", flush=True)
    print(f"camera_frame_shapes={tuple(tuple(frame.shape) for frame in camera_frames)}", flush=True)
    print(f"vjepa_raw_shape={raw_shape}", flush=True)
    print(f"vjepa_raw_shape_per_camera={(1, raw_shape[1], raw_shape[2])}", flush=True)
    print(f"vjepa_tokens_per_camera={raw_shape[1]}", flush=True)
    print(f"vjepa_dim={raw_shape[2]}", flush=True)
    print(f"tubelet_size={getattr(vjepa, 'tubelet_size', None)}", flush=True)
    print(f"patch_size={_vjepa_patch_size(vjepa)}", flush=True)
    print(f"extra_cls_register_tokens={str(extra_tokens_detected).lower()}", flush=True)
    print("spatial_order=horizontal_7x14_row_major", flush=True)
    print(f"visual_tokens_shape={tuple(visual_tokens_a.shape)}", flush=True)
    print(f"action_context_shape={tuple(teacher_context.shape)}", flush=True)
    print(f"cache_layers={len(student_cache_a)}", flush=True)
    print(f"cache_shape={tuple(student_cache_a[0]['k'].shape)}", flush=True)
    print(f"generator_parameters={generator.parameter_count}", flush=True)
    print(f"loss_total={float(loss_total):.6f}", flush=True)
    for name in ("loss_k", "loss_v", "loss_cos", "cos_first", "cos_middle", "cos_last"):
        print(f"{name}={float(loss_metrics[name]):.6f}", flush=True)
    print(f"teacher_student_action_mse={float(teacher_student_action_mse):.6f}", flush=True)
    print(f"teacher_student_action_cosine={float(teacher_student_action_cosine):.6f}", flush=True)
    print(f"teacher_action_norm={float(teacher_action_norm):.6f}", flush=True)
    print(f"student_action_norm={float(student_action_norm):.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-KV v4 sanity checks.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dummy", action="store_true", help="Run without real checkpoints.")
    mode.add_argument("--real-smoke", action="store_true", help="Run one real-weight batch.")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--expected-teacher-kind", choices=("fastwam", "fastwam_idm"), default=None)
    parser.add_argument("--vjepa-repo", default=None)
    parser.add_argument("--vjepa-checkpoint", default=None)
    parser.add_argument("--vjepa-checkpoint-key", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--layer-rank", type=int, default=16)
    parser.add_argument("--video-seq-len", type=int, default=98)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.dummy:
        run_dummy_sanity()
    else:
        run_real_smoke(args)


if __name__ == "__main__":
    main()
