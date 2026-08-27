from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT,
    PROJECT_ROOT / "tools",
    PROJECT_ROOT / "experiments" / "libero",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_fastwam_jepa_idm_v2_predictor_value as rollout_utils  # noqa: E402
from fastwam.models.wan22.v5_contract import (  # noqa: E402
    ACTION_HORIZON,
    CAMERA_WIDTH,
    EXEC_HORIZON_DEFAULT,
    temporal_metadata,
)
from fastwam_jepa_v5_data import (  # noqa: E402
    compose_cfg,
    load_v5_model_checkpoint,
    precision_dtypes,
    provenance_paths,
    require_file,
    sha256_file,
)


LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-IDM V5 LIBERO rollout.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--release-checkpoint", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--libero-data-root", required=True)
    parser.add_argument("--vjepa-repo", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint-key", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-suite", choices=LIBERO_SUITES, default="libero_spatial")
    parser.add_argument("--task-id", default="0")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--exec-horizon", type=int, default=EXEC_HORIZON_DEFAULT)
    parser.add_argument("--num-visual-inference-steps", type=int, default=10)
    parser.add_argument("--num-action-inference-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--binarize-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--pyopengl-platform", default="egl")
    parser.add_argument("--egl-device-id", type=int, default=0)
    parser.add_argument("--egl-import-device-id", default=None)
    return parser.parse_args()


def load_real_text_context(prompt: str, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = cfg.data.train.get("text_embedding_cache_dir")
    if cache_dir is None:
        raise ValueError("V5 rollout requires data.train.text_embedding_cache_dir.")
    cache_path_root = Path(str(cache_dir)).expanduser()
    if not cache_path_root.is_absolute():
        cache_path_root = (PROJECT_ROOT / cache_path_root).resolve()
    context_len = int(cfg.data.train.get("context_len", 128))
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_path_root / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing cached V5 text context: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or "context" not in payload or "mask" not in payload:
        raise ValueError(f"Invalid text context payload: {cache_path}")
    context = payload["context"]
    context_mask = payload["mask"].to(dtype=torch.bool)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    if context.ndim != 3 or int(context.shape[-1]) != 4096:
        raise ValueError(f"Cached V5 context must be [B,L,4096], got {tuple(context.shape)}.")
    if tuple(context_mask.shape) != tuple(context.shape[:2]):
        raise ValueError("Cached V5 context mask does not match context tokens.")
    if bool((context_mask.sum(dim=1) == 0).any()):
        raise ValueError("Cached V5 context has no valid text token.")
    context = context.masked_fill(~context_mask.unsqueeze(-1), 0.0)
    return context, context_mask


def postprocess_action(action: torch.Tensor, processor, *, binarize: bool) -> np.ndarray:
    from experiments.libero.libero_utils import invert_gripper_action

    denormalized = rollout_utils._denormalize_action(action, processor)[0]
    denormalized[..., -1] = denormalized[..., -1] * 2 - 1
    denormalized = invert_gripper_action(denormalized)
    if binarize:
        denormalized[..., -1] = np.sign(denormalized[..., -1])
    return denormalized


def run_episode(
    *,
    env,
    initial_state,
    model,
    processor,
    cfg,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    args: argparse.Namespace,
    task_id: int,
    episode_index: int,
) -> dict[str, Any]:
    from experiments.libero.libero_utils import get_libero_dummy_action

    env.reset()
    observation = env.set_init_state(initial_state)
    pending_actions: list[list[float]] = []
    success = False
    episode_return = 0.0
    episode_length = 0
    max_policy_steps = rollout_utils.get_max_steps(args.libero_suite, args.max_steps)
    for step_index in range(max_policy_steps + args.num_steps_wait):
        if step_index < args.num_steps_wait:
            action_to_env = get_libero_dummy_action()
        else:
            if not pending_actions:
                image, proprio, _ = rollout_utils._obs_to_model_input(
                    observation,
                    cfg=cfg,
                    processor=processor,
                    width=448,
                    height=224,
                    device=model.runtime_device(),
                    dtype=model.runtime_dtype(),
                )
                if tuple(image.shape) != (1, 3, 224, 448):
                    raise ValueError(f"V5 rollout RGB contract failed: {tuple(image.shape)}.")
                prediction = model.infer_joint(
                    agentview_rgb=image[..., :CAMERA_WIDTH],
                    wrist_rgb=image[..., CAMERA_WIDTH:],
                    context=context.to(device=model.runtime_device(), dtype=model.runtime_dtype()),
                    context_mask=context_mask.to(device=model.runtime_device()),
                    proprio=proprio.to(device=model.runtime_device(), dtype=model.runtime_dtype()),
                    num_visual_inference_steps=args.num_visual_inference_steps,
                    num_action_inference_steps=args.num_action_inference_steps,
                    seed=args.seed + task_id * 100000 + episode_index * 1000 + episode_length,
                )
                action_chunk = postprocess_action(
                    prediction["action"], processor, binarize=args.binarize_gripper
                )
                if tuple(action_chunk.shape) != (ACTION_HORIZON, 7):
                    raise ValueError(f"V5 rollout action shape failed: {tuple(action_chunk.shape)}.")
                pending_actions = action_chunk[: args.exec_horizon].tolist()
            action_to_env = pending_actions.pop(0)
        observation, reward, done, _ = env.step(action_to_env)
        episode_return += float(np.asarray(reward).reshape(-1)[0])
        episode_length += 1
        if done:
            success = True
            break
    return {
        "task_id": int(task_id),
        "episode": int(episode_index),
        "success": bool(success),
        "return": float(episode_return),
        "length": int(episode_length),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V5 rollout requires CUDA; CPU fallback is disabled.")
    if args.seed <= 0 or args.num_episodes <= 0:
        raise ValueError("--seed and --num-episodes must be positive.")
    if args.action_horizon != ACTION_HORIZON:
        raise ValueError("V5 --action-horizon is fixed at 16.")
    if not 1 <= args.exec_horizon <= ACTION_HORIZON:
        raise ValueError("--exec-horizon must be in [1,16].")
    if min(args.num_visual_inference_steps, args.num_action_inference_steps) <= 0:
        raise ValueError("V5 inference step counts must be positive.")
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    os.environ["PYOPENGL_PLATFORM"] = args.pyopengl_platform
    if args.mujoco_gl == "egl":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(
            args.egl_import_device_id
            if args.egl_import_device_id is not None
            else args.egl_device_id
        )
    from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.pytorch_utils import set_global_seed
    from libero.libero import benchmark

    set_global_seed(args.seed, get_worker_init_fn=False)
    device = torch.device(args.device)
    dtype, _ = precision_dtypes(args.precision)
    cfg = compose_cfg(args.config_name, args.task)
    OmegaConf.update(
        cfg, "data.train.pretrained_norm_stats", str(require_file(args.dataset_stats_path, name="--dataset-stats-path")), force_add=True
    )
    provenance = provenance_paths(args, rank=0)
    checkpoint = require_file(args.checkpoint, name="--checkpoint")
    model, metadata, _ = load_v5_model_checkpoint(
        args,
        cfg=cfg,
        checkpoint_path=checkpoint,
        expected_stage="stage3",
        device=device,
        dtype=dtype,
        provenance=provenance,
    )
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(provenance["stats_path"])))
    task_suite = benchmark.get_benchmark_dict()[args.libero_suite]()
    task_ids = rollout_utils.get_task_ids(task_suite, args.task_id)
    episodes = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = list(task_suite.get_task_init_states(task_id))
        if len(initial_states) < args.num_episodes:
            raise ValueError(
                f"Task {task_id} has only {len(initial_states)} initial states for {args.num_episodes} episodes."
            )
        env, description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            context, context_mask = load_real_text_context(
                DEFAULT_PROMPT.format(task=description), cfg
            )
            for episode_index in range(args.num_episodes):
                result = run_episode(
                    env=env,
                    initial_state=initial_states[episode_index],
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    context=context,
                    context_mask=context_mask,
                    args=args,
                    task_id=task_id,
                    episode_index=episode_index,
                )
                result["task_description"] = description
                episodes.append(result)
                print(
                    f"task={task_id} episode={episode_index} success={result['success']} length={result['length']}",
                    flush=True,
                )
        finally:
            env.close()
    successes = sum(int(row["success"]) for row in episodes)
    output = {
        "version": "v5",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "release_provenance": {
            "path": str(provenance["release_path"]),
            "sha256": provenance["release_sha"],
        },
        "vjepa_provenance": {
            "path": str(provenance["vjepa_path"]),
            "sha256": provenance["vjepa_sha"],
        },
        "dataset_stats_provenance": {
            "path": str(provenance["stats_path"]),
            "sha256": provenance["stats_sha"],
        },
        "task": {"suite": args.libero_suite, "task_id": args.task_id},
        "seed": args.seed,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(len(episodes), 1),
        "episode_lengths": [row["length"] for row in episodes],
        "action_horizon": ACTION_HORIZON,
        "exec_horizon": args.exec_horizon,
        "visual_inference_steps": args.num_visual_inference_steps,
        "action_inference_steps": args.num_action_inference_steps,
        "temporal_metadata": temporal_metadata(),
        "camera_order": ["agentview", "wrist"],
        "model_config": metadata["visual_config"],
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"output_json={output_path} success_rate={output['success_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
