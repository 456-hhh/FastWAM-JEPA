from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_fastwam_jepa_idm_v2_predictor_value as rollout_base
import evaluate_fastwam_jepa_idm_v2_stage2_libero_rollout as v2


TEMPORAL_METADATA = {
    "current_offset": 0,
    "current_repeat": 4,
    "future_offsets": [1, 2, 3, 4],
    "action_start_offset": 0,
    "action_horizon": 32,
    "proprio_offset": 0,
    "future_stride": 1,
    "causal": True,
}

_ORIGINAL_PARSE_ARGS = v2.parse_args
_ORIGINAL_RESOLVE_CONFIG = v2.resolve_and_validate_stage2_config
_ORIGINAL_RUN_EVAL = v2.run_rollout_eval
_ACTIVE_ARGS: argparse.Namespace | None = None


def _flag_present(name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in sys.argv[1:])


def parse_args() -> argparse.Namespace:
    global _ACTIVE_ARGS
    for required in (
        "--stage2-checkpoint",
        "--action-checkpoint",
        "--dataset-stats-path",
        "--vjepa-repo",
        "--vjepa-checkpoint",
    ):
        if not _flag_present(required):
            raise ValueError(f"v2.1 rollout requires explicit {required}")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--action-checkpoint", required=True)
    extra, remaining = pre_parser.parse_known_args(sys.argv[1:])
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv
    args.action_checkpoint = str(Path(extra.action_checkpoint).expanduser())
    args.current_frame_count = 4
    args.future_frame_count = 4
    args.action_horizon = 32
    if not _flag_present("--exec-horizon"):
        args.exec_horizon = 10
    if not _flag_present("--future-source"):
        args.future_source = "predicted"
    _ACTIVE_ARGS = args
    return args


def _validate_temporal_metadata(config: dict[str, Any]) -> None:
    for key, expected in TEMPORAL_METADATA.items():
        got = config.get(key)
        if got != expected:
            raise ValueError(
                f"Stage2 checkpoint temporal metadata mismatch for {key}: "
                f"{got!r} != {expected!r}"
            )


def resolve_and_validate_stage2_config(
    args: argparse.Namespace, checkpoint_config: dict[str, Any]
) -> None:
    _ORIGINAL_RESOLVE_CONFIG(args, checkpoint_config)
    _validate_temporal_metadata(checkpoint_config)
    recorded_action = checkpoint_config.get("action_checkpoint")
    if recorded_action is None:
        raise ValueError("Stage2 checkpoint is missing release action_checkpoint provenance")
    runtime_action = v2.require_file(args.action_checkpoint, name="--action-checkpoint")
    if Path(str(recorded_action)).expanduser().resolve() != runtime_action:
        raise ValueError(
            "Stage2 checkpoint release action path differs from --action-checkpoint: "
            f"{recorded_action!r} vs {runtime_action}"
        )
    if str(args.future_source) == "oracle":
        print("WARNING oracle future is NON-CAUSAL DIAGNOSTIC ONLY", flush=True)


def predict_env_action_chunk(
    *,
    env,
    obs: dict,
    model: torch.nn.Module,
    processor,
    cfg,
    args: argparse.Namespace,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    episode_seed: int,
    debug_requested: bool = False,
) -> tuple[np.ndarray, dict, dict[str, Any]]:
    image, proprio, imgs = rollout_base._obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model._runtime_device(),
        dtype=model.torch_dtype,
    )
    raw_current_video = image.unsqueeze(2).repeat(1, 1, 4, 1, 1)
    current_video = v2._resize_model_video(
        raw_current_video,
        frame_count=4,
        size=int(args.vjepa_img_size),
        name="current_video",
    )
    if not torch.equal(current_video[:, :, 0], current_video[:, :, 3]):
        raise RuntimeError("v2.1 rollout current-frame repeat contract failed")

    oracle_future_video = None
    if str(args.future_source) == "oracle":
        oracle_future_video = rollout_base.collect_oracle_future_video(
            env=env,
            obs=obs,
            cfg=cfg,
            processor=processor,
            args=args,
            input_w=input_w,
            input_h=input_h,
            model=model,
        )
        oracle_future_video = v2._resize_model_video(
            oracle_future_video,
            frame_count=4,
            size=int(args.vjepa_img_size),
            name="oracle_future_video",
        )

    prediction = rollout_base.sample_action_jepa_idm(
        model=model,
        current_video=current_video,
        context=context,
        context_mask=context_mask,
        proprio=proprio,
        action_horizon=action_horizon,
        future_source=str(args.future_source),
        oracle_future_video=oracle_future_video,
        num_inference_steps=int(args.num_inference_steps),
        sigma_shift=args.sigma_shift,
        seed=episode_seed,
        rand_device=str(args.rand_device),
    )
    action = v2._postprocess_env_action(prediction["action"], processor, args)
    debug_stats: dict[str, Any] = {}
    if debug_requested:
        raw_min, raw_max, raw_mean = v2._array_min_max_mean(
            raw_current_video.detach().float().cpu().numpy()
        )
        action_min, action_max, action_mean = v2._array_min_max_mean(action)
        debug_stats = {
            "raw_current_video_shape": tuple(image.unsqueeze(2).shape),
            "raw_current_video_min": raw_min,
            "raw_current_video_max": raw_max,
            "raw_current_video_mean": raw_mean,
            "model_current_video_shape": tuple(current_video.shape),
            "model_current_video_min": raw_min,
            "model_current_video_max": raw_max,
            "model_current_video_mean": raw_mean,
            "proprio_shape": tuple(proprio.shape),
            "proprio_first": proprio.detach().float().reshape(proprio.shape[0], -1)[0].cpu().numpy(),
            "action_chunk_shape": tuple(action.shape),
            "action_chunk_min": action_min,
            "action_chunk_max": action_max,
            "action_chunk_mean": action_mean,
            "action_context_shape": model.last_forward_shapes.get("action_context"),
        }
    return action, imgs, debug_stats


def run_rollout_episode(
    *,
    env,
    initial_state,
    task_description: str,
    prompt: str,
    model: torch.nn.Module,
    processor,
    cfg,
    args: argparse.Namespace,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    task_id: int,
    episode_idx: int,
    video_dir: Path | None,
) -> dict[str, Any]:
    from experiments.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_image,
        save_rollout_video,
    )

    env.reset()
    obs = env.set_init_state(initial_state)
    replay_images: list[np.ndarray] = []
    pending_actions: list[list[float]] = []
    episode_return = 0.0
    success = False
    length = 0
    video_path = None
    max_policy_steps = rollout_base.get_max_steps(str(args.libero_suite), args.max_steps)
    max_total_steps = int(max_policy_steps) + int(args.num_steps_wait)
    for step_idx in range(max_total_steps):
        if step_idx < int(args.num_steps_wait):
            action_to_env = get_libero_dummy_action()
        else:
            if not pending_actions:
                debug_step = int(getattr(args, "_debug_steps_printed", 0))
                debug_requested = 0 < int(args.debug_first_steps) > debug_step
                action, imgs, debug_stats = predict_env_action_chunk(
                    env=env,
                    obs=obs,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    args=args,
                    context=context,
                    context_mask=context_mask,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    episode_seed=int(args.effective_seed) + int(task_id) * 100_000 + int(episode_idx),
                    debug_requested=debug_requested,
                )
                if bool(args.save_videos):
                    replay_images.append(v2._frame_to_rgb(imgs))
                pending_actions = action[: int(args.exec_horizon)].tolist()
                if debug_requested and pending_actions:
                    v2._print_debug_step(
                        policy_step=debug_step,
                        task_id=task_id,
                        task_description=task_description,
                        prompt=prompt,
                        context=context,
                        debug_stats=debug_stats,
                        first_action=pending_actions[0],
                        future_source=str(args.future_source),
                    )
                    args._debug_steps_printed = debug_step + 1
            action_to_env = pending_actions.pop(0)
        obs, reward, done, _ = env.step(action_to_env)
        if bool(args.save_videos) and step_idx >= int(args.num_steps_wait):
            replay_images.append(v2._frame_to_rgb(get_libero_image(obs)))
        episode_return += float(np.asarray(reward).reshape(-1)[0])
        length += 1
        if done:
            success = True
            break
    if bool(args.save_videos) and video_dir is not None and replay_images:
        video_dir.mkdir(parents=True, exist_ok=True)
        try:
            video_path = save_rollout_video(
                video_dir,
                replay_images,
                f"v21_{args.future_source}_task{task_id}_episode{episode_idx}",
                success=success,
                task_description=task_description,
                fps=20,
            )
        except Exception as exc:
            v2._warn_video_issue("failed to save v2.1 rollout video", exc)
    return {
        "future_source": str(args.future_source),
        "task_id": int(task_id),
        "episode": int(episode_idx),
        "success": bool(success),
        "return": float(episode_return),
        "length": int(length),
        "video_path": video_path,
    }


def run_rollout_eval(**kwargs: Any) -> dict[str, Any]:
    results = _ORIGINAL_RUN_EVAL(**kwargs)
    results["temporal_metadata"] = dict(TEMPORAL_METADATA)
    results["frame_history_used"] = False
    results["oracle_note"] = (
        "NON-CAUSAL DIAGNOSTIC ONLY" if results.get("future_source") == "oracle" else None
    )
    return results


def main() -> None:
    v2.parse_args = parse_args
    v2.resolve_and_validate_stage2_config = resolve_and_validate_stage2_config
    v2.predict_env_action_chunk = predict_env_action_chunk
    v2.run_rollout_episode = run_rollout_episode
    v2.run_rollout_eval = run_rollout_eval
    v2.main()


if __name__ == "__main__":
    main()
