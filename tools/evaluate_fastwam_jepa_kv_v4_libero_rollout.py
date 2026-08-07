from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.vjepa.jepa_kv_cache_generator import JepaKVCacheGenerator  # noqa: E402
from fastwam.models.wan22.fastwam_jepa_kv_v4 import (  # noqa: E402
    FastWAMJEPAKVV4,
    prepare_v4_context,
    sha256_file,
    validate_checkpoint_context_mask_mode,
)
from train_fastwam_jepa_kv_v4_stage1_distill import (  # noqa: E402
    build_vjepa_encoder,
    camera_order_from_cfg,
    compose_cfg,
    precision_dtypes,
    require_file,
)
from train_fastwam_jepa_kv_v4_stage2_action import (  # noqa: E402
    build_action_expert,
    load_proprio_strict,
    strict_load_action,
)


LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastWAM-JEPA-KV v4 LIBERO rollout evaluator.")
    parser.add_argument("--v4-checkpoint", required=True)
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--vjepa-repo", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint-key", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--text-embedding-cache-dir", default=None)
    parser.add_argument("--libero-suite", choices=LIBERO_SUITES, default="libero_spatial")
    parser.add_argument("--task-id", default=None, help="LIBERO task id or 'all'.")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--exec-horizon", type=int, default=None)
    parser.add_argument("--num-steps-wait", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument(
        "--context-mask-mode",
        choices=("baseline_all_true", "cached_real_mask"),
        default="baseline_all_true",
    )
    parser.add_argument(
        "--allow-stats-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow-action-teacher-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--binarize-gripper",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--save-videos",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--mujoco-gl", choices=("egl", "osmesa", "glfw"), default="egl")
    parser.add_argument("--pyopengl-platform", default="egl")
    parser.add_argument("--egl-device-id", type=int, default=0)
    parser.add_argument("--egl-import-device-id", default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def configure_mujoco(args: argparse.Namespace) -> None:
    from evaluate_fastwam_jepa_idm_v2_predictor_value import configure_mujoco_env

    configure_mujoco_env(args)


def resolve_protocol(args: argparse.Namespace, cfg: DictConfig) -> None:
    evaluation = cfg.EVALUATION
    if args.task_id is None:
        args.task_id = str(evaluation.task_id)
    if args.num_episodes is None:
        args.num_episodes = int(evaluation.num_trials)
    if args.num_inference_steps is None:
        args.num_inference_steps = int(evaluation.num_inference_steps)
    if args.action_horizon is None:
        args.action_horizon = int(cfg.data.train.num_frames) - 1
    if args.exec_horizon is None:
        args.exec_horizon = int(evaluation.replan_steps)
    if args.num_steps_wait is None:
        args.num_steps_wait = int(evaluation.num_steps_wait)
    if args.seed is None:
        args.seed = int(cfg.seed)
    if args.binarize_gripper is None:
        args.binarize_gripper = bool(evaluation.binarize_gripper)
    if not 0 < int(args.seed) < int(np.iinfo(np.uint32).max):
        raise ValueError("--seed must be within the positive uint32 range.")
    if int(args.num_inference_steps) <= 0 or int(args.action_horizon) <= 0:
        raise ValueError("Inference steps and action horizon must be positive.")
    if not 0 < int(args.exec_horizon) <= int(args.action_horizon):
        raise ValueError("--exec-horizon must be in [1, action_horizon].")
    if int(args.num_steps_wait) < 0 or int(args.num_episodes) <= 0:
        raise ValueError("Wait steps must be non-negative and num episodes must be positive.")


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("v4 checkpoint payload must be a dict.")
    required = ("kv_generator", "model_configuration", "metadata")
    missing = [key for key in required if not isinstance(payload.get(key), dict)]
    if missing:
        raise ValueError(f"v4 checkpoint is missing dict fields: {missing}.")
    return payload


def checkpoint_stage(payload: dict[str, Any]) -> str:
    return "stage2" if isinstance(payload.get("action_expert"), dict) else "stage1"


def verify_sha(
    *,
    label: str,
    expected: Any,
    actual: str,
    allow_mismatch: bool = False,
) -> None:
    if expected is None:
        if not allow_mismatch:
            raise ValueError(f"v4 checkpoint metadata is missing {label} SHA256.")
        return
    if str(expected) != actual and not allow_mismatch:
        raise ValueError(f"{label} SHA256 mismatch: checkpoint={expected}, current={actual}.")


def build_generator(
    payload: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> JepaKVCacheGenerator:
    config = payload["model_configuration"]
    generator = JepaKVCacheGenerator(
        input_dim=int(config["input_dim"]),
        context_dim=int(config["context_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        attn_head_dim=int(config["head_dim"]),
        video_seq_len=int(config["video_seq_len"]),
        layer_rank=int(config["layer_rank"]),
        num_cameras=2,
    )
    generator.load_state_dict(payload["kv_generator"], strict=True)
    print("kv_generator_load strict=true missing=0 unexpected=0", flush=True)
    return generator.to(device=device, dtype=dtype)


def build_model(
    *,
    args: argparse.Namespace,
    cfg: DictConfig,
    payload: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[FastWAMJEPAKVV4, dict[str, Any]]:
    metadata = payload["metadata"]
    stage = checkpoint_stage(payload)
    camera_order = camera_order_from_cfg(cfg)
    checkpoint_args = payload.get("args")
    if isinstance(checkpoint_args, dict):
        for key in (
            "vjepa_model_name",
            "vjepa_img_size",
            "vjepa_input_range",
            "vjepa_tubelet_size",
            "vjepa_dim",
        ):
            expected = checkpoint_args.get(key)
            actual = getattr(args, key)
            if expected is not None and str(expected) != str(actual):
                raise ValueError(
                    f"v4 checkpoint {key}={expected!r} does not match evaluator {actual!r}."
                )
    if metadata.get("input_policy") != "single_current_frame_duplicated_to_2":
        raise ValueError("v4 checkpoint has an incompatible input_policy.")
    if metadata.get("selected_frame_index") != 0:
        raise ValueError("v4 checkpoint did not use selected_frame_index=0.")
    if tuple(metadata.get("camera_order", ())) != tuple(camera_order):
        raise ValueError("v4 checkpoint camera_order does not match the evaluation config.")
    validate_checkpoint_context_mask_mode(
        metadata,
        str(args.context_mask_mode),
        checkpoint_name=f"v4 {stage} checkpoint",
    )

    action_path = require_file(args.action_checkpoint, name="--action-checkpoint")
    stats_path = require_file(args.dataset_stats_path, name="--dataset-stats-path")
    vjepa_path = require_file(args.vjepa_checkpoint, name="--vjepa-checkpoint")
    action_sha = sha256_file(action_path)
    stats_sha = sha256_file(stats_path)
    vjepa_sha = sha256_file(vjepa_path)
    expected_action = (
        metadata.get("teacher_fastwam_checkpoint_sha256")
        if stage == "stage1"
        else metadata.get("action_checkpoint_sha256")
    )
    verify_sha(
        label="action checkpoint",
        expected=expected_action,
        actual=action_sha,
        allow_mismatch=bool(args.allow_action_teacher_mismatch),
    )
    verify_sha(
        label="dataset stats",
        expected=metadata.get("dataset_stats_sha256"),
        actual=stats_sha,
        allow_mismatch=bool(args.allow_stats_mismatch),
    )
    verify_sha(
        label="V-JEPA checkpoint",
        expected=metadata.get("vjepa_checkpoint_sha256"),
        actual=vjepa_sha,
    )

    action_expert, baseline_proprio_state = build_action_expert(
        cfg,
        checkpoint_path=action_path,
        device=device,
        dtype=dtype,
        rank=0,
    )
    if stage == "stage2":
        strict_load_action(action_expert, payload["action_expert"], rank=0)
        print("stage2_action_override strict=true", flush=True)
    generator = build_generator(payload, device=device, dtype=dtype)
    if int(generator.video_seq_len) != 98:
        raise ValueError(f"v4 evaluator requires 98 visual tokens, got {generator.video_seq_len}.")
    if int(generator.input_dim) != int(args.vjepa_dim):
        raise ValueError(
            f"Generator input_dim={generator.input_dim} does not match --vjepa-dim={args.vjepa_dim}."
        )
    if int(generator.num_layers) != len(action_expert.blocks):
        raise ValueError("KV generator layer count does not match ActionDiT.")
    if stage == "stage2" and int(payload["model_configuration"].get("action_horizon", 32)) != int(
        args.action_horizon
    ):
        raise ValueError("Stage2 checkpoint action_horizon does not match evaluator.")
    vjepa, vjepa_report = build_vjepa_encoder(args, device=device, dtype=dtype, rank=0)
    proprio_dim = int(OmegaConf.select(cfg, "model.proprio_dim"))
    model = FastWAMJEPAKVV4(
        action_expert=action_expert,
        vjepa_encoder=vjepa,
        kv_generator=generator,
        camera_order=camera_order,
        proprio_dim=proprio_dim,
        action_horizon=int(args.action_horizon),
        action_train_shift=float(cfg.model.action_scheduler.train_shift),
        action_infer_shift=float(cfg.model.action_scheduler.infer_shift),
        action_num_train_timesteps=int(cfg.model.action_scheduler.num_train_timesteps),
        freeze_vjepa=True,
        freeze_action=True,
        freeze_proprio=True,
        context_mask_mode=str(args.context_mask_mode),
        device=device,
        torch_dtype=dtype,
    )
    proprio_state = payload.get("proprio_encoder")
    if not isinstance(proprio_state, dict):
        proprio_state = baseline_proprio_state
    load_proprio_strict(model, proprio_state)
    print("proprio_encoder_load strict=true missing=0 unexpected=0", flush=True)
    model.requires_grad_(False)
    model.eval()
    return model, {
        "checkpoint_stage": stage,
        "checkpoint_step": int(payload.get("step", 0)),
        "v4_checkpoint_path": str(checkpoint_path),
        "v4_checkpoint_sha256": sha256_file(checkpoint_path),
        "action_checkpoint_path": str(action_path),
        "action_checkpoint_sha256": action_sha,
        "dataset_stats_path": str(stats_path),
        "dataset_stats_sha256": stats_sha,
        "vjepa_checkpoint_path": str(vjepa_path),
        "vjepa_checkpoint_sha256": vjepa_sha,
        "vjepa_load_report": vjepa_report,
        "camera_order": list(camera_order),
    }


def load_raw_text_context(
    prompt: str,
    cfg: DictConfig,
    *,
    override_dir: Optional[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_value = override_dir or cfg.data.train.get("text_embedding_cache_dir")
    if cache_value is None:
        raise ValueError("A text embedding cache directory is required.")
    cache_dir = resolve_path(str(cache_value))
    context_len = int(cfg.data.train.get("context_len", 128))
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{prompt_hash}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cached text context does not exist: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("context")):
        raise ValueError(f"Invalid text context payload: {cache_path}")
    context = payload["context"].clone()
    mask = payload.get("mask")
    if not torch.is_tensor(mask):
        raise ValueError(f"Text context payload has no tensor mask: {cache_path}")
    mask = mask.to(dtype=torch.bool)
    return prepare_v4_context(
        context,
        mask,
        mode=str(cfg.EVALUATION.context_mask_mode),
    )


def postprocess_action(action: torch.Tensor, processor, *, binarize_gripper: bool) -> np.ndarray:
    from evaluate_fastwam_jepa_idm_v2_predictor_value import _denormalize_action
    from experiments.libero.libero_utils import invert_gripper_action

    result = _denormalize_action(action, processor)[0]
    result[..., -1] = result[..., -1] * 2.0 - 1.0
    result = invert_gripper_action(result)
    if binarize_gripper:
        result[..., -1] = np.sign(result[..., -1])
    return result


def timing_summary(records: list[dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"replans": len(records)}
    for key in ("vjepa_ms", "kv_generator_ms", "action_denoise_ms", "total_replan_ms"):
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[key] = {
            "mean": None if values.size == 0 else float(values.mean()),
            "p95": None if values.size == 0 else float(np.percentile(values, 95)),
        }
    return summary


@torch.no_grad()
def run_episode(
    *,
    env,
    initial_state,
    model: FastWAMJEPAKVV4,
    processor,
    cfg: DictConfig,
    args: argparse.Namespace,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    task_description: str,
    task_id: int,
    episode_idx: int,
    video_dir: Optional[Path],
) -> dict[str, Any]:
    from evaluate_fastwam_jepa_idm_v2_predictor_value import (
        _obs_to_model_input,
        get_max_steps,
    )
    from experiments.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_image,
        save_rollout_video,
    )

    video_size = cfg.data.train.video_size
    input_h, input_w = int(video_size[0]), int(video_size[1])
    max_policy_steps = get_max_steps(str(args.libero_suite), args.max_steps)
    env.reset()
    obs = env.set_init_state(initial_state)
    pending_actions: list[list[float]] = []
    frames: list[Any] = []
    timings: list[dict[str, float]] = []
    episode_return = 0.0
    policy_steps = 0
    success = False
    replan_index = 0

    for env_step in range(int(args.num_steps_wait) + int(max_policy_steps)):
        if env_step < int(args.num_steps_wait):
            action_to_env = get_libero_dummy_action()
        else:
            if not pending_actions:
                image, proprio, images = _obs_to_model_input(
                    obs,
                    cfg=cfg,
                    processor=processor,
                    width=input_w,
                    height=input_h,
                    device=model._runtime_device(),
                    dtype=model.torch_dtype,
                )
                action = model.infer_action(
                    input_image=image,
                    context=context,
                    context_mask=context_mask,
                    proprio=proprio,
                    action_horizon=int(args.action_horizon),
                    num_inference_steps=int(args.num_inference_steps),
                    sigma_shift=args.sigma_shift,
                    seed=int(args.seed) + task_id * 100000 + episode_idx * 1000 + replan_index,
                    rand_device=str(args.rand_device),
                )
                if model.last_debug.get("selected_frame_index") != 0:
                    raise RuntimeError("v4 rollout selected a non-current frame.")
                if model.last_debug.get("duplicated_frames_equal") is not True:
                    raise RuntimeError("v4 rollout did not duplicate each camera frame identically.")
                timings.append(dict(model.last_inference_timing))
                pending_actions = postprocess_action(
                    action,
                    processor,
                    binarize_gripper=bool(args.binarize_gripper),
                )[: int(args.exec_horizon)].tolist()
                replan_index += 1
                if bool(args.save_videos):
                    frames.append(images)
            elif bool(args.save_videos):
                frames.append(get_libero_image(obs))
            action_to_env = pending_actions.pop(0)
            policy_steps += 1

        obs, reward, done, _ = env.step(action_to_env)
        episode_return += float(np.asarray(reward).reshape(-1)[0])
        if done:
            success = True
            break

    video_path = None
    if bool(args.save_videos) and frames and video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        try:
            video_path = save_rollout_video(
                video_dir,
                frames,
                f"v4_task{task_id}_episode{episode_idx}",
                success,
                task_description,
                fps=int(args.video_fps),
            )
        except Exception as exc:
            print(f"WARNING video save failed: {type(exc).__name__}: {exc}", flush=True)
    return {
        "task_id": int(task_id),
        "episode": int(episode_idx),
        "success": bool(success),
        "episode_steps": int(policy_steps),
        "return": float(episode_return),
        "inference_timing": timing_summary(timings),
        "video_path": video_path,
    }


def run_rollout(
    *,
    model: FastWAMJEPAKVV4,
    cfg: DictConfig,
    args: argparse.Namespace,
    load_info: dict[str, Any],
) -> dict[str, Any]:
    from evaluate_fastwam_jepa_idm_v2_predictor_value import get_task_ids
    from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.pytorch_utils import set_global_seed
    from libero.libero import benchmark

    set_global_seed(int(args.seed), get_worker_init_fn=False)
    stats = load_dataset_stats_from_json(load_info["dataset_stats_path"])
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    suite = benchmark.get_benchmark_dict()[str(args.libero_suite)]()
    task_ids = get_task_ids(suite, str(args.task_id))
    output_json = resolve_path(args.output_json)
    video_dir = output_json.parent / "videos" if bool(args.save_videos) else None
    episodes: list[dict[str, Any]] = []
    per_task: dict[str, Any] = {}

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = list(suite.get_task_init_states(task_id))
        if not initial_states:
            raise ValueError(f"LIBERO task {task_id} has no initial states.")
        while len(initial_states) < int(args.num_episodes):
            initial_states.extend(initial_states[: int(args.num_episodes) - len(initial_states)])
        env, description = get_libero_env(task, LIBERO_ENV_RESOLUTION, int(args.seed))
        prompt = DEFAULT_PROMPT.format(task=description)
        context, context_mask = load_raw_text_context(
            prompt,
            cfg,
            override_dir=args.text_embedding_cache_dir,
        )
        task_rows: list[dict[str, Any]] = []
        try:
            for episode_idx in range(int(args.num_episodes)):
                row = run_episode(
                    env=env,
                    initial_state=initial_states[episode_idx],
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    args=args,
                    context=context,
                    context_mask=context_mask,
                    task_description=description,
                    task_id=int(task_id),
                    episode_idx=episode_idx,
                    video_dir=video_dir,
                )
                row["task_description"] = description
                task_rows.append(row)
                episodes.append(row)
                print(
                    f"episode task={task_id} idx={episode_idx} "
                    f"success={row['success']} steps={row['episode_steps']}",
                    flush=True,
                )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        per_task[str(task_id)] = {
            "task_description": description,
            "episodes": len(task_rows),
            "successes": sum(int(row["success"]) for row in task_rows),
            "success_rate": float(np.mean([row["success"] for row in task_rows])),
        }

    successes = sum(int(row["success"]) for row in episodes)
    return {
        "model": "FastWAMJEPAKVV4",
        "input_policy": "single_current_frame_duplicated_to_2",
        "selected_frame_index": 0,
        "history_used": False,
        "future_used": False,
        "context_mask_mode": str(args.context_mask_mode),
        **load_info,
        "num_inference_steps": int(args.num_inference_steps),
        "action_horizon": int(args.action_horizon),
        "exec_horizon": int(args.exec_horizon),
        "num_steps_wait": int(args.num_steps_wait),
        "libero_suite": str(args.libero_suite),
        "task_id": str(args.task_id),
        "seed": int(args.seed),
        "successes": int(successes),
        "total_episodes": len(episodes),
        "success_rate": float(successes / max(len(episodes), 1)),
        "per_task": per_task,
        "episodes": episodes,
    }


def main() -> None:
    args = parse_args()
    configure_mujoco(args)
    device = resolve_device(str(args.device))
    dtype, _ = precision_dtypes(str(args.precision), device)
    cfg = compose_cfg(str(args.config_name), str(args.task))
    if args.text_embedding_cache_dir is not None:
        OmegaConf.update(
            cfg,
            "data.train.text_embedding_cache_dir",
            str(resolve_path(args.text_embedding_cache_dir)),
            force_add=True,
        )
    OmegaConf.update(
        cfg,
        "EVALUATION.context_mask_mode",
        str(args.context_mask_mode),
        force_add=True,
    )
    resolve_protocol(args, cfg)
    checkpoint_path = require_file(args.v4_checkpoint, name="--v4-checkpoint")
    payload = load_payload(checkpoint_path)
    model, load_info = build_model(
        args=args,
        cfg=cfg,
        payload=payload,
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
    )
    print(
        f"checkpoint={checkpoint_path} stage={load_info['checkpoint_stage']} "
        f"dataset_stats={load_info['dataset_stats_path']} camera_order={load_info['camera_order']} "
        f"num_inference_steps={args.num_inference_steps} action_horizon={args.action_horizon} "
        f"exec_horizon={args.exec_horizon} seed={args.seed} "
        f"context_mask_mode={args.context_mask_mode}",
        flush=True,
    )
    results = run_rollout(model=model, cfg=cfg, args=args, load_info=load_info)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)
    print(
        f"output_json={output_json} success_rate={results['success_rate']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
