import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_exp_name(exp_dir: Path):
    """
    exp_dir example:
      attention_outputs/step020000_libero_spatial_task0
      attention_outputs/step020000_libero_10_task1
    """
    name = exp_dir.name
    m = re.match(r"step(?P<step>\d+)_(?P<suite>.+)_task(?P<task_id>\d+)$", name)
    if not m:
        return {
            "step": None,
            "suite": "unknown",
            "task_id": -1,
        }
    return {
        "step": int(m.group("step")),
        "suite": m.group("suite"),
        "task_id": int(m.group("task_id")),
    }


def parse_replan_idx(replan_dir: Path):
    m = re.search(r"replan(\d+)", replan_dir.name)
    return int(m.group(1)) if m else -1


def parse_grid_from_layout(exp_dir: Path, tpf: int):
    layout_path = exp_dir / "layout_debug" / "idm_layout_debug.json"
    if layout_path.exists():
        with open(layout_path, "r", encoding="utf-8") as f:
            layout = json.load(f)

        meta = layout.get("video_meta", {})
        grid_size = str(meta.get("grid_size", ""))  # e.g. "(3, 7, 14)"
        nums = [int(x) for x in re.findall(r"\d+", grid_size)]
        if len(nums) >= 3:
            return nums[0], nums[1], nums[2]

        if "latents_video_shape" in layout:
            # [B, C, T, H, W], patch size [1,2,2]
            _, _, t, h, w = layout["latents_video_shape"]
            return int(t), int(h) // 2, int(w) // 2

    # fallback
    if tpf == 98:
        return 3, 7, 14

    # rough fallback
    h = int(np.sqrt(tpf))
    while h > 1 and tpf % h != 0:
        h -= 1
    w = tpf // h
    return None, h, w


def find_semantic_dirs(root: Path, pattern: str):
    sem_dirs = []
    for exp_dir in sorted(root.glob(pattern)):
        if not exp_dir.is_dir():
            continue
        for sem_dir in sorted(exp_dir.glob("task*_replan*/semantic_plots")):
            rec = sem_dir / "semantic_attention_records.csv"
            frame = sem_dir / "action_to_frame_by_call.npy"
            action = sem_dir / "action_to_action_by_call.npy"
            spatial = sem_dir / "action_to_spatial_by_call.npy"
            if rec.exists() and frame.exists() and action.exists() and spatial.exists():
                sem_dirs.append(sem_dir)
    return sem_dirs


def load_one(sem_dir: Path):
    replan_dir = sem_dir.parent
    exp_dir = replan_dir.parent
    meta = parse_exp_name(exp_dir)
    meta["exp_dir"] = str(exp_dir)
    meta["replan_dir"] = str(replan_dir)
    meta["replan_idx"] = parse_replan_idx(replan_dir)

    rec = pd.read_csv(sem_dir / "semantic_attention_records.csv")
    frame = np.load(sem_dir / "action_to_frame_by_call.npy")
    action = np.load(sem_dir / "action_to_action_by_call.npy")
    spatial = np.load(sem_dir / "action_to_spatial_by_call.npy")

    return meta, rec, frame, action, spatial


def local_window_mass(action_avg: np.ndarray):
    sa = action_avg.shape[0]
    total = action_avg.sum()
    rows = []
    for w in [0, 1, 2, 4, 8, 16]:
        mask = np.zeros_like(action_avg, dtype=bool)
        for i in range(sa):
            lo = max(0, i - w)
            hi = min(sa, i + w + 1)
            mask[i, lo:hi] = True
        rows.append({
            "window": w,
            "mass": float(action_avg[mask].sum() / max(total, 1e-12)),
        })
    return pd.DataFrame(rows)


def plot_group(
    group_name: str,
    items,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = []
    recs = []
    frame_all = []
    action_all = []
    spatial_all = []

    for meta, rec, frame, action, spatial in items:
        rec = rec.copy()
        for k, v in meta.items():
            rec[k] = v
        metas.append(meta)
        recs.append(rec)
        frame_all.append(frame)
        action_all.append(action)
        spatial_all.append(spatial)

    rec_df = pd.concat(recs, ignore_index=True)
    frame_arr = np.concatenate(frame_all, axis=0)      # [N, F]
    action_arr = np.concatenate(action_all, axis=0)    # [N, Sa, Sa]
    spatial_arr = np.concatenate(spatial_all, axis=0)  # [N, F, TPF]

    rec_df.to_csv(out_dir / "merged_semantic_attention_records.csv", index=False)
    np.save(out_dir / "merged_action_to_frame_by_call.npy", frame_arr)
    np.save(out_dir / "merged_action_to_action_by_call.npy", action_arr)
    np.save(out_dir / "merged_action_to_spatial_by_call.npy", spatial_arr)

    # Summary CSV by task/replan
    summary_rows = []
    for meta, rec, frame, action, spatial in items:
        summary_rows.append({
            "suite": meta["suite"],
            "task_id": meta["task_id"],
            "replan_idx": meta["replan_idx"],
            "video_mass_mean": float(rec["video_mass"].mean()),
            "action_mass_mean": float(rec["action_mass"].mean()),
            "frame_entropy_mean": float(
                -(frame / np.maximum(frame.sum(axis=1, keepdims=True), 1e-12)
                  * np.log(np.maximum(frame / np.maximum(frame.sum(axis=1, keepdims=True), 1e-12), 1e-12))).sum(axis=1).mean()
            ),
            "num_calls": int(len(rec)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "group_summary_by_replan.csv", index=False)

    # 1. action -> frame average
    frame_mean = frame_arr.mean(axis=0)
    num_frames = frame_mean.shape[0]

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(num_frames), frame_mean)
    plt.xlabel("video frame index")
    plt.ylabel("attention mass")
    plt.title(f"{group_name}: action → video frame average")
    plt.tight_layout()
    plt.savefig(out_dir / "aggregate_action_to_frame_average.png", dpi=200)
    plt.close()

    # 2. per-task frame heatmap
    task_frame_rows = []
    task_labels = []
    for (suite, task_id), sub in summary_df.groupby(["suite", "task_id"]):
        vals = []
        for meta, rec, frame, action, spatial in items:
            if meta["suite"] == suite and meta["task_id"] == task_id:
                vals.append(frame.mean(axis=0))
        if vals:
            task_frame_rows.append(np.stack(vals, axis=0).mean(axis=0))
            task_labels.append(f"{suite}/task{task_id}")

    if task_frame_rows:
        task_frame = np.stack(task_frame_rows, axis=0)
        plt.figure(figsize=(7, max(4, 0.35 * len(task_labels))))
        plt.imshow(task_frame, aspect="auto")
        plt.colorbar(label="attention mass")
        plt.xlabel("video frame index")
        plt.ylabel("task")
        plt.yticks(np.arange(len(task_labels)), task_labels)
        plt.title(f"{group_name}: action → frame by task")
        plt.tight_layout()
        plt.savefig(out_dir / "aggregate_action_to_frame_by_task_heatmap.png", dpi=200)
        plt.close()

    # 3. step-wise frame attention
    if "denoise_step" in rec_df.columns:
        step_rows = []
        step_labels = []
        for step in sorted(rec_df["denoise_step"].unique()):
            # Need reconstruct through records order; simpler use semantic records index matching merged arrays.
            idx = rec_df.index[rec_df["denoise_step"] == step].to_numpy()
            if len(idx) > 0 and idx.max() < len(frame_arr):
                step_rows.append(frame_arr[idx].mean(axis=0))
                step_labels.append(int(step))
        if step_rows:
            step_frame = np.stack(step_rows, axis=0)
            plt.figure(figsize=(7, 5))
            plt.imshow(step_frame, aspect="auto")
            plt.colorbar(label="attention mass")
            plt.xlabel("video frame index")
            plt.ylabel("denoising step")
            plt.yticks(np.arange(len(step_labels)), step_labels)
            plt.title(f"{group_name}: step-wise action → frame")
            plt.tight_layout()
            plt.savefig(out_dir / "aggregate_stepwise_action_to_frame_heatmap.png", dpi=200)
            plt.close()

    # 4. video vs action mass by layer
    if "layer" in rec_df.columns:
        layer_df = rec_df.groupby("layer")[["video_mass", "action_mass"]].mean().reset_index()
        layer_df.to_csv(out_dir / "aggregate_layerwise_video_action_mass.csv", index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(layer_df["layer"], layer_df["video_mass"], label="video")
        plt.plot(layer_df["layer"], layer_df["action_mass"], label="action")
        plt.xlabel("layer")
        plt.ylabel("attention mass")
        plt.title(f"{group_name}: layer-wise video/action attention mass")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "aggregate_layerwise_video_vs_action_mass.png", dpi=200)
        plt.close()

    # 5. video vs action mass by denoising step
    if "denoise_step" in rec_df.columns:
        step_df = rec_df.groupby("denoise_step")[["video_mass", "action_mass"]].mean().reset_index()
        step_df.to_csv(out_dir / "aggregate_stepwise_video_action_mass.csv", index=False)

        plt.figure(figsize=(8, 4))
        plt.plot(step_df["denoise_step"], step_df["video_mass"], label="video")
        plt.plot(step_df["denoise_step"], step_df["action_mass"], label="action")
        plt.xlabel("denoising step")
        plt.ylabel("attention mass")
        plt.title(f"{group_name}: step-wise video/action attention mass")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "aggregate_stepwise_video_vs_action_mass.png", dpi=200)
        plt.close()

    # 6. action -> action average
    action_avg = action_arr.mean(axis=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(action_avg, aspect="auto")
    plt.colorbar(label="attention")
    plt.xlabel("key action token")
    plt.ylabel("query action token")
    plt.title(f"{group_name}: action → action average")
    plt.tight_layout()
    plt.savefig(out_dir / "aggregate_action_to_action_average_heatmap.png", dpi=200)
    plt.close()

    local_df = local_window_mass(action_avg)
    local_df.to_csv(out_dir / "aggregate_action_to_action_local_window_mass.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(local_df["window"], local_df["mass"], marker="o")
    plt.xlabel("local window ±w")
    plt.ylabel("attention mass ratio")
    plt.title(f"{group_name}: action local attention mass")
    plt.tight_layout()
    plt.savefig(out_dir / "aggregate_action_to_action_local_window_mass.png", dpi=200)
    plt.close()

    # 7. spatial heatmaps
    spatial_mean = spatial_arr.mean(axis=0)  # [F, TPF]
    _, num_frames, tpf = spatial_arr.shape

    # infer grid from first item's layout
    first_exp = Path(items[0][0]["exp_dir"])
    _, grid_h, grid_w = parse_grid_from_layout(first_exp, tpf)

    for f in range(num_frames):
        heat = spatial_mean[f].reshape(grid_h, grid_w)
        plt.figure(figsize=(7, 4))
        plt.imshow(heat, aspect="auto")
        plt.colorbar(label="attention")
        plt.xlabel("latent patch x")
        plt.ylabel("latent patch y")
        plt.title(f"{group_name}: action → spatial frame {f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"aggregate_action_to_spatial_frame{f}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(12, 4))
    plt.imshow(spatial_mean, aspect="auto")
    plt.colorbar(label="attention")
    plt.xlabel("spatial token within frame")
    plt.ylabel("video frame")
    plt.title(f"{group_name}: action → spatial token by frame")
    plt.tight_layout()
    plt.savefig(out_dir / "aggregate_action_to_spatial_by_frame_token.png", dpi=200)
    plt.close()

    # Overall text summary
    summary = {
        "group": group_name,
        "num_semantic_dirs": len(items),
        "num_attention_calls": int(len(rec_df)),
        "video_mass_mean": float(rec_df["video_mass"].mean()),
        "action_mass_mean": float(rec_df["action_mass"].mean()),
        "action_local_w4_mass": float(local_df[local_df["window"] == 4]["mass"].iloc[0]),
        "action_local_w8_mass": float(local_df[local_df["window"] == 8]["mass"].iloc[0]),
        "frame_attention_mean": frame_mean.tolist(),
        "output_dir": str(out_dir),
    }

    with open(out_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[done] {group_name}: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="attention_outputs")
    parser.add_argument("--pattern", type=str, default="step020000_*")
    parser.add_argument("--out_dir", type=str, default="attention_outputs/aggregate_step020000")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sem_dirs = find_semantic_dirs(root, args.pattern)
    if not sem_dirs:
        raise RuntimeError("No semantic_plots found. Run plot_idm_semantic_attention.py first.")

    print(f"[info] found semantic dirs: {len(sem_dirs)}")

    items = []
    for sem_dir in sem_dirs:
        items.append(load_one(sem_dir))

    # per-suite groups
    suites = sorted(set(meta["suite"] for meta, *_ in items))
    for suite in suites:
        suite_items = [x for x in items if x[0]["suite"] == suite]
        plot_group(
            group_name=suite,
            items=suite_items,
            out_dir=out_root / "by_suite" / suite,
        )

    # all tasks
    plot_group(
        group_name="all_tasks",
        items=items,
        out_dir=out_root / "all_tasks",
    )

    # global index
    rows = []
    for meta, rec, frame, action, spatial in items:
        rows.append({
            "suite": meta["suite"],
            "task_id": meta["task_id"],
            "replan_idx": meta["replan_idx"],
            "exp_dir": meta["exp_dir"],
            "replan_dir": meta["replan_dir"],
            "num_records": len(rec),
            "video_mass_mean": float(rec["video_mass"].mean()),
            "action_mass_mean": float(rec["action_mass"].mean()),
        })
    pd.DataFrame(rows).to_csv(out_root / "index_by_replan.csv", index=False)

    print("[done] aggregate outputs:", out_root)


if __name__ == "__main__":
    main()