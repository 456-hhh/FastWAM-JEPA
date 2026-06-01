import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_records(attn_run: Path):
    records = []
    with open(attn_run / "records.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def parse_grid_size(video_meta):
    s = str(video_meta.get("grid_size", ""))
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    return None


def load_layout(layout_path: Path):
    with open(layout_path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    video_seq_len = int(layout["video_tokens_shape"][1])
    action_seq_len = int(layout["latents_action_shape"][1])
    tokens_per_frame = int(layout["video_tokens_per_frame"])

    grid = parse_grid_size(layout.get("video_meta", {}))
    if grid is not None:
        num_frames, grid_h, grid_w = grid
    else:
        _, _, num_frames, latent_h, latent_w = layout["latents_video_shape"]
        grid_h = latent_h // 2
        grid_w = latent_w // 2

    assert video_seq_len == tokens_per_frame * num_frames
    assert tokens_per_frame == grid_h * grid_w

    return {
        "video_seq_len": video_seq_len,
        "action_seq_len": action_seq_len,
        "tokens_per_frame": tokens_per_frame,
        "num_frames": num_frames,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }


def select_joint_mot_records(records, total_seq_len):
    selected = []
    for r in records:
        if not r.get("saved_matrix"):
            continue
        if int(r["q_len"]) == total_seq_len and int(r["k_len"]) == total_seq_len:
            if str(r.get("module")) == "mot":
                selected.append(r)
    return sorted(selected, key=lambda r: int(r["call_idx"]))


def plot_action_video_action_mass(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(12, 4))
    plt.plot(df["semantic_idx"], df["video_mass"], label="video")
    plt.plot(df["semantic_idx"], df["action_mass"], label="action")
    plt.xlabel("joint attention call index")
    plt.ylabel("mean attention mass")
    plt.title("Joint: action queries → video/action mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "joint_action_video_vs_action_mass_by_call.png", dpi=200)
    plt.close()

    layer_df = df.groupby("layer")[["video_mass", "action_mass"]].mean().reset_index()
    layer_df.to_csv(out_dir / "joint_layerwise_video_action_mass.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(layer_df["layer"], layer_df["video_mass"], label="video")
    plt.plot(layer_df["layer"], layer_df["action_mass"], label="action")
    plt.xlabel("layer")
    plt.ylabel("mean attention mass")
    plt.title("Joint: layer-wise action attention mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "joint_layerwise_video_vs_action_mass.png", dpi=200)
    plt.close()

    step_df = df.groupby("denoise_step")[["video_mass", "action_mass"]].mean().reset_index()
    step_df.to_csv(out_dir / "joint_stepwise_video_action_mass.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(step_df["denoise_step"], step_df["video_mass"], label="video")
    plt.plot(step_df["denoise_step"], step_df["action_mass"], label="action")
    plt.xlabel("joint denoising step")
    plt.ylabel("mean attention mass")
    plt.title("Joint: step-wise action attention mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "joint_stepwise_video_vs_action_mass.png", dpi=200)
    plt.close()


def plot_frame_attention(frame_arr: np.ndarray, out_dir: Path):
    num_frames = frame_arr.shape[1]
    mean_frame = frame_arr.mean(axis=0)

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(num_frames), mean_frame)
    plt.xlabel("video latent frame index")
    plt.ylabel("attention mass")
    plt.title("Joint: action → video latent frame average")
    plt.tight_layout()
    plt.savefig(out_dir / "joint_action_to_frame_average.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.imshow(frame_arr, aspect="auto")
    plt.colorbar(label="attention mass")
    plt.xlabel("video latent frame index")
    plt.ylabel("joint attention call index")
    plt.title("Joint: action → frame by call")
    plt.tight_layout()
    plt.savefig(out_dir / "joint_action_to_frame_by_call_heatmap.png", dpi=200)
    plt.close()


def plot_stepwise_frame_attention(df: pd.DataFrame, frame_arr: np.ndarray, out_dir: Path):
    rows = []
    steps = sorted(df["denoise_step"].unique())
    for step in steps:
        idx = df.index[df["denoise_step"] == step].to_numpy()
        rows.append(frame_arr[idx].mean(axis=0))

    step_frame = np.stack(rows, axis=0)

    plt.figure(figsize=(7, 5))
    plt.imshow(step_frame, aspect="auto")
    plt.colorbar(label="attention mass")
    plt.xlabel("video latent frame index")
    plt.ylabel("joint denoising step")
    plt.title("Joint: step-wise action → frame")
    plt.tight_layout()
    plt.savefig(out_dir / "joint_stepwise_action_to_frame_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    for f in range(step_frame.shape[1]):
        plt.plot(np.arange(step_frame.shape[0]), step_frame[:, f], label=f"frame {f}")
    plt.xlabel("joint denoising step")
    plt.ylabel("attention mass")
    plt.title("Joint: action → frame across denoising steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "joint_stepwise_action_to_frame_curves.png", dpi=200)
    plt.close()


def plot_action_to_action(action_arr: np.ndarray, out_dir: Path):
    avg = action_arr.mean(axis=0)

    plt.figure(figsize=(6, 5))
    plt.imshow(avg, aspect="auto")
    plt.colorbar(label="attention")
    plt.xlabel("key action token")
    plt.ylabel("query action token")
    plt.title("Joint: action → action average")
    plt.tight_layout()
    plt.savefig(out_dir / "joint_action_to_action_average_heatmap.png", dpi=200)
    plt.close()

    sa = avg.shape[0]
    total = avg.sum()
    rows = []
    for w in [0, 1, 2, 4, 8, 16]:
        mask = np.zeros_like(avg, dtype=bool)
        for i in range(sa):
            lo = max(0, i - w)
            hi = min(sa, i + w + 1)
            mask[i, lo:hi] = True
        rows.append({
            "window": w,
            "mass": float(avg[mask].sum() / max(total, 1e-12)),
        })

    local_df = pd.DataFrame(rows)
    local_df.to_csv(out_dir / "joint_action_to_action_local_window_mass.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(local_df["window"], local_df["mass"], marker="o")
    plt.xlabel("local window ±w")
    plt.ylabel("attention mass ratio")
    plt.title("Joint: action local attention mass")
    plt.tight_layout()
    plt.savefig(out_dir / "joint_action_to_action_local_window_mass.png", dpi=200)
    plt.close()


def plot_spatial_heatmaps(spatial_arr: np.ndarray, out_dir: Path, num_frames: int, grid_h: int, grid_w: int):
    mean_spatial = spatial_arr.mean(axis=0)

    for f in range(num_frames):
        heat = mean_spatial[f].reshape(grid_h, grid_w)
        plt.figure(figsize=(7, 4))
        plt.imshow(heat, aspect="auto")
        plt.colorbar(label="attention")
        plt.xlabel("latent patch x")
        plt.ylabel("latent patch y")
        plt.title(f"Joint: action → spatial video tokens, frame {f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"joint_action_to_spatial_frame{f}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(12, 4))
    plt.imshow(mean_spatial, aspect="auto")
    plt.colorbar(label="attention")
    plt.xlabel("spatial token within frame")
    plt.ylabel("video latent frame")
    plt.title("Joint: action → spatial token by frame")
    plt.tight_layout()
    plt.savefig(out_dir / "joint_action_to_spatial_by_frame_token.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_run", type=str, required=True)
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=30)
    args = parser.parse_args()

    attn_run = Path(args.attn_run)
    layout_path = Path(args.layout)
    out_dir = attn_run / "joint_semantic_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = load_layout(layout_path)
    sv = layout["video_seq_len"]
    sa = layout["action_seq_len"]
    total = sv + sa
    tpf = layout["tokens_per_frame"]
    nf = layout["num_frames"]
    gh = layout["grid_h"]
    gw = layout["grid_w"]

    print("[layout]", layout)

    records = load_records(attn_run)
    mot_records = select_joint_mot_records(records, total)
    print(f"[records] selected joint mot records: {len(mot_records)}")

    if len(mot_records) == 0:
        raise RuntimeError("No joint mot q=326,k=326 matrices found.")

    rows = []
    frame_list = []
    action_list = []
    spatial_list = []

    for i, r in enumerate(mot_records):
        mat = np.load(r["matrix_path"])  # [326, 326]

        if mat.shape != (total, total):
            print("[skip] unexpected shape", r["matrix_path"], mat.shape)
            continue

        action_rows = mat[sv:sv + sa, :]       # [Sa, Sv+Sa]
        action_to_video = action_rows[:, :sv]  # [Sa, Sv]
        action_to_action = action_rows[:, sv:sv + sa]  # [Sa, Sa]

        video_mass = float(action_to_video.sum(axis=1).mean())
        action_mass = float(action_to_action.sum(axis=1).mean())

        frame = action_to_video.reshape(sa, nf, tpf).sum(axis=2).mean(axis=0)
        spatial = action_to_video.reshape(sa, nf, tpf).mean(axis=0)

        rows.append({
            "semantic_idx": i,
            "call_idx": r["call_idx"],
            "matrix_path": r["matrix_path"],
            "denoise_step": i // args.num_layers,
            "layer": i % args.num_layers,
            "video_mass": video_mass,
            "action_mass": action_mass,
        })
        frame_list.append(frame)
        action_list.append(action_to_action)
        spatial_list.append(spatial)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "joint_semantic_attention_records.csv", index=False)

    frame_arr = np.stack(frame_list, axis=0)
    action_arr = np.stack(action_list, axis=0)
    spatial_arr = np.stack(spatial_list, axis=0)

    np.save(out_dir / "joint_action_to_frame_by_call.npy", frame_arr)
    np.save(out_dir / "joint_action_to_action_by_call.npy", action_arr)
    np.save(out_dir / "joint_action_to_spatial_by_call.npy", spatial_arr)

    plot_action_video_action_mass(df, out_dir)
    plot_frame_attention(frame_arr, out_dir)
    plot_stepwise_frame_attention(df, frame_arr, out_dir)
    plot_action_to_action(action_arr, out_dir)
    plot_spatial_heatmaps(spatial_arr, out_dir, nf, gh, gw)

    print("[done] saved joint semantic plots to:", out_dir)


if __name__ == "__main__":
    main()