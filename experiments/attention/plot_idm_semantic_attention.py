import argparse
import json
import math
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
    # video_meta["grid_size"] is usually like "(3, 7, 14)"
    s = str(video_meta.get("grid_size", ""))
    nums = []
    cur = ""
    for ch in s:
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                nums.append(int(cur))
                cur = ""
    if cur:
        nums.append(int(cur))
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
        # fallback from latent shape [B,C,T,H,W] and patch size [1,2,2]
        _, _, num_frames, latent_h, latent_w = layout["latents_video_shape"]
        grid_h = latent_h // 2
        grid_w = latent_w // 2

    assert video_seq_len == tokens_per_frame * num_frames, (
        video_seq_len,
        tokens_per_frame,
        num_frames,
    )
    assert tokens_per_frame == grid_h * grid_w, (
        tokens_per_frame,
        grid_h,
        grid_w,
    )

    return {
        "video_seq_len": video_seq_len,
        "action_seq_len": action_seq_len,
        "tokens_per_frame": tokens_per_frame,
        "num_frames": num_frames,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }


def select_mixed_action_records(records, video_seq_len, action_seq_len):
    total_k = video_seq_len + action_seq_len
    selected = []
    for r in records:
        if not r.get("saved_matrix"):
            continue
        if int(r["q_len"]) == action_seq_len and int(r["k_len"]) == total_k:
            selected.append(r)
    selected = sorted(selected, key=lambda r: int(r["call_idx"]))
    return selected


def plot_action_video_action_mass(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(12, 4))
    plt.plot(df["semantic_idx"], df["video_mass"], label="video")
    plt.plot(df["semantic_idx"], df["action_mass"], label="action")
    plt.xlabel("mixed attention call index")
    plt.ylabel("mean attention mass")
    plt.title("Action queries: video-token mass vs action-token mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "action_video_vs_action_mass_by_call.png", dpi=200)
    plt.close()

    layer_df = df.groupby("layer")[["video_mass", "action_mass"]].mean().reset_index()
    plt.figure(figsize=(10, 4))
    plt.plot(layer_df["layer"], layer_df["video_mass"], label="video")
    plt.plot(layer_df["layer"], layer_df["action_mass"], label="action")
    plt.xlabel("layer")
    plt.ylabel("mean attention mass")
    plt.title("Layer-wise action attention mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "layerwise_video_vs_action_mass.png", dpi=200)
    plt.close()

    step_df = df.groupby("denoise_step")[["video_mass", "action_mass"]].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(step_df["denoise_step"], step_df["video_mass"], label="video")
    plt.plot(step_df["denoise_step"], step_df["action_mass"], label="action")
    plt.xlabel("action denoising step")
    plt.ylabel("mean attention mass")
    plt.title("Denoising-step-wise action attention mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "stepwise_video_vs_action_mass.png", dpi=200)
    plt.close()


def plot_frame_attention(frame_arr: np.ndarray, out_dir: Path, num_frames: int):
    # frame_arr: [N_calls, num_frames]
    mean_frame = frame_arr.mean(axis=0)

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(num_frames), mean_frame)
    plt.xlabel("video frame index")
    plt.ylabel("attention mass")
    plt.title("Action → video frame attention, averaged over steps/layers/actions")
    plt.tight_layout()
    plt.savefig(out_dir / "action_to_frame_average.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.imshow(frame_arr, aspect="auto")
    plt.colorbar(label="attention mass")
    plt.xlabel("video frame index")
    plt.ylabel("mixed attention call index")
    plt.title("Action → frame attention for each mixed attention call")
    plt.tight_layout()
    plt.savefig(out_dir / "action_to_frame_by_call_heatmap.png", dpi=200)
    plt.close()


def plot_stepwise_frame_attention(df: pd.DataFrame, frame_arr: np.ndarray, out_dir: Path, num_frames: int):
    rows = []
    for step in sorted(df["denoise_step"].unique()):
        idx = df.index[df["denoise_step"] == step].to_numpy()
        rows.append(frame_arr[idx].mean(axis=0))
    step_frame = np.stack(rows, axis=0)

    plt.figure(figsize=(7, 5))
    plt.imshow(step_frame, aspect="auto")
    plt.colorbar(label="attention mass")
    plt.xlabel("video frame index")
    plt.ylabel("action denoising step")
    plt.title("Step-wise action → frame attention")
    plt.tight_layout()
    plt.savefig(out_dir / "stepwise_action_to_frame_heatmap.png", dpi=200)
    plt.close()

    for f in range(num_frames):
        plt.plot(np.arange(step_frame.shape[0]), step_frame[:, f], label=f"frame {f}")
    plt.xlabel("action denoising step")
    plt.ylabel("attention mass")
    plt.title("Action → frame attention across denoising steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "stepwise_action_to_frame_curves.png", dpi=200)
    plt.close()


def plot_action_to_action(action_mats: np.ndarray, out_dir: Path):
    # action_mats: [N_calls, Sa, Sa]
    avg = action_mats.mean(axis=0)

    plt.figure(figsize=(6, 5))
    plt.imshow(avg, aspect="auto")
    plt.colorbar(label="attention")
    plt.xlabel("key action token")
    plt.ylabel("query action token")
    plt.title("Action → action attention, averaged")
    plt.tight_layout()
    plt.savefig(out_dir / "action_to_action_average_heatmap.png", dpi=200)
    plt.close()

    # local window mass
    sa = avg.shape[0]
    total = avg.sum()
    rows = []
    for w in [0, 1, 2, 4, 8, 16]:
        mask = np.zeros_like(avg, dtype=bool)
        for i in range(sa):
            lo = max(0, i - w)
            hi = min(sa, i + w + 1)
            mask[i, lo:hi] = True
        mass = float(avg[mask].sum() / max(total, 1e-12))
        rows.append({"window": w, "mass": mass})

    local_df = pd.DataFrame(rows)
    local_df.to_csv(out_dir / "action_to_action_local_window_mass.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(local_df["window"], local_df["mass"], marker="o")
    plt.xlabel("local window ±w")
    plt.ylabel("attention mass ratio")
    plt.title("Action → action local attention mass")
    plt.tight_layout()
    plt.savefig(out_dir / "action_to_action_local_window_mass.png", dpi=200)
    plt.close()


def plot_spatial_heatmaps(spatial_arr: np.ndarray, out_dir: Path, num_frames: int, grid_h: int, grid_w: int):
    # spatial_arr: [N_calls, num_frames, tokens_per_frame]
    mean_spatial = spatial_arr.mean(axis=0)  # [F, TPF]

    for f in range(num_frames):
        heat = mean_spatial[f].reshape(grid_h, grid_w)
        plt.figure(figsize=(7, 4))
        plt.imshow(heat, aspect="auto")
        plt.colorbar(label="attention")
        plt.xlabel("latent patch x")
        plt.ylabel("latent patch y")
        plt.title(f"Action → spatial video tokens, frame {f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"action_to_spatial_frame{f}.png", dpi=200)
        plt.close()

    # one combined figure as frame-token heatmap
    combined = mean_spatial.reshape(num_frames, grid_h * grid_w)
    plt.figure(figsize=(12, 4))
    plt.imshow(combined, aspect="auto")
    plt.colorbar(label="attention")
    plt.xlabel("spatial token within frame")
    plt.ylabel("video frame")
    plt.title("Action → spatial-token attention by frame")
    plt.tight_layout()
    plt.savefig(out_dir / "action_to_spatial_by_frame_token.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_run", type=str, required=True)
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=30)
    args = parser.parse_args()

    attn_run = Path(args.attn_run)
    layout_path = Path(args.layout)
    out_dir = attn_run / "semantic_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = load_layout(layout_path)
    sv = layout["video_seq_len"]
    sa = layout["action_seq_len"]
    tpf = layout["tokens_per_frame"]
    nf = layout["num_frames"]
    gh = layout["grid_h"]
    gw = layout["grid_w"]

    print("[layout]", layout)

    records = load_records(attn_run)
    mixed_records = select_mixed_action_records(records, sv, sa)
    print(f"[records] selected mixed action attention records: {len(mixed_records)}")

    if len(mixed_records) == 0:
        raise RuntimeError("No q=action_len, k=video_len+action_len matrices found.")

    rows = []
    frame_list = []
    action_list = []
    spatial_list = []

    for i, r in enumerate(mixed_records):
        mat = np.load(r["matrix_path"])  # [Sa, Sv+Sa]
        if mat.shape != (sa, sv + sa):
            print("[skip] unexpected matrix shape", r["matrix_path"], mat.shape)
            continue

        video = mat[:, :sv]          # [Sa, Sv]
        action = mat[:, sv:sv + sa]  # [Sa, Sa]

        video_mass = float(video.sum(axis=1).mean())
        action_mass = float(action.sum(axis=1).mean())

        frame = video.reshape(sa, nf, tpf).sum(axis=2).mean(axis=0)  # [F]
        spatial = video.reshape(sa, nf, tpf).mean(axis=0)            # [F, TPF]

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
        action_list.append(action)
        spatial_list.append(spatial)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "semantic_attention_records.csv", index=False)

    frame_arr = np.stack(frame_list, axis=0)
    action_arr = np.stack(action_list, axis=0)
    spatial_arr = np.stack(spatial_list, axis=0)

    np.save(out_dir / "action_to_frame_by_call.npy", frame_arr)
    np.save(out_dir / "action_to_action_by_call.npy", action_arr)
    np.save(out_dir / "action_to_spatial_by_call.npy", spatial_arr)

    plot_action_video_action_mass(df, out_dir)
    plot_frame_attention(frame_arr, out_dir, nf)
    plot_stepwise_frame_attention(df, frame_arr, out_dir, nf)
    plot_action_to_action(action_arr, out_dir)
    plot_spatial_heatmaps(spatial_arr, out_dir, nf, gh, gw)

    print("[done] saved semantic plots to:", out_dir)


if __name__ == "__main__":
    main()