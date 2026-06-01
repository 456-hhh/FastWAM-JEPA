import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def replan_idx(p: Path):
    m = re.search(r"replan(\d+)", p.name)
    return int(m.group(1)) if m else -1


def topk_mass(heat, ratio):
    x = heat.reshape(-1)
    total = x.sum()
    if total <= 1e-12:
        return 0.0
    k = max(1, int(len(x) * ratio))
    return float(np.sort(x)[-k:].sum() / total)


def center_of_mass(heat):
    h, w = heat.shape
    total = heat.sum()
    if total <= 1e-12:
        return np.nan, np.nan

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = float((heat * xx).sum() / total)
    cy = float((heat * yy).sum() / total)
    return cx, cy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--grid_h", type=int, default=7)
    parser.add_argument("--grid_w", type=int, default=14)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    root = Path(args.dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / "spatial_trajectory_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    sem_dirs = sorted(
        root.glob("task*_replan*/semantic_plots"),
        key=lambda p: replan_idx(p.parent),
    )

    if not sem_dirs:
        raise RuntimeError("No semantic_plots found. Run plot_idm_semantic_attention.py first.")

    heatmaps = []
    rows = []

    for sem in sem_dirs:
        idx = replan_idx(sem.parent)
        spatial_path = sem / "action_to_spatial_by_call.npy"

        if not spatial_path.exists():
            continue

        spatial = np.load(spatial_path)  # [num_calls, num_frames, 98]
        # 平均 layer / denoising step / action token 后，得到该 replan 的 3 个 frame 空间图
        spatial_mean = spatial.mean(axis=0)  # [3, 98]
        spatial_heat = spatial_mean.reshape(3, args.grid_h, args.grid_w)  # [3, 7, 14]
        heatmaps.append(spatial_heat)

        for f in range(3):
            heat = spatial_heat[f]
            cx, cy = center_of_mass(heat)

            left = heat[:, :7].sum()
            right = heat[:, 7:].sum()
            top = heat[:3, :].sum()
            bottom = heat[3:, :].sum()
            total = heat.sum()

            rows.append({
                "replan_idx": idx,
                "frame": f,
                "total_mass": float(total),
                "center_x": cx,
                "center_y": cy,
                "left_camera_mass": float(left),
                "right_camera_mass": float(right),
                "left_ratio": float(left / max(total, 1e-12)),
                "right_ratio": float(right / max(total, 1e-12)),
                "top_mass": float(top),
                "bottom_mass": float(bottom),
                "top_ratio": float(top / max(total, 1e-12)),
                "bottom_ratio": float(bottom / max(total, 1e-12)),
                "top10pct_mass_ratio": topk_mass(heat, 0.10),
                "top25pct_mass_ratio": topk_mass(heat, 0.25),
            })

    heatmaps = np.stack(heatmaps, axis=0)  # [R, 3, 7, 14]
    np.save(out_dir / "replan_spatial_heatmaps.npy", heatmaps)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "replan_spatial_metrics.csv", index=False)

    replans = sorted(df["replan_idx"].unique())

    # 1. 每个 future/current frame 的 flattened spatial timeline
    for f in range(3):
        frame_heat = heatmaps[:, f]  # [R, 7, 14]
        flat = frame_heat.reshape(frame_heat.shape[0], -1)  # [R, 98]

        plt.figure(figsize=(14, 5))
        plt.imshow(flat, aspect="auto")
        plt.colorbar(label="attention")
        plt.xlabel("spatial token within frame, 0..97")
        plt.ylabel("replan index")
        plt.yticks(np.arange(len(replans)), replans)
        plt.title(f"Spatial attention over replans, video latent frame {f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_spatial_flat_frame{f}.png", dpi=200)
        plt.close()

    # 2. 每个 frame 的中心点轨迹
    for f in range(3):
        sub = df[df["frame"] == f].sort_values("replan_idx")

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["center_x"], marker="o", label="center_x")
        plt.plot(sub["replan_idx"], sub["center_y"], marker="o", label="center_y")
        plt.xlabel("replan index")
        plt.ylabel("center of mass on 7x14 grid")
        plt.title(f"Attention center trajectory, frame {f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_attention_center_curve_frame{f}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(sub["center_x"], sub["center_y"], marker="o")
        for _, r in sub.iterrows():
            plt.text(r["center_x"], r["center_y"], str(int(r["replan_idx"])), fontsize=8)
        plt.xlim(-0.5, args.grid_w - 0.5)
        plt.ylim(args.grid_h - 0.5, -0.5)
        plt.xlabel("latent patch x")
        plt.ylabel("latent patch y")
        plt.title(f"Attention center path on spatial grid, frame {f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_attention_center_path_frame{f}.png", dpi=200)
        plt.close()

    # 3. 左右相机区域注意力趋势
    for f in range(3):
        sub = df[df["frame"] == f].sort_values("replan_idx")

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["left_ratio"], marker="o", label="left half / third-person")
        plt.plot(sub["replan_idx"], sub["right_ratio"], marker="o", label="right half / wrist")
        plt.xlabel("replan index")
        plt.ylabel("attention ratio")
        plt.title(f"Left/right camera attention over replans, frame {f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_left_right_camera_ratio_frame{f}.png", dpi=200)
        plt.close()

    # 4. 上下区域注意力趋势
    for f in range(3):
        sub = df[df["frame"] == f].sort_values("replan_idx")

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["top_ratio"], marker="o", label="top region")
        plt.plot(sub["replan_idx"], sub["bottom_ratio"], marker="o", label="bottom region")
        plt.xlabel("replan index")
        plt.ylabel("attention ratio")
        plt.title(f"Top/bottom attention over replans, frame {f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_top_bottom_ratio_frame{f}.png", dpi=200)
        plt.close()

    # 5. spatial sparsity 趋势
    for f in range(3):
        sub = df[df["frame"] == f].sort_values("replan_idx")

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["top10pct_mass_ratio"], marker="o", label="top 10% tokens")
        plt.plot(sub["replan_idx"], sub["top25pct_mass_ratio"], marker="o", label="top 25% tokens")
        plt.xlabel("replan index")
        plt.ylabel("attention mass ratio")
        plt.title(f"Spatial attention concentration over replans, frame {f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_spatial_concentration_frame{f}.png", dpi=200)
        plt.close()

    # 6. contact sheet：每个 replan 一张 7x14 图，观察热点是否移动
    for f in range(3):
        frame_heat = heatmaps[:, f]
        n = frame_heat.shape[0]
        cols = min(6, n)
        rows_n = int(np.ceil(n / cols))

        vmin = frame_heat.min()
        vmax = frame_heat.max()

        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3, rows_n * 2.2))
        axes = np.array(axes).reshape(rows_n, cols)

        for i in range(rows_n * cols):
            ax = axes.flat[i]
            ax.axis("off")
            if i < n:
                im = ax.imshow(frame_heat[i], vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_title(f"replan {replans[i]}", fontsize=9)

        fig.suptitle(f"Spatial attention contact sheet, frame {f}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / f"replan_spatial_contact_sheet_frame{f}.png", dpi=200)
        plt.close(fig)

    print("saved to:", out_dir)


if __name__ == "__main__":
    main()