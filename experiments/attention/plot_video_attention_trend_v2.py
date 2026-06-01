import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def replan_idx(p: Path):
    m = re.search(r"replan(\d+)", p.name)
    return int(m.group(1)) if m else -1


def topk_mass_ratio(heat, ratio):
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


def cosine_sim(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / denom)


def make_overlay(rgb, heat, alpha=0.45):
    h, w = rgb.shape[:2]

    heat = heat.astype(np.float32)
    heat = heat - heat.min()
    heat = heat / max(heat.max(), 1e-12)

    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heat_arr = np.asarray(heat_img).astype(np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    color_heat = cmap(heat_arr)[..., :3]
    rgb_float = rgb.astype(np.float32) / 255.0

    out = (1 - alpha) * rgb_float + alpha * color_heat
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def load_rgb_files(rgb_dir: Path):
    if not rgb_dir.exists():
        return []
    files = sorted(
        rgb_dir.glob("*replan*_rgb.png"),
        key=lambda p: replan_idx(p)
    )
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--grid_h", type=int, default=7)
    parser.add_argument("--grid_w", type=int, default=14)
    parser.add_argument("--cols", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sem_dirs = sorted(
        root.glob("task*_replan*/semantic_plots"),
        key=lambda p: replan_idx(p.parent),
    )

    if not sem_dirs:
        raise RuntimeError(f"No semantic_plots found under {root}")

    replans = []
    heatmaps = []
    rows = []

    for sem in sem_dirs:
        idx = replan_idx(sem.parent)
        path = sem / "action_to_spatial_by_call.npy"
        if not path.exists():
            continue

        spatial = np.load(path)  # [calls, 3, 98]
        spatial_mean = spatial.mean(axis=0)  # [3, 98]
        spatial_heat = spatial_mean.reshape(3, args.grid_h, args.grid_w)  # [3, 7, 14]

        replans.append(idx)
        heatmaps.append(spatial_heat)

        for f in range(3):
            heat = spatial_heat[f]
            total = heat.sum()
            cx, cy = center_of_mass(heat)

            left = heat[:, : args.grid_w // 2].sum()
            right = heat[:, args.grid_w // 2 :].sum()
            top = heat[: args.grid_h // 2, :].sum()
            bottom = heat[args.grid_h // 2 :, :].sum()

            rows.append({
                "replan_idx": idx,
                "frame": f,
                "total_mass": float(total),
                "center_x": cx,
                "center_y": cy,
                "left_mass": float(left),
                "right_mass": float(right),
                "left_ratio": float(left / max(total, 1e-12)),
                "right_ratio": float(right / max(total, 1e-12)),
                "top_mass": float(top),
                "bottom_mass": float(bottom),
                "top_ratio": float(top / max(total, 1e-12)),
                "bottom_ratio": float(bottom / max(total, 1e-12)),
                "top10pct_mass_ratio": topk_mass_ratio(heat, 0.10),
                "top25pct_mass_ratio": topk_mass_ratio(heat, 0.25),
            })

    if not heatmaps:
        raise RuntimeError(f"No spatial heatmaps loaded from {root}")

    heatmaps = np.stack(heatmaps, axis=0)  # [R, 3, 7, 14]
    replans = np.asarray(replans)

    np.save(out_dir / "replan_spatial_heatmaps.npy", heatmaps)

    df = pd.DataFrame(rows).sort_values(["frame", "replan_idx"])
    df.to_csv(out_dir / "replan_spatial_metrics.csv", index=False)

    # adjacent similarity / movement
    move_rows = []
    for f in range(3):
        sub = df[df["frame"] == f].sort_values("replan_idx").reset_index(drop=True)
        for i in range(1, len(sub)):
            h0 = heatmaps[i - 1, f]
            h1 = heatmaps[i, f]
            dx = sub.loc[i, "center_x"] - sub.loc[i - 1, "center_x"]
            dy = sub.loc[i, "center_y"] - sub.loc[i - 1, "center_y"]
            move_rows.append({
                "frame": f,
                "prev_replan": int(sub.loc[i - 1, "replan_idx"]),
                "replan_idx": int(sub.loc[i, "replan_idx"]),
                "cosine_similarity": cosine_sim(h0, h1),
                "center_displacement": float(np.sqrt(dx * dx + dy * dy)),
                "dx": float(dx),
                "dy": float(dy),
            })
    pd.DataFrame(move_rows).to_csv(out_dir / "replan_spatial_change_metrics.csv", index=False)

    # 1. contact sheet per frame, fixed scale within each frame
    for f in range(3):
        frame_heat = heatmaps[:, f]
        n = frame_heat.shape[0]
        cols = min(args.cols, n)
        rows_n = int(np.ceil(n / cols))

        vmin = np.percentile(frame_heat, 1)
        vmax = np.percentile(frame_heat, 99)

        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.2, rows_n * 2.35))
        axes = np.asarray(axes).reshape(rows_n, cols)

        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < n:
                ax.imshow(frame_heat[i], aspect="auto", vmin=vmin, vmax=vmax)
                ax.set_title(f"replan {int(replans[i])}", fontsize=9)

        fig.suptitle(f"Spatial attention contact sheet, latent frame {f}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / f"replan_spatial_contact_sheet_frame{f}.png", dpi=200)
        plt.close(fig)

    # 2. flattened heatmap over replans
    for f in range(3):
        flat = heatmaps[:, f].reshape(len(heatmaps), -1)

        plt.figure(figsize=(14, max(4, 0.35 * len(replans))))
        plt.imshow(flat, aspect="auto")
        plt.colorbar(label="attention")
        plt.xlabel("spatial token within frame, 0..97")
        plt.ylabel("replan index")
        plt.yticks(np.arange(len(replans)), replans)
        plt.title(f"Spatial attention over replans, latent frame {f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_spatial_flat_frame{f}.png", dpi=200)
        plt.close()

    # 3. center path and center x/y curves
    for f in range(3):
        sub = df[df["frame"] == f].sort_values("replan_idx")

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["center_x"], marker="o", label="center_x")
        plt.plot(sub["replan_idx"], sub["center_y"], marker="o", label="center_y")
        plt.xlabel("replan index")
        plt.ylabel("center of mass on 7×14 grid")
        plt.title(f"Attention center over replans, frame {f}")
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
        plt.title(f"Attention center path, frame {f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_attention_center_path_frame{f}.png", dpi=200)
        plt.close()

    # 4. camera half / top-bottom / concentration
    for f in range(3):
        sub = df[df["frame"] == f].sort_values("replan_idx")

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["left_ratio"], marker="o", label="left half / third-person")
        plt.plot(sub["replan_idx"], sub["right_ratio"], marker="o", label="right half / wrist")
        plt.xlabel("replan index")
        plt.ylabel("attention ratio")
        plt.title(f"Left/right camera attention ratio, frame {f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_left_right_camera_ratio_frame{f}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["top_ratio"], marker="o", label="top")
        plt.plot(sub["replan_idx"], sub["bottom_ratio"], marker="o", label="bottom")
        plt.xlabel("replan index")
        plt.ylabel("attention ratio")
        plt.title(f"Top/bottom attention ratio, frame {f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_top_bottom_ratio_frame{f}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["top10pct_mass_ratio"], marker="o", label="top 10% spatial tokens")
        plt.plot(sub["replan_idx"], sub["top25pct_mass_ratio"], marker="o", label="top 25% spatial tokens")
        plt.xlabel("replan index")
        plt.ylabel("attention mass ratio")
        plt.title(f"Spatial concentration over replans, frame {f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_spatial_concentration_frame{f}.png", dpi=200)
        plt.close()

    # 5. spatial change metrics
    change_df = pd.read_csv(out_dir / "replan_spatial_change_metrics.csv")
    for f in range(3):
        sub = change_df[change_df["frame"] == f].sort_values("replan_idx")
        if len(sub) == 0:
            continue

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["center_displacement"], marker="o")
        plt.xlabel("replan index")
        plt.ylabel("COM displacement")
        plt.title(f"Attention center movement between replans, frame {f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_center_displacement_frame{f}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(sub["replan_idx"], sub["cosine_similarity"], marker="o")
        plt.xlabel("replan index")
        plt.ylabel("adjacent heatmap cosine similarity")
        plt.title(f"Spatial attention stability between replans, frame {f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"replan_spatial_similarity_frame{f}.png", dpi=200)
        plt.close()

    # 6. RGB overlay for frame0 if replan RGB exists
    rgb_files = sorted((root / "replan_rgb").glob("*replan*_rgb.png"), key=lambda p: replan_idx(p))
    if rgb_files:
        n = min(len(rgb_files), len(heatmaps))
        cols = min(4, n)
        rows_n = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.2, rows_n * 2.6))
        axes = np.asarray(axes).reshape(rows_n, cols)

        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < n:
                rgb = np.asarray(Image.open(rgb_files[i]).convert("RGB"))
                overlay = make_overlay(rgb, heatmaps[i, 0])
                ax.imshow(overlay)
                ax.set_title(f"replan {int(replans[i])}", fontsize=9)

        fig.suptitle("Current RGB + frame0 attention overlay", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / "replan_rgb_overlay_frame0_current.png", dpi=200)
        plt.close(fig)

    print("saved video trend to:", out_dir)


if __name__ == "__main__":
    main()
