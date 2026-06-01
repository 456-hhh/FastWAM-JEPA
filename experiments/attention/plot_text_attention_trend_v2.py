import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def replan_idx(p: Path):
    m = re.search(r"replan(\d+)", p.name)
    return int(m.group(1)) if m else -1


def load_records(run_dir: Path):
    with open(run_dir / "records.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def resolve_matrix_path(run_dir: Path, matrix_path: str):
    p = Path(matrix_path)
    if p.exists():
        return p
    p2 = run_dir / matrix_path
    if p2.exists():
        return p2
    p3 = Path.cwd() / matrix_path
    if p3.exists():
        return p3
    raise FileNotFoundError(matrix_path)


def entropy(x):
    p = x / max(float(x.sum()), 1e-12)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def auto_detect_valid_end(arr, min_prefix=8, min_tail=20):
    """
    arr: [R, K], replan-level text attention.
    Detects a flat high tail like token 30..127 in your current plots.
    This is a heuristic, not true tokenizer masking.
    """
    k = arr.shape[1]
    mean = arr.mean(axis=0)
    temporal_std = arr.std(axis=0)
    diffs = np.abs(np.diff(mean))

    candidates = []
    for i in range(min_prefix, k - min_tail):
        suffix = mean[i:]
        suffix_temporal = temporal_std[i:]

        suffix_cv = float(suffix.std() / max(abs(suffix.mean()), 1e-12))
        suffix_temp = float(suffix_temporal.mean())
        boundary_jump = float(abs(mean[i] - mean[i - 1]))

        # flat suffix + small temporal variation + visible transition
        score = suffix_cv + 10.0 * suffix_temp - 0.1 * boundary_jump
        if suffix_cv < 0.08 and suffix_temp < 2e-4:
            candidates.append((score, i, suffix_cv, suffix_temp, boundary_jump))

    if candidates:
        candidates.sort()
        return int(candidates[0][1])

    # fallback: look for large transition after early tokens
    if len(diffs) > min_prefix:
        i = int(np.argmax(diffs[min_prefix:]) + min_prefix + 1)
        if i < k - min_tail:
            return i

    return k


def select_text_records(records, text_k_len):
    selected = []
    for r in records:
        if not r.get("saved_matrix"):
            continue
        if int(r.get("q_len", -1)) != 32:
            continue
        if int(r.get("k_len", -1)) != text_k_len:
            continue
        module = str(r.get("module", ""))
        if "action_expert" not in module:
            continue
        if "cross_attn" not in module:
            continue
        selected.append(r)
    return sorted(selected, key=lambda x: int(x["call_idx"]))


def plot_heatmap(arr, out_path, title, xlabel="text/context token index", ylabel="replan index", ytick_labels=None):
    plt.figure(figsize=(14, max(4, 0.35 * arr.shape[0])))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="attention")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ytick_labels is not None:
        plt.yticks(np.arange(len(ytick_labels)), ytick_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_layers", type=int, default=30)
    parser.add_argument("--text_k_len", type=int, default=129)
    parser.add_argument("--valid_end", default="auto", help="'auto' or an integer, e.g. 30")
    parser.add_argument("--top_n_variable", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(root.glob("task*_replan*"), key=replan_idx)

    replans = []
    replan_text = []
    step_text_all = []
    summary_rows = []

    for run in run_dirs:
        idx = replan_idx(run)
        records = load_records(run)
        text_records = select_text_records(records, args.text_k_len)

        if not text_records:
            continue

        mats = []
        for r in text_records:
            mat_path = resolve_matrix_path(run, r["matrix_path"])
            mats.append(np.load(mat_path))  # [32, 129]

        mats = np.stack(mats, axis=0)  # [N, 32, 129]
        score = mats.mean(axis=(0, 1))  # [129]

        replans.append(idx)
        replan_text.append(score)

        summary_rows.append({
            "replan_idx": idx,
            "num_text_records": len(text_records),
            "text_entropy_full": entropy(score),
            "top1_token_full": int(score.argmax()),
            "top1_attention_full": float(score.max()),
        })

        n = len(text_records)
        if n >= args.num_layers:
            num_steps = n // args.num_layers
            step_scores = []
            for s in range(num_steps):
                sub = mats[s * args.num_layers:(s + 1) * args.num_layers]
                step_scores.append(sub.mean(axis=(0, 1)))
            step_text_all.append(np.stack(step_scores, axis=0))  # [steps, 129]

    if not replan_text:
        raise RuntimeError(f"No action text cross-attention found in {root}")

    arr = np.stack(replan_text, axis=0)  # [R, 129]
    replans = np.asarray(replans)

    np.save(out_dir / "replan_text_attention_full.npy", arr)

    if args.valid_end == "auto":
        valid_end = auto_detect_valid_end(arr)
    else:
        valid_end = int(args.valid_end)

    valid_end = max(1, min(valid_end, arr.shape[1]))
    valid_idx = np.arange(valid_end)
    arr_valid = arr[:, valid_idx]

    np.save(out_dir / "replan_text_attention_valid.npy", arr_valid)

    # token diagnostics
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    ptp = arr.max(axis=0) - arr.min(axis=0)

    diag = pd.DataFrame({
        "token_index": np.arange(arr.shape[1]),
        "mean_attention": mean,
        "std_over_replans": std,
        "peak_to_peak_over_replans": ptp,
        "is_valid_by_auto_filter": np.arange(arr.shape[1]) < valid_end,
    })
    diag.to_csv(out_dir / "text_token_diagnostics.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["valid_end"] = valid_end
    summary_df.to_csv(out_dir / "text_attention_summary_by_replan_v2.csv", index=False)

    # full heatmap
    plot_heatmap(
        arr,
        out_dir / "replan_text_attention_heatmap_full_129.png",
        "Action → text/context attention over replans, full 129 tokens",
        ytick_labels=replans,
    )

    # valid heatmap
    plot_heatmap(
        arr_valid,
        out_dir / "replan_text_attention_heatmap_valid_tokens.png",
        f"Action → text/context attention over replans, filtered tokens [0:{valid_end}]",
        xlabel=f"filtered token index, original 0..{valid_end-1}",
        ytick_labels=replans,
    )

    # early / middle / late valid tokens
    r0 = 0
    r1 = len(arr_valid) // 2
    r2 = len(arr_valid) - 1

    plt.figure(figsize=(10, 4))
    plt.plot(valid_idx, arr_valid[r0], label=f"early replan {int(replans[r0])}")
    plt.plot(valid_idx, arr_valid[r1], label=f"middle replan {int(replans[r1])}")
    plt.plot(valid_idx, arr_valid[r2], label=f"late replan {int(replans[r2])}")
    plt.xlabel("text/context token index after filtering")
    plt.ylabel("attention")
    plt.title(f"Early / middle / late action → text attention, valid tokens [0:{valid_end}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "early_middle_late_text_attention_valid_tokens.png", dpi=200)
    plt.close()

    # variable tokens over replans
    valid_std = arr_valid.std(axis=0)
    top_var_local = np.argsort(-valid_std)[: min(args.top_n_variable, len(valid_std))]
    top_var_tokens = valid_idx[top_var_local]

    top_rows = []
    for rank, tid in enumerate(top_var_tokens):
        top_rows.append({
            "rank_by_variation": rank,
            "token_index": int(tid),
            "mean_attention": float(mean[tid]),
            "std_over_replans": float(std[tid]),
            "peak_to_peak_over_replans": float(ptp[tid]),
        })
    pd.DataFrame(top_rows).to_csv(out_dir / "top_variable_text_tokens.csv", index=False)

    plt.figure(figsize=(10, 5))
    for tid in top_var_tokens:
        plt.plot(replans, arr[:, tid], marker="o", label=f"token {int(tid)}")
    plt.xlabel("replan index")
    plt.ylabel("attention")
    plt.title("Most dynamic valid text/context tokens over replans")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "top_variable_text_tokens_over_replans.png", dpi=200)
    plt.close()

    # early-mid-late difference
    early = arr_valid[r0]
    middle = arr_valid[r1]
    late = arr_valid[r2]

    plt.figure(figsize=(10, 4))
    plt.plot(valid_idx, middle - early, marker="o", label="middle - early")
    plt.plot(valid_idx, late - early, marker="o", label="late - early")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("valid token index")
    plt.ylabel("attention difference")
    plt.title("Phase difference of text attention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phase_difference_text_attention_valid_tokens.png", dpi=200)
    plt.close()

    # stepwise, both full and valid
    if step_text_all:
        min_steps = min(x.shape[0] for x in step_text_all)
        step_stack = np.stack([x[:min_steps] for x in step_text_all], axis=0)
        step_avg = step_stack.mean(axis=0)  # [steps, 129]

        np.save(out_dir / "stepwise_text_attention_avg_full.npy", step_avg)
        np.save(out_dir / "stepwise_text_attention_avg_valid.npy", step_avg[:, valid_idx])

        plot_heatmap(
            step_avg,
            out_dir / "stepwise_text_attention_heatmap_full_129.png",
            "Step-wise action → text attention, full 129 tokens",
            ylabel="denoising step",
        )

        plot_heatmap(
            step_avg[:, valid_idx],
            out_dir / "stepwise_text_attention_heatmap_valid_tokens.png",
            f"Step-wise action → text attention, filtered tokens [0:{valid_end}]",
            xlabel=f"filtered token index, original 0..{valid_end-1}",
            ylabel="denoising step",
        )

    # plain text note
    with open(out_dir / "README_text_attention_v2.txt", "w", encoding="utf-8") as f:
        f.write(
            "This analysis filters likely padding/null/context tail tokens using a heuristic.\n"
            f"Detected valid token end: {valid_end}\n"
            "Full 129-token plots are also saved for reference.\n"
            "For word-level interpretation, token_index -> actual text token mapping is still required.\n"
        )

    print("saved text trend to:", out_dir)
    print("detected valid_end =", valid_end)


if __name__ == "__main__":
    main()
