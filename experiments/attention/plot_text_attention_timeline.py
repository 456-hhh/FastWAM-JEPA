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


def entropy_from_score(x):
    p = x / max(float(x.sum()), 1e-12)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--num_layers", type=int, default=30)
    parser.add_argument("--text_k_len", type=int, default=129)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    root = Path(args.dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / "text_attention_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(root.glob("task*_replan*"), key=replan_idx)

    replan_text = []
    top_rows = []
    summary_rows = []
    step_text_all = []

    for run in run_dirs:
        idx = replan_idx(run)
        records = load_records(run)

        text_records = [
            r for r in records
            if r.get("saved_matrix")
            and int(r["q_len"]) == 32
            and int(r["k_len"]) == args.text_k_len
            and "action_expert" in str(r.get("module"))
            and "cross_attn" in str(r.get("module"))
        ]

        if not text_records:
            continue

        mats = []
        for r in text_records:
            mats.append(np.load(r["matrix_path"]))  # [32, 129]

        mats = np.stack(mats, axis=0)  # [N, 32, 129]

        token_score = mats.mean(axis=(0, 1))  # [129]
        replan_text.append(token_score)

        top_idx = np.argsort(-token_score)[:10]

        summary_rows.append({
            "replan_idx": idx,
            "num_text_records": len(text_records),
            "top1_token_index": int(top_idx[0]),
            "top1_attention": float(token_score[top_idx[0]]),
            "text_attention_max": float(token_score.max()),
            "text_attention_entropy": entropy_from_score(token_score),
        })

        for rank, tid in enumerate(top_idx):
            top_rows.append({
                "replan_idx": idx,
                "rank": rank,
                "text_token_index": int(tid),
                "attention": float(token_score[tid]),
            })

        n = len(text_records)
        if n >= args.num_layers:
            num_steps = n // args.num_layers
            step_scores = []
            for s in range(num_steps):
                sub = mats[s * args.num_layers:(s + 1) * args.num_layers]
                step_scores.append(sub.mean(axis=(0, 1)))
            step_text_all.append(np.stack(step_scores, axis=0))

    if not replan_text:
        raise RuntimeError(f"No action text cross-attention found in {root}")

    arr = np.stack(replan_text, axis=0)  # [R, 129]

    np.save(out_dir / "replan_text_attention.npy", arr)
    pd.DataFrame(summary_rows).to_csv(out_dir / "text_attention_summary_by_replan.csv", index=False)
    pd.DataFrame(top_rows).to_csv(out_dir / "top_text_tokens_by_replan.csv", index=False)

    plt.figure(figsize=(14, 5))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="attention")
    plt.xlabel("text/context token index")
    plt.ylabel("replan index")
    plt.title("Action → text/context attention over replans")
    plt.tight_layout()
    plt.savefig(out_dir / "replan_text_attention_heatmap.png", dpi=200)
    plt.close()

    r0 = 0
    r1 = len(arr) // 2
    r2 = len(arr) - 1

    plt.figure(figsize=(12, 4))
    plt.plot(arr[r0], label=f"early replan {r0}")
    plt.plot(arr[r1], label=f"middle replan {r1}")
    plt.plot(arr[r2], label=f"late replan {r2}")
    plt.xlabel("text/context token index")
    plt.ylabel("attention")
    plt.title("Early / middle / late action → text attention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "early_middle_late_text_attention_curves.png", dpi=200)
    plt.close()

    top_global = np.argsort(-arr.mean(axis=0))[:10]
    plt.figure(figsize=(10, 5))
    for tid in top_global:
        plt.plot(np.arange(arr.shape[0]), arr[:, tid], label=f"token {tid}")
    plt.xlabel("replan index")
    plt.ylabel("attention")
    plt.title("Top text-token attention over replans")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "top_text_tokens_over_replans.png", dpi=200)
    plt.close()

    if step_text_all:
        min_steps = min(x.shape[0] for x in step_text_all)
        step_stack = np.stack([x[:min_steps] for x in step_text_all], axis=0)
        step_avg = step_stack.mean(axis=0)  # [steps, 129]

        np.save(out_dir / "stepwise_text_attention_avg.npy", step_avg)

        plt.figure(figsize=(14, 5))
        plt.imshow(step_avg, aspect="auto")
        plt.colorbar(label="attention")
        plt.xlabel("text/context token index")
        plt.ylabel("denoising step")
        plt.title("Step-wise action → text attention, averaged over replans")
        plt.tight_layout()
        plt.savefig(out_dir / "stepwise_text_attention_heatmap.png", dpi=200)
        plt.close()

    print("saved to", out_dir)


if __name__ == "__main__":
    main()
