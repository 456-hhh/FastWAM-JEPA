import contextlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_MODULE_STACK: list[str] = []


def _make_jsonable(x: Any):
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_make_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _make_jsonable(v) for k, v in x.items()}
    return str(x)


class SDPAAttentionCollector:
    """
    Capture attention information from torch.nn.functional.scaled_dot_product_attention.

    Modes:
      inventory: only save metadata, no attention matrix
      full: save metadata + selected attention matrices

    Selection:
      To avoid huge memory, by default only save attention matrices when q_len <= max_q_len.
      This is suitable for action-query attention, because action query length is usually small.
    """

    def __init__(
        self,
        out_dir: str | Path,
        tag: str,
        mode: str = "full",
        max_q_len: int = 128,
        max_k_len: int = 20000,
        max_matrix_elements: int = 8_000_000,
        save_headwise: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.tag = tag
        self.mode = mode
        self.max_q_len = int(max_q_len)
        self.max_k_len = int(max_k_len)
        self.max_matrix_elements = int(max_matrix_elements)
        self.save_headwise = bool(save_headwise)

        self.call_idx = 0
        self.records = []

        self.run_dir = self.out_dir / self.tag
        self.matrix_dir = self.run_dir / "matrices"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.matrix_dir.mkdir(parents=True, exist_ok=True)

    def _should_save_matrix(self, q_len: int, k_len: int) -> bool:
        if self.mode != "full":
            return False
        if q_len > self.max_q_len:
            return False
        if k_len > self.max_k_len:
            return False
        if q_len * k_len > self.max_matrix_elements:
            return False
        return True

    @torch.no_grad()
    def record(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_mask=None,
        is_causal: bool = False,
        scale=None,
        module_name: str | None = None,
    ):
        self.call_idx += 1

        q_shape = tuple(query.shape)
        k_shape = tuple(key.shape)

        q_len = int(query.shape[-2])
        k_len = int(key.shape[-2])
        q_heads = int(query.shape[-3]) if query.ndim >= 4 else None
        k_heads = int(key.shape[-3]) if key.ndim >= 4 else None
        head_dim = int(query.shape[-1])

        record = {
            "call_idx": self.call_idx,
            "module": module_name,
            "q_shape": q_shape,
            "k_shape": k_shape,
            "q_len": q_len,
            "k_len": k_len,
            "q_heads": q_heads,
            "k_heads": k_heads,
            "head_dim": head_dim,
            "is_causal": bool(is_causal),
            "saved_matrix": False,
            "matrix_path": None,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        should_save = self._should_save_matrix(q_len, k_len)

        if should_save:
            q = query.detach().float()
            k = key.detach().float()

            # Handle GQA/MQA-like head mismatch for visualization only.
            if q.ndim >= 4 and k.ndim >= 4 and q.shape[-3] != k.shape[-3]:
                if q.shape[-3] % k.shape[-3] == 0:
                    repeat = q.shape[-3] // k.shape[-3]
                    k = k.repeat_interleave(repeat, dim=-3)

            scale_factor = (1.0 / math.sqrt(head_dim)) if scale is None else float(scale)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

            if is_causal:
                causal_mask = torch.ones(
                    (q_len, k_len),
                    device=scores.device,
                    dtype=torch.bool,
                ).tril()
                scores = scores.masked_fill(~causal_mask, float("-inf"))

            if attn_mask is not None:
                mask = attn_mask
                if isinstance(mask, torch.Tensor):
                    mask = mask.to(device=scores.device)
                    if mask.dtype == torch.bool:
                        scores = scores.masked_fill(~mask, float("-inf"))
                    else:
                        scores = scores + mask

            attn = torch.softmax(scores, dim=-1)

            # query/key attention, averaged over batch and heads: [q_len, k_len]
            if attn.ndim == 4:
                attn_mean = attn.mean(dim=(0, 1))
            elif attn.ndim == 3:
                attn_mean = attn.mean(dim=0)
            else:
                attn_mean = attn

            matrix_path = self.matrix_dir / f"call_{self.call_idx:04d}_q{q_len}_k{k_len}.npy"
            np.save(matrix_path, attn_mean.cpu().numpy().astype(np.float32))

            record["saved_matrix"] = True
            record["matrix_path"] = str(matrix_path)

            if self.save_headwise and attn.ndim == 4:
                # [heads, q, k], averaged over batch only
                headwise = attn.mean(dim=0).cpu().numpy().astype(np.float32)
                headwise_path = self.matrix_dir / f"call_{self.call_idx:04d}_headwise.npy"
                np.save(headwise_path, headwise)
                record["headwise_matrix_path"] = str(headwise_path)

        self.records.append(record)

    def close(self):
        meta_path = self.run_dir / "records.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(_make_jsonable(r), ensure_ascii=False) + "\n")

        summary = {
            "tag": self.tag,
            "num_calls": len(self.records),
            "num_saved_matrices": sum(1 for r in self.records if r["saved_matrix"]),
            "out_dir": str(self.run_dir),
        }

        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("[Attention] saved records:", meta_path)
        print("[Attention] summary:", summary)


def _register_module_stack(model: torch.nn.Module):
    handles = []

    for name, module in model.named_modules():
        def pre_hook(mod, inputs, name=name):
            _MODULE_STACK.append(name)

        def post_hook(mod, inputs, output, name=name):
            if len(_MODULE_STACK) > 0:
                _MODULE_STACK.pop()

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    return handles


@contextlib.contextmanager
def capture_sdpa_attention(
    model: torch.nn.Module,
    out_dir: str | Path,
    tag: str,
    mode: str | None = None,
    max_q_len: int | None = None,
    max_k_len: int | None = None,
    max_matrix_elements: int | None = None,
    save_headwise: bool | None = None,
):
    """
    Usage:
        with capture_sdpa_attention(model, out_dir, tag):
            pred = model.infer_action(...)
    """

    mode = mode or os.environ.get("FASTWAM_ATTN_MODE", "full")
    max_q_len = int(max_q_len or os.environ.get("FASTWAM_ATTN_MAX_Q", 128))
    max_k_len = int(max_k_len or os.environ.get("FASTWAM_ATTN_MAX_K", 20000))
    max_matrix_elements = int(
        max_matrix_elements or os.environ.get("FASTWAM_ATTN_MAX_MATRIX", 8000000)
    )
    save_headwise = bool(int(os.environ.get("FASTWAM_ATTN_SAVE_HEADWISE", "0"))) if save_headwise is None else save_headwise

    collector = SDPAAttentionCollector(
        out_dir=out_dir,
        tag=tag,
        mode=mode,
        max_q_len=max_q_len,
        max_k_len=max_k_len,
        max_matrix_elements=max_matrix_elements,
        save_headwise=save_headwise,
    )

    original_sdpa = F.scaled_dot_product_attention
    handles = _register_module_stack(model)

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        out = original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

        module_name = _MODULE_STACK[-1] if len(_MODULE_STACK) > 0 else None

        try:
            collector.record(
                query=query,
                key=key,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=scale,
                module_name=module_name,
            )
        except Exception as e:
            print(f"[Attention][warning] failed to record call {collector.call_idx + 1}: {e}")

        return out

    F.scaled_dot_product_attention = patched_sdpa

    try:
        yield collector
    finally:
        F.scaled_dot_product_attention = original_sdpa
        for h in handles:
            h.remove()
        collector.close()