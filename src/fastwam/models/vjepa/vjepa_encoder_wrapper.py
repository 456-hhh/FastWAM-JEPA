from __future__ import annotations

import torch
import torch.nn as nn


class VJepaEncoderWrapper(nn.Module):
    """Shape-compatible V-JEPA encoder wrapper for FastWAM-JEPA v1.

    v1 only supports dummy mode:
        input video:  [B, 3, T, H_img, W_img]
        output tokens: [B, num_tokens, vjepa_dim]
    """

    def __init__(
        self,
        *,
        dummy: bool = True,
        num_tokens: int = 256,
        vjepa_dim: int = 1408,
        freeze: bool = True,
        normalize_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.dummy = bool(dummy)
        self.num_tokens = int(num_tokens)
        self.vjepa_dim = int(vjepa_dim)
        self.freeze = bool(freeze)
        self.normalize_tokens = bool(normalize_tokens)

        if self.num_tokens <= 0:
            raise ValueError(f"`num_tokens` must be positive, got {self.num_tokens}.")
        if self.vjepa_dim <= 0:
            raise ValueError(f"`vjepa_dim` must be positive, got {self.vjepa_dim}.")
        if not self.dummy:
            # TODO: Integrate real frozen V-JEPA2 encoder loading on the server.
            # This v1 local wrapper intentionally does not import external/vjepa2,
            # load checkpoints, use the network, or require CUDA.
            raise NotImplementedError("Real V-JEPA2 encoder mode is TODO; use dummy=True for v1.")

        if self.freeze:
            self.requires_grad_(False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video into dummy V-JEPA feature tokens.

        Args:
            video: RGB video tensor with shape [B, 3, T, H_img, W_img].

        Returns:
            Dummy V-JEPA tokens with shape [B, num_tokens, vjepa_dim].
            The returned tensor follows the input video's device and dtype.
        """
        if video.ndim != 5:
            raise ValueError(
                "`video` must be 5D with shape [B, 3, T, H_img, W_img], "
                f"got shape {tuple(video.shape)}."
            )
        if video.shape[1] != 3:
            raise ValueError(
                "`video` channel dimension must be 3 for RGB input, "
                f"got {video.shape[1]}."
            )

        batch_size = int(video.shape[0])
        tokens = torch.zeros(
            (batch_size, self.num_tokens, self.vjepa_dim),
            device=video.device,
            dtype=video.dtype,
        )

        if self.normalize_tokens:
            tokens = torch.nn.functional.layer_norm(tokens, (self.vjepa_dim,))
        return tokens
