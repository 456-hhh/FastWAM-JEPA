from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VJepaEncoderWrapper(nn.Module):
    """V-JEPA encoder wrapper for FastWAM-JEPA.

    Shape contract:
        input video:   [B, 3, T, H_img, W_img]
        output tokens: [B, N, D_v]

    In dummy mode, N is `num_tokens` and D_v is `vjepa_dim`.
    In real mode, N comes from the official V-JEPA2 encoder tokenization.
    """

    def __init__(
        self,
        *,
        dummy: bool = True,
        num_tokens: int = 256,
        vjepa_dim: int = 1408,
        freeze: bool = True,
        normalize_tokens: bool = False,
        model_name: str = "vjepa2_vit_giant",
        external_repo_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_key: Optional[str] = None,
        pretrained: bool = False,
        img_size: int = 256,
        input_range: str = "-1_1",
        tubelet_size: int = 2,
        frame_encoding_mode: str = "clip_or_repeat",
    ) -> None:
        super().__init__()
        self.dummy = bool(dummy)
        self.num_tokens = int(num_tokens)
        self.vjepa_dim = int(vjepa_dim)
        self.freeze = bool(freeze)
        self.normalize_tokens = bool(normalize_tokens)
        self.model_name = str(model_name)
        self.external_repo_path = external_repo_path
        self.checkpoint_path = checkpoint_path
        self.checkpoint_key = checkpoint_key
        self.pretrained = bool(pretrained)
        self.img_size = int(img_size)
        self.input_range = str(input_range)
        self.tubelet_size = int(tubelet_size)
        self.frame_encoding_mode = str(frame_encoding_mode)
        self.encoder: Optional[nn.Module] = None

        if self.num_tokens <= 0:
            raise ValueError(f"`num_tokens` must be positive, got {self.num_tokens}.")
        if self.vjepa_dim <= 0:
            raise ValueError(f"`vjepa_dim` must be positive, got {self.vjepa_dim}.")

        if self.dummy:
            if self.freeze:
                self.requires_grad_(False)
            return

        self._validate_real_config()
        self.encoder = self._load_real_encoder()

        if self.checkpoint_path is not None:
            self._load_checkpoint(self.checkpoint_path)

        self.encoder.eval()

        if self.freeze:
            self.requires_grad_(False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video into V-JEPA feature tokens.

        Args:
            video: RGB video tensor with shape [B, 3, T, H_img, W_img].

        Returns:
            V-JEPA tokens with shape [B, N, D_v].
            In dummy mode, the returned tensor follows the input video's device
            and dtype.
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
        if self.dummy:
            tokens = torch.zeros(
                (batch_size, self.num_tokens, self.vjepa_dim),
                device=video.device,
                dtype=video.dtype,
            )

            if self.normalize_tokens:
                tokens = F.layer_norm(tokens, (self.vjepa_dim,))
            return tokens

        if self.encoder is None:
            raise RuntimeError("Real V-JEPA2 encoder was not initialized.")

        preprocessed = self._preprocess_real_video(video)
        preprocessed = self._adapt_temporal(preprocessed)
        preprocessed = self._move_video_to_encoder(preprocessed)

        with torch.set_grad_enabled(not self.freeze):
            tokens = self.encoder(preprocessed)

        tokens = self._validate_real_output(tokens, batch_size=batch_size)
        if self.normalize_tokens:
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))
        return tokens

    def _validate_real_config(self) -> None:
        if self.pretrained:
            raise ValueError(
                "`pretrained=True` may trigger automatic checkpoint downloads. "
                "Use `pretrained=False` and pass a local `checkpoint_path` instead."
            )
        if self.external_repo_path is None:
            raise ValueError("`external_repo_path` is required when `dummy=False`.")
        if self.img_size <= 0:
            raise ValueError(f"`img_size` must be positive, got {self.img_size}.")
        if self.input_range not in {"-1_1", "0_1"}:
            raise ValueError(
                "`input_range` must be either '-1_1' or '0_1', "
                f"got {self.input_range!r}."
            )
        if self.tubelet_size <= 0:
            raise ValueError(
                f"`tubelet_size` must be positive, got {self.tubelet_size}."
            )
        if self.frame_encoding_mode != "clip_or_repeat":
            raise ValueError(
                "`frame_encoding_mode` currently only supports 'clip_or_repeat', "
                f"got {self.frame_encoding_mode!r}."
            )

        repo_path = Path(self.external_repo_path)
        if not repo_path.exists():
            raise ValueError(
                f"`external_repo_path` does not exist: {repo_path.as_posix()}."
            )
        if not (repo_path / "hubconf.py").exists():
            raise ValueError(
                "`external_repo_path` must point to a local V-JEPA2 repo with "
                f"hubconf.py, got {repo_path.as_posix()}."
            )

    def _load_real_encoder(self) -> nn.Module:
        repo_path = Path(self.external_repo_path)  # type: ignore[arg-type]
        loaded = torch.hub.load(
            repo_path.as_posix(),
            self.model_name,
            source="local",
            pretrained=False,
            tubelet_size=self.tubelet_size,
        )

        encoder = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        if not isinstance(encoder, nn.Module):
            raise ValueError(
                "Official V-JEPA2 hub loader must return an encoder nn.Module "
                f"or a tuple/list whose first item is an nn.Module, got {type(loaded)!r}."
            )

        encoder_dim = getattr(encoder, "embed_dim", None)
        if encoder_dim is not None and int(encoder_dim) != self.vjepa_dim:
            raise ValueError(
                "Loaded V-JEPA2 encoder feature dimension does not match "
                f"`vjepa_dim`: encoder embed_dim={int(encoder_dim)}, "
                f"vjepa_dim={self.vjepa_dim}."
            )
        return encoder

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if self.encoder is None:
            raise RuntimeError("Cannot load a checkpoint before the encoder exists.")

        path = Path(checkpoint_path)
        if not path.exists():
            raise ValueError(f"`checkpoint_path` does not exist: {path.as_posix()}.")

        checkpoint = torch.load(path.as_posix(), map_location="cpu")
        state_dict, source_key = self._select_checkpoint_state_dict(checkpoint)
        state_dict = self._strip_state_dict_prefixes(state_dict)

        load_result = self.encoder.load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        print(
            "Loaded V-JEPA2 encoder checkpoint "
            f"from {path.as_posix()} using key {source_key!r}."
        )
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    def _select_checkpoint_state_dict(self, checkpoint: Any) -> tuple[dict[str, Any], str]:
        if self.checkpoint_key is not None:
            if not isinstance(checkpoint, dict) or self.checkpoint_key not in checkpoint:
                raise ValueError(
                    f"`checkpoint_key={self.checkpoint_key}` was not found in the checkpoint."
                )
            state_dict = checkpoint[self.checkpoint_key]
            source_key = self.checkpoint_key
        else:
            state_dict = checkpoint
            source_key = "<root>"
            if isinstance(checkpoint, dict):
                for candidate_key in ("target_encoder", "encoder", "model", "state_dict"):
                    if candidate_key in checkpoint:
                        state_dict = checkpoint[candidate_key]
                        source_key = candidate_key
                        break

        if not isinstance(state_dict, dict):
            raise ValueError(
                "Selected V-JEPA2 checkpoint entry must be a state_dict-like dict, "
                f"got {type(state_dict)!r} from key {source_key!r}."
            )
        return state_dict, source_key

    @staticmethod
    def _strip_state_dict_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
        prefixes = ("module.", "backbone.", "encoder.", "target_encoder.")
        stripped: dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = str(key)
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]
                        changed = True
            stripped[new_key] = value
        return stripped

    def _preprocess_real_video(self, video: torch.Tensor) -> torch.Tensor:
        # Official V-JEPA2 encoder expects normalized RGB video in [B, 3, T, H, W].
        x = video
        if self.input_range == "-1_1":
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)

        batch_size, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.reshape(batch_size, frames, channels, self.img_size, self.img_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        return (x - mean) / std

    def _adapt_temporal(self, video: torch.Tensor) -> torch.Tensor:
        # Prefer real consecutive frames. Repeat only when the clip is shorter
        # than the encoder tubelet size.
        frames = int(video.shape[2])
        if frames >= self.tubelet_size:
            return video

        repeats = (self.tubelet_size + frames - 1) // frames
        return video.repeat(1, 1, repeats, 1, 1)[:, :, : self.tubelet_size]

    def _move_video_to_encoder(self, video: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Real V-JEPA2 encoder was not initialized.")

        try:
            first_param = next(self.encoder.parameters())
        except StopIteration:
            return video
        return video.to(device=first_param.device, dtype=first_param.dtype)

    def _validate_real_output(self, tokens: Any, *, batch_size: int) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            raise ValueError(
                "Official V-JEPA2 encoder output must be a tensor with shape "
                f"[B, N, D_v], got {type(tokens)!r}."
            )
        if tokens.ndim != 3:
            raise ValueError(
                "Official V-JEPA2 encoder output must be 3D with shape [B, N, D_v], "
                f"got shape {tuple(tokens.shape)}."
            )
        if int(tokens.shape[0]) != batch_size:
            raise ValueError(
                "Official V-JEPA2 encoder output batch size mismatch: "
                f"expected {batch_size}, got {int(tokens.shape[0])}."
            )
        if int(tokens.shape[-1]) != self.vjepa_dim:
            raise ValueError(
                "Official V-JEPA2 encoder output feature dimension mismatch: "
                f"expected D_v={self.vjepa_dim}, got {int(tokens.shape[-1])}."
            )
        return tokens
