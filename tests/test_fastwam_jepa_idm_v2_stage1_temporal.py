from __future__ import annotations

import torch

from tools.train_fastwam_jepa_idm_v2_stage1_predictor_mixed import collate_stage1_samples


def _context() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(5, 32), torch.ones(5, dtype=torch.bool)


def test_stage1_collate_canonicalizes_mixed_temporal_clips() -> None:
    libero_context, libero_mask = _context()
    robotwin_context, robotwin_mask = _context()
    libero_video = torch.arange(3 * 9 * 256 * 256, dtype=torch.float32).reshape(3, 9, 256, 256)
    robotwin_video = torch.zeros(3, 4, 256, 256)
    robotwin_future = torch.ones(3, 4, 256, 256)

    batch = collate_stage1_samples(
        [
            {
                "video": libero_video,
                "context": libero_context,
                "context_mask": libero_mask,
                "dataset_name": "libero",
                "source_name": "libero",
            },
            {
                "video": robotwin_video,
                "future_video": robotwin_future,
                "context": robotwin_context,
                "context_mask": robotwin_mask,
                "dataset_name": "robotwin",
                "source_name": "robotwin",
            },
        ],
        video_size=256,
        use_proprio=False,
        current_frame_count=4,
        future_frame_count=4,
    )

    assert batch["video"].shape == (2, 3, 4, 256, 256)
    assert batch["future_video"].shape == (2, 3, 4, 256, 256)
    assert torch.equal(batch["video"][0], libero_video[:, :4])
    assert torch.equal(batch["future_video"][0], libero_video[:, -4:])
    assert torch.equal(batch["video"][1], robotwin_video)
    assert torch.equal(batch["future_video"][1], robotwin_future)