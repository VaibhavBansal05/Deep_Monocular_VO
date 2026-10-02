"""Cache-backed M1 classical-feature pose regression model."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from mvo.common.data.pose_stats import destandardize_pose, pose_mean_std
from mvo.common.geometry.se3 import mat_to_vec
from mvo.common.train.base_model import BaseVOModel
from mvo.common.train.pose_loss import geometric_pose_loss
from mvo.common.train.registry import register

from .cache import FeatureCache
from .network import PoseMLP
from .precompute import pair_key


@register("m1")
class ClassicalMLPVO(BaseVOModel):
    """Regress metric relative pose from cached classical flow descriptors."""

    seq_len = 2
    needs_depth = False

    def __init__(
        self,
        cache_path: str | Path = "data/m1_features/features.npz",
        detector: str = "orb",
        max_features: int = 1000,
        grid: Sequence[int] = (8, 8),
        hidden: Sequence[int] = (512, 256, 128),
        dropout: float = 0.2,
        rot_weight: float = 1.0,
        bias_weight: float = 0.1,
        standardize: bool = True,
        img_size: Sequence[int] = (128, 416),
    ) -> None:
        super().__init__()
        grid = tuple(int(size) for size in grid)
        if len(grid) != 2 or any(size < 1 for size in grid):
            raise ValueError(f"grid must contain two positive dimensions, got {grid}")
        img_size = tuple(int(size) for size in img_size)
        if len(img_size) != 2 or any(size < 1 for size in img_size):
            raise ValueError(f"img_size must contain two positive dimensions, got {img_size}")

        descriptor_dim = grid[0] * grid[1] * 3
        self.cache_path = str(cache_path)
        self.cache = FeatureCache.load(cache_path, expected_descriptor_dim=descriptor_dim)
        expected_cache_settings = {
            "detector": detector.lower().replace("-", "").replace("_", ""),
            "max_features": int(max_features),
            "grid": list(grid),
            "img_size": list(img_size),
        }
        for name, expected in expected_cache_settings.items():
            actual = self.cache.frontend_config.get(name)
            if actual != expected:
                raise ValueError(
                    f"M1 cache setting {name!r} is {actual!r}; model requested {expected!r}"
                )

        self.regressor = PoseMLP(
            input_dim=descriptor_dim,
            hidden_sizes=hidden,
            dropout=dropout,
        )
        self.rot_weight = float(rot_weight)
        self.bias_weight = float(bias_weight)
        self.standardize = bool(standardize)
        mean, std = pose_mean_std()
        self.register_buffer("pose_mean", mean)
        self.register_buffer("pose_std", std)

    def _cached_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        seq_ids = batch.get("seq_id")
        frame_idx = batch.get("frame_idx")
        if seq_ids is None or frame_idx is None:
            raise KeyError("M1 requires batch keys 'seq_id' and 'frame_idx' for cache lookup")
        if isinstance(seq_ids, str):
            seq_ids = [seq_ids]
        else:
            seq_ids = list(seq_ids)
        if not torch.is_tensor(frame_idx):
            frame_idx = torch.as_tensor(frame_idx)
        if frame_idx.ndim == 1:
            frame_idx = frame_idx.unsqueeze(0)
        if frame_idx.ndim != 2 or frame_idx.shape[1] != 2:
            raise ValueError(f"expected frame_idx shape (B,2), got {tuple(frame_idx.shape)}")
        if len(seq_ids) != frame_idx.shape[0]:
            raise ValueError(
                f"seq_id/frame_idx batch mismatch: {len(seq_ids)} != {frame_idx.shape[0]}"
            )

        records = [
            self.cache[pair_key(seq_id, frames)]
            for seq_id, frames in zip(seq_ids, frame_idx.detach().cpu())
        ]
        descriptors = np.stack([record.descriptor for record in records])
        descriptor_tensor = torch.from_numpy(descriptors).to(
            device=self.pose_mean.device, dtype=self.pose_mean.dtype
        )
        success = torch.tensor(
            [record.success for record in records],
            device=self.pose_mean.device,
            dtype=torch.bool,
        )
        return descriptor_tensor, success

    def forward(self, batch: dict) -> dict:
        descriptors, success = self._cached_batch(batch)
        raw_pose = self.regressor(descriptors)
        pose = (
            destandardize_pose(raw_pose, self.pose_mean, self.pose_std)
            if self.standardize
            else raw_pose
        )
        return {
            "pose": pose.unsqueeze(1),
            "pose_standardized": raw_pose.unsqueeze(1),
            "frontend_success": success.unsqueeze(1),
        }

    def compute_loss(self, batch: dict, out: dict) -> dict:
        target = mat_to_vec(batch["rel_pose"])
        return geometric_pose_loss(
            out["pose"],
            target,
            self.pose_std,
            rot_weight=self.rot_weight,
            bias_weight=self.bias_weight,
        )
