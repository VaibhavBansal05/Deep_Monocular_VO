"""Perspective projection and inverse warping.

OWNER: Abhishek Laxman Jadhav  (branch: base/se3-utils)
STATUS: STUB.

Primary consumer is Model 3 (self-supervised photometric loss), so coordinate
with Suryansh before changing signatures.
Adapted-from targets: SfmLearner-Pytorch inverse_warp.py (MIT),
monodepth2 layers.py BackprojectDepth/Project3D (non-commercial research).
Record whichever you use in THIRD_PARTY.md.
"""

import torch


def backproject(depth: torch.Tensor, K_inv: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) depth + (B,3,3) K^-1 -> (B,3,H,W) camera-frame points."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def project(points: torch.Tensor, K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) points -> (B,H,W,2) normalised [-1,1] grid for grid_sample."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def inverse_warp(src_img: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor,
                 K: torch.Tensor, K_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp src_img into the target frame. Returns (warped, valid_mask)."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")
