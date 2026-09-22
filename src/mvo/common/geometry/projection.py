# Adapted from ClementPinard/SfmLearner-Pytorch inverse_warp.py, MIT License.
# Modified: added valid mask for out-of-frame and behind-camera points,
#           refactored backproject/project as standalone functions,
#           removed homogeneous-coordinate intermediate.
"""Perspective projection and inverse warping for the photometric loss.

OWNER: Abhishek Laxman Jadhav  (branch: base/se3-utils)

Primary consumer is Model 3 (self-supervised photometric loss). Coordinate with
Suryansh before changing signatures.
"""

import torch
import torch.nn.functional as F


def backproject(depth: torch.Tensor, K_inv: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) depth + (B,3,3) K^-1 -> (B,3,H,W) points in the camera frame."""
    B, _, H, W = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=depth.dtype, device=depth.device),
        torch.arange(W, dtype=depth.dtype, device=depth.device), indexing="ij")
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0).reshape(1, 3, -1).expand(B, 3, -1)
    rays = K_inv @ pix                       # (B,3,H*W)
    return (rays * depth.reshape(B, 1, -1)).reshape(B, 3, H, W)


def project(points: torch.Tensor, K: torch.Tensor, T: torch.Tensor):
    """(B,3,H,W) camera-frame points -> (B,H,W,2) grid_sample coords in [-1,1].

    Returns (grid, valid) where valid marks points in front of the camera and
    inside the frame.
    """
    B, _, H, W = points.shape
    pts = points.reshape(B, 3, -1)
    cam = T[:, :3, :3] @ pts + T[:, :3, 3:]          # transform into the other view
    proj = K @ cam

    z = proj[:, 2:3].clamp(min=1e-6)
    uv = proj[:, :2] / z

    u = 2.0 * uv[:, 0] / (W - 1) - 1.0
    v = 2.0 * uv[:, 1] / (H - 1) - 1.0
    grid = torch.stack([u, v], dim=-1).reshape(B, H, W, 2)

    valid = ((u.abs() <= 1) & (v.abs() <= 1) & (cam[:, 2] > 0)).reshape(B, 1, H, W)
    return grid, valid.to(points.dtype)


def inverse_warp(src_img: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor,
                 K: torch.Tensor, K_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp src_img into the target view using the target's depth and the pose.

    pose is T_{target -> source} as a (B,4,4). Returns (warped, valid_mask).
    """
    pts = backproject(depth, K_inv)
    grid, valid = project(pts, K, pose)
    warped = F.grid_sample(src_img, grid, mode="bilinear",
                           padding_mode="zeros", align_corners=True)
    return warped, valid
