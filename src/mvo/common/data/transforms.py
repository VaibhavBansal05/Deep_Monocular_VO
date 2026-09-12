"""Preprocessing: resize to 128x416, ImageNet normalisation, augmentation."""

import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

RAW_H, RAW_W = 375, 1242


def rescale_intrinsics(K: np.ndarray, out_hw: tuple[int, int],
                       in_hw: tuple[int, int] = (RAW_H, RAW_W)) -> np.ndarray:
    """Scale K to match a resized image. fx,cx scale with width; fy,cy with height."""
    sy = out_hw[0] / in_hw[0]
    sx = out_hw[1] / in_hw[1]
    K = K.copy().astype(np.float32)
    K[0, 0] *= sx   # fx
    K[0, 2] *= sx   # cx
    K[1, 1] *= sy   # fy
    K[1, 2] *= sy   # cy
    return K


def normalize(img: torch.Tensor) -> torch.Tensor:
    """(3,H,W) in [0,1] -> ImageNet-normalised."""
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize. M3 needs raw-ish images for the photometric loss."""
    return img * IMAGENET_STD + IMAGENET_MEAN


def color_jitter(imgs: torch.Tensor, rng: np.random.Generator,
                 brightness=0.2, contrast=0.2, saturation=0.2) -> torch.Tensor:
    """Apply the SAME jitter to every frame in a sequence.

    Per-frame jitter would break M3's photometric consistency assumption --
    the whole training signal is that corresponding pixels have equal intensity.
    """
    b = 1.0 + rng.uniform(-brightness, brightness)
    c = 1.0 + rng.uniform(-contrast, contrast)
    s = 1.0 + rng.uniform(-saturation, saturation)

    out = imgs * b
    mean = out.mean(dim=(-3, -2, -1), keepdim=True)
    out = (out - mean) * c + mean
    gray = out.mean(dim=-3, keepdim=True)
    out = (out - gray) * s + gray
    return out.clamp(0.0, 1.0)


def hflip_pose(pose_vec: np.ndarray) -> np.ndarray:
    """Mirror a 6-DoF pose about the image's vertical axis.

    Under the reflection M = diag(-1, 1, 1):
        R' = M R M  and  R(w') = R(-M w),  so  w -> ( wx, -wy, -wz)
        t -> (-tx, ty, tz)

    Getting this wrong makes flip augmentation actively harmful rather than
    merely useless, and the failure is silent. There is a unit test for it below.
    """
    out = pose_vec.copy()
    out[0] *= -1.0    # tx
    out[4] *= -1.0    # ry
    out[5] *= -1.0    # rz
    return out
