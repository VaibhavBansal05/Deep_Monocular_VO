"""M3 RGB-only objective. OWNER: Suryansh Bajaj.

Godard et al., ICCV 2019 (arXiv:1806.01260), independently implemented.
Existing shared geometry is used at full resolution at every disparity scale.
"""

import torch
import torch.nn.functional as F

from mvo.common.geometry.projection import inverse_warp
from mvo.common.geometry.se3 import invert, vec_to_mat
from mvo.models.m3_selfsup_posedepth.depthnet import disparity_to_depth


def target_to_sources(pose):
    """Chronological adjacent poses -> middle-target warp directions."""
    transforms = vec_to_mat(pose)
    return torch.stack([invert(transforms[:, 0]), transforms[:, 1]], dim=1)


def ssim_residual(a, b):
    """3x3 SSIM dissimilarity in [0,1], retaining RGB channels."""
    def average(x):
        return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)

    ma, mb = average(a), average(b)
    va, vb = average(a * a) - ma.square(), average(b * b) - mb.square()
    covariance = average(a * b) - ma * mb
    numerator = (2 * ma * mb + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
    denominator = (ma.square() + mb.square() + 0.01 ** 2) * (va + vb + 0.03 ** 2)
    return ((1 - numerator / denominator.clamp_min(1e-12)) / 2).clamp(0, 1)


def photometric_residual(target, reconstruction, ssim_weight=0.85):
    l1 = (target - reconstruction).abs().mean(1, keepdim=True)
    if ssim_weight == 0:
        return l1
    return ssim_weight * ssim_residual(target, reconstruction).mean(1, keepdim=True) + (
        1 - ssim_weight) * l1


def erode_validity(valid):
    """Require all warped pixels in the SSIM support to be valid."""
    return F.max_pool2d(1 - valid, 3, stride=1, padding=1) == 0


def select_reprojection(errors, identity_errors, valid, automask=True, min_reproj=True):
    """Reduce (B,R,H,W) residuals, excluding invalid references before selection."""
    valid = valid.bool()
    eligible = valid.any(dim=1, keepdim=True)
    if min_reproj:
        # Residuals from [0,1] RGB are <=1; finite sentinel avoids inf*0.
        photo = torch.where(valid, errors, errors.new_full((), 1e3)).min(1, keepdim=True).values
        identity = identity_errors.min(1, keepdim=True).values
    else:
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        photo = torch.where(valid, errors, 0).sum(1, keepdim=True) / count
        identity = torch.where(valid, identity_errors, 0).sum(1, keepdim=True) / count
    active = eligible & (photo + 1e-6 < identity) if automask else eligible
    sums = torch.where(active, photo, 0).sum(dim=(1, 2, 3))
    counts = active.sum(dim=(1, 2, 3)).clamp_min(1)
    return (sums / counts).mean(), eligible.float().mean(), active.float().mean(), active


def edge_aware_smoothness(disparity, image):
    normalized = disparity / disparity.mean(dim=(2, 3), keepdim=True).clamp_min(1e-7)
    dx = (normalized[..., 1:] - normalized[..., :-1]).abs()
    dy = (normalized[..., 1:, :] - normalized[..., :-1, :]).abs()
    ix = (image[..., 1:] - image[..., :-1]).abs().mean(1, keepdim=True)
    iy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(1, keepdim=True)
    return (dx * (-ix).exp()).mean() + (dy * (-iy).exp()).mean()


def reconstruct_scale(images, depth, transforms, K, K_inv, ssim_weight):
    """Two neighbour reconstructions, residuals and validity supports."""
    target = images[:, 1]
    warped, errors, identity, masks = [], [], [], []
    for ref, index in enumerate((0, 2)):
        reconstruction, valid = inverse_warp(images[:, index], depth, transforms[:, ref], K, K_inv)
        warped.append(reconstruction)
        errors.append(photometric_residual(target, reconstruction, ssim_weight))
        identity.append(photometric_residual(target, images[:, index], ssim_weight))
        masks.append(erode_validity(valid) if ssim_weight else valid.bool())
    return torch.stack(warped, 1), torch.cat(errors, 1), torch.cat(identity, 1), torch.cat(masks, 1)


def multiscale_loss(images, disparities, pose, K, K_inv, *, min_depth=0.1,
                    max_depth=100.0, ssim_weight=0.85, smooth_weight=1e-3,
                    automask=True, min_reproj=True):
    """FP32 reconstruction losses from RGB, with no pose/depth labels."""
    with torch.amp.autocast(images.device.type, enabled=False):
        images, pose, K, K_inv = images.float(), pose.float(), K.float(), K_inv.float()
        transforms = target_to_sources(pose)
        photometric, smooth, valid_fraction, active_fraction = [], [], [], []
        for scale, disparity in enumerate(disparities):
            disparity = disparity.float()
            full = F.interpolate(disparity, images.shape[-2:], mode="bilinear", align_corners=False)
            depth = disparity_to_depth(full, min_depth, max_depth)
            _, errors, identity, valid = reconstruct_scale(
                images, depth, transforms, K, K_inv, ssim_weight)
            photo, valid_count, active_count, _ = select_reprojection(
                errors, identity, valid, automask, min_reproj)
            native_rgb = F.interpolate(images[:, 1], disparity.shape[-2:],
                                       mode="bilinear", align_corners=False)
            photometric.append(photo)
            smooth.append(edge_aware_smoothness(disparity, native_rgb) / (2 ** scale))
            valid_fraction.append(valid_count)
            active_fraction.append(active_count)
        photo, sm = torch.stack(photometric).mean(), torch.stack(smooth).mean()
        return {"total": photo + smooth_weight * sm, "photometric": photo,
                "smooth": sm, "valid_fraction": torch.stack(valid_fraction).mean(),
                "active_fraction": torch.stack(active_fraction).mean()}
