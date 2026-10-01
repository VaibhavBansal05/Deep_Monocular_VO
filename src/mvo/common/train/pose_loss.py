"""Shared geometric supervised-pose objective for M1, M2 and M4.

Translation lives in Euclidean R^3, while rotation lives on SO(3).  Treating
their concatenated coordinates as one six-dimensional Euclidean vector is only
a local approximation and makes the result depend on the chosen axis-angle
coordinates.  This module uses the geometry of each component explicitly.
"""

import torch

from mvo.common.geometry.se3 import so3_exp, so3_log


def geometric_pose_loss(pred: torch.Tensor, target: torch.Tensor,
                        pose_std: torch.Tensor, rot_weight: float = 1.0,
                        bias_weight: float = 0.1) -> dict[str, torch.Tensor]:
    """Euclidean translation + geodesic SO(3) rotation + mean-residual penalty.

    The two physical groups are divided by their training-set RMS spread so
    translation (metres) and rotation (radians) are comparable.  Individual
    translation axes are *not* equalised: a metre of forward error costs the
    same as a metre of lateral error, so the prevalent forward motion keeps
    appropriate influence without overwhelming the rotation term.

    ``bias`` penalises a coherent signed residual across the mini-batch.  Such
    a residual is especially harmful in odometry because it accumulates at
    every integration step.  It reduces train-distribution bias; it cannot by
    itself remove bias caused by a genuinely different test trajectory.
    """
    if pred.shape != target.shape or pred.shape[-1] != 6:
        raise ValueError(f"expected matching (...,6) poses, got {pred.shape} and {target.shape}")

    eps = torch.finfo(pred.dtype).eps
    trans_scale_sq = pose_std[:3].square().sum().clamp_min(eps)
    rot_scale_sq = pose_std[3:].square().sum().clamp_min(eps)

    trans_residual = pred[..., :3] - target[..., :3]
    trans_per_pose = trans_residual.square().sum(dim=-1) / trans_scale_sq

    pred_R = so3_exp(pred[..., 3:])
    target_R = so3_exp(target[..., 3:])
    relative_R = target_R.transpose(-1, -2) @ pred_R
    rotation_residual = so3_log(relative_R)
    rot_per_pose = rotation_residual.square().sum(dim=-1) / rot_scale_sq

    trans = trans_per_pose.mean()
    rot = rot_per_pose.mean()

    reduce_dims = tuple(range(trans_residual.ndim - 1))
    mean_t = trans_residual.mean(dim=reduce_dims)
    mean_r = rotation_residual.mean(dim=reduce_dims)
    bias = mean_t.square().sum() / trans_scale_sq
    bias = bias + mean_r.square().sum() / rot_scale_sq

    total = trans + float(rot_weight) * rot + float(bias_weight) * bias
    return {"total": total, "trans": trans, "rot": rot, "bias": bias}
