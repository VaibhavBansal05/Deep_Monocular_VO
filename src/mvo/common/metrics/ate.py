"""Absolute Trajectory Error with Umeyama alignment.

OWNER: Suryansh Bajaj  (branch: base/metrics). STUB.

Two numbers are reported for every model, because M1/M2/M4 use GT scale
alignment while M3 self-scales from depth (see PLAN.md 2.4):
    ate_sim3 -- 7-DoF alignment, scale solved for   (fair to M1/M2/M4)
    ate_se3  -- 6-DoF alignment, scale NOT solved   (fair to M3)
"""

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Returns ``(R, t, s)`` minimising ``||dst - (s R src + t)||^2`` for
    point arrays shaped ``(N, 3)``.
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3
    n = src.shape[0]

    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    covariance = dst_centered.T @ src_centered / n
    U, singular_values, Vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        correction[2, 2] = -1.0

    R = U @ correction @ Vt
    if with_scale:
        src_variance = np.sum(src_centered ** 2) / n
        s = (float(np.trace(np.diag(singular_values) @ correction))
             if src_variance > 0 else 1.0)
        if src_variance > 0:
            s /= src_variance
    else:
        s = 1.0

    t = mu_dst - s * R @ mu_src
    return R, t, s


def ate(pred_xyz: np.ndarray, gt_xyz: np.ndarray, with_scale: bool = True) -> dict:
    """Returns {"rmse", "mean", "median", "std", "scale"} in metres."""
    raise NotImplementedError("owner: Suryansh — branch base/metrics")
