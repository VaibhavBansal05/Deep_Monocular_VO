"""Relative Pose Error and KITTI-style drift.

OWNER: Suryansh Bajaj  (branch: base/metrics). STUB.
"""

import numpy as np


def rpe(pred: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict:
    """Returns {"trans_rmse" (m), "rot_rmse_deg" (deg)}."""
    raise NotImplementedError("owner: Suryansh — branch base/metrics")


def kitti_drift(pred: np.ndarray, gt: np.ndarray,
                lengths: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800)) -> dict:
    """Standard KITTI odometry metric. Returns {"t_rel_pct", "r_rel_deg_per_100m"}."""
    raise NotImplementedError("owner: Suryansh — branch base/metrics")
