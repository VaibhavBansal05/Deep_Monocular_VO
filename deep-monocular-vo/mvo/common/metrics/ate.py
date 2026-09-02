"""Absolute Trajectory Error with Umeyama alignment.

OWNER: Suryansh Bajaj  (branch: base/metrics). STUB.

Two numbers are reported for every model, because M1/M2/M4 use GT scale
alignment while M3 self-scales from depth (see PLAN.md 2.4):
    ate_sim3 -- 7-DoF alignment, scale solved for   (fair to M1/M2/M4)
    ate_se3  -- 6-DoF alignment, scale NOT solved   (fair to M3)
"""

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform. Returns (R, t, s)."""
    raise NotImplementedError("owner: Suryansh — branch base/metrics")


def ate(pred_xyz: np.ndarray, gt_xyz: np.ndarray, with_scale: bool = True) -> dict:
    """Returns {"rmse", "mean", "median", "std", "scale"} in metres."""
    raise NotImplementedError("owner: Suryansh — branch base/metrics")
