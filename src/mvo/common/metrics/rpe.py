"""Relative Pose Error and KITTI-style drift.

OWNER: Suryansh Bajaj  (branch: base/metrics).
"""

import numpy as np


def _rot_angle_deg(R: np.ndarray) -> float:
    """Return the geodesic angle of a 3x3 rotation matrix in degrees."""
    cosine = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rpe(pred: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict:
    """Return relative pose error over a fixed frame gap.

    ``pred`` and ``gt`` are ``(N, 4, 4)`` camera-to-world poses.
    """
    trans_error = []
    rot_error = []
    for i in range(len(gt) - delta):
        gt_relative = np.linalg.inv(gt[i]) @ gt[i + delta]
        pred_relative = np.linalg.inv(pred[i]) @ pred[i + delta]
        error = np.linalg.inv(gt_relative) @ pred_relative
        trans_error.append(np.linalg.norm(error[:3, 3]))
        rot_error.append(_rot_angle_deg(error[:3, :3]))

    trans_error = np.asarray(trans_error)
    rot_error = np.asarray(rot_error)
    return {
        "trans_rmse": float(np.sqrt(np.mean(trans_error ** 2))),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot_error ** 2))),
    }


def _path_lengths(poses: np.ndarray) -> np.ndarray:
    """Return cumulative distance travelled by camera-to-world poses."""
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def kitti_drift(pred: np.ndarray, gt: np.ndarray,
                lengths: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800),
                step: int = 10) -> dict:
    """Return KITTI-style translation and rotation drift over distance windows."""
    distance = _path_lengths(gt)
    trans_error = []
    rot_error = []

    for start in range(0, len(gt) - 1, step):
        for length in lengths:
            end = np.searchsorted(distance, distance[start] + length)
            if end >= len(gt):
                continue
            gt_relative = np.linalg.inv(gt[start]) @ gt[end]
            pred_relative = np.linalg.inv(pred[start]) @ pred[end]
            error = np.linalg.inv(gt_relative) @ pred_relative
            trans_error.append(np.linalg.norm(error[:3, 3]) / length)
            rot_error.append(np.radians(_rot_angle_deg(error[:3, :3])) / length)

    if not trans_error:
        return {"t_rel_pct": float("nan"), "r_rel_deg_per_100m": float("nan")}
    return {
        "t_rel_pct": float(np.mean(trans_error) * 100.0),
        "r_rel_deg_per_100m": float(np.degrees(np.mean(rot_error)) * 100.0),
    }
