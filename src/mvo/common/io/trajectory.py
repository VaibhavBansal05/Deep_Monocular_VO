"""Trajectory read/write in TUM format, so `evo` can cross-check our metrics.

OWNER: Suryansh Bajaj  (branch: base/metrics).

TUM format, one pose per line:  timestamp tx ty tz qx qy qz qw
"""

import numpy as np
from scipy.spatial.transform import Rotation


def write_tum(path, poses: np.ndarray, timestamps=None) -> None:
    """(N,4,4) camera-to-world -> TUM .txt"""
    if timestamps is None:
        timestamps = np.arange(len(poses), dtype=float)
    quaternions = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    with open(path, "w") as file:
        for timestamp, translation, quaternion in zip(
                timestamps, poses[:, :3, 3], quaternions):
            file.write(
                f"{timestamp:.6f} "
                f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
                f"{quaternion[0]:.9f} {quaternion[1]:.9f} "
                f"{quaternion[2]:.9f} {quaternion[3]:.9f}\n"
            )


def read_tum(path) -> np.ndarray:
    """TUM .txt -> (N,4,4)"""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None]
    poses = np.tile(np.eye(4), (len(data), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(data[:, 4:8]).as_matrix()
    poses[:, :3, 3] = data[:, 1:4]
    return poses
