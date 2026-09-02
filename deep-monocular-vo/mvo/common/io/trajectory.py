"""Trajectory read/write in TUM format, so `evo` can cross-check our metrics.

OWNER: Suryansh Bajaj  (branch: base/metrics). STUB.

TUM format, one pose per line:  timestamp tx ty tz qx qy qz qw
"""

import numpy as np


def write_tum(path, poses: np.ndarray, timestamps=None) -> None:
    """(N,4,4) camera-to-world -> TUM .txt"""
    raise NotImplementedError("owner: Suryansh — branch base/metrics")


def read_tum(path) -> np.ndarray:
    """TUM .txt -> (N,4,4)"""
    raise NotImplementedError("owner: Suryansh — branch base/metrics")
