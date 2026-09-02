"""Monocular depth metrics -- Model 3 only.

OWNER: Suryansh Bajaj  (branch: base/metrics). STUB.
"""

import numpy as np


def depth_metrics(pred: np.ndarray, gt: np.ndarray,
                  min_depth: float = 1e-3, max_depth: float = 80.0) -> dict:
    """Returns {"abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"}."""
    raise NotImplementedError("owner: Suryansh — branch base/metrics")
