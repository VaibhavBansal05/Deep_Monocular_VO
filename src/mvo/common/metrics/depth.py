"""Monocular depth metrics -- Model 3 only.

OWNER: Suryansh Bajaj  (branch: base/metrics). STUB.
"""

import numpy as np


def depth_metrics(pred: np.ndarray, gt: np.ndarray,
                  min_depth: float = 1e-3, max_depth: float = 80.0,
                  median_scale: bool = True) -> dict:
    """Return standard monocular depth metrics over valid ground-truth pixels."""
    mask = (gt > min_depth) & (gt < max_depth)
    pred, gt = pred[mask], gt[mask]

    scale = 1.0
    if median_scale and pred.size:
        scale = float(np.median(gt) / np.median(pred))
        pred = pred * scale
    pred = np.clip(pred, min_depth, max_depth)

    threshold = np.maximum(gt / pred, pred / gt)
    return {
        "abs_rel": float(np.mean(np.abs(gt - pred) / gt)),
        "sq_rel": float(np.mean((gt - pred) ** 2 / gt)),
        "rmse": float(np.sqrt(np.mean((gt - pred) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))),
        "a1": float(np.mean(threshold < 1.25)),
        "a2": float(np.mean(threshold < 1.25 ** 2)),
        "a3": float(np.mean(threshold < 1.25 ** 3)),
        "median_scale": float(scale),
    }
