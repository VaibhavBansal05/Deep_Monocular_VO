"""Monocular depth metrics -- Model 3 only.

OWNER: Suryansh Bajaj  (branch: base/metrics).
"""

import numpy as np


def depth_metrics(pred: np.ndarray, gt: np.ndarray,
                  min_depth: float = 1e-3, max_depth: float = 80.0,
                  median_scale: bool = True) -> dict:
    """Return standard monocular depth metrics over valid ground-truth pixels."""
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError("predicted and ground-truth depth shapes differ")
    if not 0 < min_depth < max_depth:
        raise ValueError("expected 0 < min_depth < max_depth")
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError("predicted depth must be finite and positive")
    mask = np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    pred, gt = pred[mask], gt[mask]
    if not pred.size:
        raise ValueError("no valid ground-truth depth pixels")

    scale = 1.0
    if median_scale:
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
