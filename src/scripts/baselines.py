#!/usr/bin/env python3
"""Reference baselines that require no training.

A learned VO model must be shown to beat trivial predictors, or its metrics
mean little on their own. Following SfMLearner (Zhou et al., 2017), which
reports a "Mean Odom." baseline alongside its results, this computes:

  identity   - the camera never moves. Bounds the metric from above.
  mean pose  - the average ground-truth relative pose applied at every frame.
               Captures "drive forward at the usual speed" while learning
               nothing about the images. This is the bar a model must clear.
  oracle     - the ground-truth relative poses re-integrated. Non-zero only
               because of the dataset's own precision floor.

    python scripts/baselines.py --data-root data/vkitti2
"""

import argparse
import json
from pathlib import Path

import torch

from mvo.common.data import splits
from mvo.common.data.vkitti import VKitti2
from mvo.common.geometry import se3
from mvo.common.io.logging import get_logger
from mvo.common.metrics.ate import ate
from mvo.common.metrics.rpe import kitti_drift, rpe

log = get_logger("baselines")


def score(pred_vec: torch.Tensor, gt_abs) -> dict:
    pred = se3.relative_to_absolute(se3.vec_to_mat(pred_vec)).numpy()
    n = min(len(pred), len(gt_abs))
    pred, gt = pred[:n], gt_abs[:n]
    return {
        "ate_sim3": ate(pred[:, :3, 3], gt[:, :3, 3], with_scale=True)["rmse"],
        "ate_se3": ate(pred[:, :3, 3], gt[:, :3, 3], with_scale=False)["rmse"],
        **{f"rpe_{k}": v for k, v in rpe(pred, gt).items()},
        **kitti_drift(pred, gt),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/vkitti2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()

    ds = VKitti2(root=args.data_root, split=args.split, seq_len=2)
    payload = {}

    for scene, variation in splits.sequences(args.split):
        sid = splits.seq_id(scene, variation)
        idx = [i for i, (s, _) in enumerate(ds.samples) if s == sid]
        gt_abs = ds.absolute_poses(sid)
        gt_vec = se3.mat_to_vec(torch.stack([ds[i]["rel_pose"][0] for i in idx]))

        payload[sid] = {
            "identity": score(torch.zeros_like(gt_vec), gt_abs),
            "mean_pose": score(gt_vec.mean(0, keepdim=True).repeat(len(gt_vec), 1), gt_abs),
            "oracle": score(gt_vec, gt_abs),
        }
        for name, m in payload[sid].items():
            log.info("%s | %-10s ATE(sim3) %8.3f  ATE(se3) %8.3f  t_rel %6.2f%%",
                     sid, name, m["ate_sim3"], m["ate_se3"], m["t_rel_pct"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
