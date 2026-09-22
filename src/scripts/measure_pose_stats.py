#!/usr/bin/env python3
"""Measure ground-truth relative-pose statistics on the training split.

The rotation weight in the pose-regression losses (M2, M4) is derived from
these numbers rather than guessed, so this script is how that constant is
justified and reproduced.

    python scripts/measure_pose_stats.py --data-root data/vkitti2
"""

import argparse

import numpy as np
import torch

from mvo.common.data.vkitti import VKitti2
from mvo.common.geometry import se3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/vkitti2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--samples", type=int, default=500)
    args = ap.parse_args()

    ds = VKitti2(root=args.data_root, split=args.split, seq_len=2, augment=False)
    idx = np.linspace(0, len(ds) - 1, min(args.samples, len(ds))).astype(int)
    vec = torch.stack([se3.mat_to_vec(ds[int(i)]["rel_pose"])[0] for i in idx])
    t, r = vec[:, :3], vec[:, 3:]

    trans_mse = float((t ** 2).mean())
    rot_mse = float((r ** 2).mean())

    print(f"split={args.split}  n={len(idx)}")
    print(f"  |t| mean {t.norm(dim=1).mean():.4f} m    max {t.norm(dim=1).max():.4f} m")
    print(f"  |r| mean {r.norm(dim=1).mean():.5f} rad  max {r.norm(dim=1).max():.5f} rad")
    print(f"  per-axis t std {t.std(0).numpy().round(4)}")
    print(f"  per-axis r std {r.std(0).numpy().round(5)}")
    print()
    print(f"  E[t^2] = {trans_mse:.5e}")
    print(f"  E[r^2] = {rot_mse:.5e}")
    print(f"  ratio  = {trans_mse / rot_mse:.0f} : 1")
    print()
    print(f"  => rot_weight for parity at initialisation: {trans_mse / rot_mse:.0f}")


if __name__ == "__main__":
    main()
