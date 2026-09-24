#!/usr/bin/env python3
"""Evaluate a trained model on the test split -> results/<model>/metrics.json

    python scripts/eval.py --model m4 --ckpt checkpoints/m4/best.pt

Every model is scored by THIS script and no other. Results are never typed by
hand into the report -- compare.py reads these JSON files directly.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvo.common.data import splits
from mvo.common.data.vkitti import VKitti2
from mvo.common.geometry import se3
from mvo.common.io.logging import get_logger
from mvo.common.io.trajectory import write_tum
from mvo.common.metrics.ate import ate
from mvo.common.metrics.rpe import kitti_drift, rpe
from mvo.common.train.registry import build, import_all_models

log = get_logger("eval")

OWNERS = {"m1": "Pranav Joshi", "m2": "Abhishek Laxman Jadhav",
          "m3": "Suryansh Bajaj", "m4": "Vaibhav Krishna Bansal"}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def predict_sequence(model, loader, device) -> tuple[np.ndarray, float]:
    """Run the model over a sequence -> (N-1, 4, 4) relative poses, ms/frame.

    The dataset yields a sliding window of stride 1, so sample i covers frames
    [i .. i+S-1] and predicts S-1 relative poses. Concatenating all of them
    counts every transition S-1 times: for S=5 that is 3332 poses where the
    trajectory needs 836, and integrating them produces a trajectory roughly
    four times too long.

    Each transition must therefore be taken from exactly one window, and for a
    recurrent model it matters WHICH. Window i predicts its S-1 transitions in
    order, so element 0 has seen a single frame pair and carries no temporal
    context at all, while element S-2 has the full window behind it. Taking
    element 0 would evaluate a sequence model as if it were pairwise and throw
    away the recurrence the model exists to provide.

    So take the LAST element of every window, which covers transitions
    S-2 .. N-2, and recover the leading S-2 transitions from the first window:

        (S - 2) + (N - S + 1) = N - 1
    """
    model.eval()
    per_window, n_frames = [], 0
    t0 = time.time()
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch)
        pose = out["pose"].float().cpu()          # (B, S-1, 6)
        per_window.append(pose)
        # One transition per window is retained, so timing per retained frame
        # is per window, not per predicted pose.
        n_frames += pose.shape[0]
    ms = (time.time() - t0) * 1000.0 / max(n_frames, 1)

    poses = torch.cat(per_window, dim=0)          # (n_windows, S-1, 6)
    if poses.shape[1] == 1:                       # pairwise model
        vec = poses[:, 0]
    else:
        head = poses[0, :-1]                      # leading S-2 transitions
        vec = torch.cat([head, poses[:, -1]], dim=0)
    return se3.vec_to_mat(vec).numpy(), ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default="data/vkitti2")
    args = ap.parse_args()

    import_all_models()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build(args.model, **cfg.get("model", {}))
    model.load_state_dict(ck["model"])
    model.to(device)

    out_dir = Path(f"results/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seq = {}

    for scene, variation in splits.sequences(args.split):
        sid = splits.seq_id(scene, variation)
        ds = VKitti2(root=args.data_root, split=args.split, seq_len=model.seq_len,
                     need_depth=model.needs_depth, augment=False)
        # Restrict to this sequence. Without it a multi-sequence split (e.g. the
        # robustness set) would be evaluated end-to-end, chaining unrelated
        # trajectories together, once per sequence.
        ds.samples = [s for s in ds.samples if s[0] == sid]
        if not ds.samples:
            raise RuntimeError(f"no samples for {sid}")
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

        rel, ms = predict_sequence(model, loader, device)
        pred_abs = se3.relative_to_absolute(torch.from_numpy(rel)).numpy()
        gt_abs = ds.absolute_poses(sid)

        n = min(len(pred_abs), len(gt_abs))
        pred_abs, gt_abs = pred_abs[:n], gt_abs[:n]

        m = {
            "ate_sim3": ate(pred_abs[:, :3, 3], gt_abs[:, :3, 3], with_scale=True)["rmse"],
            "ate_se3": ate(pred_abs[:, :3, 3], gt_abs[:, :3, 3], with_scale=False)["rmse"],
            **{f"rpe_{k}": v for k, v in rpe(pred_abs, gt_abs, delta=1).items()},
            **kitti_drift(pred_abs, gt_abs),
            "ms_per_frame": ms,
        }
        per_seq[sid] = m
        write_tum(out_dir / f"pred_{sid.replace('/', '_')}.txt", pred_abs)
        log.info("%s | ATE(sim3) %.3f m | ATE(se3) %.3f m", sid, m["ate_sim3"], m["ate_se3"])

    keys = next(iter(per_seq.values())).keys()
    agg = {k: float(np.mean([s[k] for s in per_seq.values()])) for k in keys}

    payload = {"model": args.model, "owner": OWNERS.get(args.model, "?"),
               "split": args.split, "seed": cfg.get("seed", 42),
               "params_m": model.num_params(), "commit": git_sha(),
               **agg, "per_sequence": per_seq}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
