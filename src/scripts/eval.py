#!/usr/bin/env python3
"""Evaluate a trained model on the test split -> results/<model>/metrics.json

    python scripts/eval.py --model m4 --ckpt checkpoints/m4/best.pt

Every model is scored by THIS script and no other. Results are never typed by
hand into the report -- compare.py reads these JSON files directly.
"""

import argparse
import hashlib
import json
import platform
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
from mvo.common.metrics.depth import depth_metrics
from mvo.common.metrics.rpe import kitti_drift, rpe
from mvo.common.train.registry import build, import_all_models

log = get_logger("eval")

OWNERS = {"m1": "Pranav Joshi", "m2": "Abhishek Laxman Jadhav",
          "m3": "Suryansh Bajaj", "m4": "Vaibhav Krishna Bansal"}


def signed_pose_bias(pred_rel: torch.Tensor, gt_rel: torch.Tensor) -> dict[str, float]:
    """Mean signed metric translation and geodesic rotation residuals."""
    n = min(len(pred_rel), len(gt_rel))
    pred_rel, gt_rel = pred_rel[:n], gt_rel[:n]
    trans = (pred_rel[:, :3, 3] - gt_rel[:, :3, 3]).mean(dim=0)
    rotation_error = (gt_rel[:, :3, :3].transpose(-1, -2)
                      @ pred_rel[:, :3, :3])
    rot = se3.so3_log(rotation_error).mean(dim=0)
    names = ("tx_m", "ty_m", "tz_m", "rx_rad", "ry_rad", "rz_rad")
    values = torch.cat([trans, rot]).cpu().tolist()
    return {f"bias_{name}": float(value) for name, value in zip(names, values)}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def full_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def depth_loader(root, split, sid, img_size=(128, 416)):
    """Require one depth PNG for every RGB frame before starting evaluation."""
    dataset = VKitti2(root=root, split=split, seq_len=2, need_depth=True,
                      augment=False, img_size=img_size)
    dataset.samples = [sample for sample in dataset.samples if sample[0] == sid]
    if not dataset.samples:
        raise RuntimeError(f"no depth-evaluation windows for {sid}")
    meta = dataset.meta[sid]
    depth_dir = meta["dir"] / "frames/depth" / splits.CAMERA
    expected = {f"depth_{frame:05d}.png" for frame in range(meta["n"])}
    actual = {path.name for path in depth_dir.glob("depth_*.png")}
    if actual != expected:
        raise FileNotFoundError(
            f"depth/RGB frame mismatch for {sid}: missing {len(expected - actual)}, "
            f"extra {len(actual - expected)} PNGs under {depth_dir}")
    return DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)


@torch.no_grad()
def evaluate_depth_sequence(model, loader, device):
    """Score each frame once from ordered two-frame sliding windows."""
    model.eval()
    seen, scaled, raw, factors = set(), [], [], []
    skipped = 0
    for batch in loader:
        images, truth = [], []
        for row, sid in enumerate(batch["seq_id"]):
            first, second = (int(x) for x in batch["frame_idx"][row])
            if not seen:
                seen.add((sid, first))
                images.append(batch["imgs"][row, 0])
                truth.append(batch["depth"][row, 0].numpy())
            key = (sid, second)
            if key in seen:
                raise ValueError(f"duplicate depth frame {sid}/{second}")
            seen.add(key)
            images.append(batch["imgs"][row, 1])
            truth.append(batch["depth"][row, 1].numpy())
        prediction = model.predict_depth(torch.stack(images).to(device)).float().cpu().numpy()
        if prediction.shape != np.stack(truth).shape:
            raise ValueError("predicted depth shape differs from ground truth")
        for pred, gt in zip(prediction, truth):
            if not np.isfinite(pred).all() or (pred <= 0).any():
                raise ValueError("predicted depth must be finite and positive")
            valid = np.isfinite(gt) & (gt > 0.001) & (gt < 80.0)
            if not valid.any():
                skipped += 1
                continue
            primary = depth_metrics(pred, gt, median_scale=True)
            secondary = depth_metrics(pred, gt, median_scale=False)
            factors.append(primary.pop("median_scale"))
            secondary.pop("median_scale")
            scaled.append(primary)
            raw.append(secondary)
    if not scaled:
        raise ValueError("no evaluation frames have valid ground-truth depth")

    def average(rows):
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    return {"median_scaled": average(scaled), "raw_scale": average(raw),
            "frames_seen": len(seen), "frames_scored": len(scaled),
            "frames_skipped": skipped, "median_scale_factor": float(np.median(factors))}


def aggregate_depth(per_sequence):
    scored = sum(row["frames_scored"] for row in per_sequence.values())
    return {
        "frames_seen": sum(row["frames_seen"] for row in per_sequence.values()),
        "frames_scored": scored,
        "frames_skipped": sum(row["frames_skipped"] for row in per_sequence.values()),
        **{mode: {key: float(sum(row["frames_scored"] * row[mode][key]
                                 for row in per_sequence.values()) / scored)
                  for key in next(iter(per_sequence.values()))[mode]}
           for mode in ("median_scaled", "raw_scale")},
    }


def depth_report(checkpoint, checkpoint_path, cfg, split, rgb_version,
                 depth_version, device, per_sequence):
    """Keep the depth result schema explicit and independent of pose metrics."""
    return {
        "model": "m3", "split": split,
        "datasets": {"rgb": rgb_version, "depth": depth_version},
        "protocol": {"source": "Naver Virtual KITTI 2.0.3 Camera_0",
                     "resolution": cfg.get("img_size", [128, 416]),
                     "crop": "none", "gt_valid_metres": "(0.001, 80)",
                     "aggregation": "mean of per-image metrics over valid frames",
                     "primary_scale": "per-image GT median; depth only",
                     "secondary_scale": "raw predicted scale"},
        "training_commit": checkpoint.get("training_commit", "unknown"),
        "evaluation_commit": full_git_sha(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "config": cfg,
        "environment": {"device": torch.cuda.get_device_name(device) if device.type == "cuda"
                        else "cpu", "python": platform.python_version(),
                        "torch": torch.__version__, "numpy": np.__version__},
        "per_sequence": per_sequence,
        "aggregate": aggregate_depth(per_sequence),
    }


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
    ap.add_argument("--out", default=None,
                    help="result directory; useful for M4 none/SE/CBAM ablations")
    ap.add_argument("--eval-depth", action="store_true",
                    help="also score M3 depth against official VKITTI2 PNGs")
    ap.add_argument("--rgb-dataset-version", default=None,
                    help="RGB dataset identifier and version for depth provenance")
    ap.add_argument("--depth-dataset-version", default=None,
                    help="depth dataset identifier and version for depth provenance")
    args = ap.parse_args()

    if args.eval_depth:
        if args.model != "m3":
            ap.error("--eval-depth is supported only for M3")
        if args.split not in ("val", "test"):
            ap.error("--eval-depth requires --split val or test")
        if not args.rgb_dataset_version or not args.depth_dataset_version:
            ap.error("--eval-depth requires both dataset version identifiers")

    import_all_models()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_options = dict(cfg.get("model", {}))
    if args.model == "m3":
        # The checkpoint contains both trained encoders; avoid another download.
        model_options["pretrained"] = False
    model = build(args.model, **model_options)
    model.load_state_dict(ck["model"])
    model.to(device)

    depth_loaders = {}
    if args.eval_depth:
        if not callable(getattr(model, "predict_depth", None)):
            raise ValueError("M3 checkpoint model has no predict_depth method")
        for scene, variation in splits.sequences(args.split):
            sid = splits.seq_id(scene, variation)
            depth_loaders[sid] = depth_loader(args.data_root, args.split, sid,
                                              cfg.get("img_size", (128, 416)))

    out_dir = Path(args.out or f"results/{args.model}")
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
        pred_rel = torch.from_numpy(rel[:n - 1])
        gt_rel = se3.absolute_to_relative(torch.from_numpy(gt_abs))

        m = {
            "ate_sim3": ate(pred_abs[:, :3, 3], gt_abs[:, :3, 3], with_scale=True)["rmse"],
            "ate_se3": ate(pred_abs[:, :3, 3], gt_abs[:, :3, 3], with_scale=False)["rmse"],
            **{f"rpe_{k}": v for k, v in rpe(pred_abs, gt_abs, delta=1).items()},
            **kitti_drift(pred_abs, gt_abs),
            **signed_pose_bias(pred_rel, gt_rel),
            "ms_per_frame": ms,
        }
        per_seq[sid] = m
        write_tum(out_dir / f"pred_{sid.replace('/', '_')}.txt", pred_abs)
        log.info("%s | ATE(sim3) %.3f m | ATE(se3) %.3f m", sid, m["ate_sim3"], m["ate_se3"])

    keys = next(iter(per_seq.values())).keys()
    agg = {k: float(np.mean([s[k] for s in per_seq.values()])) for k in keys}

    payload = {"model": args.model, "owner": OWNERS.get(args.model, "?"),
               "split": args.split, "seed": cfg.get("seed", 42),
               "model_config": cfg.get("model", {}),
               "params_m": model.num_params(), "commit": git_sha(),
               **agg, "per_sequence": per_seq}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", out_dir / "metrics.json")

    if depth_loaders:
        per_depth = {sid: evaluate_depth_sequence(model, loader, device)
                     for sid, loader in depth_loaders.items()}
        depth_payload = depth_report(ck, args.ckpt, cfg, args.split,
                                     args.rgb_dataset_version, args.depth_dataset_version,
                                     device, per_depth)
        depth_path = out_dir / "depth_metrics.json"
        depth_path.write_text(json.dumps(depth_payload, indent=2, allow_nan=False))
        log.info("wrote %s", depth_path)


if __name__ == "__main__":
    main()
