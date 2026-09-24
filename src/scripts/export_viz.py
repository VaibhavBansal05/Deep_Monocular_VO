#!/usr/bin/env python3
"""Export everything the results visualiser needs.

Produces, under the output directory:
  frames.jpg   a sprite sheet of the test sequence, drawn frame by frame in
               the browser to play it back as video
  data.json    ground-truth and predicted trajectories (aligned for display),
               per-frame errors, drift curves, training histories and metrics

    python scripts/export_viz.py --out ../viz_assets
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from mvo.common.data import splits
from mvo.common.io.trajectory import read_tum
from mvo.common.metrics.ate import umeyama

SEQ = "Scene20/clone"
STRIDE = 3           # every 3rd frame keeps the sheet ~1.5 MB
TILE_W, TILE_H = 240, 72
COLS = 16


def build_sprite(data_root: Path, out: Path) -> dict:
    scene, variation = SEQ.split("/")
    rgb = data_root / scene / variation / "frames" / "rgb" / splits.CAMERA
    files = sorted(rgb.glob("rgb_*.jpg"))[::STRIDE]
    rows = (len(files) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE_W, rows * TILE_H))
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS)
        sheet.paste(im, ((i % COLS) * TILE_W, (i // COLS) * TILE_H))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=72, optimize=True)
    return {"count": len(files), "stride": STRIDE, "cols": COLS, "rows": rows,
            "tileW": TILE_W, "tileH": TILE_H,
            "sizeMB": round(out.stat().st_size / 1e6, 2)}


def align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """sim(3)-align a predicted trajectory onto ground truth, as ATE does, so
    the two can be overlaid. Without this every prediction starts at the origin
    while ground truth starts at its world pose."""
    n = min(len(pred), len(gt))
    R, t, s = umeyama(pred[:n, :3, 3], gt[:n, :3, 3], with_scale=True)
    out = pred[:n].copy()
    out[:, :3, 3] = (s * (R @ pred[:n, :3, 3].T)).T + t
    out[:, :3, :3] = R @ pred[:n, :3, :3]
    return out


def drift_curve(pred: np.ndarray, gt: np.ndarray, step: int = 10) -> list:
    """Positional error against distance travelled -- how drift accumulates."""
    d = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))])
    idx = range(0, min(len(pred), len(gt)), step)
    return [{"dist": round(float(d[i]), 1),
             "err": round(float(np.linalg.norm(pred[i, :3, 3] - gt[i, :3, 3])), 3)}
            for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/vkitti2")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="viz_assets")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = Path(args.results)

    sprite = build_sprite(Path(args.data_root), out / "frames.jpg")
    print(f"sprite: {sprite['count']} frames, {sprite['sizeMB']} MB")

    from mvo.common.data.vkitti import VKitti2
    gt = VKitti2(root=args.data_root, split="test", seq_len=2).absolute_poses(SEQ)

    def pack(poses):
        """positions plus quaternions, so the 3D view can draw real camera
        orientation rather than inferring heading from the path."""
        q = Rotation.from_matrix(poses[:, :3, :3]).as_quat()   # x,y,z,w
        return {"p": [[round(float(v), 3) for v in x[:3, 3]] for x in poses],
                "q": [[round(float(v), 4) for v in r] for r in q]}

    payload = {"sequence": SEQ, "sprite": sprite,
               "gt": pack(gt), "models": {}, "baselines": {}, "metrics": {}}

    labels = {"m1": "M1 Classical + MLP", "m2": "M2 CNN + LSTM",
              "m3": "M3 Self-supervised", "m4": "M4 CNN + CBAM"}

    for d in sorted(res.glob("*/metrics.json")):
        tag = d.parent.name
        m = json.loads(d.read_text())
        traj = d.parent / f"pred_{SEQ.replace('/', '_')}.txt"
        if not traj.exists():
            continue
        pred = align(read_tum(traj), gt)
        payload["models"][tag] = {
            "label": labels.get(tag, tag),
            **pack(pred),
            "drift": drift_curve(pred, gt),
            "err": [round(float(np.linalg.norm(pred[i, :3, 3] - gt[i, :3, 3])), 3)
                    for i in range(0, len(pred), STRIDE)],
        }
        payload["metrics"][tag] = {k: m[k] for k in (
            "ate_sim3", "ate_se3", "rpe_trans_rmse", "rpe_rot_rmse_deg",
            "t_rel_pct", "r_rel_deg_per_100m", "params_m") if k in m}

        hist = d.parent / "history.csv"
        if hist.exists():
            import csv
            rows = list(csv.DictReader(open(hist)))
            payload["models"][tag]["history"] = [
                {"epoch": int(r["epoch"]),
                 "train": round(float(r["train_total"]), 5),
                 "val": round(float(r["val_total"]), 5)} for r in rows]

    bl = res / "baselines.json"
    if bl.exists():
        payload["baselines"] = json.loads(bl.read_text()).get(SEQ, {})

    (out / "data.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"data.json: {(out / 'data.json').stat().st_size / 1e6:.2f} MB")
    print("models:", list(payload["models"]))


if __name__ == "__main__":
    main()
