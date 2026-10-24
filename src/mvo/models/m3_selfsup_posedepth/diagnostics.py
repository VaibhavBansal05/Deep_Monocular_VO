"""M3 smoke checks and evidence export; training uses the shared Trainer.

Run from src: python -m mvo.models.m3_selfsup_posedepth.diagnostics --help
Smoke checkpoints are disposable diagnostics, not benchmark results.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mvo.common.data.vkitti import VKitti2
from mvo.common.io.trajectory import read_tum
from mvo.common.metrics.ate import align
from mvo.common.train.engine import Trainer, set_seed, to_device
from mvo.models.m3_selfsup_posedepth.losses import (
    reconstruct_scale,
    select_reprojection,
    target_to_sources,
)
from mvo.models.m3_selfsup_posedepth.model import M3PoseDepth


def load_config(path):
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
    cfg.update(yaml.safe_load(Path(path).read_text()))
    return cfg


def smoke(cfg, out, steps):
    """Exercise the actual shared trainer on real RGB without a full training run."""
    cfg = dict(cfg, epochs=1, selection_key="total", scheduler="none", augment=False)
    set_seed(cfg.get("seed", 42))
    model = M3PoseDepth(**cfg["model"])
    common = dict(root=cfg["data_root"], seq_len=3, need_depth=False,
                  img_size=cfg.get("img_size", (128, 416)), augment=False)
    train, val = VKitti2(split="train", **common), VKitti2(split="val", **common)
    bs = cfg.get("batch_size", 2)
    count = min(steps * bs, len(train))
    if count < bs:
        raise ValueError("not enough samples for a training batch")
    # Spread a short check over a route rather than scoring only its first frames.
    indices = np.linspace(0, min(len(train) - 1, max(count, 100)), count, dtype=int).tolist()
    loaders = [DataLoader(Subset(ds, ids), batch_size=bs, shuffle=False, num_workers=0)
               for ds, ids in ((train, indices), (val, list(range(min(bs, len(val))))))]
    monitored = [model.depthnet.encoder.stem[0].weight, model.posenet.encoder.stem[0].weight]
    before = [p.detach().clone() for p in monitored]
    trainer = Trainer(model, *loaders, cfg, out)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.fit()
    changes = [float((p.detach().cpu() - old.cpu()).abs().max())
               for p, old in zip(monitored, before)]
    if any(value == 0 for value in changes):
        raise RuntimeError("a network did not update; inspect active_fraction and gradients")
    with (Path(out) / "history.csv").open() as f:
        row = list(csv.DictReader(f))[-1]
    if float(row["train_active_fraction"]) == 0:
        raise RuntimeError("no active photometric pixels; this smoke check did not train on RGB")
    report = {"kind": "real_RGB_smoke_only_not_a_training_result", "steps": len(loaders[0]),
              "pretrained": cfg["model"].get("pretrained", True), "seconds": time.perf_counter() - start,
              "params_m": model.num_params(), "device": str(trainer.device),
              "encoder_max_updates": dict(zip(("depth", "pose"), changes)),
              "history": row,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
              if trainer.device.type == "cuda" else None}
    (Path(out) / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    model = M3PoseDepth(**dict(cfg["model"], pretrained=False))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), cfg


def _imshow_rgb(ax, image):
    """Render CHW RGB tensors, including boolean masks, with a supported dtype."""
    ax.imshow(image.permute(1, 2, 0).float().cpu().numpy())


@torch.no_grad()
def visualize(checkpoint, root, split, out, device, with_depth=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg = checkpoint_model(checkpoint, device)
    ds = VKitti2(root or cfg["data_root"], split, seq_len=3, need_depth=with_depth,
                augment=False, img_size=cfg.get("img_size", (128, 416)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(ds) - 1, min(4, len(ds)), dtype=int).tolist()
    loader = DataLoader(Subset(ds, indices), batch_size=1)
    reconstruction_figure, axes = plt.subplots(len(indices), 4, figsize=(14, 3 * len(indices)), squeeze=False)
    depth_columns = 3 if with_depth else 2
    depth_figure, depth_axes = plt.subplots(len(indices), depth_columns,
                                           figsize=(5 * depth_columns, 3 * len(indices)), squeeze=False)
    for row, batch in enumerate(loader):
        batch = to_device(batch, device)
        result = model(batch)
        rgb = model.denormalize(batch["imgs"])
        warped, errors, identity, valid = reconstruct_scale(
            rgb, result["depth"], target_to_sources(result["pose"].float()),
            batch["K"], batch["K_inv"], model.loss_options["ssim_weight"])
        _, _, active_fraction, mask = select_reprojection(
            errors, identity, valid, model.loss_options["automask"], model.loss_options["min_reproj"])
        for col, (image, title) in enumerate([
            (rgb[0, 1], f"target frame {int(batch['frame_idx'][0, 1])}"),
            (warped[0, 0], "previous -> target"), (warped[0, 1], "next -> target"),
            (mask[0].expand(3, -1, -1), f"active pixels: {float(active_fraction):.1%}")]):
            _imshow_rgb(axes[row, col], image)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
        depth_axes[row, 0].imshow(rgb[0, 1].permute(1, 2, 0).cpu().numpy())
        prediction = result["depth"][0, 0].cpu().numpy()
        if with_depth:
            gt = batch["depth"][0, 1, 0].cpu().numpy()
            valid_gt = (gt > 1e-3) & (gt < 80)
            if valid_gt.any():
                prediction = prediction * np.median(gt[valid_gt]) / np.median(prediction[valid_gt])
            depth_axes[row, 2].imshow(np.ma.masked_where(~valid_gt, gt), cmap="magma", vmin=0, vmax=80)
            depth_axes[row, 2].set_title("GT metres, valid pixels only")
        depth_axes[row, 1].imshow(prediction, cmap="magma", vmin=0, vmax=80 if with_depth else None)
        depth_axes[row, 1].set_title("GT-median scaled metres" if with_depth else "predicted depth, arbitrary scale")
        for ax in depth_axes[row]:
            ax.axis("off")
    for figure, name in [(reconstruction_figure, "reprojection_examples.png"), (depth_figure, "depth_examples.png")]:
        figure.tight_layout()
        figure.savefig(out / name, dpi=150)
        plt.close(figure)
    history = Path(checkpoint).parent / "history.csv"
    if history.exists():
        with history.open() as f:
            rows = list(csv.DictReader(f))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for key in ("train_total", "val_total"):
            axes[0].plot([int(r["epoch"]) for r in rows], [float(r[key]) for r in rows], label=key)
        for key in ("train_active_fraction", "val_active_fraction"):
            axes[1].plot([int(r["epoch"]) for r in rows], [float(r[key]) for r in rows], label=key)
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "loss_curves.png", dpi=150)
        plt.close(fig)
    for path in out.glob("pred_*.txt"):
        sid = path.stem.removeprefix("pred_").replace("_", "/", 1)
        if sid not in ds.meta:
            continue
        predicted, truth = read_tum(path)[:, :3, 3], ds.absolute_poses(sid)[:, :3, 3]
        if len(predicted) != len(truth):
            raise ValueError("trajectory length differs from ground truth")
        corrected, _ = align(predicted, truth, with_scale=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(truth[:, 0], truth[:, 2], label="GT")
        ax.plot(corrected[:, 0], corrected[:, 2], label="M3, Sim(3)-aligned")
        ax.set(xlabel="x [m]", ylabel="z [m]", aspect="equal")
        ax.legend()
        fig.savefig(out / "trajectory.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "visualize"))
    parser.add_argument("--config", default="configs/m3_smoke.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ckpt")
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "robustness"))
    parser.add_argument("--device")
    parser.add_argument("--with-depth", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.mode == "smoke":
        cfg = load_config(args.config)
        if args.data_root:
            cfg["data_root"] = args.data_root
        if args.device:
            cfg["device"] = args.device
        smoke(cfg, args.out, args.steps)
    else:
        if not args.ckpt:
            parser.error("visualize requires --ckpt")
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        visualize(args.ckpt, args.data_root, args.split, args.out, device, args.with_depth)


if __name__ == "__main__":
    main()
