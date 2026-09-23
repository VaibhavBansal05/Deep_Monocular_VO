#!/usr/bin/env python3
"""Single entry point for training any of the four models.

    python scripts/train.py --model m4 --config configs/m4.yaml

Nobody adds a second training script. If your model needs a knob, add it to
your YAML config -- that keeps the four models comparable and keeps the diff
in your own folder.
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from mvo.common.data.vkitti import VKitti2
from mvo.common.io.logging import get_logger
from mvo.common.train.engine import Trainer, set_seed
from mvo.common.train.registry import available, build, import_all_models

log = get_logger("train")


def load_config(path: str) -> dict:
    """Load a model YAML, merged over configs/base.yaml."""
    base = yaml.safe_load(Path("configs/base.yaml").read_text())
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    base.update(cfg)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="m1 | m2 | m3 | m4")
    ap.add_argument("--config", default=None, help="defaults to configs/<model>.yaml")
    ap.add_argument("--data-root", default=None, help="overrides cfg.data_root")
    ap.add_argument("--out", default=None, help="defaults to checkpoints/<model>")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="dataloader workers; lower it on hosts with few vCPUs "
                         "(Kaggle allocates 2-4, and oversubscribing slows loading)")
    args = ap.parse_args()

    import_all_models()
    if args.model not in available():
        raise SystemExit(f"model {args.model!r} not implemented yet. Available: {available()}")

    cfg = load_config(args.config or f"configs/{args.model}.yaml")
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    set_seed(cfg.get("seed", 42))

    model = build(args.model, **cfg.get("model", {}))
    log.info("built %s (%.2fM params, seq_len=%d, needs_depth=%s)",
             args.model, model.num_params(), model.seq_len, model.needs_depth)

    common = dict(root=cfg["data_root"], seq_len=model.seq_len,
                  need_depth=model.needs_depth)
    train_ds = VKitti2(split="train", augment=cfg.get("augment", True), **common)
    val_ds = VKitti2(split="val", augment=False, **common)

    loader_kw = dict(batch_size=cfg.get("batch_size", 8),
                     num_workers=cfg.get("num_workers", 4),
                     pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    out_dir = Path(args.out or f"checkpoints/{args.model}")
    trainer = Trainer(model, train_loader, val_loader, cfg, out_dir)
    if args.resume:
        trainer.load(args.resume)
    trainer.fit()
    log.info("done -> %s", out_dir)


if __name__ == "__main__":
    main()
