"""Generic train/val loop shared by all four models.

Nobody writes their own training loop. If your model needs something this
engine cannot express, raise it in a PR rather than forking the loop --
divergent training loops make the comparison table meaningless.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvo.common.io.logging import CSVLogger, get_logger

log = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed every RNG. Same seed across all four models -> fair comparison."""
    # Required by deterministic CUDA matrix multiplication on supported GPUs;
    # it must be set before the first CUDA GEMM is launched.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only preserves portability if a future model uses an operation with
    # no deterministic CUDA implementation, while making that limitation loud.
    torch.use_deterministic_algorithms(True, warn_only=True)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


class Trainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader,
                 cfg: dict, out_dir: str | Path, val_hook=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(cfg.get("device")
                                   or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self.epochs = int(cfg.get("epochs", 30))
        self.amp = bool(cfg.get("amp", True)) and self.device.type == "cuda"
        self.clip = float(cfg.get("grad_clip", 0.0))

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        # Per-frame validation loss is a poor selector here: validation is a
        # single short sequence (232 samples), so the loss is noisy enough that
        # an early lucky epoch can beat every later one -- M4 once selected
        # epoch 2 of 25 that way. val_hook may return trajectory-level metrics,
        # and selection_key chooses which to minimise.
        self.val_hook = val_hook
        self.selection_key = str(cfg.get("selection_key", "total"))

        self.csv = CSVLogger(self.out_dir / "history.csv")
        self.best_val = float("inf")
        self.start_epoch = 0

    def _build_optimizer(self):
        name = self.cfg.get("optimizer", "adam").lower()
        lr = float(self.cfg.get("lr", 1e-4))
        wd = float(self.cfg.get("weight_decay", 0.0))
        params = [p for p in self.model.parameters() if p.requires_grad]
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"unknown optimizer {name!r}")

    def _build_scheduler(self):
        name = str(self.cfg.get("scheduler", "none")).lower()
        if name == "none":
            return None
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(self.cfg.get("lr_step", 15)),
                gamma=float(self.cfg.get("lr_gamma", 0.1)))
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs)
        raise ValueError(f"unknown scheduler {name!r}")

    # ------------------------------------------------------------------ #

    def _run_epoch(self, loader, train: bool) -> dict:
        self.model.train(train)
        sums: dict[str, float] = {}
        n = 0
        t0 = time.time()

        for batch in loader:
            batch = to_device(batch, self.device)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast("cuda", enabled=self.amp):
                    out = self.model(batch)
                    losses = self.model.compute_loss(batch, out)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(losses["total"]).backward()
                if self.clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            bs = batch["imgs"].shape[0]
            n += bs
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * bs

        stats = {k: v / max(n, 1) for k, v in sums.items()}
        stats["sec"] = time.time() - t0
        return stats

    def fit(self) -> dict:
        log.info("training on %s | %.2fM params | %d epochs",
                 self.device, self.model.num_params(), self.epochs)

        for epoch in range(self.start_epoch, self.epochs):
            # Augmentation is a deterministic function of (seed, epoch, idx).
            # Set this before workers are spawned for the epoch.
            set_epoch = getattr(self.train_loader.dataset, "set_epoch", None)
            if set_epoch is not None:
                set_epoch(epoch)
            tr = self._run_epoch(self.train_loader, train=True)
            va = self._run_epoch(self.val_loader, train=False)
            if self.val_hook is not None:
                va.update(self.val_hook(self.model))
            if self.scheduler is not None:
                self.scheduler.step()

            row = {"epoch": epoch,
                   **{f"train_{k}": v for k, v in tr.items()},
                   **{f"val_{k}": v for k, v in va.items()},
                   "lr": self.optimizer.param_groups[0]["lr"]}
            self.csv.append(row)
            extra = ""
            if self.selection_key != "total" and self.selection_key in va:
                extra = f" | {self.selection_key} {va[self.selection_key]:.4f}"
            log.info("epoch %3d | train %.5f | val %.5f%s | %.1fs",
                     epoch, tr["total"], va["total"], extra, tr["sec"])

            self.save("last.pt", epoch)
            score = va.get(self.selection_key, va["total"])
            if score < self.best_val:
                self.best_val = score
                self.save("best.pt", epoch)
                log.info("  new best %s %.5f", self.selection_key, self.best_val)

        (self.out_dir / "summary.json").write_text(
            json.dumps({"best_val_loss": self.best_val,
                        "epochs": self.epochs,
                        "params_m": self.model.num_params()}, indent=2))
        return {"best_val_loss": self.best_val}

    def save(self, name: str, epoch: int) -> None:
        torch.save({"epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_val": self.best_val,
                    "cfg": self.cfg}, self.out_dir / name)

    def load(self, path: str | Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.best_val = ck.get("best_val", float("inf"))
        self.start_epoch = ck.get("epoch", -1) + 1
