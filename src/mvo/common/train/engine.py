"""Generic train/val loop shared by all four models.

Nobody writes their own training loop. If your model needs something this
engine cannot express, raise it in a PR rather than forking the loop --
divergent training loops make the comparison table meaningless.
"""

import json
import os
import random
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvo.common.io.logging import CSVLogger, get_logger

log = get_logger(__name__)


def training_revision():
    """Identify this checkout even when training is launched from another directory."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parent,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rng_state():
    """Capture global RNGs used by the shared non-persistent data loaders."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            warnings.warn("CUDA is unavailable; exact resume is unavailable", RuntimeWarning)
        elif len(state["cuda"]) != torch.cuda.device_count():
            warnings.warn("CUDA device count changed; exact resume is unavailable", RuntimeWarning)
        else:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


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
        self.training_commit = training_revision()

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
                self.scaler.unscale_(self.optimizer)
                # Never let AMP silently skip an entire run. The previous M2
                # run produced finite losses but non-finite gradients, so
                # GradScaler skipped all updates and the model stayed exactly
                # at its initial mean-pose prior for 60 epochs.
                max_norm = self.clip if self.clip > 0 else float("inf")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm, error_if_nonfinite=True)
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

            score = va.get(self.selection_key, va["total"])
            if score < self.best_val:
                self.best_val = score
                self.save("best.pt", epoch)
                log.info("  new best %s %.5f", self.selection_key, self.best_val)
            self.save("last.pt", epoch)

        (self.out_dir / "summary.json").write_text(
            json.dumps({"best_val_loss": self.best_val,
                        "epochs": self.epochs,
                        "params_m": self.model.num_params()}, indent=2))
        return {"best_val_loss": self.best_val}

    def save(self, name: str, epoch: int) -> None:
        """Publish a complete epoch checkpoint without overwriting it on a failed write."""
        path = self.out_dir / name
        temporary = path.with_suffix(".pt.tmp")
        try:
            torch.save({"epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_val": self.best_val,
                        "cfg": self.cfg,
                        "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                        "scaler": self.scaler.state_dict(),
                        "rng": rng_state(), "selection_key": self.selection_key,
                        "history_fields": self.csv.fields,
                        "training_commit": self.training_commit}, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, path: str | Path) -> None:
        """Resume at the next epoch; exact replay requires the same configuration/environment.

        Keep the original run directory, including history.csv and best.pt. Global
        RNG restoration covers the shared loaders; custom generators or persistent
        worker RNGs are not captured. Legacy checkpoints permit best-effort loading.
        """
        # Load on CPU so optimizer counters keep their original placement; the
        # state loaders move model weights and optimizer buffers onto the device.
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("selection_key", self.selection_key) != self.selection_key:
            raise ValueError("checkpoint selection_key differs from the current configuration")
        if "scheduler" in ck:
            saved_name = str(ck.get("cfg", {}).get("scheduler", "none")).lower()
            current_name = str(self.cfg.get("scheduler", "none")).lower()
            if (saved_name != current_name
                    or (ck["scheduler"] is None) != (self.scheduler is None)):
                raise ValueError("checkpoint scheduler differs from the current configuration")
        # Validate history before changing any model, optimiser, scheduler or RNG state.
        self.csv.resume(ck.get("epoch", -1), expected_fields=ck.get("history_fields"))
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.best_val = ck.get("best_val", float("inf"))
        self.start_epoch = ck.get("epoch", -1) + 1
        self.training_commit = ck.get("training_commit", "unknown")
        required = {"scheduler", "scaler", "rng", "selection_key", "training_commit"}
        missing = required - ck.keys()
        if missing:
            warnings.warn("legacy checkpoint lacks resume state; exact resume is unavailable: "
                          + ", ".join(sorted(missing)),
                          RuntimeWarning)
        if "scheduler" in ck and self.scheduler is not None:
            self.scheduler.load_state_dict(ck["scheduler"])
        if "scaler" in ck:
            self.scaler.load_state_dict(ck["scaler"])
        if "rng" in ck:
            restore_rng(ck["rng"])
