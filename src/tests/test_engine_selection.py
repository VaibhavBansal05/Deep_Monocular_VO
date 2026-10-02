"""Checkpoint selection in the training engine.

Validation here is one short sequence, so per-frame validation loss is noisy
enough that an early epoch can win by chance -- M4 once selected epoch 2 of 25
that way. The engine therefore supports selecting on a trajectory-level metric
supplied by a hook.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mvo.common.train.base_model import BaseVOModel
from mvo.common.train.engine import Trainer


class _Tiny(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return {"imgs": torch.zeros(2, 3, 8, 8),
                "rel_pose": torch.eye(4).unsqueeze(0)}


class _Toy(BaseVOModel):
    seq_len = 2

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 6)

    def forward(self, batch):
        b = batch["imgs"].shape[0]
        x = batch["imgs"].new_zeros(b, 1)
        return {"pose": self.lin(x).unsqueeze(1)}

    def compute_loss(self, batch, out):
        return {"total": out["pose"].pow(2).mean()}


def _trainer(tmp_path, cfg, hook=None):
    dl = DataLoader(_Tiny(), batch_size=2)
    return Trainer(_Toy(), dl, dl, cfg, tmp_path, val_hook=hook)


def test_defaults_to_selecting_on_total_loss(tmp_path):
    t = _trainer(tmp_path, {"epochs": 2, "device": "cpu", "amp": False})
    assert t.selection_key == "total"
    t.fit()
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()


def test_hook_metrics_reach_the_history(tmp_path):
    seen = []

    def hook(model):
        seen.append(1)
        return {"val_ate": 10.0 - len(seen)}      # improves every epoch

    t = _trainer(tmp_path, {"epochs": 3, "device": "cpu", "amp": False,
                            "selection_key": "val_ate"}, hook)
    t.fit()
    assert len(seen) == 3, "hook must run once per validation epoch"
    assert "val_ate" in (tmp_path / "history.csv").read_text()


def test_selection_follows_the_chosen_key_not_the_loss(tmp_path):
    """The decisive case: the hook metric is best at the LAST epoch while the
    loss would favour an earlier one. The saved checkpoint must follow the key.
    """
    scores = [5.0, 9.0, 1.0]
    calls = {"n": 0}

    def hook(model):
        v = scores[calls["n"] % len(scores)]
        calls["n"] += 1
        return {"val_ate": v}

    t = _trainer(tmp_path, {"epochs": 3, "device": "cpu", "amp": False,
                            "selection_key": "val_ate"}, hook)
    t.fit()
    assert t.best_val == 1.0, f"selected {t.best_val}, expected the minimum 1.0"
    ck = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == 2, "best.pt must be the epoch with the best val_ate"


def test_missing_hook_key_falls_back_to_loss(tmp_path):
    """A hook that fails to supply the key must not crash training."""
    t = _trainer(tmp_path, {"epochs": 2, "device": "cpu", "amp": False,
                            "selection_key": "val_ate"}, lambda m: {})
    t.fit()
    assert (tmp_path / "best.pt").exists()


def test_weight_decay_reaches_the_optimizer(tmp_path):
    t = _trainer(tmp_path, {"epochs": 1, "device": "cpu", "amp": False,
                            "weight_decay": 1e-4})
    assert t.optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_nonfinite_gradients_fail_instead_of_silently_skipping_steps(tmp_path):
    class NonFiniteToy(_Toy):
        def compute_loss(self, batch, out):
            return {"total": out["pose"].sum() * torch.tensor(float("nan"))}

    dl = DataLoader(_Tiny(), batch_size=2)
    t = Trainer(NonFiniteToy(), dl, dl,
                {"epochs": 1, "device": "cpu", "amp": False}, tmp_path)
    with pytest.raises(RuntimeError, match="non-finite"):
        t._run_epoch(dl, train=True)
