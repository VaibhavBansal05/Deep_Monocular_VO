"""Contract tests for the shared base.

These run on every PR. They guard the interfaces the four models depend on,
so one member's branch cannot silently break the other three.
"""

import inspect

import pytest
import torch

from mvo.common.data import splits
from mvo.common.train import registry
from mvo.common.train.base_model import BaseVOModel


def test_splits_are_disjoint_by_scene():
    train = set(splits.SPLITS["train"]["scenes"])
    val = set(splits.SPLITS["val"]["scenes"])
    test = set(splits.SPLITS["test"]["scenes"])
    assert not (train & val), "train/val scene leakage"
    assert not (train & test), "train/test scene leakage"
    assert not (val & test), "val/test scene leakage"


def test_test_split_is_clone_only():
    """All four models must be scored on identical trajectories."""
    assert splits.SPLITS["test"]["variations"] == ["clone"]


def test_robustness_variations_never_seen_in_training():
    seen = set(splits.SPLITS["train"]["variations"])
    held = set(splits.SPLITS["robustness"]["variations"])
    assert not (seen & held), "robustness ablation is contaminated"


def test_sequences_helper():
    seqs = splits.sequences("train")
    assert len(seqs) == 3 * 4
    assert ("Scene01", "clone") in seqs


def test_base_model_contract_is_stable():
    """If this test needs changing, all four models need changing. Discuss first."""
    assert inspect.isabstract(BaseVOModel)
    assert {"forward", "compute_loss"} <= set(BaseVOModel.__abstractmethods__)
    assert BaseVOModel.seq_len == 2
    assert BaseVOModel.needs_depth is False


def test_registry_round_trip():
    @registry.register("_dummy")
    class _Dummy(BaseVOModel):
        def forward(self, batch):
            b, s = batch["imgs"].shape[:2]
            return {"pose": torch.zeros(b, s - 1, 6)}

        def compute_loss(self, batch, out):
            return {"total": out["pose"].abs().mean()}

    m = registry.build("_dummy")
    out = m({"imgs": torch.zeros(2, 2, 3, 128, 416)})
    assert out["pose"].shape == (2, 1, 6), "pose must be (B, S-1, 6)"
    assert "total" in m.compute_loss({}, out)
    registry._REGISTRY.pop("_dummy")


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        registry.build("does_not_exist")
