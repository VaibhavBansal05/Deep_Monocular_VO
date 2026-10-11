"""Epoch-boundary resume through the shared trainer, without model weights or data."""

import copy
import csv
import multiprocessing
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mvo.common.io.logging import CSVLogger
from mvo.common.train import engine
from mvo.common.train.engine import Trainer, restore_rng, rng_state, set_seed


@pytest.fixture(autouse=True)
def preserve_deterministic_settings():
    """Keep checkpoint tests' set_seed calls from changing subsequent tests."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Dropout(0.2),
                                        torch.nn.Linear(8, 1))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def forward(self, batch):
        if self.training:
            self.updates.add_(1)
        return self.head(batch["imgs"])

    def compute_loss(self, batch, out):
        # Dropout, shuffle and these factors exercise all three global RNGs.
        factor = 1 + random.random() + np.random.random()
        return {"total": ((out - batch["imgs"]) ** 2).mean() * factor}

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6


class Samples(Dataset):
    """Like the shared dataset, augmentation depends on epoch and sample identity."""

    def __init__(self):
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 8

    def __getitem__(self, index):
        jitter = np.random.default_rng([42, self.epoch, index]).uniform(-0.01, 0.01)
        return {"imgs": torch.tensor([index / 8. + jitter], dtype=torch.float32)}


def make_trainer(path, *, workers=0, val_hook=None, **overrides):
    # Keep the platform's default worker mode. On Python 3.14/Linux, preload
    # torch once in the forkserver instead of reimporting CUDA libraries per worker.
    if workers and multiprocessing.get_start_method() == "forkserver":
        multiprocessing.set_forkserver_preload(["torch"])
    loader_kw = {"batch_size": 2, "num_workers": workers}
    train = DataLoader(Samples(), shuffle=True, **loader_kw)
    val = DataLoader(Samples(), **loader_kw)
    cfg = {"device": "cpu", "amp": False, "epochs": 4,
           "scheduler": "step", "lr_step": 2, "lr": 0.01, **overrides}
    return Trainer(Toy(), train, val, cfg, path, val_hook=val_hook)


def assert_exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            assert_exact(left, right)
    else:
        assert a == b


def state(trainer):
    return copy.deepcopy({"model": trainer.model.state_dict(),
                          "optimizer": trainer.optimizer.state_dict(),
                          "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
                          "scaler": trainer.scaler.state_dict(), "best": trainer.best_val,
                          "epoch": trainer.start_epoch, "commit": trainer.training_commit,
                          "rng": rng_state()})


def next_random_draws():
    return random.random(), np.random.random(3), torch.rand(3)


def history(path):
    with path.open(newline="") as stream:
        return [{key: value for key, value in row.items() if not key.endswith("_sec")}
                for row in csv.DictReader(stream)]


@pytest.mark.parametrize("scheduler", ["step", "cosine", "none"])
@pytest.mark.parametrize("workers", [0, 2])
def test_epoch_boundary_resume_replays_complete_training_state_and_history(tmp_path, scheduler, workers):
    set_seed(42)
    full = make_trainer(tmp_path / "full", scheduler=scheduler, workers=workers)
    full.fit()
    expected = state(full)
    expected_draws = next_random_draws()

    set_seed(42)
    part = make_trainer(tmp_path / "resumed", scheduler=scheduler, workers=workers)
    # Keep the original horizon configured, especially for cosine scheduling.
    part.epochs = 2
    part.fit()
    old_history = (part.out_dir / "history.csv").read_bytes()
    resumed = make_trainer(part.out_dir, scheduler=scheduler, workers=workers)
    resumed.load(part.out_dir / "last.pt")
    assert resumed.start_epoch == 2
    resumed.fit()

    actual = state(resumed)
    # start_epoch records the resume point rather than the completed epoch.
    actual["epoch"] = expected["epoch"]
    assert_exact(expected, actual)
    assert_exact(expected_draws, next_random_draws())
    assert (resumed.out_dir / "history.csv").read_bytes().startswith(old_history)
    assert history(full.out_dir / "history.csv") == history(resumed.out_dir / "history.csv")
    assert [int(row["epoch"]) for row in history(resumed.out_dir / "history.csv")] == [0, 1, 2, 3]
    for name in ("best.pt", "last.pt"):
        assert_exact(torch.load(full.out_dir / name, weights_only=False),
                     torch.load(resumed.out_dir / name, weights_only=False))
    assert resumed.training_commit == part.training_commit != "unknown"


@pytest.mark.parametrize("scheduler", ["step", "cosine", "none"])
def test_checkpoint_restores_state_before_any_further_training(tmp_path, scheduler):
    source = make_trainer(tmp_path / "run", scheduler=scheduler)
    source.epochs = 2
    source.fit()
    expected = state(source)
    expected["epoch"] = 2
    expected_draws = next_random_draws()
    resumed = make_trainer(source.out_dir, scheduler=scheduler)
    resumed.load(source.out_dir / "last.pt")
    assert_exact(expected, state(resumed))
    assert_exact(expected_draws, next_random_draws())


def test_old_checkpoints_still_load_with_explicit_best_effort_warning(tmp_path):
    source = make_trainer(tmp_path / "source")
    source.epochs = 1
    source.fit()
    old = tmp_path / "old.pt"
    torch.save({"model": source.model.state_dict(), "optimizer": source.optimizer.state_dict(),
                "epoch": 0, "best_val": source.best_val}, old)
    resumed = make_trainer(tmp_path / "destination")
    with pytest.warns(RuntimeWarning, match="legacy checkpoint.*exact resume is unavailable"):
        resumed.load(old)
    assert resumed.start_epoch == 1
    assert resumed.training_commit == "unknown"
    assert_exact(source.model.state_dict(), resumed.model.state_dict())
    assert_exact(source.optimizer.state_dict(), resumed.optimizer.state_dict())


@pytest.mark.parametrize("scheduler", ["cosine", "none"])
def test_resume_rejects_real_scheduler_change_before_mutating_training_state(tmp_path, scheduler):
    source = make_trainer(tmp_path / "source")
    source.save("last.pt", 0)
    resumed = make_trainer(tmp_path / "destination", scheduler=scheduler)
    before = state(resumed)
    with pytest.raises(ValueError, match="scheduler differs"):
        resumed.load(source.out_dir / "last.pt")
    assert_exact(before, state(resumed))


def test_resume_rejects_scheduler_presence_mismatch_before_mutating_state(tmp_path):
    source = make_trainer(tmp_path / "source")
    source.save("last.pt", 0)
    ck = torch.load(source.out_dir / "last.pt", weights_only=False)
    ck["scheduler"] = None
    torch.save(ck, source.out_dir / "last.pt")
    resumed = make_trainer(tmp_path / "destination")
    before = state(resumed)
    with pytest.raises(ValueError, match="scheduler differs"):
        resumed.load(source.out_dir / "last.pt")
    assert_exact(before, state(resumed))


def test_scheduler_names_use_the_same_normalization_as_construction(tmp_path):
    source = make_trainer(tmp_path / "source", scheduler="STEP")
    source.save("last.pt", 0)
    resumed = make_trainer(tmp_path / "destination", scheduler="step")
    resumed.load(source.out_dir / "last.pt")
    assert_exact(source.scheduler.state_dict(), resumed.scheduler.state_dict())


def test_selection_key_change_is_rejected_before_mutating_state(tmp_path):
    source = make_trainer(tmp_path / "source", selection_key="val_ate")
    source.save("last.pt", 0)
    resumed = make_trainer(tmp_path / "destination")
    before = state(resumed)
    with pytest.raises(ValueError, match="selection_key differs"):
        resumed.load(source.out_dir / "last.pt")
    assert_exact(before, state(resumed))


def test_best_and_last_share_updated_best_score_without_replacing_prior_best(tmp_path):
    scores = iter([2., 1., 3.])
    trainer = make_trainer(tmp_path, epochs=3, selection_key="val_ate",
                           val_hook=lambda model: {"val_ate": next(scores)})
    trainer.fit()
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "last.pt", weights_only=False)
    assert best["epoch"] == 1 and last["epoch"] == 2
    assert best["best_val"] == last["best_val"] == trainer.best_val == 1.
    assert best["selection_key"] == last["selection_key"] == "val_ate"


@pytest.mark.parametrize("contents", [
    "epoch,loss\n2,1\n",  # newer than checkpoint
    "epoch,loss\n0,1\n0,2\n",  # duplicate
    "epoch,loss\n1,1\n0,2\n",  # out of order
    "epoch,loss\n-1,1\n",  # negative epoch
    "epoch,loss\nbad,1\n",  # invalid epoch
    "loss\n1\n",  # missing epoch column
    "epoch,epoch\n0,0\n",  # duplicate header
    "epoch,loss\n0\n",  # missing field
    "epoch,loss\n0,1,extra\n",  # extra field
    'epoch,loss\n0,"1\n',  # unterminated quoted field
])
def test_invalid_history_fails_before_mutating_state_or_log(tmp_path, contents):
    source = make_trainer(tmp_path / "source")
    source.save("last.pt", 1)
    resumed = make_trainer(tmp_path / "destination")
    resumed.csv.path.write_text(contents)
    before = state(resumed)
    with pytest.raises(ValueError, match="history"):
        resumed.load(source.out_dir / "last.pt")
    assert_exact(before, state(resumed))
    assert resumed.csv.path.read_text() == contents
    assert resumed.csv.fields is None


def test_checkpoint_history_schema_mismatch_fails_before_mutating_training_state(tmp_path):
    source = make_trainer(tmp_path / "run")
    source.epochs = 1
    source.fit()
    contents = "epoch,another_loss\n0,1\n"
    source.csv.path.write_text(contents)
    resumed = make_trainer(source.out_dir)
    before = state(resumed)
    with pytest.raises(ValueError, match="history schema changed"):
        resumed.load(source.out_dir / "last.pt")
    assert_exact(before, state(resumed))
    assert resumed.csv.path.read_text() == contents


@pytest.mark.parametrize("contents", [None, "", "epoch,loss\n", "epoch,loss\n0,1\n"])
def test_missing_empty_header_only_or_older_history_can_continue(tmp_path, contents):
    path = tmp_path / "history.csv"
    if contents is not None:
        path.write_text(contents)
    logger = CSVLogger(path)
    logger.resume(1)
    logger.append({"epoch": 2, "loss": 0.5})
    if contents:
        assert path.read_text().startswith(contents)
    assert [row["epoch"] for row in history(path)] == (["0", "2"] if contents and "0,1" in contents else ["2"])


@pytest.mark.parametrize("row", [{"epoch": 4, "another_loss": 2.}, {"epoch": 3, "loss": 2.}])
def test_history_schema_change_or_repeated_epoch_does_not_corrupt_log(tmp_path, row):
    logger = CSVLogger(tmp_path / "history.csv")
    logger.append({"epoch": 3, "loss": 1.})
    before = logger.path.read_bytes()
    resumed = CSVLogger(logger.path)
    resumed.resume(3)
    with pytest.raises(ValueError, match="schema changed|duplicates"):
        resumed.append(row)
    assert logger.path.read_bytes() == before


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_checkpoint_publication_preserves_previous_file(tmp_path, monkeypatch, failure):
    trainer = make_trainer(tmp_path)
    trainer.save("last.pt", 0)
    before = (tmp_path / "last.pt").read_bytes()

    if failure == "write":
        def broken_save(payload, path):
            Path(path).write_bytes(b"incomplete checkpoint")
            raise OSError("simulated write failure")
        monkeypatch.setattr(torch, "save", broken_save)
    else:
        def broken_replace(source, destination):
            raise OSError("simulated replace failure")
        monkeypatch.setattr(Path, "replace", broken_replace)
    with pytest.raises(OSError, match="simulated"):
        trainer.save("last.pt", 1)
    assert (tmp_path / "last.pt").read_bytes() == before
    assert not (tmp_path / "last.pt.tmp").exists()


def test_training_revision_is_checkout_relative_and_falls_back_without_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert engine.training_revision() != "unknown"

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")
    monkeypatch.setattr(engine.subprocess, "check_output", unavailable)
    assert engine.training_revision() == "unknown"


def test_changed_cuda_device_count_warns_and_restores_cpu_rngs(monkeypatch):
    saved = rng_state()
    saved["cuda"] = [torch.zeros(1, dtype=torch.uint8)]
    expected = next_random_draws()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.warns(RuntimeWarning, match="CUDA device count changed"):
        restore_rng(saved)
    assert_exact(expected, next_random_draws())


def test_unavailable_cuda_warns_instead_of_silently_discarding_saved_rng(monkeypatch):
    saved = rng_state()
    saved["cuda"] = [torch.zeros(1, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="CUDA is unavailable"):
        restore_rng(saved)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_and_enabled_amp_scaler_are_restored(tmp_path):
    set_seed(42)
    source = make_trainer(tmp_path / "run", device="cuda", amp=True, epochs=2)
    # Make restoration distinguishable from a fresh scaler's defaults.
    source.scaler = torch.amp.GradScaler("cuda", init_scale=128., growth_interval=2)
    source.fit()
    assert source.scaler.is_enabled()
    expected = state(source)
    expected["epoch"] = 2
    expected_cuda = [torch.rand(5, device=f"cuda:{index}")
                     for index in range(torch.cuda.device_count())]
    resumed = make_trainer(source.out_dir, device="cuda", amp=True, epochs=2)
    resumed.load(source.out_dir / "last.pt")
    assert_exact(expected, state(resumed))
    assert_exact(expected_cuda, [torch.rand(5, device=f"cuda:{index}")
                                 for index in range(torch.cuda.device_count())])
