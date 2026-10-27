"""M3 depth evaluation contract on synthetic pairs; no dataset or weights download."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mvo.common.data import splits
from mvo.common.data.vkitti import VKitti2
from mvo.common.metrics.depth import depth_metrics


def load_eval():
    path = Path(__file__).parents[1] / "scripts/eval.py"
    spec = importlib.util.spec_from_file_location("m3_depth_eval_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_eval()


class ImageDepth:
    def eval(self):
        return self

    def predict_depth(self, images):
        return images[:, :1]


def depth_batches(n=837, zero_truth=False):
    for start in range(0, n - 1, 32):
        ids = torch.tensor([[i, i + 1] for i in range(start, min(start + 32, n - 1))])
        values = (ids % 20 + 1).float()
        images = values[..., None, None, None].expand(-1, -1, 3, 4, 4)
        truth = 2 * images[:, :, :1].clone()
        if zero_truth:
            truth.zero_()
        yield {"seq_id": ["Scene20/clone"] * len(ids), "frame_idx": ids,
               "imgs": images, "depth": truth}


def test_unique_frame_count_and_both_depth_scales():
    result = evaluation.evaluate_depth_sequence(ImageDepth(), depth_batches(), torch.device("cpu"))
    assert result["frames_seen"] == result["frames_scored"] == 837
    assert result["frames_skipped"] == 0
    assert result["median_scaled"]["abs_rel"] == 0
    assert result["median_scaled"]["a1"] == 1
    assert result["raw_scale"]["abs_rel"] == 0.5
    assert result["median_scale_factor"] == 2


def test_empty_and_nonfinite_depth_predictions_fail():
    with pytest.raises(ValueError, match="no evaluation frames"):
        evaluation.evaluate_depth_sequence(ImageDepth(), depth_batches(3, zero_truth=True),
                                           torch.device("cpu"))

    class Nonfinite(ImageDepth):
        def predict_depth(self, images):
            return torch.full_like(images[:, :1], float("nan"))

    with pytest.raises(ValueError, match="finite and positive"):
        evaluation.evaluate_depth_sequence(Nonfinite(), depth_batches(3), torch.device("cpu"))


def test_empty_frame_is_counted_and_aggregate_is_weighted_by_scored_frames():
    batches = list(depth_batches(3))
    batches[0]["depth"][0, 0].zero_()
    result = evaluation.evaluate_depth_sequence(ImageDepth(), batches, torch.device("cpu"))
    assert (result["frames_seen"], result["frames_scored"], result["frames_skipped"]) == (3, 2, 1)
    other = dict(result, frames_seen=2, frames_scored=1, frames_skipped=1,
                 raw_scale=dict(result["raw_scale"], abs_rel=0.2))
    aggregate = evaluation.aggregate_depth({"first": result, "second": other})
    assert aggregate["frames_seen"] == 5
    assert aggregate["frames_scored"] == 3
    assert aggregate["frames_skipped"] == 2
    assert aggregate["raw_scale"]["abs_rel"] == pytest.approx(0.4)


def test_depth_result_schema_records_protocol_and_full_provenance(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    revision = "a" * 40
    monkeypatch.setattr(evaluation, "full_git_sha", lambda: revision)
    sequence = evaluation.evaluate_depth_sequence(
        ImageDepth(), depth_batches(3), torch.device("cpu"))
    report = evaluation.depth_report(
        {"training_commit": "b" * 40, "epoch": 17}, checkpoint,
        {"img_size": [128, 416], "seed": 42}, "test",
        "vaibhavkrishnabansal/vkitti2-subset/1", "suryanshbajaj/depth/1",
        torch.device("cpu"), {"Scene20/clone": sequence})
    assert report["training_commit"] == "b" * 40
    assert report["evaluation_commit"] == revision
    assert report["checkpoint_epoch"] == 17
    assert len(report["checkpoint_sha256"]) == 64
    assert report["datasets"]["rgb"].endswith("/1")
    assert report["protocol"]["gt_valid_metres"] == "(0.001, 80)"
    assert report["aggregate"]["frames_scored"] == 3
    assert report["environment"]["device"] == "cpu"
    assert json.loads(json.dumps(report, allow_nan=False))["model"] == "m3"


def test_metric_valid_interval_median_scale_and_invalid_inputs():
    gt = np.array([0., 0.001, 10., 20., 80., 655.35, np.nan])
    pred = np.array([1., 1., 5., 10., 1., 1., 1.])
    scaled = depth_metrics(pred, gt)
    raw = depth_metrics(pred, gt, median_scale=False)
    assert scaled["median_scale"] == 2
    assert scaled["abs_rel"] == scaled["rmse"] == scaled["rmse_log"] == 0
    assert raw["abs_rel"] == 0.5
    assert set(("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")) <= scaled.keys()
    with pytest.raises(ValueError, match="no valid"):
        depth_metrics(np.ones(2), np.zeros(2))
    with pytest.raises(ValueError, match="finite and positive"):
        depth_metrics(np.array([np.inf]), np.array([1.]))
    with pytest.raises(ValueError, match="shapes differ"):
        depth_metrics(np.ones(2), np.ones(3))


def tiny_tree(root, count=2):
    sequence = root / "Scene02/clone"
    rgb = sequence / "frames/rgb/Camera_0"
    depth = sequence / "frames/depth/Camera_0"
    rgb.mkdir(parents=True)
    depth.mkdir(parents=True)
    extrinsic = " ".join(str(x) for x in np.eye(4).flatten())
    (sequence / "extrinsic.txt").write_text("header\n" + "".join(
        f"{i} 0 {extrinsic}\n" for i in range(count)))
    (sequence / "intrinsic.txt").write_text("header\n0 0 725 725 620.5 187\n")
    for frame in range(count):
        Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(rgb / f"rgb_{frame:05d}.jpg")
        Image.fromarray(np.array([[0, 100, 65535], [1000, 2000, 3000]], dtype=np.uint16)).save(
            depth / f"depth_{frame:05d}.png")
    return depth


def test_loader_centimetres_sentinel_and_missing_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "sequences", lambda split: [("Scene02", "clone")])
    depth = tiny_tree(tmp_path)
    sample = VKitti2(tmp_path, "val", seq_len=2, need_depth=True, img_size=(2, 3))[0]
    torch.testing.assert_close(sample["depth"][0, 0],
                               torch.tensor([[0., 1., 0.], [10., 20., 30.]]))
    loader = evaluation.depth_loader(tmp_path, "val", "Scene02/clone", img_size=(2, 3))
    assert len(loader.dataset.samples) == 1
    (depth / "depth_00001.png").unlink()
    with pytest.raises(FileNotFoundError, match="depth/RGB frame mismatch"):
        evaluation.depth_loader(tmp_path, "val", "Scene02/clone", img_size=(2, 3))


def test_depth_cli_is_m3_only_and_requires_provenance(monkeypatch):
    for args, message in [(["--model", "m4"], "only for M3"),
                          (["--model", "m3", "--split", "robustness"], "val or test"),
                          (["--model", "m3"], "dataset version")]:
        monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", "unused.pt", "--eval-depth", *args])
        with pytest.raises(SystemExit, match="2"):
            evaluation.main()


def test_pose_only_cli_does_not_require_depth_or_dataset_versions(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    torch.save({"cfg": {"model": {"pretrained": True}}, "model": {}}, checkpoint)
    seen = {}

    class Model:
        seq_len = 2
        needs_depth = False

        def load_state_dict(self, state):
            assert state == {}

        def to(self, device):
            return self

        def num_params(self):
            return 1.0

    def build(name, **options):
        seen.update(options)
        return Model()

    monkeypatch.setattr(evaluation, "build", build)
    monkeypatch.setattr(evaluation, "import_all_models", lambda: None)
    monkeypatch.setattr(splits, "sequences", lambda split: [("Scene20", "clone")])

    class Dataset:
        samples = [("Scene20/clone", 0)]

        def absolute_poses(self, sid):
            return np.stack((np.eye(4), np.eye(4)))

    monkeypatch.setattr(evaluation, "VKitti2", lambda **kwargs: Dataset())
    monkeypatch.setattr(evaluation, "DataLoader", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation, "predict_sequence", lambda *args: (np.eye(4)[None], 1.0))
    monkeypatch.setattr(evaluation, "ate", lambda *args, **kwargs: {"rmse": 0.0})
    monkeypatch.setattr(evaluation, "rpe", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "kitti_drift", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "signed_pose_bias", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "write_tum", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["eval.py", "--model", "m3", "--ckpt", str(checkpoint),
                                     "--out", str(tmp_path / "results")])
    evaluation.main()
    assert seen["pretrained"] is False
    assert (tmp_path / "results/metrics.json").is_file()
    assert not (tmp_path / "results/depth_metrics.json").exists()
    assert json.loads((tmp_path / "results/metrics.json").read_text())["model"] == "m3"
