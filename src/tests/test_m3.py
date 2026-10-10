import pytest
import torch
from torch.utils.data import DataLoader

from mvo.common.data.transforms import normalize
from mvo.common.train.engine import Trainer
from mvo.common.train.registry import available, build, import_all_models
from mvo.models.m3_selfsup_posedepth.model import M3PoseDepth


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _batch():
    K = torch.tensor([[[50., 0, 47.5], [0, 50., 31.5], [0, 0, 1.]]]).expand(2, -1, -1)
    return {"imgs": normalize(torch.rand(2, 3, 3, 64, 96)), "K": K,
            "K_inv": torch.linalg.inv(K)}


def test_registry_interface_and_label_independence():
    import_all_models()
    assert "m3" in available()
    model = build("m3", pretrained=False).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch)
        loss = model.compute_loss(batch, out)
        labelled = {**batch, "rel_pose": torch.randn(2, 2, 4, 4),
                    "abs_pose": torch.randn(2, 3, 4, 4), "depth": torch.randn(2, 3, 1, 64, 96)}
        again = model(labelled)
        other_loss = model.compute_loss(labelled, again)
        torch.testing.assert_close(out["pose"], again["pose"])
        for key in loss:
            torch.testing.assert_close(loss[key], other_loss[key])
        torch.testing.assert_close(model.predict_depth(batch["imgs"][:, 1]), out["depth"])
    assert out["pose"].shape == (2, 2, 6) and out["depth"].shape == (2, 1, 64, 96)
    assert not model.needs_depth
    assert all(v.ndim == 0 and torch.isfinite(v) for v in loss.values())
    rgb = model.denormalize(batch["imgs"])
    assert rgb.min() >= 0 and rgb.max() <= 1


def test_shared_trainer_updates_both_networks_without_depth(tmp_path):
    torch.manual_seed(42)
    model = M3PoseDepth(pretrained=False, num_scales=1, automask=False)
    batch = _batch()
    samples = [{k: v[i] for k, v in batch.items()} for i in range(2)]
    loader = DataLoader(samples, batch_size=2)
    parameters = [model.depthnet.encoder.stem[0].weight, model.posenet.encoder.stem[0].weight]
    before = [p.detach().clone() for p in parameters]
    trainer = Trainer(model, loader, loader, {"epochs": 1, "amp": False, "device": "cpu",
                                            "selection_key": "val_ate", "grad_clip": 1.},
                      tmp_path, val_hook=lambda model: {"val_ate": 1.})
    trainer.fit()
    for old, parameter in zip(before, parameters):
        assert not torch.equal(old, parameter) and torch.isfinite(parameter.grad).all()
    assert (tmp_path / "best.pt").exists()
    restored = M3PoseDepth(pretrained=False, num_scales=1, automask=False)
    restored.load_state_dict(torch.load(tmp_path / "best.pt", weights_only=False)["model"])
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["pose"], restored(batch)["pose"])


@pytest.mark.parametrize("kwargs", [{"seq_len": 5}, {"backbone": "resnet34"},
                                    {"min_depth": 0}, {"ssim_weight": 2}])
def test_invalid_baseline_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        M3PoseDepth(pretrained=False, **kwargs)


def test_three_frame_evaluation_keeps_exactly_836_scene20_transitions():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/eval.py"
    spec = importlib.util.spec_from_file_location("m3_pose_eval", path)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    truth = torch.zeros(836, 6)
    truth[:, 2] = torch.linspace(0.1, 0.5, 836)

    class Replay:
        def __init__(self):
            self.index = 0

        def eval(self):
            pass

        def __call__(self, batch):
            pose = truth[self.index:self.index + 2].unsqueeze(0)
            self.index += 1
            return {"pose": pose}

    relative, _ = evaluation.predict_sequence(Replay(), [{"imgs": torch.zeros(1)}] * 835,
                                               torch.device("cpu"))
    assert relative.shape == (836, 4, 4)
    torch.testing.assert_close(torch.from_numpy(relative[:, 2, 3]), truth[:, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
def test_cuda_denormalization_buffers_and_fp32_backward():
    model = M3PoseDepth(pretrained=False, num_scales=1).cuda()
    batch = {k: v.cuda() for k, v in _batch().items()}
    loss = model.compute_loss(batch, model(batch))
    assert model.rgb_mean.is_cuda and loss["total"].dtype == torch.float32
    loss["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
