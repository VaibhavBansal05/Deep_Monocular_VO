import pytest
import torch

from mvo.models.m3_selfsup_posedepth.depthnet import DepthEncoder, DepthNet, disparity_to_depth


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_encoder_and_disparity_shapes_at_project_resolution():
    net = DepthNet(pretrained=False).eval()
    images = torch.randn(1, 3, 128, 416)
    with torch.no_grad():
        features = net.encoder(images)
        disparities = net(images)
    assert [tuple(f.shape[1:]) for f in features] == [
        (64, 64, 208), (64, 32, 104), (128, 16, 52), (256, 8, 26), (512, 4, 13)]
    assert [tuple(d.shape) for d in disparities] == [
        (1, 1, 128, 416), (1, 1, 64, 208), (1, 1, 32, 104), (1, 1, 16, 52)]
    for d in disparities:
        depth = disparity_to_depth(d)
        assert torch.isfinite(depth).all() and (depth >= 0.1).all() and (depth <= 100).all()


def test_every_depth_head_and_encoder_get_finite_gradients():
    net = DepthNet(pretrained=False)
    sum(d.mean() for d in net(torch.randn(2, 3, 64, 96))).backward()
    for parameter in [net.encoder.stem[0].weight, *[h[1].weight for h in net.heads.values()]]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_depth_conversion_bounds_and_half_precision():
    bounds = disparity_to_depth(torch.tensor([0., 1.]))
    torch.testing.assert_close(bounds, torch.tensor([100., 0.1]))
    d = torch.tensor([0.1, 1., 0.5], dtype=torch.float16, requires_grad=True)
    depth = disparity_to_depth(d)
    assert depth.dtype == torch.float32
    assert torch.isfinite(depth).all()
    depth.sum().backward()
    assert torch.isfinite(d.grad).all()


def test_pretraining_flag_and_download_failure_are_not_silenced(monkeypatch):
    from torchvision.models import ResNet18_Weights, resnet18

    import mvo.models.m3_selfsup_posedepth.depthnet as module
    seen = []

    def fake(*, weights):
        seen.append(weights)
        return resnet18(weights=None)

    monkeypatch.setattr(module, "resnet18", fake)
    DepthEncoder(pretrained=False)
    DepthEncoder(pretrained=True)
    assert seen == [None, ResNet18_Weights.IMAGENET1K_V1]

    def fail(**kwargs):
        raise OSError("weights unavailable")

    monkeypatch.setattr(module, "resnet18", fail)
    with pytest.raises(OSError, match="weights unavailable"):
        DepthEncoder(pretrained=True)


def test_single_scale_and_unsupported_inputs():
    net = DepthNet(pretrained=False, num_scales=1).eval()
    with torch.no_grad():
        assert len(net(torch.randn(1, 3, 65, 97))) == 1
    with pytest.raises(ValueError, match="num_scales"):
        DepthNet(pretrained=False, num_scales=2)
    with pytest.raises(ValueError, match="64 pixels"):
        net(torch.randn(1, 3, 32, 64))
