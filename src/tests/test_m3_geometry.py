import numpy as np
import pytest
import torch

from mvo.common.data.transforms import rescale_intrinsics
from mvo.common.geometry.projection import inverse_warp
from mvo.common.geometry.se3 import invert, vec_to_mat
from mvo.models.m3_selfsup_posedepth.losses import target_to_sources


def test_middle_target_uses_matrix_inverse_for_previous_reference():
    pose = torch.tensor([[[0.4, -0.2, 0.5, 0., 0.3, 0.],
                          [0.1, 0.2, -0.5, 0.1, 0., 0.]]])
    got = target_to_sources(pose)
    T = vec_to_mat(pose)
    torch.testing.assert_close(got[:, 0], invert(T[:, 0]))
    torch.testing.assert_close(got[:, 1], T[:, 1])
    assert not torch.allclose(got[:, 0], vec_to_mat(-pose[:, 0]))


@pytest.mark.parametrize("scale", [0, 1, 2, 3])
def test_intrinsics_and_known_pixel_translation_at_each_scale(scale):
    H, W = 128 // 2**scale, 416 // 2**scale
    K0 = np.array([[200., 0, 207.5], [0, 200., 63.5], [0, 0, 1.]], dtype=np.float32)
    Ks = rescale_intrinsics(K0, (H, W), in_hw=(128, 416))
    assert np.allclose(Ks, np.diag([W / 416, H / 128, 1]) @ K0)
    K = torch.from_numpy(Ks).unsqueeze(0)
    x = torch.arange(W).view(1, 1, 1, W).expand(1, 3, H, W).float()
    depth = torch.full((1, 1, H, W), 5.)
    pose = torch.zeros(1, 6)
    pose[:, 0] = 5 / K[0, 0, 0]
    warp, valid = inverse_warp(x, depth, vec_to_mat(pose), K, torch.linalg.inv(K))
    torch.testing.assert_close((warp - x)[valid.expand_as(x).bool()],
                               torch.ones_like(warp)[valid.expand_as(x).bool()], atol=1e-4, rtol=0)


def test_identity_invalid_geometry_and_tiny_pose_gradients():
    image = torch.rand(1, 3, 16, 24)
    K = torch.tensor([[[20., 0, 11.5], [0, 20., 7.5], [0, 0, 1.]]])
    depth = torch.full((1, 1, 16, 24), 5., requires_grad=True)
    pose = torch.zeros(1, 6, requires_grad=True)
    warp, valid = inverse_warp(image, depth, vec_to_mat(pose), K, torch.linalg.inv(K))
    torch.testing.assert_close(warp * valid, image * valid, atol=2e-6, rtol=0)
    (warp * valid).sum().backward()
    assert torch.isfinite(pose.grad).all() and torch.isfinite(depth.grad).all()
    for offset in ([0, 0, -10, 0, 0, 0], [100, 0, 0, 0, 0, 0]):
        _, mask = inverse_warp(image, depth, vec_to_mat(torch.tensor([offset]).float()),
                               K, torch.linalg.inv(K))
        assert mask.sum() == 0
