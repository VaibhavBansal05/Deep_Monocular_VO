"""Tests for mvo/common/geometry/projection.py.

OWNER: Abhishek Laxman Jadhav  (branch: base/se3-utils)
"""

import torch

from mvo.common.geometry.projection import backproject, inverse_warp, project


def _K(B=2, H=32, W=64):
    K = torch.eye(3).expand(B, 3, 3).clone().double()
    K[:, 0, 0] = K[:, 1, 1] = 50.0
    K[:, 0, 2], K[:, 1, 2] = W / 2, H / 2
    return K, torch.linalg.inv(K)


def test_backproject_then_project_is_identity_under_no_motion():
    """The round trip must land every pixel back on itself."""
    B, H, W = 2, 32, 64
    K, K_inv = _K(B, H, W)
    depth = torch.rand(B, 1, H, W).double() * 10 + 1
    pts = backproject(depth, K_inv)
    grid, valid = project(pts, K, torch.eye(4).expand(B, 4, 4).double())

    ys, xs = torch.meshgrid(torch.arange(H).double(), torch.arange(W).double(),
                            indexing="ij")
    expect_u = 2 * xs / (W - 1) - 1
    expect_v = 2 * ys / (H - 1) - 1
    assert torch.allclose(grid[0, ..., 0], expect_u, atol=1e-6)
    assert torch.allclose(grid[0, ..., 1], expect_v, atol=1e-6)
    assert valid.mean() > 0.99


def test_identity_warp_reproduces_the_source_image():
    B, H, W = 1, 32, 64
    K, K_inv = _K(B, H, W)
    img = torch.rand(B, 3, H, W).double()
    depth = torch.ones(B, 1, H, W).double() * 5
    warped, valid = inverse_warp(img, depth, torch.eye(4).expand(B, 4, 4).double(),
                                 K, K_inv)
    m = valid > 0
    assert torch.allclose(warped * m, img * m, atol=1e-5)


def test_points_behind_the_camera_are_invalid():
    B, H, W = 1, 16, 16
    K, K_inv = _K(B, H, W)
    depth = torch.ones(B, 1, H, W).double()
    pts = backproject(depth, K_inv)
    T = torch.eye(4).expand(B, 4, 4).clone().double()
    T[:, 2, 3] = -50.0                      # push everything behind the camera
    _, valid = project(pts, K, T)
    assert valid.sum() == 0
