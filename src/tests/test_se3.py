"""Tests for mvo/common/geometry/se3.py.

OWNER: Abhishek Laxman Jadhav  (branch: base/se3-utils)

These run on every PR. If any of these fail, all four models are suspect.
test_absolute_relative_round_trip is the single most important test in the repo.
"""

import pytest
import torch

from mvo.common.geometry import se3


def _random_poses(n=8, scale=0.3):
    g = torch.Generator().manual_seed(0)
    return torch.cat([torch.randn(n, 3, generator=g) * scale,
                      torch.randn(n, 3, generator=g) * scale], dim=-1).double()


def test_exp_log_round_trip():
    w = torch.randn(16, 3).double() * 0.5
    assert torch.allclose(se3.so3_log(se3.so3_exp(w)), w, atol=1e-9)


def test_vec_mat_round_trip():
    v = _random_poses()
    assert torch.allclose(se3.mat_to_vec(se3.vec_to_mat(v)), v, atol=1e-9)


def test_exp_is_a_valid_rotation():
    R = se3.so3_exp(torch.randn(32, 3).double())
    eye = torch.eye(3).double().expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-9)
    assert torch.allclose(torch.linalg.det(R), torch.ones(32).double(), atol=1e-9)


@pytest.mark.parametrize("mag", [0.0, 1e-12, 1e-8, 1e-4])
def test_zero_and_tiny_rotations_are_stable(mag):
    """The pose head starts near zero -- this must not produce NaN."""
    w = torch.full((4, 3), mag, dtype=torch.float64, requires_grad=True)
    R = se3.so3_exp(w)
    assert torch.isfinite(R).all()
    R.sum().backward()
    assert torch.isfinite(w.grad).all(), "NaN gradient at small angle"


def test_invert_is_a_true_inverse():
    T = se3.vec_to_mat(_random_poses())
    eye = torch.eye(4).double().expand_as(T)
    assert torch.allclose(se3.compose(T, se3.invert(T)), eye, atol=1e-9)


def test_absolute_relative_round_trip():
    """The single most important test in the repo."""
    rel = se3.vec_to_mat(_random_poses(10))
    abs_poses = se3.relative_to_absolute(rel)
    assert abs_poses.shape == (11, 4, 4)
    assert torch.allclose(se3.absolute_to_relative(abs_poses), rel, atol=1e-9)


def test_identity_relative_poses_mean_a_stationary_camera():
    rel = torch.eye(4).double().expand(5, 4, 4)
    traj = se3.relative_to_absolute(rel)
    assert torch.allclose(traj[:, :3, 3], torch.zeros(6, 3).double(), atol=1e-12)


def test_rotation_angle_matches_axis_angle_norm():
    w = torch.randn(16, 3).double() * 0.4
    ang = se3.rotation_angle_deg(se3.so3_exp(w))
    assert torch.allclose(ang, torch.rad2deg(w.norm(dim=-1)), atol=1e-8)
