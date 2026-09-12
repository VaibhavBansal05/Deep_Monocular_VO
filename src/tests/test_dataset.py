import numpy as np
import torch

from mvo.common.data.transforms import hflip_pose, rescale_intrinsics


def test_rescale_intrinsics_preserves_principal_point_ratio():
    K = np.array([[725.0, 0, 620.5], [0, 725.0, 187.0], [0, 0, 1]], dtype=np.float32)
    Ks = rescale_intrinsics(K, (128, 416))
    assert np.isclose(Ks[0, 2] / 416, K[0, 2] / 1242, atol=1e-5)
    assert np.isclose(Ks[1, 2] / 128, K[1, 2] / 375, atol=1e-5)


def test_hflip_pose_is_an_involution():
    p = np.array([0.1, -0.2, 1.5, 0.01, -0.03, 0.02], dtype=np.float32)
    assert np.allclose(hflip_pose(hflip_pose(p)), p)


def test_hflip_flips_only_tx_ry_rz():
    p = np.arange(6, dtype=np.float32) + 1
    f = hflip_pose(p)
    assert f[0] == -p[0] and f[4] == -p[4] and f[5] == -p[5]
    assert f[1] == p[1] and f[2] == p[2] and f[3] == p[3]