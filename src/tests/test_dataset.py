import numpy as np

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


# --------------------------------------------------- augmentation regressions

def _train_ds(**kw):
    import pytest

    from mvo.common.data.vkitti import VKitti2
    try:
        return VKitti2(root="data/vkitti2", split="train", seq_len=2, **kw)
    except FileNotFoundError:
        pytest.skip("dataset not present")


def test_augmentation_is_reproducible_but_changes_by_epoch():
    """Same seed/epoch/index must replay exactly; a new epoch must not freeze it."""
    import torch
    a = _train_ds(augment=True, seed=123)
    b = _train_ds(augment=True, seed=123)
    assert torch.equal(a[7]["imgs"], b[7]["imgs"])
    first = a[7]["imgs"]
    a.set_epoch(1)
    assert not torch.equal(first, a[7]["imgs"]), "augmentation is frozen across epochs"


def test_augmented_samples_keep_a_stable_schema():
    """Regression: dropping abs_pose only on flipped samples made flipped and
    unflipped items carry different keys, and default_collate raised KeyError
    on the first mixed batch."""
    ds = _train_ds(augment=True)
    keys = [frozenset(ds[i].keys()) for i in range(24)]
    assert len(set(keys)) == 1, f"schema varies across samples: {set(keys)}"


def test_rel_pose_and_abs_pose_agree_under_augmentation():
    """rel_pose[i] = inv(A[i+1]) @ A[i] must hold whether or not the sample was
    flipped; the contract claims the two fields describe the same motion."""
    import torch
    ds = _train_ds(augment=True)
    for i in range(0, 60, 5):
        s = ds[i]
        E = torch.linalg.inv(s["abs_pose"])
        implied = E[1] @ torch.linalg.inv(E[0])
        assert torch.allclose(implied, s["rel_pose"][0], atol=1e-3), \
            f"sample {i}: abs_pose and rel_pose disagree"
