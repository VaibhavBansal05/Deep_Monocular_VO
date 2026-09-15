import numpy as np

from mvo.common.metrics.ate import umeyama


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(60, 3))
    angle = 0.7
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.array([1.0, -2.0, 3.0])
    dst = 2.5 * (rotation @ src.T).T + translation

    estimated_rotation, estimated_translation, estimated_scale = umeyama(src, dst)

    assert np.allclose(estimated_rotation, rotation, atol=1e-8)
    assert np.allclose(estimated_translation, translation, atol=1e-8)
    assert np.isclose(estimated_scale, 2.5, atol=1e-8)


def test_umeyama_never_returns_a_reflection():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    dst = src.copy()
    dst[:, 0] *= -1.0

    rotation, _, _ = umeyama(src, dst)

    assert np.linalg.det(rotation) > 0.0


def test_umeyama_without_scale_keeps_unit_scale():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(30, 3))
    dst = 3.0 * src + np.array([4.0, -2.0, 1.0])

    _, _, scale = umeyama(src, dst, with_scale=False)

    assert scale == 1.0
