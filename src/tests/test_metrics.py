import numpy as np

from mvo.common.metrics.ate import ate, umeyama


def _traj(n=400, span=200.0):
    t = np.linspace(0, span, n)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.sin(t) * 10
    poses[:, 2, 3] = t * 5
    for i in range(n):
        angle = 0.1 * t[i]
        poses[i, :3, :3] = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
    return poses


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


def test_perfect_prediction_scores_zero():
    poses = _traj()

    assert ate(poses[:, :3, 3], poses[:, :3, 3])["rmse"] < 1e-9


def test_sim3_absorbs_a_global_scale_error_but_se3_does_not():
    gt = _traj()[:, :3, 3]
    pred = gt * 0.5

    assert ate(pred, gt, with_scale=True)["rmse"] < 1e-9
    assert ate(pred, gt, with_scale=False)["rmse"] > 1.0
