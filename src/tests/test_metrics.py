import numpy as np

from mvo.common.io.trajectory import read_tum, write_tum
from mvo.common.metrics.ate import ate, umeyama
from mvo.common.metrics.depth import depth_metrics
from mvo.common.metrics.rpe import kitti_drift, rpe


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
    assert type(estimated_scale) is float


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
    assert type(scale) is float


def test_perfect_prediction_scores_zero():
    poses = _traj()

    assert ate(poses[:, :3, 3], poses[:, :3, 3])["rmse"] < 1e-9
    assert rpe(poses, poses)["trans_rmse"] < 1e-9
    assert rpe(poses, poses)["rot_rmse_deg"] < 1e-9
    assert kitti_drift(poses, poses)["t_rel_pct"] < 1e-6


def test_sim3_absorbs_a_global_scale_error_but_se3_does_not():
    gt = _traj()[:, :3, 3]
    pred = gt * 0.5

    assert ate(pred, gt, with_scale=True)["rmse"] < 1e-9
    assert ate(pred, gt, with_scale=False)["rmse"] > 1.0


def test_kitti_drift_returns_nan_on_a_short_sequence():
    short = _traj(n=20, span=2.0)

    metrics = kitti_drift(short, short)

    assert np.isnan(metrics["t_rel_pct"])
    assert np.isnan(metrics["r_rel_deg_per_100m"])


def test_depth_metrics_on_a_perfect_prediction():
    gt = np.random.default_rng(2).uniform(2, 50, size=(64, 64))

    metrics = depth_metrics(gt.copy(), gt)

    assert metrics["abs_rel"] < 1e-9
    assert metrics["a1"] > 0.999


def test_tum_trajectory_round_trip(tmp_path):
    poses = _traj(n=4, span=3.0)
    path = tmp_path / "trajectory.txt"

    write_tum(path, poses)
    loaded = read_tum(path)

    assert loaded.shape == poses.shape
    assert np.allclose(loaded, poses, atol=1e-8)


def test_read_tum_supports_a_single_pose(tmp_path):
    pose = _traj(n=2, span=1.0)[1:]
    path = tmp_path / "single_pose.txt"

    write_tum(path, pose)
    loaded = read_tum(path)

    assert loaded.shape == (1, 4, 4)
    assert np.allclose(loaded, pose, atol=1e-8)
