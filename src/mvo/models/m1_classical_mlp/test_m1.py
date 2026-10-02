"""Contract and geometry tests for the M1 classical front end."""

import cv2
import numpy as np
import pytest

from mvo.models.m1_classical_mlp.cache import FeatureCache
from mvo.models.m1_classical_mlp.frontend import ClassicalFrontend

IMAGE_SHAPE = (128, 416)
K = np.array([[400.0, 0.0, 208.0], [0.0, 400.0, 64.0], [0.0, 0.0, 1.0]])


def _project(points: np.ndarray, camera: np.ndarray) -> np.ndarray:
    homogeneous = (camera @ points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _synthetic_correspondences():
    """Non-planar points viewed under a known T_0->1 transform."""
    rng = np.random.default_rng(7)
    points3d = np.column_stack(
        (
            rng.uniform(-2.0, 2.0, 300),
            rng.uniform(-0.6, 0.6, 300),
            rng.uniform(5.0, 18.0, 300),
        )
    )
    rotation_vector = np.array([0.01, -0.025, 0.008])
    rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = np.array([0.15, 0.02, -0.45])

    points0 = _project(points3d, K)
    points1 = _project((rotation @ points3d.T).T + translation, K)
    height, width = IMAGE_SHAPE
    visible = (
        (points0[:, 0] >= 0.0)
        & (points0[:, 0] < width)
        & (points0[:, 1] >= 0.0)
        & (points0[:, 1] < height)
        & (points1[:, 0] >= 0.0)
        & (points1[:, 0] < width)
        & (points1[:, 1] >= 0.0)
        & (points1[:, 1] < height)
    )
    return points0[visible], points1[visible], rotation, translation


def _rotation_error(estimated: np.ndarray, expected: np.ndarray) -> float:
    residual_vector, _ = cv2.Rodrigues(estimated @ expected.T)
    return float(np.linalg.norm(residual_vector))


def test_synthetic_correspondences_recover_t_to_t_plus_one_pose():
    points0, points1, expected_rotation, expected_translation = _synthetic_correspondences()
    result = ClassicalFrontend().estimate_from_correspondences(points0, points1, K, IMAGE_SHAPE)

    assert result.success, result.failure_reason
    assert _rotation_error(result.rotation, expected_rotation) < 1e-3
    expected_direction = expected_translation / np.linalg.norm(expected_translation)
    assert float(result.translation_direction @ expected_direction) > 0.999


def test_mirrored_correspondences_flip_tx_ry_rz_signs():
    points0, points1, expected_rotation, expected_translation = _synthetic_correspondences()
    _, width = IMAGE_SHAPE
    mirrored0, mirrored1 = points0.copy(), points1.copy()
    mirrored0[:, 0] = (width - 1) - mirrored0[:, 0]
    mirrored1[:, 0] = (width - 1) - mirrored1[:, 0]
    mirrored_K = K.copy()
    mirrored_K[0, 2] = (width - 1) - mirrored_K[0, 2]

    result = ClassicalFrontend().estimate_from_correspondences(
        mirrored0, mirrored1, mirrored_K, IMAGE_SHAPE
    )

    assert result.success, result.failure_reason
    reflection = np.diag([-1.0, 1.0, 1.0])
    expected_mirrored_rotation = reflection @ expected_rotation @ reflection
    expected_mirrored_translation = reflection @ expected_translation
    expected_mirrored_translation /= np.linalg.norm(expected_mirrored_translation)
    assert _rotation_error(result.rotation, expected_mirrored_rotation) < 1e-3
    assert float(result.translation_direction @ expected_mirrored_translation) > 0.999

    original_rotation_vector, _ = cv2.Rodrigues(expected_rotation)
    mirrored_rotation_vector, _ = cv2.Rodrigues(result.rotation)
    np.testing.assert_allclose(
        mirrored_rotation_vector.reshape(3),
        original_rotation_vector.reshape(3) * np.array([1.0, -1.0, -1.0]),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "points0, points1",
    [
        (np.zeros((4, 2)), np.zeros((4, 2))),
        (np.zeros((12, 2)), np.zeros((12, 2))),
        (
            np.vstack((np.full((1, 2), np.nan), np.zeros((11, 2)))),
            np.zeros((12, 2)),
        ),
    ],
)
def test_degenerate_correspondences_return_finite_controlled_failure(points0, points1):
    result = ClassicalFrontend().estimate_from_correspondences(points0, points1, K, IMAGE_SHAPE)

    assert not result.success
    assert result.failure_reason
    assert np.isfinite(result.descriptor).all()
    assert np.isfinite(result.rotation).all()
    assert np.isfinite(result.translation_direction).all()


def test_descriptor_has_fixed_192_values_and_is_finite():
    points0, points1, _, _ = _synthetic_correspondences()
    result = ClassicalFrontend(grid=(8, 8)).estimate_from_correspondences(
        points0, points1, K, IMAGE_SHAPE
    )

    assert result.success
    assert result.descriptor.shape == (192,)
    assert result.descriptor.dtype == np.float32
    assert np.isfinite(result.descriptor).all()


@pytest.mark.parametrize("detector", ["orb", "shitomasi"])
def test_image_pair_extraction_is_deterministic(detector):
    rng = np.random.default_rng(19)
    image0 = rng.integers(0, 256, size=(*IMAGE_SHAPE, 3), dtype=np.uint8)
    image1 = np.roll(image0, 3, axis=1)
    frontend = ClassicalFrontend(detector=detector, max_features=400)

    first = frontend.extract(image0, image1, K)
    second = frontend.extract(image0, image1, K)

    assert first.success == second.success
    assert first.failure_reason == second.failure_reason
    assert np.array_equal(first.points0, second.points0)
    assert np.array_equal(first.points1, second.points1)
    assert np.array_equal(first.inlier_mask, second.inlier_mask)
    assert np.array_equal(first.descriptor, second.descriptor)
    assert np.array_equal(first.rotation, second.rotation)
    assert np.array_equal(first.translation_direction, second.translation_direction)


def test_feature_cache_round_trip_preserves_results(tmp_path):
    points0, points1, _, _ = _synthetic_correspondences()
    frontend = ClassicalFrontend()
    success = frontend.estimate_from_correspondences(points0, points1, K, IMAGE_SHAPE)
    failure = frontend.estimate_from_correspondences(
        np.zeros((4, 2)), np.zeros((4, 2)), K, IMAGE_SHAPE
    )
    config = {"detector": "orb", "grid": (8, 8), "max_features": 1000}
    cache = FeatureCache.from_results(
        ["Scene01/clone:00000:00001", "Scene01/clone:00001:00002"],
        [success, failure],
        frontend_config=config,
    )
    path = tmp_path / "features.npz"

    cache.save(path)
    loaded = FeatureCache.load(path, expected_descriptor_dim=192, expected_frontend_config=config)

    assert len(loaded) == 2
    np.testing.assert_array_equal(
        loaded["Scene01/clone:00000:00001"].descriptor, success.descriptor
    )
    failed = loaded["Scene01/clone:00001:00002"]
    assert not failed.success
    assert failed.failure_reason == "too_few_correspondences"
    assert failed.correspondences == 4


def test_feature_cache_rejects_duplicate_keys():
    points0, points1, _, _ = _synthetic_correspondences()
    result = ClassicalFrontend().estimate_from_correspondences(points0, points1, K, IMAGE_SHAPE)

    with pytest.raises(ValueError, match="unique"):
        FeatureCache.from_results(["same", "same"], [result, result])


def test_feature_cache_rejects_stale_frontend_configuration(tmp_path):
    points0, points1, _, _ = _synthetic_correspondences()
    result = ClassicalFrontend().estimate_from_correspondences(points0, points1, K, IMAGE_SHAPE)
    cache = FeatureCache.from_results(
        ["Scene01/clone:00000:00001"],
        [result],
        frontend_config={"detector": "orb"},
    )
    path = tmp_path / "features.npz"
    cache.save(path)

    with pytest.raises(ValueError, match="configuration"):
        FeatureCache.load(path, expected_frontend_config={"detector": "shitomasi"})
