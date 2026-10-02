"""Deterministic classical correspondence front end for M1.

This module owns the OpenCV portion of M1 so feature precomputation, the MLP,
and the essential-matrix baseline all use exactly the same correspondences and
pose convention.  It deliberately contains no dataset, training, or metric
code; those are provided by :mod:`mvo.common`.

The recovered pose follows the project-wide convention: ``T_{t->t+1}``, with
translation expressed in camera ``t+1`` coordinates.  As with every monocular
essential-matrix estimate, translation has unit direction but unknown scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np

_OPENCV_RNG_LOCK = Lock()


@dataclass(frozen=True)
class FrontendResult:
    """Result of processing one image pair.

    ``descriptor`` is the fixed-length grid summary consumed by the future
    MLP. ``rotation`` and ``translation_direction`` are the pure classical
    essential-matrix estimate.  Failed estimates remain finite and carry an
    explicit reason instead of leaking NaNs into a cache or training run.
    """

    descriptor: np.ndarray
    rotation: np.ndarray
    translation_direction: np.ndarray
    points0: np.ndarray
    points1: np.ndarray
    inlier_mask: np.ndarray
    success: bool
    failure_reason: str | None
    detected0: int
    detected1: int
    correspondences: int
    inliers: int


class ClassicalFrontend:
    """ORB matching or Shi--Tomasi/Lucas--Kanade followed by five-point RANSAC."""

    _DETECTORS = {"orb", "shitomasi"}

    def __init__(
        self,
        detector: str = "orb",
        max_features: int = 1000,
        grid: tuple[int, int] = (8, 8),
        ratio_test: float = 0.75,
        min_correspondences: int = 8,
        min_inliers: int = 8,
        ransac_threshold: float = 1.0,
        ransac_confidence: float = 0.999,
        shi_quality: float = 0.01,
        shi_min_distance: float = 7.0,
        lk_window: int = 21,
        lk_levels: int = 3,
        lk_fb_threshold: float = 1.0,
        seed: int = 42,
    ) -> None:
        detector = detector.lower().replace("-", "").replace("_", "")
        if detector not in self._DETECTORS:
            raise ValueError(f"detector must be one of {sorted(self._DETECTORS)}, got {detector!r}")
        if max_features < 1:
            raise ValueError("max_features must be positive")
        if len(grid) != 2 or grid[0] < 1 or grid[1] < 1:
            raise ValueError(f"grid must contain two positive dimensions, got {grid!r}")
        if not 0.0 < ratio_test < 1.0:
            raise ValueError("ratio_test must lie strictly between zero and one")
        if min_correspondences < 5 or min_inliers < 5:
            raise ValueError(
                "five-point geometry requires at least five correspondences and inliers"
            )
        if ransac_threshold <= 0.0 or not 0.0 < ransac_confidence < 1.0:
            raise ValueError("invalid essential-matrix RANSAC parameters")
        if lk_window < 3 or lk_levels < 0 or lk_fb_threshold <= 0.0:
            raise ValueError("invalid Lucas--Kanade parameters")

        self.detector = detector
        self.max_features = int(max_features)
        self.grid = (int(grid[0]), int(grid[1]))
        self.ratio_test = float(ratio_test)
        self.min_correspondences = int(min_correspondences)
        self.min_inliers = int(min_inliers)
        self.ransac_threshold = float(ransac_threshold)
        self.ransac_confidence = float(ransac_confidence)
        self.shi_quality = float(shi_quality)
        self.shi_min_distance = float(shi_min_distance)
        self.lk_window = int(lk_window)
        self.lk_levels = int(lk_levels)
        self.lk_fb_threshold = float(lk_fb_threshold)
        self.seed = int(seed)

    @property
    def descriptor_dim(self) -> int:
        """Number of MLP inputs: mean dx, mean dy, and inlier density per cell."""
        return self.grid[0] * self.grid[1] * 3

    def extract(self, image0: np.ndarray, image1: np.ndarray, K: np.ndarray) -> FrontendResult:
        """Extract correspondences and estimate motion from an RGB or grayscale pair."""
        gray0 = self._to_gray(image0)
        gray1 = self._to_gray(image1)
        if gray0.shape != gray1.shape:
            raise ValueError(f"image shapes must match, got {gray0.shape} and {gray1.shape}")

        if self.detector == "orb":
            points0, points1, detected0, detected1 = self._orb_correspondences(gray0, gray1)
        else:
            points0, points1, detected0, detected1 = self._lk_correspondences(gray0, gray1)

        return self.estimate_from_correspondences(
            points0,
            points1,
            K,
            gray0.shape,
            detected0=detected0,
            detected1=detected1,
        )

    def estimate_from_correspondences(
        self,
        points0: np.ndarray,
        points1: np.ndarray,
        K: np.ndarray,
        image_shape: tuple[int, int] | tuple[int, int, int],
        *,
        detected0: int | None = None,
        detected1: int | None = None,
    ) -> FrontendResult:
        """Estimate ``T_{0->1}`` from matched pixels.

        This public geometry boundary lets tests use exact synthetic matches and
        lets the baseline reuse the estimator without duplicating OpenCV calls.
        """
        p0 = self._as_points(points0)
        p1 = self._as_points(points1)
        if len(p0) != len(p1):
            raise ValueError(f"point counts must match, got {len(p0)} and {len(p1)}")

        finite = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
        if not finite.all():
            return self._failure(
                "non_finite_correspondences",
                p0[finite],
                p1[finite],
                len(p0) if detected0 is None else int(detected0),
                len(p1) if detected1 is None else int(detected1),
            )

        height, width = int(image_shape[0]), int(image_shape[1])
        if height < 1 or width < 1:
            raise ValueError(f"invalid image shape {image_shape!r}")

        camera = np.asarray(K, dtype=np.float64)
        if camera.shape != (3, 3) or not np.isfinite(camera).all():
            raise ValueError("K must be a finite 3x3 matrix")

        n = len(p0)
        d0 = n if detected0 is None else int(detected0)
        d1 = n if detected1 is None else int(detected1)
        if n < self.min_correspondences:
            return self._failure("too_few_correspondences", p0, p1, d0, d1)

        # OpenCV's RANSAC uses process-global RNG state. Serialising only this
        # short section makes repeated calls deterministic even when callers
        # precompute several sequences concurrently.
        with _OPENCV_RNG_LOCK:
            cv2.setRNGSeed(self.seed)
            try:
                essential, ransac_mask = cv2.findEssentialMat(
                    p0,
                    p1,
                    camera,
                    method=cv2.RANSAC,
                    prob=self.ransac_confidence,
                    threshold=self.ransac_threshold,
                )
            except cv2.error:
                return self._failure("essential_matrix_failed", p0, p1, d0, d1)

            if essential is None or ransac_mask is None:
                return self._failure("essential_matrix_failed", p0, p1, d0, d1)

            essential = np.asarray(essential, dtype=np.float64)
            if (
                essential.ndim != 2
                or essential.shape[0] < 3
                or essential.shape[1] != 3
                or not np.isfinite(essential).all()
            ):
                return self._failure("invalid_essential_matrix", p0, p1, d0, d1)
            # findEssentialMat may stack multiple 3x3 solutions vertically.
            essential = essential[:3]
            mask = np.asarray(ransac_mask, dtype=np.uint8).reshape(-1, 1)

            try:
                _, rotation, translation, pose_mask = cv2.recoverPose(
                    essential, p0, p1, camera, mask=mask.copy()
                )
            except cv2.error:
                return self._failure("pose_recovery_failed", p0, p1, d0, d1)

        if pose_mask is None:
            return self._failure("pose_recovery_failed", p0, p1, d0, d1)
        inlier_mask = np.asarray(pose_mask, dtype=np.uint8).reshape(-1) > 0
        inlier_count = int(inlier_mask.sum())
        if inlier_count < self.min_inliers:
            return self._failure("too_few_inliers", p0, p1, d0, d1, inlier_mask=inlier_mask)

        rotation = np.asarray(rotation, dtype=np.float32)
        direction = np.asarray(translation, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(rotation).all() or not np.isfinite(direction).all():
            return self._failure("non_finite_pose", p0, p1, d0, d1, inlier_mask=inlier_mask)
        if norm <= np.finfo(np.float32).eps:
            return self._failure("near_zero_translation", p0, p1, d0, d1, inlier_mask=inlier_mask)
        direction /= norm

        descriptor = self._grid_descriptor(p0[inlier_mask], p1[inlier_mask], height, width)
        return FrontendResult(
            descriptor=descriptor,
            rotation=rotation,
            translation_direction=direction,
            points0=p0,
            points1=p1,
            inlier_mask=inlier_mask,
            success=True,
            failure_reason=None,
            detected0=d0,
            detected1=d1,
            correspondences=n,
            inliers=inlier_count,
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            gray = array
        elif array.ndim == 3 and array.shape[2] == 3:
            if np.issubdtype(array.dtype, np.floating):
                scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
                array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
            elif array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"expected HxW or HxWx3 image, got shape {array.shape}")

        if np.issubdtype(gray.dtype, np.floating):
            scale = 255.0 if float(gray.max(initial=0.0)) <= 1.0 else 1.0
            gray = np.clip(gray * scale, 0.0, 255.0).astype(np.uint8)
        elif gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(gray)

    @staticmethod
    def _as_points(points: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if array.ndim == 3 and array.shape[1:] == (1, 2):
            array = array[:, 0]
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"points must have shape (N,2) or (N,1,2), got {array.shape}")
        return np.ascontiguousarray(array)

    def _orb_correspondences(
        self, gray0: np.ndarray, gray1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        orb = cv2.ORB_create(nfeatures=self.max_features)
        keypoints0, descriptors0 = orb.detectAndCompute(gray0, None)
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        n0, n1 = len(keypoints0), len(keypoints1)
        if descriptors0 is None or descriptors1 is None or n0 < 2 or n1 < 2:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy(), n0, n1

        candidates = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors0, descriptors1, k=2)
        good = [
            pair[0]
            for pair in candidates
            if len(pair) == 2 and pair[0].distance < self.ratio_test * pair[1].distance
        ]
        good.sort(key=lambda match: (match.distance, match.queryIdx, match.trainIdx))

        # The ratio test is one-way and can map several source features to one
        # destination feature. Greedy uniqueness removes that ambiguity while
        # retaining the strongest deterministic match.
        unique = []
        used_queries: set[int] = set()
        used_trains: set[int] = set()
        for match in good:
            if match.queryIdx in used_queries or match.trainIdx in used_trains:
                continue
            unique.append(match)
            used_queries.add(match.queryIdx)
            used_trains.add(match.trainIdx)

        p0 = np.asarray([keypoints0[m.queryIdx].pt for m in unique], dtype=np.float32)
        p1 = np.asarray([keypoints1[m.trainIdx].pt for m in unique], dtype=np.float32)
        return p0.reshape(-1, 2), p1.reshape(-1, 2), n0, n1

    def _lk_correspondences(
        self, gray0: np.ndarray, gray1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        corners = cv2.goodFeaturesToTrack(
            gray0,
            maxCorners=self.max_features,
            qualityLevel=self.shi_quality,
            minDistance=self.shi_min_distance,
        )
        if corners is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy(), 0, 0

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        lk = {
            "winSize": (self.lk_window, self.lk_window),
            "maxLevel": self.lk_levels,
            "criteria": criteria,
        }
        forward, status_f, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, corners, None, **lk)
        if forward is None or status_f is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy(), len(corners), 0
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, forward, None, **lk)
        if backward is None or status_b is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy(), len(corners), 0

        source = corners.reshape(-1, 2)
        target = forward.reshape(-1, 2)
        reverse = backward.reshape(-1, 2)
        fb_error = np.linalg.norm(source - reverse, axis=1)
        height, width = gray0.shape
        valid = status_f.reshape(-1).astype(bool) & status_b.reshape(-1).astype(bool)
        valid &= np.isfinite(target).all(axis=1) & (fb_error <= self.lk_fb_threshold)
        valid &= (target[:, 0] >= 0.0) & (target[:, 0] < width)
        valid &= (target[:, 1] >= 0.0) & (target[:, 1] < height)
        return source[valid], target[valid], len(corners), int(valid.sum())

    def _grid_descriptor(
        self,
        points0: np.ndarray,
        points1: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        rows, cols = self.grid
        descriptor = np.zeros((rows, cols, 3), dtype=np.float32)
        if len(points0) == 0:
            return descriptor.reshape(-1)

        x_cell = np.clip((points0[:, 0] * cols / width).astype(np.int64), 0, cols - 1)
        y_cell = np.clip((points0[:, 1] * rows / height).astype(np.int64), 0, rows - 1)
        flow = points1 - points0
        flow[:, 0] /= float(width)
        flow[:, 1] /= float(height)

        for row in range(rows):
            for col in range(cols):
                selected = (y_cell == row) & (x_cell == col)
                count = int(selected.sum())
                if count:
                    descriptor[row, col, :2] = flow[selected].mean(axis=0)
                    descriptor[row, col, 2] = count / float(self.max_features)
        return descriptor.reshape(-1)

    def _failure(
        self,
        reason: str,
        points0: np.ndarray,
        points1: np.ndarray,
        detected0: int,
        detected1: int,
        *,
        inlier_mask: np.ndarray | None = None,
    ) -> FrontendResult:
        if inlier_mask is None:
            inlier_mask = np.zeros(len(points0), dtype=bool)
        else:
            inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        return FrontendResult(
            descriptor=np.zeros(self.descriptor_dim, dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
            translation_direction=np.zeros(3, dtype=np.float32),
            points0=points0,
            points1=points1,
            inlier_mask=inlier_mask,
            success=False,
            failure_reason=reason,
            detected0=int(detected0),
            detected1=int(detected1),
            correspondences=len(points0),
            inliers=int(inlier_mask.sum()),
        )
