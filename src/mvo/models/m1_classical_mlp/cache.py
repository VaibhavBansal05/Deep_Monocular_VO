"""Versioned on-disk cache for deterministic M1 front-end outputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .frontend import FrontendResult

CACHE_SCHEMA_VERSION = 1
_DIAGNOSTIC_NAMES = ("detected0", "detected1", "correspondences", "inliers")


def _canonical_config(config: dict | None) -> dict:
    """Return the exact JSON-compatible representation stored in the archive."""
    try:
        value = json.loads(json.dumps(config or {}, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"M1 front-end configuration must be JSON serializable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("M1 front-end configuration must be a mapping")
    return value


@dataclass(frozen=True)
class CachedFeature:
    """One cached image-pair result returned by :class:`FeatureCache`."""

    descriptor: np.ndarray
    rotation: np.ndarray
    translation_direction: np.ndarray
    success: bool
    failure_reason: str | None
    detected0: int
    detected1: int
    correspondences: int
    inliers: int


class FeatureCache:
    """Validated collection of M1 descriptors and classical pose diagnostics.

    The cache contains no ground-truth pose.  It is therefore safe to generate
    for validation and test images without leaking labels into the learned MLP.
    """

    def __init__(
        self,
        *,
        keys: np.ndarray,
        descriptors: np.ndarray,
        rotations: np.ndarray,
        translation_directions: np.ndarray,
        success: np.ndarray,
        failure_reasons: np.ndarray,
        diagnostics: np.ndarray,
        frontend_config: dict | None = None,
    ) -> None:
        self.keys = np.asarray(keys, dtype=str).reshape(-1)
        self.descriptors = np.asarray(descriptors, dtype=np.float32)
        self.rotations = np.asarray(rotations, dtype=np.float32)
        self.translation_directions = np.asarray(translation_directions, dtype=np.float32)
        self.success = np.asarray(success, dtype=bool).reshape(-1)
        self.failure_reasons = np.asarray(failure_reasons, dtype=str).reshape(-1)
        self.diagnostics = np.asarray(diagnostics, dtype=np.int32)
        self.frontend_config = _canonical_config(frontend_config)
        self._validate()
        self._index = {key: index for index, key in enumerate(self.keys.tolist())}

    @classmethod
    def from_results(
        cls,
        keys: Sequence[str],
        results: Sequence[FrontendResult],
        *,
        frontend_config: dict | None = None,
    ) -> FeatureCache:
        """Build a cache from ordered front-end results."""
        keys = list(keys)
        results = list(results)
        if not results:
            raise ValueError("cannot build an M1 feature cache with no results")
        if len(keys) != len(results):
            raise ValueError(f"key/result count mismatch: {len(keys)} != {len(results)}")

        return cls(
            keys=np.asarray(keys, dtype=str),
            descriptors=np.stack([result.descriptor for result in results]),
            rotations=np.stack([result.rotation for result in results]),
            translation_directions=np.stack([result.translation_direction for result in results]),
            success=np.asarray([result.success for result in results], dtype=bool),
            failure_reasons=np.asarray(
                [result.failure_reason or "" for result in results], dtype=str
            ),
            diagnostics=np.asarray(
                [
                    (
                        result.detected0,
                        result.detected1,
                        result.correspondences,
                        result.inliers,
                    )
                    for result in results
                ],
                dtype=np.int32,
            ),
            frontend_config=frontend_config,
        )

    @property
    def descriptor_dim(self) -> int:
        return int(self.descriptors.shape[1])

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __getitem__(self, key: str) -> CachedFeature:
        try:
            index = self._index[key]
        except KeyError as exc:
            raise KeyError(f"M1 feature cache has no entry for {key!r}") from exc
        diagnostic = self.diagnostics[index]
        reason = self.failure_reasons[index]
        return CachedFeature(
            descriptor=self.descriptors[index],
            rotation=self.rotations[index],
            translation_direction=self.translation_directions[index],
            success=bool(self.success[index]),
            failure_reason=reason or None,
            detected0=int(diagnostic[0]),
            detected1=int(diagnostic[1]),
            correspondences=int(diagnostic[2]),
            inliers=int(diagnostic[3]),
        )

    def save(self, path: str | Path) -> None:
        """Atomically write a compressed NumPy archive without pickled objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "descriptor_dim": self.descriptor_dim,
                "frontend_config": self.frontend_config,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    metadata=np.asarray(metadata),
                    keys=self.keys,
                    descriptors=self.descriptors,
                    rotations=self.rotations,
                    translation_directions=self.translation_directions,
                    success=self.success,
                    failure_reasons=self.failure_reasons,
                    diagnostics=self.diagnostics,
                )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_descriptor_dim: int | None = None,
        expected_frontend_config: dict | None = None,
    ) -> FeatureCache:
        """Load and validate a cache, rejecting stale schemas/configurations."""
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                arrays = {
                    name: archive[name].copy()
                    for name in (
                        "keys",
                        "descriptors",
                        "rotations",
                        "translation_directions",
                        "success",
                        "failure_reasons",
                        "diagnostics",
                    )
                }
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid M1 feature cache {path}: {exc}") from exc

        version = metadata.get("schema_version")
        if version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"M1 cache schema {version!r} is stale; expected {CACHE_SCHEMA_VERSION}"
            )
        cache = cls(**arrays, frontend_config=metadata.get("frontend_config", {}))
        stored_dim = metadata.get("descriptor_dim")
        if stored_dim != cache.descriptor_dim:
            raise ValueError(
                f"M1 cache metadata says {stored_dim} features, archive has {cache.descriptor_dim}"
            )
        if expected_descriptor_dim is not None and cache.descriptor_dim != expected_descriptor_dim:
            raise ValueError(
                f"M1 cache has {cache.descriptor_dim} features; expected {expected_descriptor_dim}"
            )
        if expected_frontend_config is not None and cache.frontend_config != _canonical_config(
            expected_frontend_config
        ):
            raise ValueError("M1 cache front-end configuration does not match the requested run")
        return cache

    def _validate(self) -> None:
        count = len(self.keys)
        if count == 0:
            raise ValueError("M1 feature cache must contain at least one entry")
        if len(set(self.keys.tolist())) != count:
            raise ValueError("M1 feature cache keys must be unique")
        if any(not key for key in self.keys):
            raise ValueError("M1 feature cache keys must be non-empty")
        expected_shapes = {
            "descriptors": (count, None),
            "rotations": (count, 3, 3),
            "translation_directions": (count, 3),
            "success": (count,),
            "failure_reasons": (count,),
            "diagnostics": (count, len(_DIAGNOSTIC_NAMES)),
        }
        for name, expected in expected_shapes.items():
            value = getattr(self, name)
            if value.ndim != len(expected) or any(
                size is not None and value.shape[axis] != size for axis, size in enumerate(expected)
            ):
                raise ValueError(f"invalid {name} shape {value.shape}; expected {expected}")
        if self.descriptor_dim < 1:
            raise ValueError("M1 descriptors must have at least one feature")
        for name in ("descriptors", "rotations", "translation_directions"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"M1 cache {name} must be finite")
        if (self.diagnostics < 0).any():
            raise ValueError("M1 cache diagnostic counts cannot be negative")
