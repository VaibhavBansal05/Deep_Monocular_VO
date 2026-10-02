"""Precompute deterministic M1 pair descriptors using the shared VKitti2 dataset.

Run from ``src/``:

    python -m mvo.models.m1_classical_mlp.precompute --data-root data/vkitti2
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from mvo.common.data import splits as split_definitions
from mvo.common.data.transforms import denormalize
from mvo.common.data.vkitti import VKitti2

from .cache import FeatureCache
from .frontend import ClassicalFrontend, FrontendResult


def pair_key(seq_id: str, frame_idx: Sequence[int] | torch.Tensor | np.ndarray) -> str:
    """Canonical cache key for one adjacent image pair."""
    if torch.is_tensor(frame_idx):
        frames = frame_idx.detach().cpu().tolist()
    else:
        frames = np.asarray(frame_idx).reshape(-1).tolist()
    if len(frames) != 2:
        raise ValueError(f"M1 cache keys require exactly two frames, got {frames}")
    first, second = (int(frame) for frame in frames)
    if second != first + 1:
        raise ValueError(f"M1 expects adjacent frames, got {first} and {second}")
    if not seq_id or ":" in seq_id:
        raise ValueError(f"invalid sequence id {seq_id!r}")
    return f"{seq_id}:{first:05d}:{second:05d}"


def images_to_uint8(imgs: torch.Tensor) -> np.ndarray:
    """Convert shared ImageNet-normalised ``(2,3,H,W)`` tensors to RGB uint8."""
    if imgs.ndim != 4 or imgs.shape[0] != 2 or imgs.shape[1] != 3:
        raise ValueError(f"expected image pair with shape (2,3,H,W), got {tuple(imgs.shape)}")
    rgb = denormalize(imgs.detach().cpu()).clamp(0.0, 1.0)
    rgb = rgb.permute(0, 2, 3, 1).numpy()
    # Half-up quantisation is stable even when normalise/denormalise leaves a
    # value infinitesimally below an exact half-integer (for example 0.1*255).
    return np.floor(rgb * 255.0 + 0.500001).astype(np.uint8)


def precompute_feature_cache(
    data_root: str | Path,
    output: str | Path,
    *,
    split_names: Sequence[str] = ("train", "val", "test"),
    frontend: ClassicalFrontend | None = None,
    img_size: tuple[int, int] = (128, 416),
    show_progress: bool = True,
) -> tuple[FeatureCache, dict[str, dict]]:
    """Extract all requested splits and write one label-free M1 feature cache."""
    if not split_names:
        raise ValueError("at least one dataset split is required")
    unknown = [name for name in split_names if name not in split_definitions.SPLITS]
    if unknown:
        raise ValueError(
            f"unknown dataset splits {unknown}; expected {list(split_definitions.SPLITS)}"
        )
    if len(set(split_names)) != len(split_names):
        raise ValueError(f"dataset splits must be unique, got {list(split_names)}")

    frontend = frontend or ClassicalFrontend()
    keys: list[str] = []
    results: list[FrontendResult] = []
    summary: dict[str, dict] = {}

    for split_name in split_names:
        dataset = VKitti2(
            root=data_root,
            split=split_name,
            seq_len=2,
            need_depth=False,
            augment=False,
            img_size=img_size,
        )
        failures: Counter[str] = Counter()
        for index in range(len(dataset)):
            sample = dataset[index]
            images = images_to_uint8(sample["imgs"])
            result = frontend.extract(images[0], images[1], sample["K"].cpu().numpy())
            keys.append(pair_key(sample["seq_id"], sample["frame_idx"]))
            results.append(result)
            if not result.success:
                failures[result.failure_reason or "unknown"] += 1
            if show_progress and ((index + 1) % 250 == 0 or index + 1 == len(dataset)):
                print(
                    f"[m1-cache] {split_name}: {index + 1}/{len(dataset)} pairs, "
                    f"{sum(failures.values())} failures"
                )
        summary[split_name] = {
            "pairs": len(dataset),
            "failures": int(sum(failures.values())),
            "failure_reasons": dict(sorted(failures.items())),
        }

    config = frontend.configuration()
    config["img_size"] = list(img_size)
    cache = FeatureCache.from_results(keys, results, frontend_config=config)
    cache.save(output)
    return cache, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/vkitti2")
    parser.add_argument("--output", default="data/m1_features/features.npz")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=sorted(split_definitions.SPLITS),
    )
    parser.add_argument("--detector", choices=["orb", "shitomasi"], default="orb")
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--grid", nargs=2, type=int, default=(8, 8), metavar=("ROWS", "COLS"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    frontend = ClassicalFrontend(
        detector=args.detector,
        max_features=args.max_features,
        grid=tuple(args.grid),
        seed=args.seed,
    )
    cache, summary = precompute_feature_cache(
        args.data_root,
        args.output,
        split_names=args.splits,
        frontend=frontend,
        show_progress=not args.quiet,
    )
    print(f"[m1-cache] wrote {len(cache)} pairs x {cache.descriptor_dim} features -> {args.output}")
    for split_name, stats in summary.items():
        print(
            f"[m1-cache] {split_name}: {stats['failures']}/{stats['pairs']} failures "
            f"{stats['failure_reasons']}"
        )


if __name__ == "__main__":
    main()
