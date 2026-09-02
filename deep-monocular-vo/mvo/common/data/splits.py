"""Canonical train/val/test split. SINGLE SOURCE OF TRUTH.

Every model MUST import from here. Do not hardcode scene lists anywhere else.
Split is BY SCENE so there is zero geometric leakage between train and test.

Rationale (see PLAN.md 2.3):
  - Test is `clone` only, so all four models are scored on identical trajectories.
  - Train uses 4 photometric variations of the train scenes as augmentation.
  - fog / rain / 15-deg / 30-deg are held out entirely for the robustness ablation.

Changing anything in this file invalidates every result in results/.
It requires a PR approved by all four members.
"""

SCENES = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]

# Approximate frame counts per scene (Camera_0, one variation).
SCENE_LENGTHS = {
    "Scene01": 447,
    "Scene02": 233,
    "Scene06": 270,
    "Scene18": 339,
    "Scene20": 837,
}

ALL_VARIATIONS = [
    "clone", "morning", "overcast", "sunset", "rain", "fog",
    "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right",
]

CAMERA = "Camera_0"

SPLITS = {
    "train": {
        "scenes": ["Scene01", "Scene06", "Scene18"],
        "variations": ["clone", "morning", "overcast", "sunset"],
    },
    "val": {
        "scenes": ["Scene02"],
        "variations": ["clone"],
    },
    "test": {
        "scenes": ["Scene20"],
        "variations": ["clone"],
    },
    # Held out entirely. Never trained on. Used once, at the end, for the
    # robustness ablation in Results & Discussion.
    "robustness": {
        "scenes": ["Scene20"],
        "variations": ["rain", "fog", "15-deg-left", "30-deg-right"],
    },
}


def sequences(split: str) -> list[tuple[str, str]]:
    """Return [(scene, variation), ...] for a split name."""
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
    cfg = SPLITS[split]
    return [(s, v) for s in cfg["scenes"] for v in cfg["variations"]]


def seq_id(scene: str, variation: str) -> str:
    """Canonical string id used in batches, filenames and results."""
    return f"{scene}/{variation}"
