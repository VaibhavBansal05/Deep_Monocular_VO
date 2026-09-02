"""The model contract. All four models subclass this.

Pose convention — FROZEN, do not deviate:
    6-vector, translation first, axis-angle rotation:
        [tx, ty, tz, rx, ry, rz]
    meaning the relative transform T_{t -> t+1} (maps a point in camera frame t
    into camera frame t+1), in the camera coordinate frame (x right, y down,
    z forward).

Every convention bug in a VO project comes from four people picking four
conventions. This is the one place it is defined.
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseVOModel(nn.Module, ABC):
    """Base class every model in mvo/models/ must implement.

    Attributes set by subclasses:
        seq_len:    number of frames per sample the model expects (2 for pairwise
                    models like M1/M4, 5-10 for sequence models like M2/M3).
        needs_depth: whether the dataloader should load GT depth maps.
    """

    seq_len: int = 2
    needs_depth: bool = False

    @abstractmethod
    def forward(self, batch: dict) -> dict:
        """Run the model on a batch dict (see mvo/common/data/vkitti.py).

        Returns a dict that MUST contain:
            "pose": (B, S-1, 6) float tensor — relative poses T_{t->t+1}.
        May additionally contain any model-specific outputs, e.g.:
            "depth": (B, S, 1, H, W), "uncertainty": ..., "disp": [...]
        These extras are passed straight back into compute_loss.
        """

    @abstractmethod
    def compute_loss(self, batch: dict, out: dict) -> dict:
        """Return a dict of scalar tensors.

        MUST contain key "total" — that is what .backward() is called on.
        Every other key is logged verbatim (e.g. "photometric", "smooth",
        "trans", "rot") and shows up in the training curves.
        """

    def num_params(self) -> float:
        """Trainable parameter count in millions (goes in the comparison table)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
