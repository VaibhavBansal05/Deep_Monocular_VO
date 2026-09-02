"""SE(3) / SO(3) utilities.

OWNER: Abhishek Laxman Jadhav  (branch: base/se3-utils)
STATUS: STUB -- implement on your branch, do not implement here.

Frozen convention (mvo/common/train/base_model.py):
    pose 6-vector = [tx, ty, tz, rx, ry, rz], translation first,
    rotation as axis-angle (rotation vector), meaning T_{t -> t+1}.

Everything is batched: leading dims (..., 6) <-> (..., 4, 4).
See handoff packet for the full spec and required tests.
"""

import torch


def vec_to_mat(vec: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 4, 4) homogeneous SE(3) matrix."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def mat_to_vec(mat: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) -> (..., 6) pose vector."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrix (Rodrigues)."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 3) axis-angle."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def invert(T: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) -> inverse transform, exploiting R^-1 = R^T."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix product A @ B for stacks of SE(3)."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def relative_to_absolute(rel: torch.Tensor, T0: torch.Tensor | None = None) -> torch.Tensor:
    """(N, 4, 4) chain of T_{t->t+1} -> (N+1, 4, 4) absolute cam-to-world poses."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def absolute_to_relative(abs_poses: torch.Tensor) -> torch.Tensor:
    """(N, 4, 4) absolute cam-to-world -> (N-1, 4, 4) chain of T_{t->t+1}."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")


def rotation_angle_deg(R: torch.Tensor) -> torch.Tensor:
    """Geodesic angle of a rotation, in degrees. Used by RPE."""
    raise NotImplementedError("owner: Abhishek — branch base/se3-utils")
