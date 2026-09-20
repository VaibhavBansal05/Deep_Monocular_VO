"""SE(3) / SO(3) utilities.

OWNER: Abhishek Laxman Jadhav  (branch: base/se3-utils)

Frozen convention (mvo/common/train/base_model.py):
    pose 6-vector = [tx, ty, tz, rx, ry, rz], translation first,
    rotation as axis-angle, meaning T_{t -> t+1}.

All functions are batched: leading dims (..., 6) <-> (..., 4, 4), and
differentiable, so M2/M3/M4 can backprop through them.
"""

import torch

_EPS = 1e-7


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrix, via Rodrigues.

    The naive formula divides by theta, which is NaN at theta=0 -- and theta=0
    is exactly where training starts, since the pose head is initialised near
    zero. torch.where is not enough on its own: the untaken branch still
    produces NaN gradients. So the angle is clamped before any division.
    """
    theta = w.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    axis = w / theta

    kx, ky, kz = axis.unbind(-1)
    zero = torch.zeros_like(kx)
    K = torch.stack([zero, -kz, ky,
                     kz, zero, -kx,
                     -ky, kx, zero], dim=-1).reshape(*w.shape[:-1], 3, 3)

    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    s = torch.sin(theta).unsqueeze(-1)
    c = (1.0 - torch.cos(theta)).unsqueeze(-1)
    return eye + s * K + c * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 3) axis-angle."""
    raise NotImplementedError("se3-utils: so3_log coming next commit")


def vec_to_mat(vec: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 4, 4) homogeneous SE(3) matrix."""
    raise NotImplementedError("se3-utils: vec_to_mat coming in a later commit")


def mat_to_vec(mat: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) -> (..., 6) pose vector."""
    raise NotImplementedError("se3-utils: mat_to_vec coming in a later commit")


def invert(T: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) -> inverse transform, exploiting R^-1 = R^T."""
    raise NotImplementedError("se3-utils: invert coming in a later commit")


def compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix product A @ B for stacks of SE(3)."""
    raise NotImplementedError("se3-utils: compose coming in a later commit")


def relative_to_absolute(rel: torch.Tensor, T0: torch.Tensor | None = None) -> torch.Tensor:
    """(N, 4, 4) chain of T_{t->t+1} -> (N+1, 4, 4) absolute cam-to-world poses."""
    raise NotImplementedError("se3-utils: chaining coming in a later commit")


def absolute_to_relative(abs_poses: torch.Tensor) -> torch.Tensor:
    """(N, 4, 4) absolute cam-to-world -> (N-1, 4, 4) chain of T_{t->t+1}."""
    raise NotImplementedError("se3-utils: chaining coming in a later commit")


def rotation_angle_deg(R: torch.Tensor) -> torch.Tensor:
    """Geodesic angle of a rotation, in degrees. Used by RPE."""
    raise NotImplementedError("se3-utils: rotation_angle_deg coming in a later commit")
