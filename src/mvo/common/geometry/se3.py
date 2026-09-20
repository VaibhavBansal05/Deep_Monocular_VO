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
    """(..., 3, 3) rotation matrix -> (..., 3) axis-angle.

    Uses the limit theta / (2 sin theta) -> 1/2 as theta -> 0, applied via
    torch.where so gradients remain finite near the identity.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_t = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_t)

    skew = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                        R[..., 0, 2] - R[..., 2, 0],
                        R[..., 1, 0] - R[..., 0, 1]], dim=-1)

    # theta / (2 sin theta) -> 1/2 as theta -> 0. Use the limit directly rather
    # than dividing by a vanishing sin.
    sin_t = torch.sin(theta).clamp(min=_EPS)
    scale = torch.where(theta.abs() < 1e-4,
                        torch.full_like(theta, 0.5),
                        theta / (2.0 * sin_t))
    return skew * scale.unsqueeze(-1)


def vec_to_mat(vec: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 4, 4) homogeneous SE(3)."""
    t, w = vec[..., :3], vec[..., 3:]
    R = so3_exp(w)
    T = torch.zeros(*vec.shape[:-1], 4, 4, dtype=vec.dtype, device=vec.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def mat_to_vec(mat: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) -> (..., 6)."""
    return torch.cat([mat[..., :3, 3], so3_log(mat[..., :3, :3])], dim=-1)


def invert(T: torch.Tensor) -> torch.Tensor:
    """Inverse of an SE(3) stack, using R^-1 = R^T (never torch.inverse)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix product A @ B for stacks of SE(3)."""
    return A @ B


def relative_to_absolute(rel: torch.Tensor, T0: torch.Tensor | None = None) -> torch.Tensor:
    """(N, 4, 4) chain of T_{t->t+1} -> (N+1, 4, 4) absolute cam-to-world poses."""
    raise NotImplementedError("se3-utils: chaining coming in next commit")


def absolute_to_relative(abs_poses: torch.Tensor) -> torch.Tensor:
    """(N, 4, 4) absolute cam-to-world -> (N-1, 4, 4) chain of T_{t->t+1}."""
    raise NotImplementedError("se3-utils: chaining coming in next commit")


def rotation_angle_deg(R: torch.Tensor) -> torch.Tensor:
    """Geodesic angle of a rotation, in degrees. Used by RPE."""
    raise NotImplementedError("se3-utils: rotation_angle_deg coming in next commit")
