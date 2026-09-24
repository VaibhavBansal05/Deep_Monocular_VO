"""Per-axis statistics of the ground-truth relative pose, and the whitening
transform built from them.

Measured over the training split by scripts/measure_pose_stats.py. The spread
between axes is 424x:

    axis    mean        std        share of a plain-MSE gradient
    tx      +0.001471   0.010589    0.05 %
    ty      +0.006244   0.006491    0.02 %
    tz      -0.606309   0.471248   99.89 %
    rx      +0.000068   0.001297    0.00 %
    ry      +0.002316   0.009766    0.04 %
    rz      +0.000039   0.001112    0.00 %

An unweighted MSE over the 6-vector therefore trains forward translation and
almost nothing else, which is precisely the behaviour observed: both the
recurrent and the attention model scored close to a baseline that simply
applies the average relative pose at every frame.

Two consequences follow, and models use both:

1. Predicting a WHITENED pose gives every axis an equal share of the gradient.
2. Adding the mean back means a zero-initialised head starts by predicting the
   average motion rather than the identity. That is a far better prior: on
   Scene20 the identity baseline scores t_rel 99.16% while the mean-pose
   baseline scores 22.22%, so the model begins where the trivial predictor
   ends and learns the deviation from it.

Re-measure with:
    python scripts/measure_pose_stats.py --data-root data/vkitti2
"""

import torch

POSE_MEAN = (0.001471, 0.006244, -0.606309, 0.000068, 0.002316, 0.000039)
POSE_STD = (0.010589, 0.006491, 0.471248, 0.001297, 0.009766, 0.001112)


def pose_mean_std(dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, std) as (6,) tensors, for register_buffer in a model."""
    return (torch.tensor(POSE_MEAN, dtype=dtype),
            torch.tensor(POSE_STD, dtype=dtype))


def denormalize(norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """whitened -> metric pose."""
    return norm * std + mean


def normalize(pose: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """metric pose -> whitened."""
    return (pose - mean) / std
