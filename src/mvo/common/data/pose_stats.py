"""Per-axis statistics and standardisation for relative-pose outputs.

Measured over the training split by scripts/measure_pose_stats.py. The spread
between axes is 424x:

    axis    mean        std        share of plain-MSE target energy
    tx      +0.001471   0.010589    0.05 %
    ty      +0.006244   0.006491    0.02 %
    tz      -0.606309   0.471248   99.89 %
    rx      +0.000068   0.001297    0.00 %
    ry      +0.002316   0.009766    0.04 %
    rz      +0.000039   0.001112    0.00 %

An unweighted MSE over the 6-vector is therefore dominated by forward
translation.  The percentages above describe squared target energy, not a
literal fixed share of every optimisation gradient.

Two consequences follow:

1. Predicting a STANDARDISED pose improves output-layer conditioning.  This is
   not full whitening: axes are scaled independently and covariance is not
   decorrelated.
2. Adding a mean back means a zero-initialised head starts by predicting
   typical motion rather than the identity, which is a far better starting
   point than a stationary camera. Only the forward-translation mean is used;
   see POSE_MEAN below for why the rest is discarded.

Re-measure with:
    python scripts/measure_pose_stats.py --data-root data/vkitti2
"""

import torch

# Measured mean of the training split, retained for reference and ablation.
POSE_MEAN_MEASURED = (0.001471, 0.006244, -0.606309, 0.000068, 0.002316, 0.000039)

# The prior actually used. Only forward translation keeps its mean.
#
# Not every component of the measured mean is a prior that transfers. Forward
# speed is physical: vehicles move forward, and a typical speed is a reasonable
# expectation on any route. The others are accidents of which three scenes
# happened to fall in the training split:
#
#            train mean    test mean     offset
#     tz     -0.607461     -0.849390     +0.241929   physical, magnitude varies
#     ry     +0.002353     +0.000205     +0.002148   route artifact, 10x the test value
#     ty     +0.006355     +0.011814     -0.005459   route artifact
#
# A vehicle has no intrinsic tendency to turn one way, so a non-zero mean yaw
# rate says only that the training routes happened to curve right on average.
# Adding it back to every prediction makes a constant yaw offset, and a constant
# yaw integrates into a circular arc -- which is exactly the bow observed in M2's
# trajectory, whose measured ry bias (+0.002838) matches this offset.
#
# Horizontal-flip augmentation makes the training distribution symmetric in tx
# and ry, so a non-zero offset on those axes also contradicts the data the model
# actually sees: the network has to learn to cancel an offset the architecture
# keeps adding.
POSE_MEAN = (0.0, 0.0, -0.606309, 0.0, 0.0, 0.0)

POSE_STD = (0.010589, 0.006491, 0.471248, 0.001297, 0.009766, 0.001112)


def pose_mean_std(dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, std) as (6,) tensors, for register_buffer in a model."""
    return (torch.tensor(POSE_MEAN, dtype=dtype),
            torch.tensor(POSE_STD, dtype=dtype))


def destandardize_pose(value: torch.Tensor, mean: torch.Tensor,
                       std: torch.Tensor) -> torch.Tensor:
    """Per-axis standardised pose -> metric pose."""
    return value * std + mean


def standardize_pose(pose: torch.Tensor, mean: torch.Tensor,
                     std: torch.Tensor) -> torch.Tensor:
    """Metric pose -> per-axis standardised pose."""
    return (pose - mean) / std


# Backwards-compatible aliases for old notebooks/checkpoints.  New code uses
# the precise terminology above.
denormalize = destandardize_pose
normalize = standardize_pose
