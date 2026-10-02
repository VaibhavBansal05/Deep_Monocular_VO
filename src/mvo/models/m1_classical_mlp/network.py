"""Learned pose regressor for M1's fixed-length classical descriptor."""

from collections.abc import Sequence

import torch
import torch.nn as nn


class PoseMLP(nn.Module):
    """Batch-normalised MLP mapping a flow-grid descriptor to six pose values.

    The six outputs are still in per-axis standardised coordinates.  Converting
    them to metric ``[tx, ty, tz, rx, ry, rz]`` poses belongs to the M1 model
    wrapper, which keeps this network independently testable.
    """

    def __init__(
        self,
        input_dim: int = 192,
        hidden_sizes: Sequence[int] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError(f"hidden_sizes must contain positive widths, got {hidden_sizes}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0,1), got {dropout}")

        self.input_dim = int(input_dim)
        self.hidden_sizes = hidden_sizes
        self.dropout_probability = float(dropout)

        layers: list[nn.Module] = []
        in_features = self.input_dim
        for out_features in hidden_sizes:
            linear = nn.Linear(in_features, out_features)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.extend(
                (
                    linear,
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout_probability),
                )
            )
            in_features = out_features
        self.features = nn.Sequential(*layers)
        self.pose_head = nn.Linear(in_features, 6)
        # A zero raw output maps to the shared metric motion prior after
        # destandardisation.  It is a better initial prediction than a random
        # six-axis pose and matches the supervised M2/M4 initialisation policy.
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim != 2 or descriptor.shape[1] != self.input_dim:
            raise ValueError(
                f"expected descriptor shape (B,{self.input_dim}), got {tuple(descriptor.shape)}"
            )
        return self.pose_head(self.features(descriptor))
