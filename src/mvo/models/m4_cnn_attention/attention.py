"""Convolutional attention blocks for M4.

CBAM  - Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
SE    - Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.

Both are implemented from the papers rather than adapted from a repository, so
they can be defended in the viva. The one departure is documented on
`ResidualAttention` below and is switchable.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """CBAM channel branch: what to emphasise.

    Average-pooling and max-pooling over space each produce a channel
    descriptor; a shared two-layer MLP maps both, the results are summed, and a
    sigmoid gives a per-channel weight. Using both pools is the paper's point:
    average pooling summarises the whole map, max pooling keeps the strongest
    response, and the two are complementary.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx)                      # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM spatial branch: where to look.

    Average and max over the channel axis are concatenated into a 2-channel map
    and convolved with a single large kernel, so the receptive field covers
    enough of the frame to separate static structure from moving objects --
    which is exactly the distinction that matters for ego-motion.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))   # (B, 1, H, W)


class CBAM(nn.Module):
    """Channel attention then spatial attention, applied in sequence."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)
        self.last_spatial_map: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        smap = self.spatial(x)
        # Kept for visualisation: the spatial map is what can be overlaid on a
        # frame to show where the model is looking.
        self.last_spatial_map = smap.detach()
        return x * smap


class SEBlock(nn.Module):
    """Squeeze-and-excitation: CBAM's channel branch with average pooling only.

    Strictly weaker than CBAM -- no max pooling, no spatial branch -- which is
    why the two make a meaningful pair in the ablation.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.fc(x.mean(dim=(2, 3), keepdim=True)))


class ResidualAttention(nn.Module):
    """Wraps an attention block so it starts as the identity.

    Inserting CBAM into a *pretrained* backbone is disruptive at
    initialisation: a sigmoid gate outputs about 0.5, so every activation is
    roughly halved and the ImageNet features the encoder was chosen for are
    degraded before training begins. With ~1k unique training frames there is
    not enough data to recover from that.

    So the block is applied residually behind a scalar gate initialised to
    zero:

        y = x + gamma * (attn(x) - x),   gamma = 0 at init

    At initialisation y = x exactly, and gamma is learned, so the network opts
    into attention only to the extent it helps. Same mechanism as LayerScale in
    vision transformers and ControlNet's zero convolutions.

    Set residual=False for the paper-faithful formulation; the ablation can
    report both.
    """

    def __init__(self, block: nn.Module, residual: bool = True):
        super().__init__()
        self.block = block
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if not self.residual:
            return out
        return x + self.gamma * (out - x)


def build_attention(kind: str, channels: int, reduction: int = 16,
                    kernel_size: int = 7, residual: bool = True) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "cbam":
        block = CBAM(channels, reduction, kernel_size)
    elif kind == "se":
        block = SEBlock(channels, reduction)
    else:
        raise ValueError(f"unknown attention {kind!r}; expected none|cbam|se")
    return ResidualAttention(block, residual=residual)
