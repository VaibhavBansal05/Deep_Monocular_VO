"""Shared stacked-frame encoder and pose head for controlled M2/M4 ablations."""

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, (64, 128, 256, 512)),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1, (64, 128, 256, 512)),
}


def inflate_conv1(conv1: nn.Conv2d, in_channels: int = 6) -> nn.Conv2d:
    """Tile a pretrained RGB stem and preserve its response scale."""
    if in_channels % conv1.in_channels:
        raise ValueError("new input channels must be a multiple of the RGB channels")
    repeats = in_channels // conv1.in_channels
    new = nn.Conv2d(in_channels, conv1.out_channels,
                    kernel_size=conv1.kernel_size, stride=conv1.stride,
                    padding=conv1.padding, bias=conv1.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv1.weight.repeat(1, repeats, 1, 1) / repeats)
        if conv1.bias is not None:
            new.bias.copy_(conv1.bias)
    return new


class PairEncoder(nn.Module):
    """ResNet feature encoder for a six-channel consecutive-frame pair."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; expected {list(_BACKBONES)}")
        ctor, weights, self.stage_widths = _BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        net.conv1 = inflate_conv1(net.conv1, 6)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.out_channels = self.stage_widths[-1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def pool_features(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.pool(feature_map).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_features(self.feature_map(x))


class PoseHead(nn.Module):
    """Common dropout-plus-linear pose regressor used by M2 and M4."""

    def __init__(self, in_features: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_features, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(features))
