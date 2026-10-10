"""M3 depth network. OWNER: Suryansh Bajaj.

Independently written from Godard et al. (ICCV 2019, arXiv:1806.01260).
Inputs already have the shared dataset's ImageNet normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def disparity_to_depth(disparity, min_depth=0.1, max_depth=100.0):
    """Map sigmoid disparity to bounded depth in the model's arbitrary scale."""
    if not 0 < min_depth < max_depth:
        raise ValueError("expected 0 < min_depth < max_depth")
    with torch.amp.autocast(disparity.device.type, enabled=False):
        d = disparity.float() if disparity.dtype in (torch.float16, torch.bfloat16) else disparity
        return (1.0 / max_depth + d * (1.0 / min_depth - 1.0 / max_depth)).reciprocal()


class DepthEncoder(nn.Module):
    """ResNet18 features at /2, /4, /8, /16, /32; pretrained errors propagate."""

    widths = (64, 64, 128, 256, 512)

    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DepthEncoder expects (B,3,H,W) normalized images")
        x = self.stem(images)
        features = [x]
        x = self.pool(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


def _block(inputs, outputs):
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(inputs, outputs, 3), nn.ELU())


class DepthNet(nn.Module):
    """Five skip-connected upsampling stages, with one or four sigmoid heads."""

    def __init__(self, pretrained=True, num_scales=4):
        super().__init__()
        if num_scales not in (1, 4):
            raise ValueError("num_scales must be 1 or 4")
        self.num_scales = num_scales
        self.encoder = DepthEncoder(pretrained)
        widths = (256, 128, 64, 32, 16)
        inputs = 512
        self.reduce, self.fuse = nn.ModuleList(), nn.ModuleList()
        self.heads = nn.ModuleDict()
        for level, width in enumerate(widths):
            skip = self.encoder.widths[3 - level] if level < 4 else 0
            self.reduce.append(_block(inputs, width))
            self.fuse.append(_block(width + skip, width))
            scale = 4 - level
            if scale < num_scales:
                self.heads[str(scale)] = nn.Sequential(nn.ReflectionPad2d(1),
                                                       nn.Conv2d(width, 1, 3), nn.Sigmoid())
            inputs = width

    def forward(self, images):
        if min(images.shape[-2:]) < 64:
            raise ValueError("M3 depth input must be at least 64 pixels in each dimension")
        features = self.encoder(images)
        x, outputs = features[-1], {}
        for level, (reduce, fuse) in enumerate(zip(self.reduce, self.fuse)):
            x = reduce(x)
            size = features[3 - level].shape[-2:] if level < 4 else images.shape[-2:]
            x = F.interpolate(x, size=size, mode="nearest")
            if level < 4:
                x = torch.cat([x, features[3 - level]], dim=1)
            x = fuse(x)
            scale = 4 - level
            if str(scale) in self.heads:
                outputs[scale] = self.heads[str(scale)](x)
        return [outputs[s] for s in range(self.num_scales)]
