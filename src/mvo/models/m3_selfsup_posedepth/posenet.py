"""M3 chronological pair pose network. OWNER: Suryansh Bajaj."""

import torch
import torch.nn as nn

from mvo.common.train.pair_encoder import PairEncoder


class PoseNet(nn.Module):
    """Predict T_0->1 and T_1->2, translation first and axis-angle rotation."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = PairEncoder(backbone="resnet18", pretrained=pretrained)
        self.hidden = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.head = nn.Linear(256, 6)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, images):
        if images.ndim != 5 or images.shape[1:3] != (3, 3):
            raise ValueError("PoseNet expects (B,3,3,H,W) chronological images")
        pairs = torch.cat([images[:, :-1], images[:, 1:]], dim=2)
        b, n, c, h, w = pairs.shape
        features = self.encoder(pairs.reshape(b * n, c, h, w))
        return (0.01 * self.head(self.hidden(features))).reshape(b, n, 6)
