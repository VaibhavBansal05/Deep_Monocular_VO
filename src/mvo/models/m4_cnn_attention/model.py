"""M4 — CNN + convolutional attention for monocular pose regression.

OWNER: Vaibhav Krishna Bansal

A ResNet18 encoder over a stacked image pair regresses the relative pose
T_{t -> t+1} directly. Attention blocks (CBAM / SE) are inserted after each
residual stage; `attention: none` gives the identical network without them,
which is the controlled ablation this model exists to run.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

from mvo.common.geometry import se3
from mvo.common.train.base_model import BaseVOModel
from mvo.common.train.registry import register

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1, 512),
}


def _inflate_conv1(conv1: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Widen a pretrained 3-channel stem to `in_channels`.

    The pretrained kernel is tiled along the input axis and divided by the
    number of repeats. Tiling alone would scale the pre-activation by that
    factor and push BatchNorm far off its running statistics; dividing keeps
    the activation distribution where pretraining left it, so the pretrained
    features stay usable from step one.
    """
    assert in_channels % conv1.in_channels == 0
    repeats = in_channels // conv1.in_channels

    new = nn.Conv2d(in_channels, conv1.out_channels,
                    kernel_size=conv1.kernel_size, stride=conv1.stride,
                    padding=conv1.padding, bias=conv1.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv1.weight.repeat(1, repeats, 1, 1) / repeats)
        if conv1.bias is not None:
            new.bias.copy_(conv1.bias)
    return new


class PoseDecoder(nn.Module):
    """Fully-convolutional pose head: 6-DoF averaged over the feature map.

    Regressing from a flattened feature map would need a ~14M-parameter fully
    connected layer and tie the model to one input resolution. Convolutions
    followed by a spatial mean keep it small and resolution-agnostic, and let
    every location vote on the motion.
    """

    def __init__(self, in_channels: int = 512, width: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Conv2d(width, 6, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            self.head,
        )
        # Zero-init the last layer so the model predicts EXACTLY the identity
        # transform at initialisation -- the right prior for adjacent video
        # frames, and it keeps so3_exp away from its large-angle regime early.
        #
        # This replaces the constant output scaling used by monodepth2's pose
        # decoder. That 0.01 factor suits a network predicting scale-ambiguous
        # translation, but vKITTI 2 supplies METRIC poses: measured over the
        # training split, mean |t| = 0.61 m, so the decoder would have to drive
        # its raw output to ~61 while its gradient was damped 100x. Zero-init
        # gives the same near-identity start with no damping.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).mean(dim=(2, 3))


@register("m4")
class AttentionVO(BaseVOModel):
    seq_len = 2
    needs_depth = False

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 attention: str = "none", reduction: int = 16,
                 spatial_kernel: int = 7, dropout: float = 0.2,
                 rot_weight: float = 5900.0):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; expected {list(_BACKBONES)}")
        if attention not in ("none", "cbam", "se"):
            raise ValueError(f"unknown attention {attention!r}; expected none|cbam|se")
        if attention != "none":
            raise NotImplementedError(
                f"attention={attention!r} lands on branch m4/cbam (owner: Vaibhav). "
                "attention='none' is the ablation baseline and works today.")

        ctor, weights, feat_channels = _BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)

        # Two stacked RGB frames -> 6 input channels.
        net.conv1 = _inflate_conv1(net.conv1, 6)

        self.attention = attention
        self.rot_weight = float(rot_weight)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.decoder = PoseDecoder(feat_channels, dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B,6,H,W) -> (B,512,H/32,W/32).

        Attention blocks will be applied after each layer here on m4/cbam,
        which is why the stages are kept separate rather than nn.Sequential.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, batch: dict) -> dict:
        imgs = batch["imgs"]                       # (B, S, 3, H, W)
        if imgs.shape[1] != self.seq_len:
            raise ValueError(f"M4 expects seq_len={self.seq_len}, got {imgs.shape[1]}")

        pair = torch.cat([imgs[:, 0], imgs[:, 1]], dim=1)   # (B, 6, H, W)
        pose = self.decoder(self.encode(pair))              # (B, 6)
        return {"pose": pose.unsqueeze(1)}                  # (B, S-1, 6)

    def compute_loss(self, batch: dict, out: dict) -> dict:
        pred = out["pose"]                                  # (B, 1, 6)
        target = se3.mat_to_vec(batch["rel_pose"])          # (B, 1, 6)

        trans = ((pred[..., :3] - target[..., :3]) ** 2).mean()
        rot = ((pred[..., 3:] - target[..., 3:]) ** 2).mean()

        # Rotation and translation live on wildly different scales, so an
        # unweighted sum is dominated by translation and the model learns to
        # predict a straight line. DeepVO introduces a weight k for this.
        #
        # k is measured, not guessed. Over the training split the variance of
        # the targets is trans 2.00e-1 vs rot 3.41e-5, a ratio of ~5900:1, so
        # k=5900 puts the two terms on equal footing at initialisation.
        # DeepVO's published k=100 was tuned for KITTI's scale convention and
        # leaves rotation ~59x under-weighted here.
        # Reproduce with: scripts/measure_pose_stats.py
        return {"total": trans + self.rot_weight * rot, "trans": trans, "rot": rot}
