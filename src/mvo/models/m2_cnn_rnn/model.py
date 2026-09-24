"""M2 — CNN (ResNet18) + GRU/LSTM for monocular visual odometry.

OWNER: Abhishek Laxman Jadhav
BRANCH: m2/backbone → m2/lstm-head → m2/training

Architecture:
    1. ResNet18 backbone with inflated 6-channel conv1 (stacked pairs I_t, I_{t+1}).
       Inflation: duplicate pretrained 3-ch weights along input axis, halve them —
       preserves activation scale at initialisation.
    2. Configurable RNN head: LSTM | GRU | none (pairwise ablation).
       rnn=none yields the same backbone as M4-without-attention, making
       m2(rnn=none) vs m4(attention=none) a direct cross-member consistency check.
    3. Linear pose head: 6-DoF per step [tx, ty, tz, rx, ry, rz].
       Initialised near zero so the pose starts near identity.

Loss (DeepVO-style):
    L = ||t̂ - t||² + rot_weight * ||r̂ - r||²
    rot_weight ≈ 100  (radians are tiny between adjacent frames; without
    upweighting the network predicts straight-line trajectories).

Viva notes:
    - The LSTM state carries accumulated scene context / motion history, NOT a map.
    - Per-frame errors compound into drift — evaluation must report ATE over full
      trajectories, not per-step pose error.
    - At sequence boundaries during evaluation, hidden state is zeroed (each
      subsequence is independent).  For overlapping windows, hidden state can be
      propagated, but this is not done here.
    - LSTM vs GRU: LSTM has separate cell and hidden state (two gates for memory
      control), making it better at long-range temporal dependencies; GRU is lighter.
      For seq_len=5 the difference is small — the ablation quantifies it.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

from mvo.common.data.pose_stats import denormalize, normalize, pose_mean_std
from mvo.common.geometry.se3 import mat_to_vec
from mvo.common.train.base_model import BaseVOModel
from mvo.common.train.registry import register


def _inflate_conv1(pretrained_conv: nn.Conv2d) -> nn.Conv2d:
    """Duplicate 3-channel pretrained weights to 6-channel input, halving to
    preserve the expected activation scale (E[|Wx|] ≈ constant if input stats
    are similar for both channels of a stereo pair)."""
    w3 = pretrained_conv.weight.data          # (64, 3, 7, 7)
    w6 = torch.cat([w3, w3], dim=1) * 0.5    # (64, 6, 7, 7)
    conv6 = nn.Conv2d(6, pretrained_conv.out_channels,
                      kernel_size=pretrained_conv.kernel_size,
                      stride=pretrained_conv.stride,
                      padding=pretrained_conv.padding,
                      bias=pretrained_conv.bias is not None)
    conv6.weight = nn.Parameter(w6)
    if pretrained_conv.bias is not None:
        conv6.bias = nn.Parameter(pretrained_conv.bias.data.clone())
    return conv6


@register("m2")
class M2CnnRnn(BaseVOModel):
    """ResNet18 + optional LSTM/GRU for sequential visual odometry.

    Config keys (from configs/m2.yaml):
        rnn:         "lstm" | "gru" | "none"
        hidden_size: int (default 512)
        num_layers:  int (default 2)
        dropout:     float (default 0.2)
        rot_weight:  float (default 100.0)
    """

    seq_len: int = 5
    needs_depth: bool = False

    def __init__(self, rnn: str = "lstm", hidden_size: int = 512,
                 num_layers: int = 2, dropout: float = 0.2,
                 rot_weight: float = 1.0, pretrained: bool = True,
                 whiten: bool = True, **kwargs):
        super().__init__()
        self.rnn_type = rnn.lower()
        self.rot_weight = rot_weight
        self.whiten = bool(whiten)
        _mean, _std = pose_mean_std()
        self.register_buffer("pose_mean", _mean)
        self.register_buffer("pose_std", _std)

        # ---- Backbone -------------------------------------------------------
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tvm.resnet18(weights=weights)

        # Inflate conv1: 3 channels → 6 channels (stacked frame pair)
        resnet.conv1 = _inflate_conv1(resnet.conv1)

        # Strip the classification head; keep layers up to layer4
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),   # → (B*(S-1), 512, 1, 1)
        )
        feat_dim = 512  # ResNet18 layer4 output channels

        # ---- RNN head -------------------------------------------------------
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(feat_dim, hidden_size, num_layers,
                               batch_first=True, dropout=rnn_dropout)
            out_dim = hidden_size
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(feat_dim, hidden_size, num_layers,
                              batch_first=True, dropout=rnn_dropout)
            out_dim = hidden_size
        elif self.rnn_type == "none":
            self.rnn = None
            out_dim = feat_dim
        else:
            raise ValueError(f"rnn must be 'lstm', 'gru', or 'none'; got {rnn!r}")

        # ---- Pose head -------------------------------------------------------
        self.pose_head = nn.Linear(out_dim, 6)
        # Initialise near zero so early poses are near identity
        nn.init.normal_(self.pose_head.weight, std=1e-4)
        nn.init.zeros_(self.pose_head.bias)

    # ---------------------------------------------------------------------- #

    def forward(self, batch: dict) -> dict:
        """
        batch["imgs"]: (B, S, 3, H, W)  — ImageNet-normalised frames
        Returns {"pose": (B, S-1, 6)} — T_{t->t+1} in [tx,ty,tz,rx,ry,rz].
        """
        imgs = batch["imgs"]                          # (B, S, 3, H, W)
        B, S, C, H, W = imgs.shape

        # Stack consecutive frame pairs along the channel dimension
        pairs = torch.cat([imgs[:, :-1], imgs[:, 1:]], dim=2)  # (B, S-1, 6, H, W)
        pairs_flat = pairs.flatten(0, 1)                        # (B*(S-1), 6, H, W)

        # CNN features
        feats = self.backbone(pairs_flat)             # (B*(S-1), 512, 1, 1)
        feats = feats.flatten(1)                      # (B*(S-1), 512)
        feats = feats.reshape(B, S - 1, -1)          # (B, S-1, 512)

        # RNN (temporal modelling)
        if self.rnn is not None:
            feats, _ = self.rnn(feats)                # (B, S-1, hidden_size)

        # 6-DoF pose per step
        raw = self.pose_head(feats)                   # (B, S-1, 6)
        # Head predicts a whitened pose; the contract requires a metric one.
        pose = denormalize(raw, self.pose_mean, self.pose_std) if self.whiten else raw
        return {"pose": pose}

    def compute_loss(self, batch: dict, out: dict) -> dict:
        """DeepVO-style weighted MSE.  L = ||t̂-t||² + k*||r̂-r||²

        GT rel_pose is (B, S-1, 4, 4); convert to 6-vec via mat_to_vec.
        With whiten=True the six axes are already commensurate and rot_weight
        is 1. Without it, k must be measured rather than copied from the paper:
        E[t^2]/E[r^2] is about 5869 on this data, not the 100 DeepVO used.
        """
        pose_pred = out["pose"]                        # (B, S-1, 6)
        rel_mat = batch["rel_pose"].to(pose_pred.dtype)  # (B, S-1, 4, 4)
        pose_gt = mat_to_vec(rel_mat)                  # (B, S-1, 6)

        if self.whiten:
            # Per-axis variance spans 424x on this data: a plain MSE sends
            # 99.89% of its gradient to tz and trains forward translation
            # alone. Whitening equalises the six axes.
            pose_pred = normalize(pose_pred, self.pose_mean, self.pose_std)
            pose_gt = normalize(pose_gt, self.pose_mean, self.pose_std)

        t_err = (pose_pred[..., :3] - pose_gt[..., :3]).pow(2).mean()
        r_err = (pose_pred[..., 3:] - pose_gt[..., 3:]).pow(2).mean()
        total = t_err + self.rot_weight * r_err

        return {"total": total, "trans": t_err.detach(), "rot": r_err.detach()}
