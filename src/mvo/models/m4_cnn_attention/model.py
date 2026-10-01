"""M4 — CNN + convolutional attention for monocular pose regression.

OWNER: Vaibhav Krishna Bansal

A ResNet18 encoder over a stacked image pair regresses the relative pose
T_{t -> t+1} directly. Attention blocks (CBAM / SE) are inserted after each
residual stage; `attention: none` gives the identical network without them,
which is the controlled ablation this model exists to run.
"""

import torch

from mvo.common.data.pose_stats import destandardize_pose, pose_mean_std
from mvo.common.geometry import se3
from mvo.common.train.base_model import BaseVOModel
from mvo.common.train.pair_encoder import PairEncoder, PoseHead, inflate_conv1
from mvo.common.train.pose_loss import geometric_pose_loss
from mvo.common.train.registry import register
from mvo.models.m4_cnn_attention.attention import build_attention

# Backwards-compatible test/import name.
_inflate_conv1 = inflate_conv1


@register("m4")
class AttentionVO(BaseVOModel):
    seq_len = 2
    needs_depth = False

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 attention: str = "none", reduction: int = 16,
                 spatial_kernel: int = 7, dropout: float = 0.2,
                 rot_weight: float = 1.0, bias_weight: float = 0.1,
                 standardize: bool | None = None, whiten: bool | None = None,
                 seq_len: int = 2, attention_residual: bool = True,
                 attention_gate_init: float = 0.1):
        super().__init__()
        if attention not in ("none", "cbam", "se"):
            raise ValueError(f"unknown attention {attention!r}; expected none|cbam|se")

        if seq_len != 2:
            raise ValueError("M4 is a pairwise model; seq_len must be 2")
        self.seq_len = int(seq_len)
        self.attention = attention
        self.rot_weight = float(rot_weight)
        self.bias_weight = float(bias_weight)
        self.standardize = bool(whiten if standardize is None and whiten is not None
                                else True if standardize is None else standardize)
        self.whiten = self.standardize
        mean, std = pose_mean_std()
        self.register_buffer("pose_mean", mean)
        self.register_buffer("pose_std", std)
        self.encoder = PairEncoder(backbone, pretrained)
        # One block after each residual stage. ResNet18 stage widths.
        widths = self.encoder.stage_widths
        self.attn1, self.attn2, self.attn3, self.attn4 = (
            build_attention(attention, w, reduction, spatial_kernel,
                            residual=attention_residual,
                            gate_init=attention_gate_init) for w in widths)

        self.decoder = PoseHead(self.encoder.out_channels, dropout)

    @property
    def stem(self):
        return self.encoder.stem

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B,6,H,W) -> (B,512,H/32,W/32).

        Stages are kept separate so an attention block can be applied after
        each one.
        """
        x = self.encoder.stem(x)
        x = self.attn1(self.encoder.layer1(x))
        x = self.attn2(self.encoder.layer2(x))
        x = self.attn3(self.encoder.layer3(x))
        x = self.attn4(self.encoder.layer4(x))
        return x

    def spatial_maps(self) -> list[torch.Tensor]:
        """Spatial attention maps from the last forward pass, coarse to fine.

        Empty unless attention='cbam'; SE has no spatial branch. Used to
        overlay where the model looked onto the input frame.
        """
        out = []
        for a in (self.attn1, self.attn2, self.attn3, self.attn4):
            blk = getattr(a, "block", None)
            m = getattr(blk, "last_spatial_map", None)
            if m is not None:
                out.append(m)
        return out

    def forward(self, batch: dict) -> dict:
        imgs = batch["imgs"]                       # (B, S, 3, H, W)
        if imgs.shape[1] != self.seq_len:
            raise ValueError(f"M4 expects seq_len={self.seq_len}, got {imgs.shape[1]}")

        pair = torch.cat([imgs[:, 0], imgs[:, 1]], dim=1)   # (B, 6, H, W)
        feat = self.encoder.pool_features(self.encode(pair))
        raw = self.decoder(feat)                            # (B, 6)

        # The head predicts a standardised pose; the contract requires metric.
        # With the zero-initialised head this means the model starts by
        # predicting the AVERAGE relative pose rather than the identity -- the
        # mean-pose baseline scores t_rel 22.22% on Scene20 against the identity
        # baseline's 99.16%, so it begins where the trivial predictor ends.
        pose = (destandardize_pose(raw, self.pose_mean, self.pose_std)
                if self.standardize else raw)
        return {"pose": pose.unsqueeze(1)}                  # (B, S-1, 6)

    def compute_loss(self, batch: dict, out: dict) -> dict:
        pred = out["pose"]                                  # (B, 1, 6), metric
        target = se3.mat_to_vec(batch["rel_pose"])          # (B, 1, 6), metric

        return geometric_pose_loss(pred, target, self.pose_std,
                                   self.rot_weight, self.bias_weight)
