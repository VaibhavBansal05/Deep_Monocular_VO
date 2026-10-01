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

Loss:
    metric Euclidean translation + SO(3) geodesic rotation, with each group
    normalised by its training-set RMS and a small signed-bias penalty.

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

from mvo.common.data.pose_stats import destandardize_pose, pose_mean_std
from mvo.common.geometry.se3 import mat_to_vec
from mvo.common.train.base_model import BaseVOModel
from mvo.common.train.pair_encoder import PairEncoder, PoseHead, inflate_conv1
from mvo.common.train.pose_loss import geometric_pose_loss
from mvo.common.train.registry import register

# Kept for old imports/tests; both M2 and M4 now use the shared implementation.
_inflate_conv1 = inflate_conv1


@register("m2")
class M2CnnRnn(BaseVOModel):
    """ResNet18 + optional LSTM/GRU for sequential visual odometry.

    Config keys (from configs/m2.yaml):
        rnn:         "lstm" | "gru" | "none"
        hidden_size: int (default 512)
        num_layers:  int (default 2)
        dropout:     float (default 0.2)
        rot_weight:  relative weight of the geodesic rotation group
        bias_weight: penalty on the signed mean residual in a mini-batch
    """

    seq_len: int = 5
    needs_depth: bool = False

    def __init__(self, rnn: str = "lstm", hidden_size: int = 512,
                 num_layers: int = 2, dropout: float = 0.2,
                 rot_weight: float = 1.0, bias_weight: float = 0.1,
                 pretrained: bool = True, backbone: str = "resnet18",
                 standardize: bool | None = None, whiten: bool | None = None,
                 seq_len: int = 5, **kwargs):
        super().__init__()
        self.rnn_type = rnn.lower()
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        self.seq_len = int(seq_len)
        self.rot_weight = float(rot_weight)
        self.bias_weight = float(bias_weight)
        # ``whiten`` is accepted only so old checkpoint configs still build.
        self.standardize = bool(whiten if standardize is None and whiten is not None
                                else True if standardize is None else standardize)
        self.whiten = self.standardize
        _mean, _std = pose_mean_std()
        self.register_buffer("pose_mean", _mean)
        self.register_buffer("pose_std", _std)

        # Shared with M4.  Therefore M2(rnn=none) and M4(attention=none)
        # differ only in their sequence wrapper, not in backbone or pose head.
        self.encoder = PairEncoder(backbone, pretrained)
        feat_dim = self.encoder.out_channels

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
        self.pose_head = PoseHead(out_dim, dropout)

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
        feats = self.encoder(pairs_flat)              # (B*(S-1), 512)
        feats = feats.reshape(B, S - 1, -1)          # (B, S-1, 512)

        # RNN (temporal modelling)
        if self.rnn is not None:
            feats, _ = self.rnn(feats)                # (B, S-1, hidden_size)

        # 6-DoF pose per step
        raw = self.pose_head(feats)                   # (B, S-1, 6)
        # Standardisation conditions the output parameters; all public poses
        # and the geometric loss remain in metric units.
        pose = (destandardize_pose(raw, self.pose_mean, self.pose_std)
                if self.standardize else raw)
        return {"pose": pose}

    def compute_loss(self, batch: dict, out: dict) -> dict:
        """Shared Euclidean-translation and geodesic-rotation objective."""
        pose_pred = out["pose"]                        # (B, S-1, 6)
        rel_mat = batch["rel_pose"].to(pose_pred.dtype)  # (B, S-1, 4, 4)
        pose_gt = mat_to_vec(rel_mat)                  # (B, S-1, 6)

        return geometric_pose_loss(pose_pred, pose_gt, self.pose_std,
                                   self.rot_weight, self.bias_weight)
