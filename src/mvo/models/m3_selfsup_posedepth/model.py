"""RGB-only self-supervised M3. OWNER: Suryansh Bajaj."""

import torch

from mvo.common.train.base_model import BaseVOModel
from mvo.common.train.registry import register
from mvo.models.m3_selfsup_posedepth.depthnet import DepthNet, disparity_to_depth
from mvo.models.m3_selfsup_posedepth.losses import multiscale_loss
from mvo.models.m3_selfsup_posedepth.posenet import PoseNet


@register("m3")
class M3PoseDepth(BaseVOModel):
    """Three-frame middle-target reconstruction; labels are never consumed."""

    seq_len = 3
    needs_depth = False

    def __init__(self, seq_len=3, backbone="resnet18", pretrained=True, num_scales=4,
                 min_depth=0.1, max_depth=100.0, ssim_weight=0.85,
                 smooth_weight=1e-3, automask=True, min_reproj=True):
        super().__init__()
        if seq_len != 3 or backbone != "resnet18":
            raise ValueError("M3 baseline requires seq_len=3 and backbone=resnet18")
        if not 0 < min_depth < max_depth:
            raise ValueError("expected 0 < min_depth < max_depth")
        if not 0 <= ssim_weight <= 1 or smooth_weight < 0:
            raise ValueError("expected SSIM weight in [0,1] and nonnegative smoothness")
        self.depthnet = DepthNet(pretrained, num_scales)
        self.posenet = PoseNet(pretrained)
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)
        self.loss_options = dict(min_depth=min_depth, max_depth=max_depth,
                                 ssim_weight=ssim_weight, smooth_weight=smooth_weight,
                                 automask=automask, min_reproj=min_reproj)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def denormalize(self, images):
        with torch.amp.autocast(images.device.type, enabled=False):
            return (images.float() * self.rgb_std + self.rgb_mean).clamp(0, 1)

    def predict_depth(self, images):
        return disparity_to_depth(self.depthnet(images)[0], self.min_depth, self.max_depth)

    def forward(self, batch):
        images = batch["imgs"]
        if images.ndim != 5 or images.shape[1:3] != (3, 3):
            raise ValueError("M3 expects imgs with shape (B,3,3,H,W)")
        disparities = self.depthnet(images[:, 1])
        return {"pose": self.posenet(images), "disp": disparities,
                "depth": disparity_to_depth(disparities[0], self.min_depth, self.max_depth)}

    def compute_loss(self, batch, out):
        return multiscale_loss(self.denormalize(batch["imgs"]), out["disp"], out["pose"],
                               batch["K"], batch["K_inv"], **self.loss_options)
