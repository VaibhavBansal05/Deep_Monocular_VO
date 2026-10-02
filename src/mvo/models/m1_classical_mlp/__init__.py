"""M1 — Classical ORB/Lucas-Kanade + MLP

OWNER: Pranav Joshi
BRANCH: m1/feature-extraction

Implement your model here. It must subclass BaseVOModel and register itself:

    from mvo.common.train.base_model import BaseVOModel
    from mvo.common.train.registry import register

    @register("m1")
    class YourModel(BaseVOModel):
        seq_len = 2
        needs_depth = False
        def forward(self, batch): ...      # -> {"pose": (B, S-1, 6)}
        def compute_loss(self, batch, out): ...   # -> {"total": ...}

Then export it below so the @register decorator fires on import:

    from .model import YourModel   # noqa: F401

Nobody else edits this folder. See handoff/ for the full spec.
"""

from .frontend import ClassicalFrontend, FrontendResult

__all__ = ["ClassicalFrontend", "FrontendResult"]
