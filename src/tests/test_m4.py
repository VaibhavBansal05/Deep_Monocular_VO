"""M4 tests that need no dataset.

The batch dict is a frozen contract, so the model can be driven with synthetic
tensors long before the real dataloader exists. That is the point of the
contract: M4 is verified end-to-end while base/dataset is still in review.
"""

import pytest
import torch
import torch.nn as nn

from mvo.common.train.registry import build, import_all_models
from mvo.models.m4_cnn_attention.model import AttentionVO, _inflate_conv1

import_all_models()

H, W = 128, 416


def _model(**kw):
    kw.setdefault("pretrained", False)      # no network access in CI
    return build("m4", **kw)


def _batch(b=2, s=2):
    return {
        "imgs": torch.randn(b, s, 3, H, W),
        "rel_pose": torch.eye(4).expand(b, s - 1, 4, 4).contiguous(),
        "K": torch.eye(3).expand(b, 3, 3),
    }


# --------------------------------------------------------------------- shapes


def test_forward_returns_the_contract_shape():
    out = _model()(_batch(b=3))
    assert out["pose"].shape == (3, 1, 6), "pose must be (B, S-1, 6)"
    assert torch.isfinite(out["pose"]).all()


def test_registered_under_m4():
    assert isinstance(_model(), AttentionVO)


def test_declares_the_right_contract():
    m = _model()
    assert m.seq_len == 2
    assert m.needs_depth is False


def test_rejects_a_wrong_length_sequence():
    with pytest.raises(ValueError, match="seq_len"):
        _model()(_batch(s=5))


def test_head_is_resolution_agnostic():
    """The decoder means over space, so a different input size must still work."""
    m = _model()
    out = m({"imgs": torch.randn(1, 2, 3, 192, 640)})
    assert out["pose"].shape == (1, 1, 6)


# ------------------------------------------------------------ conv1 inflation


def test_inflation_preserves_the_pretrained_response():
    """The reason for dividing by the repeat count.

    Tiling a pretrained 3-channel kernel across 6 inputs and halving it means a
    duplicated image pair produces exactly the response the original kernel
    gave on one frame. Without the division the pre-activation doubles and
    BatchNorm's running statistics are meaningless from the first step.
    """
    # In float64 the identity is exact, so assert it exactly. float32 differs by
    # ~2e-6 purely from summing 6 channels instead of 3 -- accumulation order,
    # not a modelling error -- so it gets a realistic tolerance below.
    conv3 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False).double()
    conv6 = _inflate_conv1(conv3, 6).double()

    x3 = torch.randn(1, 3, 64, 64, dtype=torch.float64)
    x6 = torch.cat([x3, x3], dim=1)

    assert torch.equal(conv6(x6), conv3(x3))


def test_inflation_holds_in_float32_within_accumulation_error():
    conv3 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
    conv6 = _inflate_conv1(conv3, 6)

    x3 = torch.randn(1, 3, 64, 64)
    x6 = torch.cat([x3, x3], dim=1)

    assert torch.allclose(conv6(x6), conv3(x3), atol=1e-5)


def test_inflation_keeps_the_stem_shape():
    m = _model()
    assert m.stem[0].in_channels == 6
    assert m.stem[0].out_channels == 64


# ------------------------------------------------------------ initialisation


def test_initial_predictions_are_near_identity():
    """Adjacent video frames are a near-identity transform, and the 6-vector is
    an axis-angle, so a near-zero output is the right prior at initialisation."""
    out = _model()(_batch(b=4))
    assert out["pose"].abs().max() < 0.1


# ------------------------------------------------------------------ backward


def test_gradients_reach_the_stem_and_stay_finite():
    m = _model()
    out = m(_batch())
    out["pose"].abs().mean().backward()

    g = m.stem[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g.abs().sum() > 0, "no gradient reached the inflated stem"


# ------------------------------------------------------------------ ablation


@pytest.mark.parametrize("attn", ["cbam", "se"])
def test_attention_variants_are_not_implemented_yet(attn):
    """Guards the ablation contract: these must exist by the m4/cbam branch."""
    with pytest.raises(NotImplementedError, match="m4/cbam"):
        _model(attention=attn)


def test_rejects_an_unknown_attention_name():
    with pytest.raises(ValueError, match="attention"):
        _model(attention="squeeze-excite")


def test_rejects_an_unknown_backbone():
    with pytest.raises(ValueError, match="backbone"):
        _model(backbone="vgg16")


# ---------------------------------------------------------------------- loss


def _se3_ready() -> bool:
    from mvo.common.geometry import se3
    try:
        se3.mat_to_vec(torch.eye(4).expand(1, 4, 4))
        return True
    except NotImplementedError:
        return False


@pytest.mark.skipif(not _se3_ready(),
                    reason="needs se3.mat_to_vec from base/se3-utils (Abhishek)")
def test_loss_is_finite_and_weights_rotation():
    m = _model()
    batch = _batch()
    losses = m.compute_loss(batch, m(batch))

    assert {"total", "trans", "rot"} <= set(losses)
    assert torch.isfinite(losses["total"])
    assert torch.allclose(losses["total"],
                          losses["trans"] + m.rot_weight * losses["rot"])
