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


# Per-axis standard deviations measured on the training split by
# scripts/measure_pose_stats.py. Translation is almost entirely forward (z);
# rotation is almost entirely yaw (ry). Identity targets are NOT a stand-in:
# a zero target makes dL/dout vanish at a zero-initialised head, which hides
# whether gradients flow at all.
T_STD = torch.tensor([0.0107, 0.0066, 0.4760])
R_STD = torch.tensor([0.00137, 0.00969, 0.00116])


def _realistic_batch(b=4, s=2, seed=0):
    """A batch whose rel_pose has the real magnitude of vKITTI 2 frame pairs."""
    from mvo.common.geometry import se3

    g = torch.Generator().manual_seed(seed)
    n = b * (s - 1)
    vec = torch.cat([torch.randn(n, 3, generator=g) * T_STD,
                     torch.randn(n, 3, generator=g) * R_STD], dim=-1)
    batch = _batch(b, s)
    batch["rel_pose"] = se3.vec_to_mat(vec).reshape(b, s - 1, 4, 4)
    return batch


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


def test_initial_prediction_is_exactly_the_identity():
    """Adjacent video frames are a near-identity transform, and the 6-vector is
    an axis-angle, so zero output is the right prior at initialisation.

    The head is zero-initialised, so this is exact rather than merely small.
    Guards against reintroducing a constant output scale: vKITTI 2 poses are
    metric (mean |t| = 0.61 m), so scaling the head down damps translation
    gradients without buying anything.
    """
    out = _model()(_batch(b=4))
    assert torch.equal(out["pose"], torch.zeros_like(out["pose"]))


def test_head_is_zero_initialised_but_trainable():
    m = _model()
    assert torch.equal(m.decoder.head.weight, torch.zeros_like(m.decoder.head.weight))
    assert m.decoder.head.weight.requires_grad

    # A zero-initialised final layer must still receive gradient, or the model
    # is frozen at identity forever. The target must be non-zero: against a
    # zero target, dL/dout = -2*target = 0 and nothing would flow -- that is a
    # property of the test, not of the model.
    b = _realistic_batch()
    m.compute_loss(b, m(b))["total"].backward()
    assert m.decoder.head.weight.grad.abs().sum() > 0


def test_rotation_weight_balances_the_two_loss_terms():
    """rot_weight is derived from measured target variance (E[t^2]/E[r^2] ~ 5869
    on the training split), so the weighted rotation term must be comparable to
    the translation term rather than ~59x smaller as it was with k=100."""
    m = _model()
    assert m.rot_weight > 1000, "k=100 leaves rotation effectively ignored"

    # The balance is defined at initialisation, where the model predicts zero
    # and each term reduces to the variance of its target. k is chosen so that
    #     E[t^2]  ~=  k * E[r^2]
    b = _realistic_batch(b=256, seed=1)
    zero_pred = {"pose": torch.zeros(256, 1, 6)}
    losses = m.compute_loss(b, zero_pred)

    # Band is deliberately loose: T_STD/R_STD are a Gaussian approximation of
    # the real per-axis spread, so the synthetic ratio (~2374:1) differs from
    # the measured one (~5869:1). k=5900 lands at ~2.5 here; the old k=100
    # lands at 0.04 and is still caught.
    ratio = float(m.rot_weight * losses["rot"] / losses["trans"])
    assert 0.2 < ratio < 5.0, (
        f"weighted rotation term is {ratio:.2f}x the translation term; "
        "k no longer matches the measured target statistics")


# ------------------------------------------------------------------ backward


def test_backbone_receives_no_gradient_on_the_very_first_step():
    """A consequence of zero-initialising the head, and it is intended.

    For y = Wx with W = 0:  dL/dW = x != 0, so the head learns, but
    dL/dx = W = 0, so nothing propagates into the encoder. The backbone is
    therefore frozen for exactly one step. This is the same "zero convolution"
    behaviour ControlNet relies on.

    Asserted rather than merely tolerated, so that swapping the head's
    initialisation is a deliberate decision and not a silent change.
    """
    m = _model()
    b = _realistic_batch()
    m.compute_loss(b, m(b))["total"].backward()

    assert m.decoder.head.weight.grad.abs().sum() > 0, "the head itself must learn"
    assert m.stem[0].weight.grad.abs().sum() == 0, (
        "expected no gradient into the backbone while head.weight is zero")


def test_gradients_reach_the_stem_after_one_optimiser_step():
    """The freeze above must last exactly one step, not forever."""
    m = _model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    b = _realistic_batch()

    opt.zero_grad()
    m.compute_loss(b, m(b))["total"].backward()
    opt.step()                      # head.weight departs from zero here

    opt.zero_grad()
    m.compute_loss(b, m(b))["total"].backward()

    g = m.stem[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g.abs().sum() > 0, "backbone still frozen after an optimiser step"


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
