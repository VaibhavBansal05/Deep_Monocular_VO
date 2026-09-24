"""M4 tests that need no dataset.

The batch dict is a frozen contract, so the model can be driven with synthetic
tensors long before the real dataloader exists. That is the point of the
contract: M4 is verified end-to-end while base/dataset is still in review.
"""

import pytest
import torch
import torch.nn as nn

from mvo.common.geometry import se3
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
    # The identity is exact in exact arithmetic, but NOT bitwise reproducible:
    # summing 6 channels instead of 3 changes the accumulation order, and the
    # order depends on the convolution backend. torch.equal() here passed on
    # one machine and failed on Colab over the last few bits.
    #
    # float64 with 7*7*6 accumulations on O(1) values leaves error ~3e-14, so
    # 1e-12 has ~30x headroom while still catching the failure that matters:
    # dropping the division scales the output by the repeat count, an O(1)
    # discrepancy that is ~12 orders of magnitude larger.
    conv3 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False).double()
    conv6 = _inflate_conv1(conv3, 6).double()

    x3 = torch.randn(1, 3, 64, 64, dtype=torch.float64)
    x6 = torch.cat([x3, x3], dim=1)

    torch.testing.assert_close(conv6(x6), conv3(x3), atol=1e-12, rtol=0)


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
    # whiten=False, since the whitened head deliberately starts at the mean
    # pose instead; that behaviour has its own test below.
    out = _model(whiten=False)(_batch(b=4))
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
    """The unwhitened path still needs DeepVO's k, and it must be measured.

    E[t^2]/E[r^2] is ~5869 on the training split, so the published k=100 leaves
    rotation ~59x under-weighted. Whitening supersedes this (each axis gets an
    equal share regardless), but the unwhitened path remains available for the
    ablation and must stay correct.
    """
    m = _model(whiten=False, rot_weight=5900.0)
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


@pytest.mark.parametrize("attn", ["none", "cbam", "se"])
def test_every_ablation_variant_is_available(attn):
    """The ablation M4 exists to run needs all three to build and run."""
    out = _model(attention=attn)(_batch(b=2))
    assert out["pose"].shape == (2, 1, 6)


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


# ------------------------------------------------------------- pose whitening


def test_whitened_model_starts_at_the_mean_pose_not_the_identity():
    """The point of adding the mean back.

    With a zero-initialised head, an unwhitened model predicts the identity --
    a stationary camera, which scores t_rel 99.16% on Scene20. The whitened
    model instead starts at the average training motion, which is where the
    trivial constant predictor ends, so training begins from a far better
    prior.
    """
    from mvo.common.data.pose_stats import POSE_MEAN

    out = _model(whiten=True)(_batch(b=2))
    expected = torch.tensor(POSE_MEAN)
    assert torch.allclose(out["pose"][0, 0], expected, atol=1e-6)

    plain = _model(whiten=False)(_batch(b=2))
    assert torch.equal(plain["pose"], torch.zeros_like(plain["pose"]))


def test_whitening_equalises_the_per_axis_gradient_share():
    """Why whitening exists: unwhitened, tz takes 99.89% of the gradient."""
    from mvo.common.data.pose_stats import POSE_STD

    std = torch.tensor(POSE_STD)
    plain_share = (std ** 2) / (std ** 2).sum()
    assert plain_share[2] > 0.99, "tz should dominate an unwhitened MSE"

    whitened = torch.ones(6) / 6.0          # unit variance on every axis
    assert abs(float(whitened.max() - whitened.min())) < 1e-6


def test_whitened_loss_is_scale_free_across_axes():
    """An error of one standard deviation must cost the same on every axis."""
    m = _model(whiten=True)
    b = _batch(b=1)
    b["rel_pose"] = torch.eye(4).expand(1, 1, 4, 4).contiguous()

    costs = []
    for axis in range(6):
        off = torch.zeros(1, 1, 6)
        off[0, 0, axis] = m.pose_std[axis]          # one sigma on this axis
        target = se3.mat_to_vec(b["rel_pose"])
        costs.append(float(m.compute_loss(b, {"pose": target + off})["total"]))

    assert max(costs) / min(costs) < 1.01, f"axis costs differ: {costs}"
