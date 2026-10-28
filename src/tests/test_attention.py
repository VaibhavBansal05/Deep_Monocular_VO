"""CBAM and SE blocks for M4."""

import pytest
import torch

from mvo.common.train.registry import build, import_all_models
from mvo.models.m4_cnn_attention.attention import (
    ChannelAttention,
    SEBlock,
    SpatialAttention,
    build_attention,
)

import_all_models()


def _x(b=2, c=32, h=8, w=26):
    return torch.randn(b, c, h, w)


# ------------------------------------------------------------------ branches


def test_channel_attention_is_a_per_channel_gate_in_0_1():
    a = ChannelAttention(32)(_x())
    assert a.shape == (2, 32, 1, 1), "channel attention must not vary over space"
    assert (a >= 0).all() and (a <= 1).all()


def test_spatial_attention_is_a_single_map_in_0_1():
    a = SpatialAttention(7)(_x())
    assert a.shape == (2, 1, 8, 26), "spatial attention must be one map per sample"
    assert (a >= 0).all() and (a <= 1).all()


def test_channel_attention_uses_both_pools():
    """Equal channel means with different maxima must produce different gates."""
    m = ChannelAttention(16)
    with torch.no_grad():
        m.mlp[0].weight.zero_()
        m.mlp[2].weight.zero_()
        m.mlp[0].weight[0, 3, 0, 0] = 1.0
        m.mlp[2].weight[0, 0, 0, 0] = 1.0
    spread = torch.zeros(1, 16, 4, 4)
    spread[0, 3, 0, :] = 1.0
    spike = torch.zeros_like(spread)
    spike[0, 3, 0, 0] = 4.0
    assert torch.equal(spread.mean((2, 3)), spike.mean((2, 3)))
    assert not torch.allclose(m(spread), m(spike), atol=1e-4)


def test_se_has_no_spatial_branch():
    """SE is strictly weaker than CBAM; that is what makes the pair a
    meaningful ablation rather than two arbitrary variants."""
    se = SEBlock(32)
    assert not any("spatial" in n for n, _ in se.named_modules())


# ------------------------------------------------------------------- wrapper


@pytest.mark.parametrize("kind", ["cbam", "se"])
def test_residual_wrapper_starts_near_the_identity(kind):
    """A 0.1 gate preserves most pretrained features and learns immediately."""
    blk = build_attention(kind, 32, residual=True)
    x = _x()
    assert torch.allclose(blk.gamma, torch.tensor([0.1]))
    assert (blk(x) - x).abs().mean() < 0.2 * x.abs().mean()


@pytest.mark.parametrize("kind", ["cbam", "se"])
def test_non_residual_form_does_modify_the_input(kind):
    """The paper-faithful path must actually gate, or the ablation is empty."""
    torch.manual_seed(0)
    blk = build_attention(kind, 32, residual=False)
    x = _x()
    assert not torch.allclose(blk(x), x, atol=1e-3)


def test_gate_receives_gradient_so_attention_can_switch_on():
    blk = build_attention("cbam", 32, residual=True)
    blk(_x()).pow(2).mean().backward()
    assert blk.gamma.grad is not None and blk.gamma.grad.abs() > 0


def test_shape_is_preserved():
    for kind in ("none", "se", "cbam"):
        x = _x()
        assert build_attention(kind, 32)(x).shape == x.shape


# --------------------------------------------------------------- integration


@pytest.mark.parametrize("kind", ["none", "se", "cbam"])
def test_m4_builds_and_runs_with_each_variant(kind):
    m = build("m4", pretrained=False, attention=kind)
    out = m({"imgs": torch.randn(2, 2, 3, 128, 416)})
    assert out["pose"].shape == (2, 1, 6)


def test_attention_is_a_small_fraction_of_the_parameter_budget():
    """The ablation is only controlled if the variants are comparable in size;
    a large attention module would confound capacity with attention."""
    base = build("m4", pretrained=False, attention="none").num_params()
    cbam = build("m4", pretrained=False, attention="cbam").num_params()
    assert (cbam - base) / base < 0.01, "attention adds more than 1% of parameters"


def test_cbam_exposes_spatial_maps_and_se_does_not():
    b = {"imgs": torch.randn(1, 2, 3, 128, 416)}
    m = build("m4", pretrained=False, attention="cbam")
    m(b)
    maps = m.spatial_maps()
    assert len(maps) == 4, "one spatial map per residual stage"
    assert all(mp.shape[1] == 1 for mp in maps)
    assert all(((mp >= 0) & (mp <= 1)).all() for mp in maps)

    se = build("m4", pretrained=False, attention="se")
    se(b)
    assert se.spatial_maps() == []


def test_residual_variant_has_the_same_initial_pose_prior_as_baseline():
    """Both zero heads start from the same metric motion prior."""
    torch.manual_seed(0)
    base = build("m4", pretrained=False, attention="none")
    torch.manual_seed(0)
    attn = build("m4", pretrained=False, attention="cbam")
    b = {"imgs": torch.randn(2, 2, 3, 128, 416)}
    assert torch.allclose(base(b)["pose"], attn(b)["pose"], atol=1e-6)


def test_unknown_attention_is_rejected():
    with pytest.raises(ValueError, match="attention"):
        build("m4", pretrained=False, attention="nonlocal")
