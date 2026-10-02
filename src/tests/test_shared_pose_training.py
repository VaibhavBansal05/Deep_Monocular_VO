import torch

from mvo.common.train.pose_loss import geometric_pose_loss
from mvo.models.m2_cnn_rnn.model import M2CnnRnn
from mvo.models.m4_cnn_attention.model import AttentionVO


def test_m2_and_m4_no_module_baselines_are_identical():
    """The claimed no-RNN/no-attention controls must be the same network."""
    torch.manual_seed(7)
    m2 = M2CnnRnn(rnn="none", seq_len=2, pretrained=False, dropout=0.0)
    torch.manual_seed(7)
    m4 = AttentionVO(attention="none", pretrained=False, dropout=0.0)

    assert type(m2.encoder) is type(m4.encoder)
    assert type(m2.pose_head) is type(m4.decoder)
    assert sum(p.numel() for p in m2.parameters()) == sum(p.numel() for p in m4.parameters())

    batch = {"imgs": torch.randn(2, 2, 3, 64, 96)}
    m2.eval()
    m4.eval()
    torch.testing.assert_close(m2(batch)["pose"], m4(batch)["pose"])


def test_geodesic_rotation_uses_relative_rotation_not_axis_difference():
    std = torch.tensor([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    target = torch.zeros(1, 1, 6)
    pred = target.clone()
    pred[..., 3:] = torch.tensor([0.0, 0.0, 0.1])
    loss = geometric_pose_loss(pred, target, std, bias_weight=0.0)
    expected = 0.1 ** 2 / (3 * 0.1 ** 2)
    assert torch.allclose(loss["rot"], torch.tensor(expected), atol=1e-5)


def test_bias_penalty_detects_coherent_but_not_cancelling_residuals():
    std = torch.ones(6)
    target = torch.zeros(2, 1, 6)
    coherent = target.clone()
    coherent[..., 2] = 1.0
    cancelling = coherent.clone()
    cancelling[1, ..., 2] = -1.0
    a = geometric_pose_loss(coherent, target, std)
    b = geometric_pose_loss(cancelling, target, std)
    assert a["bias"] > 0
    assert torch.allclose(b["bias"], torch.tensor(0.0))


def test_geometric_loss_stays_fp32_inside_autocast():
    """SO(3) math must not be downcast by the surrounding AMP context."""
    pred = torch.zeros(2, 1, 6, requires_grad=True)
    target = pred.detach().clone()
    target[..., 2] = -0.6
    target[..., 4] = 0.01
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss = geometric_pose_loss(pred, target, torch.ones(6))
    assert loss["total"].dtype == torch.float32
    loss["total"].backward()
    assert torch.isfinite(pred.grad).all()
