import pytest
import torch

from mvo.models.m3_selfsup_posedepth.losses import (
    edge_aware_smoothness,
    erode_validity,
    multiscale_loss,
    select_reprojection,
    ssim_residual,
)


@pytest.mark.parametrize("value", [0., 0.5, 1.])
def test_identical_constant_images_have_zero_ssim_error(value):
    a = torch.full((1, 3, 8, 12), value)
    assert ssim_residual(a, a).abs().max() < 1e-6


def test_invalid_source_cannot_win_minimum_or_mean():
    errors = torch.tensor([[[[0.0]], [[0.4]]]], requires_grad=True)
    identity = torch.ones_like(errors)
    valid = torch.tensor([[[[False]], [[True]]]])
    for use_min in (True, False):
        loss, coverage, active, _ = select_reprojection(errors, identity, valid, min_reproj=use_min)
        torch.testing.assert_close(loss, torch.tensor(0.4))
        assert coverage == active == 1
    loss.backward()
    assert errors.grad[0, 0, 0, 0] == 0 and errors.grad[0, 1, 0, 0] == 1


def test_minimum_selects_best_source_per_pixel():
    errors = torch.tensor([[[[0.1, 0.8]], [[0.7, 0.2]]]])
    valid = torch.ones_like(errors, dtype=torch.bool)
    loss, _, _, _ = select_reprojection(errors, torch.ones_like(errors), valid)
    torch.testing.assert_close(loss, torch.tensor(0.15))


@pytest.mark.parametrize("automask", [False, True])
def test_empty_masks_have_connected_finite_zero_loss(automask):
    errors = torch.rand(2, 2, 8, 12, requires_grad=True)
    loss, valid, active, _ = select_reprojection(
        errors, torch.ones_like(errors), torch.zeros_like(errors), automask=automask)
    assert loss == valid == active == 0
    loss.backward()
    assert torch.isfinite(errors.grad).all() and errors.grad.abs().sum() == 0


def test_automask_rejects_stationary_ties_and_worse_warps():
    errors = torch.tensor([[[[0.1, 0.5, 0.8]], [[0.2, 0.5, 0.9]]]])
    identity = torch.full_like(errors, 0.5)
    loss, _, active, mask = select_reprojection(errors, identity, torch.ones_like(errors))
    assert mask.flatten().tolist() == [True, False, False]
    torch.testing.assert_close(loss, torch.tensor(0.1))
    torch.testing.assert_close(active, torch.tensor(1 / 3))


def test_ssim_mask_rejects_neighbours_of_invalid_pixel():
    valid = torch.ones(1, 1, 7, 7)
    valid[..., 3, 3] = 0
    assert (~erode_validity(valid)).sum() == 9


def test_smoothness_is_scale_invariant_and_edges_relax_it():
    constant = torch.ones(1, 1, 8, 12)
    rgb = torch.zeros(1, 3, 8, 12)
    assert edge_aware_smoothness(constant, rgb) == 0
    step = constant.clone()
    step[..., 6:] = 2
    edge = rgb.clone()
    edge[..., 6:] = 1
    torch.testing.assert_close(edge_aware_smoothness(step, rgb),
                               edge_aware_smoothness(step * 10, rgb))
    assert edge_aware_smoothness(step, edge) < edge_aware_smoothness(step, rgb)


def test_loss_is_fp32_and_all_scales_backpropagate_at_zero_pose():
    images = torch.rand(1, 3, 3, 16, 24)
    K = torch.tensor([[[20., 0, 11.5], [0, 20., 7.5], [0, 0, 1.]]])
    pose = torch.zeros(1, 2, 6, requires_grad=True)
    disparities = [torch.full((1, 1, 16 // 2**s, 24 // 2**s), 0.4,
                              requires_grad=True) for s in range(4)]
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        losses = multiscale_loss(images, disparities, pose, K, torch.linalg.inv(K), automask=False)
    assert losses["total"].dtype == torch.float32
    losses["total"].backward()
    for p in [pose, *disparities]:
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_static_sequence_has_no_active_photometric_pixels():
    image = torch.rand(1, 1, 3, 16, 24).expand(-1, 3, -1, -1, -1)
    K = torch.eye(3).unsqueeze(0)
    loss = multiscale_loss(image, [torch.ones(1, 1, 16, 24) * 0.5],
                           torch.zeros(1, 2, 6), K, K)
    assert loss["active_fraction"] == loss["photometric"] == 0


def test_synthetic_textured_plane_optimizes_joint_depth_and_translation():
    # Reference images differ by exactly one horizontal pixel. Geometry must
    # learn fx*tx/depth = 1; neither tx nor depth has a uniquely observable scale.
    torch.manual_seed(9)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        target = torch.rand(1, 3, 16, 32)
        images = torch.stack([torch.roll(target, -1, -1), target,
                              torch.roll(target, 1, -1)], dim=1)
        K = torch.tensor([[[25., 0, 15.5], [0, 25., 7.5], [0, 0, 1.]]])
        tx = torch.tensor(0.05, requires_grad=True)
        logit = torch.tensor(-3.94, requires_grad=True)
        optimizer = torch.optim.Adam([tx, logit], lr=0.02)

        def objective():
            vector = torch.cat([tx.view(1), torch.zeros(5)]).view(1, 1, 6).expand(1, 2, 6)
            disp = logit.sigmoid().expand(1, 1, 16, 32)
            return multiscale_loss(images, [disp], vector, K, torch.linalg.inv(K),
                                   ssim_weight=0, smooth_weight=0, automask=False)["total"]

        initial = float(objective().detach())
        for _ in range(40):
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            assert torch.isfinite(tx.grad) and torch.isfinite(logit.grad)
            optimizer.step()
        assert float(objective().detach()) < initial * 0.3
        assert logit.detach() != -3.94 and tx.detach() != 0.05
    finally:
        torch.set_num_threads(previous)
