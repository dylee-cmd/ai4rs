# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.structures import InstanceData

from mmrotate.models.losses import GDLoss, PSFGDLoss, inflate_rbox
from mmrotate.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma

KLD = dict(loss_type='kld', fun='log1p', tau=1, sqrt=False)


def boxes(n, seed=0, size=(8, 120)):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * 1000
    wh = torch.rand(n, 2, generator=g) * (size[1] - size[0]) + size[0]
    a = (torch.rand(n, 1, generator=g) - 0.5) * 3
    return torch.cat([xy, wh, a], 1)


def test_inflate_rbox_adds_isotropic_psf_variance():
    b = boxes(10)
    for s in (0.5, 1.0, 3.0):
        _, sigma = xy_wh_r_2_xy_sigma(b)
        _, sigma_psf = xy_wh_r_2_xy_sigma(inflate_rbox(b, s))
        torch.testing.assert_close(
            sigma_psf, sigma + s**2 * torch.eye(2), rtol=1e-5, atol=1e-3)
    assert inflate_rbox(b, 0) is b


@pytest.mark.parametrize('loss_type', ['kld', 'gwd'])
def test_psf_std_zero_is_gdloss(loss_type):
    pred, target = boxes(16, 0), boxes(16, 1)
    weight = torch.rand(16)
    cfg = dict(KLD, loss_type=loss_type)
    if loss_type == 'gwd':
        cfg.pop('sqrt')  # gwd_loss has no sqrt argument
    p1, p2 = pred.clone().requires_grad_(), pred.clone().requires_grad_()
    ref = GDLoss(**cfg)(p1, target, weight, avg_factor=4)
    out = PSFGDLoss(psf_std=0, **cfg)(p2, target, weight, avg_factor=4)
    torch.testing.assert_close(out, ref)
    ref.backward()
    out.backward()
    torch.testing.assert_close(p2.grad, p1.grad)


def test_psf_floor_damps_small_target_errors():
    """The same 1 px center offset: huge KLD for a 3 px box, small for a
    100 px box. The PSF floor lowers the small-box loss much more."""
    small = torch.tensor([[100., 100., 3., 3., 0.]])
    large = torch.tensor([[100., 100., 100., 100., 0.]])
    shift = torch.tensor([[1., 0., 0., 0., 0.]])
    losses = {}
    for s in (0.0, 0.5, 1.0, 2.0):
        loss = PSFGDLoss(psf_std=s, reduction='none', **KLD)
        losses[s] = (float(loss(small + shift, small)),
                     float(loss(large + shift, large)))
    small_l = [v[0] for v in losses.values()]
    assert all(a > b for a, b in zip(small_l, small_l[1:]))  # monotone
    assert losses[2.0][0] < 0.5 * losses[0.0][0]
    assert abs(losses[2.0][1] - losses[0.0][1]) < 0.05 * losses[0.0][1]


def test_tiny_box_gradients_are_finite():
    pred = torch.tensor([[50., 50., 0., 1e-4, 0.3]], requires_grad=True)
    target = torch.tensor([[51., 50., 2., 2., 0.]])
    PSFGDLoss(psf_std=1.0, **KLD)(pred, target).backward()
    assert torch.isfinite(pred.grad).all()


def test_half_precision_inputs_and_autocast():
    pred, target = boxes(32, 0, (8, 900)), boxes(32, 1, (8, 900))
    loss = PSFGDLoss(psf_std=1.0, **KLD)
    ref = loss(pred.half().float(), target.half().float())
    out = loss(pred.half(), target.half())
    assert out.dtype == torch.float32 and torch.isfinite(out)
    torch.testing.assert_close(out, ref)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out_amp = loss(pred, target)
    assert out_amp.dtype == torch.float32
    torch.testing.assert_close(out_amp, loss(pred, target))


def test_psf_cost_matches_psf_loss():
    pytest.importorskip('mmdet')
    from projects.rotated_dino.rotated_dino.match_cost import (GDCost,
                                                               PSFGDCost)
    pred, gt = boxes(5, 0, (2, 60)), boxes(3, 1, (2, 60))
    for s in (0.0, 1.5):
        cost = PSFGDCost(psf_std=s, weight=2.0, **KLD)(
            InstanceData(bboxes=pred), InstanceData(bboxes=gt))
        loss = PSFGDLoss(psf_std=s, reduction='none', **KLD)
        pairwise = torch.stack([
            loss(pred[i:i + 1].expand(len(gt), 5), gt) for i in range(5)
        ])
        torch.testing.assert_close(cost, 2.0 * pairwise, rtol=1e-4,
                                   atol=1e-5)
    plain = GDCost(weight=2.0, **KLD)(
        InstanceData(bboxes=pred), InstanceData(bboxes=gt))
    zero = PSFGDCost(psf_std=0.0, weight=2.0, **KLD)(
        InstanceData(bboxes=pred), InstanceData(bboxes=gt))
    torch.testing.assert_close(zero, plain)
