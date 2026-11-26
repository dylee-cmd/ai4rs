"""The PSF config wires PSFGDLoss / PSFGDCost into rotated RT-DETR.

Run from the repo root:
    python -m pytest projects/rotated_rtdetr/tests/test_psf.py
"""
import math

import torch
from mmdet.structures import DetDataSample
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmrotate.models.losses import PSFGDLoss
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from projects.rotated_dino.rotated_dino.match_cost import PSFGDCost

CFG = 'projects/rotated_rtdetr/configs/'
SIZE = 128


def build(name, psf_std=None):
    init_default_scope('mmrotate')
    cfg = Config.fromfile(CFG + name + '.py')
    cfg.model.backbone.init_cfg = None
    if psf_std is not None:
        cfg.merge_from_dict({
            'model.bbox_head.loss_iou.psf_std': psf_std,
            'model.train_cfg.assigner.match_costs.2.psf_std': psf_std
        })
    torch.manual_seed(0)
    model = MODELS.build(cfg.model)
    model.init_weights()
    return model.train()


def batch():
    torch.manual_seed(1)
    samples = []
    for k in (4, 2):
        ds = DetDataSample(metainfo=dict(
            img_shape=(SIZE, SIZE), pad_shape=(SIZE, SIZE),
            batch_input_shape=(SIZE, SIZE)))
        ds.gt_instances = InstanceData(
            bboxes=RotatedBoxes(torch.cat([
                torch.rand(k, 2) * 100 + 14,
                torch.rand(k, 2) * 8 + 2,  # small targets
                (torch.rand(k, 1) - 0.5) * 3
            ], 1)),
            labels=torch.randint(0, 15, (k, )))
        samples.append(ds)
    return torch.randn(2, 3, SIZE, SIZE), samples


def losses_of(model):
    inputs, samples = batch()
    torch.manual_seed(2)  # same denoising noise
    return model.loss(inputs, samples)


def test_config_uses_psf_loss_and_cost():
    model = build('o2_rtdetr_r50vd_psf_2xb4_72e_dota')
    assert isinstance(model.bbox_head.loss_iou, PSFGDLoss)
    assert model.bbox_head.loss_iou.psf_std == 1.0
    costs = model.bbox_head.assigner.match_costs
    assert isinstance(costs[2], PSFGDCost) and costs[2].psf_std == 1.0


def test_psf_zero_is_the_original_model():
    ref = losses_of(build('o2_rtdetr_r50vd_2xb4_72e_dota'))
    out = losses_of(build('o2_rtdetr_r50vd_psf_2xb4_72e_dota', psf_std=0.0))
    assert ref.keys() == out.keys()
    for k in ref:
        torch.testing.assert_close(out[k], ref[k])


def test_psf_changes_only_box_losses_and_matching():
    ref = losses_of(build('o2_rtdetr_r50vd_psf_2xb4_72e_dota', 0.0))
    out = losses_of(build('o2_rtdetr_r50vd_psf_2xb4_72e_dota', 2.0))
    assert all(math.isfinite(float(v)) for v in out.values())
    # denoising queries have fixed targets (no matching): only loss_iou moves
    for k in ref:
        if 'dn_' in k and 'loss_iou' not in k:
            torch.testing.assert_close(out[k], ref[k])
    assert float(out['dn_loss_iou']) < float(ref['dn_loss_iou'])
