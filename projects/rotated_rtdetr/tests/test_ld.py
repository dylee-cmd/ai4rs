"""Tests for localization distillation (rotated_rtdetr_ld.py).

Run from the repo root:
    python -m pytest projects/rotated_rtdetr/tests/test_ld.py
"""
import copy
import math

import pytest
import torch
from mmdet.models.losses.kd_loss import knowledge_distillation_kl_div_loss
from mmdet.structures import DetDataSample
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from projects.rotated_rtdetr.rotated_rtdetr import (Integral,
                                                    RotatedRTDETRKD)
from projects.rotated_rtdetr.rotated_rtdetr.rotated_rtdetr_ld import (
    distribution_kl_div, generate_shared_dn_queries, shift_distribution)

CFG = 'projects/rotated_rtdetr/configs/'
SIZE = 128


def build(name, **head):
    init_default_scope('mmrotate')
    cfg = Config.fromfile(CFG + name + '.py')
    cfg.model.backbone.init_cfg = None
    cfg.model.bbox_head.update(head)
    if 'teacher_config' in cfg.model:
        teacher = Config.fromfile(cfg.model.teacher_config)
        teacher.model.backbone.init_cfg = None
        cfg.model.teacher_config = teacher
        cfg.model.teacher_ckpt = None
    model = MODELS.build(cfg.model)
    model.init_weights()
    return model


def batch(n_gts=(3, 1)):
    torch.manual_seed(0)
    samples = []
    for k in n_gts:
        ds = DetDataSample(metainfo=dict(
            img_shape=(SIZE, SIZE), ori_shape=(SIZE, SIZE),
            pad_shape=(SIZE, SIZE), batch_input_shape=(SIZE, SIZE),
            scale_factor=(1., 1.)))
        boxes = torch.cat([
            torch.rand(k, 2) * 80 + 24, torch.rand(k, 2) * 30 + 10,
            (torch.rand(k, 1) - 0.5) * 3
        ], 1)
        ds.gt_instances = InstanceData(
            bboxes=RotatedBoxes(boxes), labels=torch.randint(0, 15, (k, )))
        samples.append(ds)
    return torch.randn(len(n_gts), 3, SIZE, SIZE), samples


# ----------------------------------------------------------------- pieces
def test_integral():
    integral = Integral(reg_max=8, reg_range=4.0, num_coords=5)
    assert torch.allclose(
        integral(torch.zeros(2, 45)), torch.zeros(2, 5), atol=1e-6)
    one_hot = torch.full((1, 5, 9), -1e4)
    one_hot[0, :, 6] = 0
    torch.testing.assert_close(integral(one_hot.view(1, 45)),
                               torch.full((1, 5), 2.0))


def test_distribution_head_starts_like_plain_head():
    model = build('o2_rtdetr_r18vd_dist_2xb4_72e_dota')
    for branch in model.bbox_head.reg_branches:
        assert isinstance(branch[-1], Integral)
        out = branch(torch.randn(4, 256))
        assert out.shape == (4, 5)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_shift_distribution():
    torch.manual_seed(0)
    prob = torch.rand(6, 17).softmax(-1)
    # no shift: identity
    torch.testing.assert_close(
        shift_distribution(prob, torch.zeros(6), 4.0), prob)
    # half a bin: one hot splits in two
    one_hot = torch.zeros(1, 17)
    one_hot[0, 8] = 1
    out = shift_distribution(one_hot, torch.tensor([0.25]), 4.0)
    torch.testing.assert_close(out[0, 8:10], torch.tensor([0.5, 0.5]))
    # mass is kept and the expectation moves by the shift
    bins = torch.linspace(-4, 4, 17)
    centered = torch.exp(-(bins - 0.3)**2)[None].repeat(3, 1)
    centered = centered / centered.sum(-1, keepdim=True)
    shift = torch.tensor([-0.7, 0.1, 0.55])
    out = shift_distribution(centered, shift, 4.0)
    torch.testing.assert_close(out.sum(-1), torch.ones(3))
    torch.testing.assert_close((out * bins).sum(-1),
                               (centered * bins).sum(-1) + shift,
                               atol=2e-3, rtol=0)


def test_distribution_kl_matches_mmdet_kd_loss():
    torch.manual_seed(0)
    pred, soft = torch.randn(7, 33), torch.randn(7, 33)
    T = 5
    ref = knowledge_distillation_kl_div_loss(
        pred, soft, T=T, reduction='none')
    ours = distribution_kl_div(pred, (soft / T).softmax(-1), T)
    torch.testing.assert_close(ours, ref)


def test_shared_dn_queries():
    student = build('o2_rtdetr_r18vd_2xb4_72e_dota')
    teacher = build('o2_rtdetr_r50vd_2xb4_72e_dota')
    _, samples = batch()
    (s_label, s_box, s_mask, s_meta), (t_label, t_box, t_mask, t_meta) = \
        generate_shared_dn_queries(
            [student.dn_query_generator, teacher.dn_query_generator],
            samples)
    assert torch.equal(s_box, t_box) and torch.equal(s_mask, t_mask)
    assert s_meta == t_meta
    # each label query is the model's own embedding of the same label
    s_emb = student.dn_query_generator.label_embedding.weight
    t_emb = teacher.dn_query_generator.label_embedding.weight
    valid = s_label.abs().sum(-1) > 0  # not padding
    s_idx = (s_label[valid][:, None] == s_emb[None]).all(-1).float().argmax(1)
    t_idx = (t_label[valid][:, None] == t_emb[None]).all(-1).float().argmax(1)
    assert torch.equal(s_idx, t_idx)
    assert torch.equal(valid, t_label.abs().sum(-1) > 0)


# ------------------------------------------------------- whole detector
def self_distill(name, **head):
    """A student whose teacher is an exact copy of itself."""
    model = build(name, **head)
    model.teacher_model = copy.deepcopy(model)
    model.teacher_model.__class__ = RotatedRTDETRKD.__mro__[1]  # plain RTDETR
    model.teacher_model.requires_grad_(False)
    # The encoder score layer only picks the top-k matching queries: with a
    # different one, the teacher must still use the student's positions.
    enc_cls = model.teacher_model.bbox_head.cls_branches[
        model.teacher_model.decoder.num_layers]
    torch.nn.init.normal_(enc_cls.weight, std=1.0)
    model.train()
    for m in model.modules():  # identical numerics in train / eval
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    return model


@pytest.mark.parametrize('name', [
    'o2_rtdetr_r18vd_ld_r50vd_dist_2xb4_72e_dota',
    'o2_rtdetr_r18vd_ld_gauss_r50vd_2xb4_72e_dota'
])
def test_self_distillation_is_zero(name):
    """Teacher == student: the LD / KD losses of the last layer vanish, which
    needs the denoising queries, the matching queries (shared top-k encoder
    positions) and the reference frames of teacher and student to line up.
    A low IoU threshold makes the matching queries count as well."""
    model = self_distill(name, vlr_iou_thr=0.0)
    inputs, samples = batch()
    losses = model.loss(inputs, samples)
    assert float(losses['loss_kd']) < 1e-5, losses['loss_kd']
    assert float(losses['loss_ld']) < 1e-4, losses['loss_ld']
    # earlier layers differ from the teacher's last layer
    assert float(losses['d0.loss_kd']) > 1e-5


def test_distillation_gradients_reach_student_only():
    model = build('o2_rtdetr_r18vd_ld_r50vd_dist_2xb4_72e_dota')
    model.train()
    inputs, samples = batch()
    losses = model.loss(inputs, samples)
    distill = sum(v for k, v in losses.items() if 'loss_ld' in k
                  or 'loss_kd' in k)
    assert math.isfinite(float(distill)) and float(distill) > 0
    distill.backward()
    grads = [p.grad for p in model.bbox_head.reg_branches.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in model.teacher_model.parameters())


def test_teacher_is_not_part_of_the_student():
    model = build('o2_rtdetr_r18vd_ld_gauss_r50vd_2xb4_72e_dota')
    teacher_params = {id(p) for p in model.teacher_model.parameters()}
    assert not teacher_params & {id(p) for p in model.parameters()}
    assert not any(k.startswith('teacher') for k in model.state_dict())
    model.train()
    assert model.training and not model.teacher_model.training
    assert all(not p.requires_grad for p in model.teacher_model.parameters())


def test_plain_model_unchanged():
    """Non-distribution decoders still return (classes, coords) only."""
    model = build('o2_rtdetr_r18vd_2xb4_72e_dota').eval()
    inputs, samples = batch((2, ))
    with torch.no_grad():
        feats = model.extract_feat(inputs[:1])
        out = model.forward_transformer(feats, samples)
    assert len(out['references']) == 2
