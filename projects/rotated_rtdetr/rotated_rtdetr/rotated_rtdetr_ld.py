"""Localization Distillation (LD) + classification KD for rotated RT-DETR.

Adapted from `Rotated-LD <https://github.com/Zzh-tju/Rotated-LD>`_
(Localization Distillation for Object Detection, CVPR 2022 / TPAMI 2023),
which distills a dense, anchor-based rotated detector:

- ``KnowledgeDistillationRotatedSingleStageDetector`` keeps a frozen teacher
  (a plain attribute, not a submodule) and feeds it the same images.
- ``LDRotatedRetinaHead`` adds ``loss_ld`` (KL divergence between the
  teacher's and the student's box distributions, temperature ``T``) and
  ``loss_kd`` (KL divergence between the classification logits), both
  restricted to the *valuable localization region* and averaged over it.

RT-DETR has no anchor grid shared by teacher and student, so queries are put
in 1-to-1 correspondence instead:

- **Denoising queries**: the noisy GT boxes/labels are generated once and fed
  to both models (each with its own label embedding).
- **Matching queries**: the teacher takes its queries from the encoder
  positions the student selected (``topk_indices``), i.e. the same "anchors".

Valuable localization region: positive denoising queries for LD and all
denoising queries for KD; for the matching queries, those whose teacher box
overlaps a GT with rotated IoU >= ``vlr_iou_thr`` (LD and KD).

Two LD modes:

- ``'distribution'`` (Rotated-LD): teacher and student both have
  distribution regression heads (``reg_max``). The teacher distribution is
  shifted into the frame of the student's reference points (their references
  differ after the first layer) and re-binned, then KL(teacher || student)
  with temperature. The teacher must be trained as a distribution model
  first, as in Rotated-LD.
- ``'gaussian'``: works with any (e.g. already released) teacher. The boxes
  are modeled as 2-D Gaussians and the student box is pulled to the teacher
  box with a Gaussian distance (``GDLoss``, KLD by default).
"""
import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import Tensor, nn
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, reduce_mean
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbbox_overlaps

from .rotated_rtdetr import RotatedRTDETR
from .rotated_rtdetr_head import RotatedRTDETRHead
from .utils import cast_tensors_to_float, disable_autocast, find_device_type


def generate_shared_dn_queries(generators: List[nn.Module],
                               batch_data_samples: SampleList) -> List[tuple]:
    """Denoising queries with the same noise for several models.

    Same as ``RotatedCdnQueryGenerator.__call__`` but the label and box noise
    is drawn once (with ``generators[0]``) and each generator only embeds the
    noisy labels with its own ``label_embedding``.

    Returns:
        list[tuple]: ``(dn_label_query, dn_bbox_query, attn_mask, dn_meta)``
        for every generator.
    """
    gen = generators[0]
    gt_labels_list, gt_bboxes_list = [], []
    for sample in batch_data_samples:
        img_h, img_w = sample.img_shape
        bboxes = sample.gt_instances.bboxes.regularize_boxes(**gen.angle_cfg)
        factor = bboxes.new_tensor(
            [img_w, img_h, img_w, img_h, gen.angle_factor]).unsqueeze(0)
        gt_bboxes_list.append(bboxes / factor)
        gt_labels_list.append(sample.gt_instances.labels)
    gt_labels = torch.cat(gt_labels_list)
    gt_bboxes = torch.cat(gt_bboxes_list)
    max_num_target = max(len(b) for b in gt_bboxes_list)
    num_groups = gen.get_num_groups(max_num_target)

    # noisy labels, as in CdnQueryGenerator.generate_dn_label_query
    assert gen.label_noise_scale > 0
    gt_labels_expand = gt_labels.repeat(2 * num_groups, 1).view(-1)
    p = torch.rand_like(gt_labels_expand.float())
    chosen_indice = torch.nonzero(p < (gen.label_noise_scale * 0.5)).view(-1)
    new_labels = torch.randint_like(chosen_indice, 0, gen.num_classes)
    noisy_labels = gt_labels_expand.scatter(0, chosen_indice, new_labels)

    dn_bbox_query = gen.generate_dn_bbox_query(gt_bboxes, num_groups)
    batch_idx = torch.cat([
        torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
    ])
    dn_meta = dict(
        num_denoising_queries=int(max_num_target * 2 * num_groups),
        num_denoising_groups=num_groups)

    outputs = []
    for g in generators:
        assert g.num_classes == gen.num_classes
        label_query, bbox_query = g.collate_dn_queries(
            g.label_embedding(noisy_labels), dn_bbox_query, batch_idx,
            len(batch_data_samples), num_groups)
        attn_mask = g.generate_dn_mask(
            max_num_target, num_groups, device=label_query.device)
        outputs.append((label_query, bbox_query, attn_mask, dict(dn_meta)))
    return outputs


def shift_distribution(prob: Tensor, shift: Tensor, reg_range: float) -> Tensor:
    """Move a distribution over evenly spaced bins by ``shift`` (values).

    Every bin's mass is moved to ``bin + shift`` and split linearly between
    the two nearest bins (as GFL builds its distribution targets), clamped to
    the bin range.

    Args:
        prob (Tensor): (N, reg_max + 1) probabilities.
        shift (Tensor): (N, ) shift in value units.
        reg_range (float): Bins span ``[-reg_range, reg_range]``.
    """
    reg_max = prob.size(1) - 1
    step = 2 * reg_range / reg_max
    pos = (torch.arange(reg_max + 1, device=prob.device, dtype=prob.dtype)
           + shift[:, None] / step).clamp(0, reg_max)
    left = pos.floor().long().clamp(max=reg_max - 1)
    w_right = pos - left
    out = torch.zeros_like(prob)
    out.scatter_add_(1, left, prob * (1 - w_right))
    out.scatter_add_(1, left + 1, prob * w_right)
    return out


def distribution_kl_div(pred: Tensor, target_prob: Tensor, T: float) -> Tensor:
    """KL(target || softmax(pred / T)) * T^2 per row, like
    ``knowledge_distillation_kl_div_loss`` but with target probabilities."""
    log_p = (pred / T).log_softmax(-1)
    t = target_prob.detach()
    kl = (t * (t.clamp(min=1e-12).log() - log_p)).sum(-1)
    return kl * (T * T) / pred.size(-1)


class RotatedRTDETRLDHead(RotatedRTDETRHead):
    """Rotated RT-DETR head with localization (LD) and classification (KD)
    distillation losses, the counterpart of Rotated-LD's
    ``LDRotatedRetinaHead``.

    Args:
        loss_ld (dict): LD loss. For ``ld_mode='distribution'`` a
            ``KnowledgeDistillationKLDivLoss`` config (its ``T`` and
            ``loss_weight`` are used); for ``'gaussian'`` a box loss on
            (cx, cy, w, h, rad) pixel boxes, e.g. ``GDLoss``.
        loss_kd (dict): KD loss on the classification logits,
            ``KnowledgeDistillationKLDivLoss``.
        ld_mode (str): ``'distribution'`` or ``'gaussian'``.
        vlr_iou_thr (float): Matching queries whose teacher box has a rotated
            IoU >= this with a GT form the valuable localization region.
        distill_all_layers (bool): Distill every student decoder layer (each
            from the teacher's last layer) or only the last one.
    """

    def __init__(self,
                 *args,
                 loss_ld: dict,
                 loss_kd: dict,
                 ld_mode: str = 'distribution',
                 vlr_iou_thr: float = 0.4,
                 distill_all_layers: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert ld_mode in ('distribution', 'gaussian')
        if ld_mode == 'distribution':
            assert self.reg_max is not None, \
                "ld_mode='distribution' needs a distribution head (reg_max)"
        self.ld_mode = ld_mode
        self.loss_ld = MODELS.build(loss_ld)
        self.loss_kd = MODELS.build(loss_kd)
        self.vlr_iou_thr = vlr_iou_thr
        self.distill_all_layers = distill_all_layers

    def get_distill_regions(self, teacher_bboxes: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict],
                            dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Per query weights (B, Q) of the LD and KD regions."""
        bs, num_queries = teacher_bboxes.shape[:2]
        num_dn = dn_meta['num_denoising_queries']
        num_groups = dn_meta['num_denoising_groups']
        ld_region = teacher_bboxes.new_zeros(bs, num_queries)
        kd_region = teacher_bboxes.new_zeros(bs, num_queries)
        group_size = num_dn // num_groups if num_groups else 0
        for i, (gt, meta) in enumerate(zip(batch_gt_instances,
                                           batch_img_metas)):
            num_gt = len(gt.labels)
            if num_gt > 0:
                # same layout as RotatedDINOHead._get_dn_targets_single
                t = torch.arange(num_gt, device=ld_region.device)
                pos = (torch.arange(num_groups, device=ld_region.device)
                       [:, None] * group_size + t).flatten()
                neg = pos + group_size // 2
                ld_region[i, pos] = 1
                kd_region[i, pos] = 1
                kd_region[i, neg] = 1
                img_h, img_w = meta['img_shape']
                factor = teacher_bboxes.new_tensor(
                    [img_w, img_h, img_w, img_h, self.angle_factor])
                ious = rbbox_overlaps(teacher_bboxes[i, num_dn:] * factor,
                                      gt.bboxes.tensor.float())
                vlr = (ious.max(1)[0] >= self.vlr_iou_thr).float()
                ld_region[i, num_dn:] = vlr
                kd_region[i, num_dn:] = vlr
        return ld_region, kd_region

    def loss_distill(self, student_refs: List[List[Tensor]],
                     teacher_refs: List[List[Tensor]],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     dn_meta: Dict[str, int]) -> Dict[str, Tensor]:
        """LD and KD losses.

        Args:
            student_refs, teacher_refs (list): Decoder ``references``:
                per layer class logits (B, Q, C), sigmoid boxes (B, Q, 5) and,
                for distribution heads, distribution logits
                (B, Q, 5 * (reg_max + 1)) and unactivated reference points
                (B, Q, 5). The teacher's last layer is used.
        """
        # float32 and no autocast, as for the other losses
        student_refs = cast_tensors_to_float(student_refs)
        teacher_refs = cast_tensors_to_float(teacher_refs)
        device_type = find_device_type(student_refs) or 'cuda'
        with disable_autocast(device_type):
            return self._loss_distill(student_refs, teacher_refs,
                                      batch_gt_instances, batch_img_metas,
                                      dn_meta)

    def _loss_distill(self, student_refs, teacher_refs, batch_gt_instances,
                      batch_img_metas, dn_meta):
        t_cls, t_box = teacher_refs[0][-1], teacher_refs[1][-1]
        assert t_cls.shape == student_refs[0][-1].shape, \
            'teacher and student must have the same queries and classes'
        ld_region, kd_region = self.get_distill_regions(
            t_box, batch_gt_instances, batch_img_metas, dn_meta)
        ld_num = reduce_mean(ld_region.sum()).clamp(min=1).item()
        kd_num = reduce_mean(kd_region.sum()).clamp(min=1).item()
        ld_w, kd_w = ld_region.flatten(), kd_region.flatten()

        if self.ld_mode == 'distribution':
            assert len(teacher_refs) == 4 and len(student_refs) == 4, \
                'distribution LD needs distribution heads in both models'
            bins = self.reg_max + 1
            T = self.loss_ld.T
            t_prob = (teacher_refs[2][-1].reshape(-1, bins) / T).softmax(-1)
            t_ref = teacher_refs[3][-1].reshape(-1)
        else:
            img_h, img_w = batch_img_metas[0]['img_shape']
            factor = t_box.new_tensor(
                [img_w, img_h, img_w, img_h, self.angle_factor])

        num_layers = len(student_refs[0])
        layers = range(num_layers) if self.distill_all_layers \
            else [num_layers - 1]
        losses = {}
        for lid in layers:
            prefix = '' if lid == num_layers - 1 else f'd{lid}.'
            s_cls = student_refs[0][lid]
            losses[f'{prefix}loss_kd'] = self.loss_kd(
                s_cls.reshape(-1, s_cls.size(-1)),
                t_cls.reshape(-1, t_cls.size(-1)),
                weight=kd_w, avg_factor=kd_num)
            if self.ld_mode == 'distribution':
                # teacher distribution, moved into this layer's frame:
                # teacher value = t_ref + d = s_ref + (d + t_ref - s_ref)
                s_ref = student_refs[3][lid].reshape(-1)
                target = shift_distribution(t_prob, t_ref - s_ref,
                                            self.reg_range)
                loss = distribution_kl_div(
                    student_refs[2][lid].reshape(-1, bins), target, T)
                w = ld_w[:, None].expand(-1, self.reg_dim).reshape(-1)
                # summed over the 5 coordinates, averaged over the region,
                # as Rotated-LD
                losses[f'{prefix}loss_ld'] = self.loss_ld.loss_weight * (
                    loss * w).sum() / ld_num
            else:
                s_box = student_refs[1][lid]
                losses[f'{prefix}loss_ld'] = self.loss_ld(
                    (s_box * factor).reshape(-1, 5),
                    (t_box * factor).reshape(-1, 5),
                    weight=ld_w, avg_factor=ld_num)
        return losses


class RotatedRTDETRKD(RotatedRTDETR):
    """Rotated RT-DETR student distilled from a rotated RT-DETR teacher,
    the counterpart of Rotated-LD's
    ``KnowledgeDistillationRotatedSingleStageDetector``.

    The student's ``bbox_head`` must be a :class:`RotatedRTDETRLDHead`.

    Args:
        teacher_config (str | dict | Config): Teacher config file or config
            (with a ``model`` key).
        teacher_ckpt (str, optional): Teacher checkpoint.
        eval_teacher (bool): Keep the teacher in eval mode. Defaults to True.
    """

    def __init__(self,
                 *args,
                 teacher_config: Union[str, dict, Config],
                 teacher_ckpt: Optional[str] = None,
                 eval_teacher: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.bbox_head, RotatedRTDETRLDHead)
        if isinstance(teacher_config, str):
            teacher_config = Config.fromfile(teacher_config)
        teacher_model_cfg = copy.deepcopy(teacher_config['model'])
        if teacher_ckpt is not None:
            # the checkpoint holds all weights, skip pretrained downloads
            teacher_model_cfg['backbone']['init_cfg'] = None
        self.teacher_model = MODELS.build(teacher_model_cfg)
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        self.eval_teacher = eval_teacher
        self.teacher_model.train(not eval_teacher and self.training)
        self.teacher_model.requires_grad_(False)
        assert self.teacher_model.num_queries == self.num_queries
        assert self.teacher_model.bbox_head.num_classes == \
            self.bbox_head.num_classes

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        teacher = self.teacher_model
        dn_student, dn_teacher = generate_shared_dn_queries(
            [self.dn_query_generator, teacher.dn_query_generator],
            batch_data_samples)

        # student (sets distill_ctx['topk_indices'])
        ctx = dict(dn=dn_student)
        self.distill_ctx = ctx
        try:
            img_feats = self.extract_feat(batch_inputs)
            head_inputs_dict = self.forward_transformer(
                img_feats, batch_data_samples)
        finally:
            self.distill_ctx = None
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        # teacher on the same denoising queries and encoder positions
        with torch.no_grad():
            teacher.distill_ctx = dict(
                dn=dn_teacher, topk_indices=ctx['topk_indices'])
            try:
                t_feats = teacher.extract_feat(batch_inputs)
                t_head_inputs = teacher.forward_transformer(
                    t_feats, batch_data_samples)
            finally:
                teacher.distill_ctx = None

        batch_gt_instances = [s.gt_instances for s in batch_data_samples]
        batch_img_metas = [s.metainfo for s in batch_data_samples]
        losses.update(
            self.bbox_head.loss_distill(
                head_inputs_dict['references'], t_head_inputs['references'],
                batch_gt_instances, batch_img_metas,
                head_inputs_dict['dn_meta']))
        return losses

    def cuda(self, device=None) -> nn.Module:
        """The teacher is a plain attribute: move it explicitly."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def to(self, *args, **kwargs) -> nn.Module:
        self.teacher_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True) -> nn.Module:
        """Keep the teacher in eval mode (if ``eval_teacher``)."""
        if hasattr(self, 'teacher_model'):
            self.teacher_model.train(mode and not self.eval_teacher)
        return super().train(mode)

    def __setattr__(self, name, value) -> None:
        """Keep the teacher out of the student's modules, parameters,
        state_dict and optimizer (as in Rotated-LD)."""
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
