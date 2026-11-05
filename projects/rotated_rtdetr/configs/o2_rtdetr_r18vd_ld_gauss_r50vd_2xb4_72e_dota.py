# Localization Distillation for a regular (non distribution) teacher:
# R50vd teacher -> R18vd student. The boxes are modeled as 2-D Gaussians and
# the student is pulled to the teacher boxes with the KLD (GDLoss), plus
# classification KD as in Rotated-LD. Works with the released checkpoint.
from mmengine.config import read_base
from mmdet.models.losses import KnowledgeDistillationKLDivLoss
from mmrotate.models.losses import GDLoss
from projects.rotated_rtdetr.rotated_rtdetr import (RotatedRTDETRKD,
                                                    RotatedRTDETRLDHead)

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_dota import *

teacher_config = \
    'projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py'
teacher_ckpt = ('https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/'
                'master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dota/epoch_72.pth')

model.update(
    type=RotatedRTDETRKD,
    teacher_config=teacher_config,
    teacher_ckpt=teacher_ckpt)
model.bbox_head.update(
    type=RotatedRTDETRLDHead,
    ld_mode='gaussian',
    loss_ld=dict(
        type=GDLoss,
        loss_type='kld',
        fun='log1p',
        tau=1,
        sqrt=False,
        loss_weight=2.0),
    loss_kd=dict(type=KnowledgeDistillationKLDivLoss, loss_weight=15, T=5),
    vlr_iou_thr=0.4,
    distill_all_layers=True)
