# Localization Distillation (Rotated-LD, https://github.com/Zzh-tju/Rotated-LD)
# R50vd teacher -> R18vd student, distribution LD + classification KD.
# 1. train the distribution-kind teacher:
#    bash tools/dist_train.sh \
#        projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_dist_2xb4_72e_dota.py 2
# 2. set `teacher_ckpt` below and train this config.
from mmengine.config import read_base
from mmdet.models.losses import KnowledgeDistillationKLDivLoss
from projects.rotated_rtdetr.rotated_rtdetr import (RotatedRTDETRKD,
                                                    RotatedRTDETRLDHead)

with read_base():
    from .o2_rtdetr_r18vd_dist_2xb4_72e_dota import *

teacher_config = \
    'projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_dist_2xb4_72e_dota.py'
teacher_ckpt = 'work_dirs/o2_rtdetr_r50vd_dist_2xb4_72e_dota/epoch_72.pth'

model.update(
    type=RotatedRTDETRKD,
    teacher_config=teacher_config,
    teacher_ckpt=teacher_ckpt)
model.bbox_head.update(
    type=RotatedRTDETRLDHead,
    ld_mode='distribution',
    # T as Rotated-LD. Its loss weights (15, for a single RetinaNet head)
    # were too strong here (distilled per decoder layer over many queries):
    # lower AP75 than without distillation in our experiments; 3 helped.
    loss_ld=dict(type=KnowledgeDistillationKLDivLoss, loss_weight=3, T=5),
    loss_kd=dict(type=KnowledgeDistillationKLDivLoss, loss_weight=3, T=5),
    vlr_iou_thr=0.4,
    matching_ld_weight='score',
    distill_all_layers=True)
