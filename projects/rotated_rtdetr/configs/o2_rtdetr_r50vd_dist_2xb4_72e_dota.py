# Distribution-kind O2-RTDETR R50vd: every box coordinate is regressed as a
# general distribution (reg_max + 1 bins). Train it first and use it as the
# teacher of distribution localization distillation (Rotated-LD), see
# o2_rtdetr_r18vd_ld_r50vd_dist_2xb4_72e_dota.py.
from mmengine.config import read_base
with read_base():
    from .o2_rtdetr_r50vd_2xb4_72e_dota import *

model.bbox_head.update(reg_max=32, reg_range=8.0)
