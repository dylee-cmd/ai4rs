# Distribution-kind O2-RTDETR R18vd (baseline of the distilled student in
# o2_rtdetr_r18vd_ld_r50vd_dist_2xb4_72e_dota.py).
from mmengine.config import read_base
with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_dota import *

model.bbox_head.update(reg_max=32, reg_range=8.0)
