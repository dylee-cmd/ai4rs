# O2-RTDETR R50vd with a PSF floor in the KLD loss and the KLD matching cost
# (for small targets). Both boxes become Gaussians with Sigma + s^2 I, i.e.
# w, h -> sqrt(w^2 + 4 s^2), sqrt(h^2 + 4 s^2). psf_std = 0 is the original
# o2_rtdetr_r50vd_2xb4_72e_dota.py.
#
# psf_std is in pixels of the network input (after Resize; the loss and the
# matching cost both get unnormalized boxes). Sweep e.g. {0, 0.5, 1, 2},
# starting around half of the resolution cell. The effect is a hypothesis to
# be tested. Keep the loss and the cost on the same value, e.g.
#   --cfg-options model.bbox_head.loss_iou.psf_std=2.0 \
#       model.train_cfg.assigner.match_costs.2.psf_std=2.0
from mmengine.config import read_base
from mmrotate.models.losses import PSFGDLoss
from projects.rotated_dino.rotated_dino.match_cost import PSFGDCost

with read_base():
    from .o2_rtdetr_r50vd_2xb4_72e_dota import *

psf_std = 1.0

model.bbox_head.loss_iou = dict(
    type=PSFGDLoss,
    psf_std=psf_std,
    loss_type='kld',
    fun='log1p',
    tau=1,
    sqrt=False,
    loss_weight=2.0)
model.train_cfg.assigner.match_costs[2] = dict(
    type=PSFGDCost,
    psf_std=psf_std,
    loss_type='kld',
    fun='log1p',
    tau=1,
    sqrt=False,
    weight=2.0)
