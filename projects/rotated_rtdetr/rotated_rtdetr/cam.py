"""Class Activation Maps (Grad-CAM, EigenCAM, AblationCAM, ...) for rotated
RT-DETR, built on `pytorch-grad-cam <https://github.com/jacobgil/pytorch-grad-cam>`_.

Follows the pytorch-grad-cam tutorial "Class Activation Maps for Object
Detection with Faster RCNN", adapted to this detector:

- ``RotatedRTDETRCAMModel`` wraps the detector so that it maps an input tensor
  to differentiable per-image outputs (``bboxes`` / ``labels`` / ``scores``),
  the same post-processing as ``RotatedRTDETRHead.predict`` but without
  ``no_grad`` and in-place ops.
- The multi-level outputs of the backbone / neck / encoder are a tuple, which
  pytorch-grad-cam cannot hook (it only records gradients of a single
  tensor). The wrapper therefore packs the levels into one tensor and routes
  it through ``cam_tap``, an identity module used as the CAM target layer.
  ``reshape_transform`` turns the packed tensor back into a
  (B, sum(C), H, W) map, like ``fasterrcnn_reshape_transform``.
- ``RotatedBoxScoreTarget`` is the rotated counterpart of
  ``FasterRCNNBoxScoreTarget`` (IoU + classification score per box).
- ``AblationLayerRotatedRTDETR`` is the counterpart of
  ``AblationLayerFasterRCNN``.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from mmdet.structures import DetDataSample
from pytorch_grad_cam.ablation_layer import AblationLayer

from .prob_iou import probiou

TARGET_LAYERS = ('backbone', 'neck', 'encoder')


class CAMTap(nn.Identity):
    """Identity layer on the packed multi-level features.

    Pass it as the CAM target layer: ``target_layers=[model.cam_tap]``.
    """


class RotatedRTDETRCAMModel(nn.Module):
    """Wrap a rotated RT-DETR detector for pytorch-grad-cam.

    Args:
        detector (nn.Module): A built ``RotatedRTDETR``.
        target_layer (str): Which multi-level features to explain:
            ``'encoder'`` (hybrid encoder output, i.e. the memory the decoder
            attends to), ``'neck'`` or ``'backbone'``. Defaults to
            ``'encoder'``.
        img_shape (tuple[int, int], optional): (h, w) of the valid image
            region inside the (padded) input, used to scale the normalized
            boxes like the detector does. Defaults to the input size.
    """

    def __init__(self,
                 detector: nn.Module,
                 target_layer: str = 'encoder',
                 img_shape: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        assert target_layer in TARGET_LAYERS, \
            f'target_layer must be one of {TARGET_LAYERS}'
        self.detector = detector
        self.cam_tap = CAMTap()
        self.img_shape = img_shape
        self._shapes: List[torch.Size] = []
        self._handle = getattr(detector, target_layer).register_forward_hook(
            self._tap_hook)

    def remove(self) -> None:
        """Remove the hook from the detector."""
        self._handle.remove()

    @property
    def level_shapes(self) -> List[torch.Size]:
        """(C, H, W) of every level seen in the last forward."""
        return self._shapes

    def _tap_hook(self, module, inputs, outputs):
        # Look `cam_tap` up at call time: AblationCAM swaps it for its
        # ablation layer.
        return self.unpack(self.cam_tap(self.pack(outputs)))

    def pack(self, feats: Sequence[Tensor]) -> Tensor:
        """Flatten the levels into one (B, N) tensor."""
        self._shapes = [f.shape[1:] for f in feats]
        packed = torch.cat([f.flatten(1) for f in feats], dim=1)
        if torch.is_grad_enabled() and not packed.requires_grad:
            # e.g. a fully frozen backbone: still record gradients here
            packed.requires_grad_()
        return packed

    def unpack(self, packed: Tensor) -> Tuple[Tensor]:
        """Inverse of :meth:`pack`."""
        sizes = [s.numel() for s in self._shapes]
        return tuple(
            p.reshape(packed.size(0), *s)
            for p, s in zip(packed.split(sizes, dim=1), self._shapes))

    def reshape_transform(self, packed: Tensor, abs: bool = False) -> Tensor:
        """Packed tensor -> (B, sum(C), H, W) at the finest level resolution.

        Same idea as ``fasterrcnn_reshape_transform``. Use ``abs=True`` (as in
        the tutorial) for gradient free methods such as EigenCAM and
        AblationCAM, since these activations are signed. Keep ``abs=False``
        for gradient based methods: the transform is applied to the
        gradients as well and their sign matters.
        """
        levels = self.unpack(packed)
        target_size = max((lvl.shape[-2:] for lvl in levels),
                          key=lambda s: s[0] * s[1])
        out = []
        for lvl in levels:
            if abs:
                lvl = lvl.abs()
            if lvl.shape[-2:] != target_size:
                lvl = F.interpolate(
                    lvl, size=tuple(target_size), mode='bilinear',
                    align_corners=False)
            out.append(lvl)
        return torch.cat(out, dim=1)

    def forward(self, inputs: Tensor) -> List[dict]:
        """Run the detector on normalized inputs of shape (B, 3, H, W).

        Returns:
            list[dict]: Per image ``bboxes`` (N, 5) as (cx, cy, w, h, rad) in
            input pixels, ``labels`` (N, ) and ``scores`` (N, ), the top
            ``max_per_img`` (query, class) pairs like ``predict``.
        """
        detector = self.detector
        head = detector.bbox_head
        h, w = inputs.shape[-2:]
        img_h, img_w = self.img_shape or (h, w)
        batch_data_samples = [
            DetDataSample(
                metainfo=dict(
                    img_shape=(img_h, img_w),
                    pad_shape=(h, w),
                    batch_input_shape=(h, w))) for _ in range(len(inputs))
        ]
        img_feats = detector.extract_feat(inputs)
        head_inputs_dict = detector.forward_transformer(
            img_feats, batch_data_samples)
        all_cls_scores, all_bbox_preds = head(
            hidden_states=head_inputs_dict['hidden_states'],
            references=head_inputs_dict['references'])
        cls_scores, bbox_preds = all_cls_scores[-1], all_bbox_preds[-1]

        num_classes = head.num_classes
        max_per_img = head.test_cfg.get('max_per_img', cls_scores.size(1))
        factor = bbox_preds.new_tensor(
            [img_w, img_h, img_w, img_h, head.angle_factor])
        outputs = []
        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            scores, indexes = cls_score.sigmoid().flatten().topk(
                min(max_per_img, cls_score.numel()))
            bboxes = bbox_pred[indexes // num_classes] * factor
            outputs.append(
                dict(
                    bboxes=bboxes,
                    labels=indexes % num_classes,
                    scores=scores))
        return outputs


class RotatedBoxScoreTarget:
    """Rotated version of ``FasterRCNNBoxScoreTarget``.

    For every reference detection, find the output box of the same class that
    overlaps it the most. If the overlap is above ``iou_threshold``, add
    ``overlap + score`` to the target. The overlap is the probabilistic IoU
    (``probiou``), which is differentiable, so gradient based CAMs also see
    how the box regression depends on the features.

    Args:
        labels (Sequence[int]): Labels of the reference detections.
        bounding_boxes (array-like): (N, 5) reference rotated boxes
            (cx, cy, w, h, rad) in input pixels.
        iou_threshold (float): Minimum overlap. Defaults to 0.5.
    """

    def __init__(self, labels, bounding_boxes,
                 iou_threshold: float = 0.5) -> None:
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs: dict) -> Tensor:
        boxes = model_outputs['bboxes']
        labels = model_outputs['labels']
        output = boxes.new_zeros(1)
        if len(boxes) == 0:
            return output
        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.as_tensor(
                box, dtype=boxes.dtype, device=boxes.device).view(1, 5)
            ious = probiou(box.expand_as(boxes), boxes)[:, 0]
            ious = torch.where(labels == int(label), ious,
                               torch.full_like(ious, -1))
            index = ious.argmax()
            if ious[index] > self.iou_threshold:
                output = output + ious[index] + model_outputs['scores'][index]
        return output


class AblationLayerRotatedRTDETR(AblationLayer):
    """Ablation layer for ``AblationCAM`` on ``RotatedRTDETRCAMModel``.

    Like ``AblationLayerFasterRCNN``, it maps the index of a channel in the
    reshaped (B, sum(C), H, W) activations back to its pyramid level and
    zeroes that channel in the packed features.

    Args:
        cam_model (RotatedRTDETRCAMModel): The wrapped model.
        ablation_value (float): Value written into ablated channels.
            Defaults to 0.
    """

    def __init__(self, cam_model: RotatedRTDETRCAMModel,
                 ablation_value: float = 0.) -> None:
        super().__init__()
        # A plain function, not a submodule: this layer is inserted into
        # `cam_model` itself.
        self._level_shapes = lambda: cam_model.level_shapes
        self.ablation_value = ablation_value

    def set_next_batch(self, input_batch_index, activations,
                       num_channels_to_ablate):
        """Repeat the packed activations of one image for every ablation."""
        self.activations = activations[input_batch_index].clone().unsqueeze(
            0).repeat(num_channels_to_ablate, 1)

    def __call__(self, x):
        output = self.activations
        shapes = self._level_shapes()
        channel_ends = np.cumsum([s[0] for s in shapes])
        level_starts = np.cumsum([0] + [s.numel() for s in shapes])
        for i in range(output.size(0)):
            channel = int(self.indices[i])
            level = int(np.searchsorted(channel_ends, channel, side='right'))
            c = channel - (channel_ends[level - 1] if level > 0 else 0)
            hw = shapes[level][1] * shapes[level][2]
            start = int(level_starts[level] + c * hw)
            output[i, start:start + hw] = self.ablation_value
        return output
