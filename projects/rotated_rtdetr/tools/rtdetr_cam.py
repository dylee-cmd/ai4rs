"""Visualize class activation maps (Grad-CAM, EigenCAM, AblationCAM, ...) of
rotated RT-DETR detections.

Example:
    python projects/rotated_rtdetr/tools/rtdetr_cam.py \
        demo/dota_demo.jpg \
        projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py \
        epoch_72.pth --method gradcam --out-file cam.jpg

Requires ``pip install grad-cam``.
"""
import argparse
import os
import os.path as osp
import sys
from functools import partial

import cv2
import numpy as np
import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mmcv.transforms import Compose  # noqa: E402
from mmdet.apis import init_detector  # noqa: E402
from mmengine.config import DictAction  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from pytorch_grad_cam import (AblationCAM, EigenCAM, EigenGradCAM,  # noqa
                              GradCAM, GradCAMPlusPlus, HiResCAM, LayerCAM,
                              XGradCAM)
from pytorch_grad_cam.utils.image import (scale_cam_image,  # noqa: E402
                                          show_cam_on_image)

from mmrotate.registry import DATASETS  # noqa: E402
from mmrotate.structures.bbox import RotatedBoxes  # noqa: E402
from mmrotate.utils import register_all_modules  # noqa: E402
from projects.rotated_rtdetr.rotated_rtdetr.cam import (  # noqa: E402
    TARGET_LAYERS, AblationLayerRotatedRTDETR, RotatedBoxScoreTarget,
    RotatedRTDETRCAMModel)

METHODS = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'hirescam': HiResCAM,
    'layercam': LayerCAM,
    'eigengradcam': EigenGradCAM,
    'eigencam': EigenCAM,
    'ablationcam': AblationCAM,
}
GRADIENT_FREE = ('eigencam', 'ablationcam')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Class activation maps for rotated RT-DETR')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', nargs='?', help='Checkpoint file')
    parser.add_argument(
        '--method', default='gradcam', choices=list(METHODS),
        help='CAM method')
    parser.add_argument(
        '--target-layer', default='encoder', choices=TARGET_LAYERS,
        help='Multi-level features to explain')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Detection threshold')
    parser.add_argument(
        '--max-dets', type=int, default=50,
        help='Explain at most this many detections (highest scores)')
    parser.add_argument(
        '--iou-thr', type=float, default=0.5,
        help='Overlap threshold of the box score target')
    parser.add_argument(
        '--renormalize', action='store_true',
        help='Re-normalize the CAM inside every box, zero outside')
    parser.add_argument(
        '--ratio-channels-to-ablate', type=float, default=1.0,
        help='AblationCAM: fraction of channels to ablate (e.g. 0.1)')
    parser.add_argument(
        '--ablation-batch-size', type=int, default=8,
        help='AblationCAM: forward batch size')
    parser.add_argument(
        '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out-file', default='cam.jpg', help='Output image')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override settings in the config, e.g. '
        'test_dataloader.dataset.pipeline.1.scale="(512,512)"')
    return parser.parse_args()


def load_image(model, img_path):
    """Run the test pipeline and data preprocessor of the config.

    Returns the normalized input tensor, the valid (h, w) region inside it and
    the resized RGB image in [0, 1] for visualization.
    """
    cfg = model.cfg
    pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
    data = pipeline(dict(img_path=img_path, img_id=0))
    data = model.data_preprocessor(pseudo_collate([data]), False)
    inputs = data['inputs']
    img_h, img_w = data['data_samples'][0].img_shape
    # undo the normalization for display
    mean = model.data_preprocessor.mean
    std = model.data_preprocessor.std
    img = (inputs[0] * std + mean)[:, :img_h, :img_w]
    img = img.permute(1, 2, 0).cpu().numpy()
    if not model.data_preprocessor._channel_conversion:
        img = img[..., ::-1]  # BGR -> RGB
    img = np.ascontiguousarray(np.clip(img / 255., 0, 1), dtype=np.float32)
    return inputs, (img_h, img_w), img


def get_classes(model):
    """Class names from the checkpoint, else from the config's dataset."""
    num_classes = model.bbox_head.num_classes
    classes = (model.dataset_meta or {}).get('classes')
    if classes is None or len(classes) != num_classes:
        dataset_type = model.cfg.test_dataloader.dataset.type
        dataset_cls = DATASETS.get(dataset_type) \
            if isinstance(dataset_type, str) else dataset_type
        classes = getattr(dataset_cls, 'METAINFO', {}).get('classes')
    if classes is None or len(classes) != num_classes:
        classes = [str(i) for i in range(num_classes)]
    return list(classes)


def box_polygons(boxes):
    return RotatedBoxes(torch.as_tensor(boxes)).convert_to('qbox').tensor \
        .reshape(-1, 4, 2).cpu().numpy().astype(np.int32)


def draw_boxes(image, boxes, labels, classes, colors):
    for poly, label in zip(box_polygons(boxes), labels):
        color = tuple(int(c) for c in colors[label])
        cv2.polylines(image, [poly], True, color, 2)
        x, y = poly.min(0)
        cv2.putText(image, classes[label], (int(x), int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    lineType=cv2.LINE_AA)
    return image


def renormalize_cam_in_boxes(boxes, grayscale_cam):
    """Normalize the CAM to [0, 1] inside every rotated box, 0 outside."""
    cams = []
    for poly in box_polygons(boxes):
        mask = np.zeros(grayscale_cam.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        mask = mask.astype(bool)
        cam = np.zeros_like(grayscale_cam)
        if mask.any():
            values = grayscale_cam[mask]
            values = values - values.min()
            cam[mask] = values / (values.max() + 1e-7)
        cams.append(cam)
    if not cams:
        return grayscale_cam
    return scale_cam_image(np.max(np.float32(cams), axis=0))


def build_cam(args, cam_model):
    reshape_transform = partial(
        cam_model.reshape_transform, abs=args.method in GRADIENT_FREE)
    kwargs = dict(
        model=cam_model,
        target_layers=[cam_model.cam_tap],
        reshape_transform=reshape_transform)
    if args.method == 'ablationcam':
        kwargs.update(
            ablation_layer=AblationLayerRotatedRTDETR(cam_model),
            batch_size=args.ablation_batch_size,
            ratio_channels_to_ablate=args.ratio_channels_to_ablate)
    return METHODS[args.method](**kwargs)


def main():
    args = parse_args()
    register_all_modules()
    model = init_detector(
        args.config, args.checkpoint, device=args.device,
        cfg_options=args.cfg_options)
    classes = get_classes(model)
    colors = np.random.RandomState(0).randint(
        0, 255, size=(len(classes), 3))

    inputs, img_shape, image = load_image(model, args.img)
    cam_model = RotatedRTDETRCAMModel(
        model, target_layer=args.target_layer, img_shape=img_shape)
    cam_model.eval()

    # 1) detections to explain
    with torch.no_grad():
        outputs = cam_model(inputs)[0]
    keep = outputs['scores'] >= args.score_thr
    boxes = outputs['bboxes'][keep][:args.max_dets].cpu().numpy()
    labels = outputs['labels'][keep][:args.max_dets].cpu().numpy()
    scores = outputs['scores'][keep][:args.max_dets].cpu().numpy()
    print(f'{len(boxes)} detections with score >= {args.score_thr}')
    for box, label, score in zip(boxes, labels, scores):
        print(f'  {classes[label]:20s} {score:.3f} '
              f'{np.array2string(box, precision=1)}')
    if len(boxes) == 0:
        print('Nothing to explain, lower --score-thr.')
        return

    # 2) CAM for the target "these boxes, with these classes"
    targets = [
        RotatedBoxScoreTarget(
            labels=labels, bounding_boxes=boxes, iou_threshold=args.iou_thr)
    ]
    cam = build_cam(args, cam_model)
    grayscale_cam = cam(inputs, targets=targets)[0]
    grayscale_cam = grayscale_cam[:img_shape[0], :img_shape[1]]
    if args.renormalize:
        grayscale_cam = renormalize_cam_in_boxes(boxes, grayscale_cam)

    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    cam_image = draw_boxes(cam_image, boxes, labels, classes, colors)
    out_dir = osp.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.out_file, cam_image[..., ::-1])
    print(f'Saved {args.method} ({args.target_layer}) to {args.out_file}')
    cam_model.remove()


if __name__ == '__main__':
    main()
