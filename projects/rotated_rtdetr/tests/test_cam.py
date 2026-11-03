"""Tests for projects/rotated_rtdetr/rotated_rtdetr/cam.py.

Run from the repo root:
    python -m pytest projects/rotated_rtdetr/tests/test_cam.py
"""
import math

import numpy as np
import pytest
import torch

pytest.importorskip('pytorch_grad_cam')

from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from pytorch_grad_cam import (AblationCAM, EigenCAM, GradCAM,  # noqa: E402
                              GradCAMPlusPlus, LayerCAM, XGradCAM)

from mmrotate.registry import MODELS  # noqa: E402
from projects.rotated_rtdetr.rotated_rtdetr.cam import (  # noqa: E402
    AblationLayerRotatedRTDETR, RotatedBoxScoreTarget, RotatedRTDETRCAMModel)

CFG = 'projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dota.py'
SIZE = 128


@pytest.fixture(scope='module')
def detector():
    torch.manual_seed(0)
    init_default_scope('mmrotate')
    cfg = Config.fromfile(CFG)
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    model.init_weights()
    return model.eval()


@pytest.fixture
def inputs():
    torch.manual_seed(1)
    return torch.randn(1, 3, SIZE, SIZE)


@pytest.fixture
def cam_model(detector):
    model = RotatedRTDETRCAMModel(detector, target_layer='encoder')
    yield model.eval()
    model.remove()


def top_detections(cam_model, inputs, k=3):
    with torch.no_grad():
        out = cam_model(inputs)[0]
    return out['bboxes'][:k].numpy(), out['labels'][:k].numpy()


def test_hook_does_not_change_outputs(detector, inputs):
    with torch.no_grad():
        ref = RotatedRTDETRCAMModel(detector, 'encoder')
        ref.remove()  # no tap
        out_ref = ref(inputs)[0]
        for layer in ('backbone', 'neck', 'encoder'):
            m = RotatedRTDETRCAMModel(detector, layer)
            out = m(inputs)[0]
            m.remove()
            torch.testing.assert_close(out['bboxes'], out_ref['bboxes'])
            torch.testing.assert_close(out['scores'], out_ref['scores'])
            assert torch.equal(out['labels'], out_ref['labels'])


def test_outputs_match_predict(detector, inputs):
    """Wrapper post-processing == RotatedRTDETRHead.predict."""
    from mmdet.structures import DetDataSample
    sample = DetDataSample(metainfo=dict(
        img_shape=(SIZE, SIZE), pad_shape=(SIZE, SIZE),
        batch_input_shape=(SIZE, SIZE), scale_factor=(1., 1.)))
    with torch.no_grad():
        pred = detector.predict(inputs, [sample], rescale=False)[0]
        m = RotatedRTDETRCAMModel(detector, 'encoder')
        out = m(inputs)[0]
        m.remove()
    inst = pred.pred_instances
    torch.testing.assert_close(out['scores'], inst.scores)
    assert torch.equal(out['labels'], inst.labels)
    # predict() clamps box coordinates to the image, the wrapper does not
    bboxes = out['bboxes'].clone()
    bboxes[:, :4] = bboxes[:, :4].clamp(0, SIZE)
    torch.testing.assert_close(bboxes, inst.bboxes)


def test_pack_unpack_and_reshape(cam_model, inputs):
    with torch.no_grad():
        cam_model(inputs)
    shapes = cam_model.level_shapes
    assert len(shapes) == 3
    feats = tuple(torch.randn(2, *s) for s in shapes)
    packed = cam_model.pack(feats)
    for a, b in zip(cam_model.unpack(packed), feats):
        assert torch.equal(a, b)
    reshaped = cam_model.reshape_transform(packed)
    finest = max(shapes, key=lambda s: s[1] * s[2])
    assert reshaped.shape == (2, sum(s[0] for s in shapes), *finest[1:])
    assert (cam_model.reshape_transform(packed, abs=True) >= 0).all()


def test_target(cam_model, inputs):
    boxes, labels = top_detections(cam_model, inputs, k=2)
    out = cam_model(inputs)[0]
    score = RotatedBoxScoreTarget(labels, boxes)(out)
    # each box matches itself: overlap 1 + its score
    expected = 2 + out['scores'][:2].sum()
    torch.testing.assert_close(score, expected.view(1), atol=2e-3, rtol=0)
    # no output of that class -> no match
    missing = RotatedBoxScoreTarget([15], boxes[:1])(out)
    assert float(missing) == 0
    # far away box -> no match
    far = boxes[:1].copy()
    far[0, :2] = -1000
    assert float(RotatedBoxScoreTarget(labels[:1], far)(out)) == 0


def test_gradcam_matches_manual_autograd(cam_model, inputs):
    boxes, labels = top_detections(cam_model, inputs)
    target = RotatedBoxScoreTarget(labels, boxes)
    cam = GradCAM(cam_model, [cam_model.cam_tap],
                  reshape_transform=cam_model.reshape_transform)
    grayscale = cam(inputs, targets=[target])
    assert grayscale.shape == (1, SIZE, SIZE)
    assert np.isfinite(grayscale).all()
    assert grayscale.min() >= 0 and grayscale.max() <= 1
    assert grayscale.max() > 0, 'empty CAM: no gradient reached the tap'

    # recompute Grad-CAM by hand from autograd on the encoder outputs
    feats = {}

    def hook(module, inp, out):
        feats['out'] = out
        for o in out:
            o.retain_grad()

    cam_model.remove()
    handle = cam_model.detector.encoder.register_forward_hook(hook)
    try:
        plain = RotatedRTDETRCAMModel(cam_model.detector, 'encoder')
        plain.remove()
        target(plain(inputs)[0]).sum().backward()
    finally:
        handle.remove()
    with torch.no_grad():
        acts = cam_model.reshape_transform(
            cam_model.pack([o.detach() for o in feats['out']]))
        grads = cam_model.reshape_transform(
            cam_model.pack([o.grad for o in feats['out']]))
    torch.testing.assert_close(
        torch.as_tensor(cam.activations_and_grads.gradients[0]), grads,
        atol=1e-6, rtol=1e-4)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    manual = torch.relu((weights * acts).sum(1))[0].numpy()
    from pytorch_grad_cam.utils.image import scale_cam_image
    # per layer: normalize + resize, then normalize the aggregate again
    manual = scale_cam_image(scale_cam_image(manual[None], (SIZE, SIZE)))[0]
    np.testing.assert_allclose(grayscale[0], manual, atol=1e-4)


@pytest.mark.parametrize('method', [
    GradCAMPlusPlus, XGradCAM, LayerCAM, EigenCAM])
def test_other_methods(cam_model, inputs, method):
    boxes, labels = top_detections(cam_model, inputs)
    cam = method(cam_model, [cam_model.cam_tap],
                 reshape_transform=lambda x: cam_model.reshape_transform(
                     x, abs=method is EigenCAM))
    grayscale = cam(inputs, targets=[RotatedBoxScoreTarget(labels, boxes)])
    assert grayscale.shape == (1, SIZE, SIZE)
    assert np.isfinite(grayscale).all()
    if method is not EigenCAM:
        # random weights: the CAM itself may be all zero after ReLU
        assert cam.activations_and_grads.gradients[0].abs().sum() > 0


@pytest.mark.parametrize('layer', ['backbone', 'neck'])
def test_other_target_layers(detector, inputs, layer):
    cam_model = RotatedRTDETRCAMModel(detector, layer).eval()
    boxes, labels = top_detections(cam_model, inputs)
    cam = GradCAM(cam_model, [cam_model.cam_tap],
                  reshape_transform=cam_model.reshape_transform)
    grayscale = cam(inputs, targets=[RotatedBoxScoreTarget(labels, boxes)])
    cam_model.remove()
    assert np.isfinite(grayscale).all()
    assert cam.activations_and_grads.gradients[0].abs().sum() > 0


def test_ablation_layer_zeroes_one_channel(cam_model, inputs):
    with torch.no_grad():
        cam_model(inputs)
    shapes = cam_model.level_shapes
    feats = tuple(torch.rand(1, *s) + 1 for s in shapes)
    packed = cam_model.pack(feats)
    layer = AblationLayerRotatedRTDETR(cam_model)
    c0 = shapes[0][0]
    # last channel of level 0, first and 5th channel of level 1, last overall
    channels = [c0 - 1, c0, c0 + 4, sum(s[0] for s in shapes) - 1]
    expected = [(0, c0 - 1), (1, 0), (1, 4), (len(shapes) - 1,
                                              shapes[-1][0] - 1)]
    layer.indices = np.int32(channels)
    layer.set_next_batch(0, packed, len(channels))
    out = cam_model.unpack(layer(None))
    for i, (lvl, ch) in enumerate(expected):
        for li, level in enumerate(out):
            zero = (level[i] == 0).flatten(1).all(dim=1)
            want = torch.zeros_like(zero)
            if li == lvl:
                want[ch] = True
            assert torch.equal(zero, want), (i, li)


def test_ablation_cam(cam_model, inputs):
    boxes, labels = top_detections(cam_model, inputs)
    cam = AblationCAM(
        cam_model, [cam_model.cam_tap],
        reshape_transform=lambda x: cam_model.reshape_transform(x, abs=True),
        ablation_layer=AblationLayerRotatedRTDETR(cam_model),
        batch_size=8, ratio_channels_to_ablate=0.02)
    grayscale = cam(inputs, targets=[RotatedBoxScoreTarget(labels, boxes)])
    assert grayscale.shape == (1, SIZE, SIZE)
    assert np.isfinite(grayscale).all()
    # the tap is restored after ablation
    assert type(cam_model.cam_tap).__name__ == 'CAMTap'
    assert math.isfinite(float(grayscale.sum()))
