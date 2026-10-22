from typing import Any, Optional
import torch
from torch import Tensor
from mmcv.cnn import Linear, build_activation_layer
from mmengine.model import BaseModule, ModuleList
from mmdet.utils import ConfigType


class MLP(BaseModule):
    """Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            act_cfg: ConfigType = dict(type='ReLU'),
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.

        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def cast_tensors_to_float(obj: Any) -> Any:
    """Recursively cast floating point tensors in ``obj`` to float32.

    Non-tensor objects (e.g. ``InstanceData``, ints, dicts of ints) are
    returned unchanged, so this is safe to apply to the inputs of
    ``loss_by_feat``.
    """
    if isinstance(obj, Tensor):
        return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(cast_tensors_to_float(o) for o in obj)
    if isinstance(obj, dict):
        return {k: cast_tensors_to_float(v) for k, v in obj.items()}
    return obj


def find_device_type(obj: Any) -> Optional[str]:
    """Return the device type of the first tensor found in ``obj``."""
    if isinstance(obj, Tensor):
        return obj.device.type
    if isinstance(obj, (list, tuple)):
        obj = list(obj)
    elif isinstance(obj, dict):
        obj = list(obj.values())
    else:
        obj = []
    for o in obj:
        device_type = find_device_type(o)
        if device_type is not None:
            return device_type
    return None


def disable_autocast(device_type: str):
    """Context manager that disables autocast on ``device_type``."""
    return torch.autocast(device_type=device_type, enabled=False)
