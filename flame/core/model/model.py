import torch
from torch import nn
from typing import Optional, List, Tuple

from .darknet53 import YOLOv3


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 80,
        weight_path: Optional[str] = None,
    ) -> None:
        super(Model, self).__init__()
        self.model = YOLOv3(
            in_channels=in_channels,
            num_classes=num_classes
        )

        if weight_path is not None:
            state_dict = torch.load(f=weight_path, map_location='cpu')['state_dict']

            state_dict.pop('layers.15.pred.1.conv.weight')
            state_dict.pop('layers.15.pred.1.conv.bias')
            state_dict.pop('layers.15.pred.1.bn.weight')
            state_dict.pop('layers.15.pred.1.bn.bias')
            state_dict.pop('layers.15.pred.1.bn.running_mean')
            state_dict.pop('layers.15.pred.1.bn.running_var')

            state_dict.pop('layers.22.pred.1.conv.weight')
            state_dict.pop('layers.22.pred.1.conv.bias')
            state_dict.pop('layers.22.pred.1.bn.weight')
            state_dict.pop('layers.22.pred.1.bn.bias')
            state_dict.pop('layers.22.pred.1.bn.running_mean')
            state_dict.pop('layers.22.pred.1.bn.running_var')

            state_dict.pop('layers.29.pred.1.conv.weight')
            state_dict.pop('layers.29.pred.1.conv.bias')
            state_dict.pop('layers.29.pred.1.bn.weight')
            state_dict.pop('layers.29.pred.1.bn.bias')
            state_dict.pop('layers.29.pred.1.bn.running_mean')
            state_dict.pop('layers.29.pred.1.bn.running_var')

            self.model.load_state_dict(state_dict, strict=False)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, inputs):
        return self.model(inputs)
