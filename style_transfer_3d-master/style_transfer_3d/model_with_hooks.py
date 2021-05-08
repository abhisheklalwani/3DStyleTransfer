import torch
import torch.nn as nn
import torchvision.models as models
from typing import Type, Any, Callable, Union, List, Optional, cast
from torch import Tensor
from collections import OrderedDict

from typing import Dict, Iterable, Callable

class FeatureExtractor(nn.Module):
    def __init__(self, layers: Iterable[str]):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])['features'][layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        with torch.autograd.no_grad():
            _ = self.model(x)
            return self._features