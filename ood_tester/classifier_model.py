from typing import Tuple
from collections import OrderedDict
import logging

import torch as th
import torch.nn as nn
from torch import Tensor
import torchvision

from network.resnet_tv import resnet50


class ImageNormalizer(nn.Module):
    """Perform center crop and normalization.
    """

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float], crop=None) -> None:
        super(ImageNormalizer, self).__init__()

        if crop is None:
            self.crop = torchvision.transforms.Compose([])
        else:
            self.crop = torchvision.transforms.CenterCrop(crop)
        self.register_buffer('mean', th.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', th.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        input = self.crop(input)
        return (input - self.mean) / self.std


def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
                    std: Tuple[float, float, float], crop=None) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std, crop)),
        ('model', model)
    ])
    return nn.Sequential(layers)


def load_classifier(model_name, NormWrapper=True, mu=None, sigma=None,
                    state_file=None, **kwargs):
    if model_name == "resnet50":
        classifier = resnet50(pretrained=True)
    else:
        raise NotImplementedError(f"Unknown model name {model_name}.")
    if NormWrapper:
        classifier = normalize_model(classifier, mean=mu, std=sigma)
    return classifier
