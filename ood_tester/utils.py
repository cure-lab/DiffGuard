import logging
import torch
import torchvision.transforms as tvs_trans

normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
}

interpolation_modes = {
    'nearest': tvs_trans.InterpolationMode.NEAREST,
    'bilinear': tvs_trans.InterpolationMode.BILINEAR,
}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def minus_1_1_to_01(image):
    """convert range [-1, 1] to range [0, 1], and apply hard clamp.
    """
    if (image > 0).all() or (image < 0).all():
        logging.warn(f"Please check your input, in range "
                     f"[{image.max():.2f}, {image.min():.2f}]")
    return torch.clamp((image + 1) * 0.5, 0, 1.)
