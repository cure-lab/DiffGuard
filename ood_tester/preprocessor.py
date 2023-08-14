import torchvision.transforms as tvs_trans
from omegaconf import DictConfig

from .utils import Convert, interpolation_modes, normalization_dict


class BasePreprocessor():
    """For train dataset standard transformation."""

    def __init__(self, config: DictConfig):
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        self.interpolation = interpolation_modes[config.dataset.interpolation]
        normalization_type = getattr(config.dataset, "normalization_type", None)
        if normalization_type in normalization_dict.keys():
            self.mean = normalization_dict[normalization_type][0]
            self.std = normalization_dict[normalization_type][1]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(self.pre_size, interpolation=self.interpolation),
            tvs_trans.CenterCrop(self.image_size),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.RandomCrop(self.image_size, padding=4),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std),
        ])

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)

    def __repr__(self) -> str:
        return f"BasePreprocessor with {self.transform}"


class TestStandardPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""

    def __init__(self, config: DictConfig):
        super(TestStandardPreProcessor, self).__init__(config)
        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(self.pre_size, interpolation=self.interpolation),
            tvs_trans.CenterCrop(self.image_size),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std),
        ])

    def __repr__(self) -> str:
        return f"TestStandardPreProcessor with {self.transform}"
