import torch as th
import piq


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = th.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        return 20 * th.log10(255.0 / th.sqrt(mse))


class PiqDistWrapper:
    def __init__(self, func, range01=False) -> None:
        super().__init__()
        self.func = func
        self.range01 = range01

    def __call__(self, img1, img2):
        img1 = img1 * 0.5 + 0.5
        img2 = img2 * 0.5 + 0.5
        if self.range01:
            return 1. - self.func(img1, img2)
        else:
            return - self.func(img1, img2)


_METRICS = None


def global_metrics():  # singleton
    global _METRICS
    if _METRICS is None:
        _METRICS = {
            'psnr': PSNR(),
            'lpips': PiqDistWrapper(
                piq.LPIPS(reduction='none').cuda(), True),
            'dists': PiqDistWrapper(
                piq.DISTS(reduction='none').cuda(), True),
        }
    return _METRICS


class CombinedMetrics:
    """combine several metrics for early-stop selection.
    distance will be calculated for early metrics in `names`.
    `threshs[i] = -1` will disable selection for metric `i`.
    """

    def __init__(self, names, threshs):
        _metrics = global_metrics()
        self.funcs = {name: _metrics[name] for name in names}
        self.threshs = threshs
        assert len(threshs) == len(names)

    def __call__(self, img1, img2):
        '''sel from value < thresh on any metrics'''
        sel = th.zeros(len(img1), dtype=th.bool, device=img1.device)
        all_v_dict = {}
        for (func_n, func), thresh in zip(self.funcs.items(), self.threshs):
            _v = func(img1, img2)  # tensor
            if thresh != -1:
                sel = th.logical_or(_v < thresh, sel)
            all_v_dict[func_n] = _v.tolist()
        return sel, all_v_dict


def get_metrics(name, metric_thresh):
    if name is None:
        name = 'psnr'  # default
    if not isinstance(name, list):
        names, metric_threshs = [name], [metric_thresh]  # old config
    else:
        names, metric_threshs = name, metric_thresh
    return [n.upper() for n in names], CombinedMetrics(names, metric_threshs)
