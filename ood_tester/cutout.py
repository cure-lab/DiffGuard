import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., out_size=None,
                 local_rand_gen=None):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.out_size = None if cut_size == out_size else out_size
        self.local_rand_gen = local_rand_gen

    def __repr__(self):
        return (
            f"[Cutouts Module] with cut_size={self.cut_size}, cutn={self.cutn}, "
            + f"cut_pow={self.cut_pow}, out_size={self.out_size}"
        )

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)  # 256
        min_size = min(sideX, sideY, self.cut_size)  # 224
        cutouts = []
        for _ in range(self.cutn):
            size = int(th.rand(
                [], generator=self.local_rand_gen.rand_gen
            )**self.cut_pow * (max_size - min_size) + min_size)  # 224-256
            offsetx = th.randint(0, sideX - size + 1, (),
                                 generator=self.local_rand_gen.rand_gen)
            offsety = th.randint(0, sideY - size + 1, (),
                                 generator=self.local_rand_gen.rand_gen)
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.interpolate(cutout, self.cut_size))
        cutouts = th.cat(cutouts)
        if self.out_size is not None:
            cutouts = F.interpolate(cutouts, self.out_size)
        return cutouts
