import logging
from functools import partial

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from omegaconf import OmegaConf

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict_with_defaults,
)

from ood_tester.cutout import MakeCutouts


class GuidedDiffusionModel:
    def __init__(self, config, local_rand_gen, **kwargs):
        self.config = config
        self.local_rand_gen = local_rand_gen
        # diffusion model
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict_with_defaults(
                self.config, model_and_diffusion_defaults())
        )
        self.model.load_state_dict(dist_util.load_state_dict(
            self.config.model_path, map_location='cpu'))
        if self.config.use_fp16:
            self.model.convert_to_fp16()
        self.model.eval()

        # other data
        self.metric_crop = -1
        if self.config.image_size != self.config.input_size:
            if self.config.process_method == "resize":
                self.preprocess = transforms.Resize(self.config.image_size)
                self.postprocess = transforms.Resize(self.config.input_size)
            elif self.config.process_method == "pad":
                self.preprocess = transforms.Pad(
                    (self.config.image_size - self.config.input_size) // 2,
                    fill=-1)
                self.postprocess = transforms.CenterCrop(self.config.input_size)
                self.metric_crop = self.config.input_size
            logging.info(f"GuidedDiffusion apply preprocess: {self.preprocess}")
            logging.info(
                f"GuidedDiffusion apply postprocess: {self.postprocess}")
        else:
            self.preprocess = lambda x: x
            self.postprocess = lambda x: x
        # check early_stop settings
        self.early_stop_metric = None
        self.early_stop_thresh = -1
        if hasattr(self.config, "early_stop_metric"):
            assert hasattr(self.config, "early_stop_thresh")
            if isinstance(self.config.early_stop_metric, (str, type(None))):
                # string or None, can set directly
                self.early_stop_metric = self.config.early_stop_metric
                self.early_stop_thresh = self.config.early_stop_thresh
            else:
                # convert to lists
                self.early_stop_metric = OmegaConf.to_object(
                    self.config.early_stop_metric)
                self.early_stop_thresh = OmegaConf.to_object(
                    self.config.early_stop_thresh)
        # if any ele != -1, change bs = 1
        if self.early_stop_thresh != -1 and any(
                ele != -1 for ele in self.early_stop_thresh):
            self.config.batch_size = 1
        self.batch_size = self.config.batch_size
        logging.info(f"GuidedDiffusion forward bs={self.batch_size}")

        # param for cond_fn
        self.no_x0_hat = getattr(self.config, "no_x0_hat", False)
        if self.config.cutouts_num > 0:
            self.make_cutouts = MakeCutouts(
                self.config.cutouts_size, self.config.cutouts_num,
                out_size=self.config.input_size,
                local_rand_gen=self.local_rand_gen)
        else:
            self.make_cutouts = lambda x: x
        logging.info(
            f"cond_fn setting: no_x0_hat={self.no_x0_hat}; "
            f"cutout={self.make_cutouts}"
        )

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, x0, target_label, classifier, uncond_aux=False):
        # empty cache is used to expose real memory usage
        # th.cuda.empty_cache()
        xT, rev_t = self.diffuse(x0)
        # th.cuda.empty_cache()
        all_recons = []
        xT_splits = xT.split(self.batch_size)
        label_splits = target_label.split(self.batch_size)
        rev_t_splits = rev_t.split(self.batch_size)
        # full GPU usage, just do it twice; otherwise, double on batch
        uncond_twice = (self.batch_size != 1 and uncond_aux)
        uncond_batch = uncond_aux and not uncond_twice
        logging.debug(f"mode: twice={uncond_twice}, batch={uncond_batch}")
        for xTi, labeli, rev_ti in zip(xT_splits, label_splits, rev_t_splits):
            self.local_rand_gen.reset()
            recon = self.denoise(xTi, labeli, rev_ti, classifier,
                                 uncond_aux=uncond_batch)
            if uncond_twice:
                recon_uncond = self.denoise(xTi, None, rev_ti, None)
                recon = th.stack([recon, recon_uncond], dim=1)
            all_recons.append(recon)
        # th.cuda.empty_cache()
        return th.cat(all_recons, dim=0)

    def diffuse(self, x0):
        # assume x0 in [-1, 1]
        model_kwargs = {}
        x0 = self.preprocess(x0)
        assert x0.shape[-1] == self.config.image_size
        # it's important to convert ListConfig to list with to_object
        ddim_reverse_latent, rev_t = self.diffusion.ddim_reverse_ode_loop(
            self.model_fn, x0, clip_denoised=self.config.clip_denoised,
            model_kwargs=model_kwargs, cond_fn=None, device=dist_util.dev(),
            early_stop_steps=self.config.early_stop_steps,
            early_stop_metric=self.early_stop_metric,
            early_stop_thresh=self.early_stop_thresh,
            crop_size=self.metric_crop, aux=True)
        xT = ddim_reverse_latent
        return xT, rev_t

    def denoise(self, xT, target_label, rev_t, classifier, uncond_aux=False):
        model_kwargs = {}
        if uncond_aux:
            assert xT.shape[0] == 1, "wrong usage, assume bs=1, please check!"
            xT = xT.repeat(2, 1, 1, 1)
            # label=-1 means uncond
            target_label = th.cat(
                [target_label, th.ones_like(target_label) * -1], dim=0)
        model_kwargs["y"] = target_label
        # `skip_first_steps` is the offset from total steps
        if self.config.forback_skip_diff > 0:
            # typically, we use denoising steps more than inversion steps.
            # len=1 or all with the same value, since `skip_first_steps` can
            # only be set by int.
            assert len(th.unique(rev_t)) == 1
            _skip_first_steps = max(
                0, self.diffusion.num_timesteps - int(rev_t[0]) - self.config.forback_skip_diff)
        else:
            _skip_first_steps = self.config.skip_first_steps
        reconstruction = self.diffusion.ddim_sample_loop(
            self.model_fn,
            noise=xT,
            clip_denoised=self.config.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=partial(self.cond_fn, classifier=classifier),
            device=dist_util.dev(),
            skip_first_steps=_skip_first_steps,
        )
        reconstruction = self.postprocess(reconstruction)
        # reconstruction in [-1, 1]
        if uncond_aux:
            # uncond should stack at dim=1
            reconstruction = reconstruction.reshape(
                1, 2, *reconstruction.shape[1:])
        return reconstruction

    def model_fn(self, x, t, y=None):
        return self.model(x, t, y if self.config.class_cond else None)

    def cond_fn(self, x, t, y=None, classifier=None):
        # assert y is not None
        if y is None:
            return th.zeros_like(x)
        assert classifier is not None
        results_holder = th.zeros_like(x)
        place_indicator = th.BoolTensor([yi != -1 for yi in y])

        # start, only do on y != -1
        x_for_compute = x[place_indicator]
        y_for_compute = y[place_indicator]
        ce_func = nn.CrossEntropyLoss(reduction='none')
        with th.enable_grad():
            # NOTE: here we should not assume the data range of x. cuz x can be
            # pure gaussian with mean 0.
            x_in = x_for_compute.detach().requires_grad_(True)

            n = x_in.shape[0]
            cur_t = self.diffusion._unscale_timesteps(t)

            # use x0 hat
            if not self.no_x0_hat:
                # get parameters
                my_t = th.ones(
                    [n], device=dist_util.dev(), dtype=th.long
                ) * cur_t

                # recompute x0
                out = self.diffusion.p_mean_variance(
                    self.model_fn, x_in, my_t, clip_denoised=False,
                    model_kwargs={})

                # apply weight
                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x_in * (1 - fac)

            # cutout aug
            clip_in = self.make_cutouts(x_in.add(1).div(2))

            # classifier inference
            logits = classifier(clip_in)
            gt = y_for_compute.repeat(len(clip_in) // n)  # same form as clip_in

            ce_loss = ce_func(logits, gt)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), gt.view(-1)]
            final_loss = selected.sum()

            for i in range(x_in.shape[0]):
                logging.debug(f"loss on {i} for this inference: "
                              f"{ce_loss[i]:.4f}, log_prob: {selected[i]:.4f}")
        grad = th.autograd.grad(final_loss, x_in)[0]
        grad = grad * self.config.classifier_scale
        logging.debug(f"cur_t = {cur_t}, t={t[0].item()} done " + "=" * 10)
        results_holder[place_indicator] = grad  # put back
        return results_holder
