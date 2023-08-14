from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import logging

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMForward
import external.pytorch_grad_cam as grad_cam
from .utils import minus_1_1_to_01


def load_model_from_config(config_file, ckpt):
    config = OmegaConf.load(config_file)
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    logging.info(f"missing: {m}; unexpected: {u}")
    return model


def scale_each(to_scale):
    results = []
    for img in to_scale:
        _min = np.min(img)
        _max = np.max(img)
        img = (img - _min) / (_max - _min + 1e-7)  # normalize to [0, 1]
        results.append(img)
    return np.array(results)


class LDMWrapper:
    def __init__(self, config, cam=None, local_rand_gen=None):
        super().__init__()
        self.config = config
        self.model = load_model_from_config(
            self.config.ldm_config_path, self.config.ldm_weight_path)
        self.model.eval()

        self.sampler = DDIMSampler(self.model)
        self.noiser = DDIMForward(self.model)
        self.ddim_steps = self.config.ddim_steps
        self.ddim_eta = self.config.ddim_eta
        self.denoise_scale = self.config.denoise_guidance_scale   # for unconditional guidance
        # scale param for forward pass
        self.forward_scale = self.config.diffuse_guidance_scale
        self.unconditional_forward = self.config.unconditional_diffuse
        if self.config.use_cam:
            assert cam is not None
            self.cam_cut = self.config.cam_cut
            self.cam = cam

        # other data
        if self.config.image_size != self.config.input_size:
            if self.config.process_method == "resize":
                self.preprocess = transforms.Resize(self.config.image_size)
                self.postprocess = transforms.Resize(self.config.input_size)
            elif self.config.process_method == "pad":
                self.preprocess = transforms.Pad(
                    (self.config.image_size - self.config.input_size) // 2,
                    fill=-1)
                self.postprocess = transforms.CenterCrop(self.config.input_size)
                # self.metric_crop = self.config.input_size
            logging.info(f"LatentDiffusion apply preprocess: {self.preprocess}")
            logging.info(
                f"LatentDiffusion apply postprocess: {self.postprocess}")
        else:
            self.preprocess = lambda x: x
            self.postprocess = lambda x: x
        self.batch_size = self.config.batch_size
        self.log_inter_every = -1

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, x0, target_label, uncond_aux=False, **kwargs):
        # empty cache is used to expose real memory usage
        # torch.cuda.empty_cache()
        xT = self.diffuse(x0, target_label)
        logging.debug(f"{len(x0)} images diffuse done, start denoise")
        # torch.cuda.empty_cache()

        all_recons = []
        x0_splits = x0.split(self.batch_size)
        xT_splits = xT.split(self.batch_size)
        label_splits = target_label.split(self.batch_size)
        for x0i, xTi, labeli in zip(x0_splits, xT_splits, label_splits):
            activation_map = self.get_activation_map(x0i, labeli)
            recon = self.denoise(xTi, labeli, activation_map)
            if uncond_aux:
                # ldm can take all GPU in one pass, we just do it twice
                recon_uncond = self.denoise(xTi, None, None)
                recon = torch.stack([recon, recon_uncond], dim=1)
            all_recons.append(recon)
            logging.debug(f"{len(x0i)} images done, target label: {labeli}")
            # torch.cuda.empty_cache()
        return torch.cat(all_recons, dim=0)

    def get_uc_and_c(self, label, scale, batch_size):
        if label is None or scale == 1.0:  # disable uc
            uc = None
        else:
            uc = self.model.get_learned_conditioning(
                {self.model.cond_stage_key: torch.tensor(
                    batch_size * [1000]).to(self.model.device)})
        if label is None:  # use unconditional condition, i.e. 1000
            c = self.model.get_learned_conditioning(
                {self.model.cond_stage_key: torch.tensor(
                    batch_size * [1000]).to(self.model.device)})
        else:
            c = self.model.get_learned_conditioning(
                {self.model.cond_stage_key: label.to(self.model.device)})
        return uc, c

    @torch.no_grad()
    def diffuse(self, x0, label=None):
        if self.unconditional_forward:
            label = None
        # assume x0 in [-1, 1]
        _uc, _c = self.get_uc_and_c(label, self.forward_scale, len(x0))
        # start
        x0 = self.preprocess(x0)
        assert x0.shape[-1] == self.config.image_size
        _embedding = self.model.encode_first_stage(x0)
        # if we need quality measure, apply 224 / 8 = 28 centercrop
        latent_ddim, inter = self.noiser.sample(
            S=self.ddim_steps, conditioning=_c,
            batch_size=len(x0), shape=_embedding.shape[1:],
            verbose=False, unconditional_guidance_scale=self.forward_scale,
            unconditional_conditioning=_uc, eta=self.ddim_eta,
            x_T=_embedding, log_every_t=self.log_inter_every)
        if self.log_inter_every > 0:
            for i, x_in in zip(inter['t'], inter['x_inter']):
                _inter_ddim = self.model.decode_first_stage(x_in)
                _inter_ddim = torch.clamp(_inter_ddim, min=-1.0, max=1.0)
                _inter_ddim = self.postprocess(_inter_ddim)
                torchvision.utils.save_image(
                    _inter_ddim * 0.5 + 0.5, f"add_noise{i}.png")
        return latent_ddim

    @torch.no_grad()
    def denoise(self, xT, label, activation_map=None):
        # generate image for given noise with label
        _uc, _c = self.get_uc_and_c(label, self.denoise_scale, len(xT))

        re_samples_ddim, inter = self.sampler.sample(
            S=self.ddim_steps, conditioning=_c,
            batch_size=len(xT), shape=xT.shape[1:],
            verbose=False, unconditional_guidance_scale=self.denoise_scale,
            unconditional_conditioning=_uc, eta=self.ddim_eta,
            x_T=xT, activation_map=activation_map,
            log_every_t=self.log_inter_every)
        _samples_ddim = self.model.decode_first_stage(re_samples_ddim)
        _samples_ddim = torch.clamp(_samples_ddim, min=-1.0, max=1.0)
        _samples_ddim = self.postprocess(_samples_ddim)
        if self.log_inter_every > 0:
            for i, x_in in zip(inter['t'], inter['x_inter']):
                _inter_ddim = self.model.decode_first_stage(x_in)
                _inter_ddim = torch.clamp(_inter_ddim, min=-1.0, max=1.0)
                _inter_ddim = self.postprocess(_inter_ddim)
                torchvision.utils.save_image(
                    _inter_ddim * 0.5 + 0.5, f"denoise{i}.png")
        return _samples_ddim

    def get_activation_map(self, x0, label):
        if not self.config.use_cam:
            return None
        targets = [grad_cam.utils.model_targets.ClassifierOutputTarget(i.item())
                   for i in label.cpu()]
        x0 = minus_1_1_to_01(x0)

        with torch.enable_grad():
            grayscale_cam = self.cam(input_tensor=x0, targets=targets)

        grayscale_cam_scaled = scale_each(grayscale_cam)
        if self.cam_cut < 0:
            # low with label, high with input image.
            activation_map = torch.from_numpy(
                grayscale_cam_scaled < abs(self.cam_cut)).float().cuda()
        else:
            # high with label, low with input image.
            activation_map = torch.from_numpy(
                grayscale_cam_scaled > abs(self.cam_cut)).float().cuda()
        # activation_edge = dilate(activation_map) - activation_map
        activation_map = activation_map.unsqueeze(1).repeat(1, 3, 1, 1)
        return activation_map
