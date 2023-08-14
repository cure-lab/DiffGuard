import os
import logging
import importlib
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from guided_diffusion import dist_util
from ood_tester.classifier_model import load_classifier
from ood_tester.dataloader import get_dataloader, get_ood_dataloader
from ood_tester.test_runner import TestRunner
from external.pytorch_grad_cam import grad_cam
from testOOD.test_openood import fix_seed


def setup(rank, world_size, init_method, port="12345"):
    if init_method.startswith("env://"):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
    # this is important to use dist_util
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method=init_method)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # only log debug info to log file
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    logging.debug("start!")
    logging.getLogger('PIL').setLevel(logging.WARNING)
    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)
    setattr(cfg, "rank", 0)
    setup(0, 1, "env://")

    local_rand_gen = fix_seed(cfg.seed)

    classifier = load_classifier(**cfg.classifier.params)
    classifier = classifier.to(dist_util.dev())
    classifier.eval()
    logging.info("classifier loaded")

    # if we do not need cam, do not call it.
    if cfg.diffusion.use_cam:
        target_layers = [eval(f"classifier.model.{cfg.classifier.cam_layer}")]
        cam = grad_cam.GradCAM(
            model=classifier, target_layers=target_layers, use_cuda=True,
            norm_each=False)
    else:
        cam = None

    module, cls = cfg.diffusion.target.rsplit(".", 1)
    diffusion_model = getattr(
        importlib.import_module(module, package=None), cls)(
            config=cfg.diffusion, cam=cam, local_rand_gen=local_rand_gen)
    diffusion_model.to(dist_util.dev())
    if hasattr(diffusion_model, "log_inter_every"):
        diffusion_model.log_inter_every = 5
    logging.info("diffusion loaded")

    dataloader = get_dataloader(cfg)
    # ood_dataloader = get_ood_dataloader(cfg)
    logging.info("dataset loaded")

    runner = TestRunner(cfg)
    logging.info("start testing")
    runner.run(diffusion_model, classifier, dataloader, {})


if __name__ == "__main__":
    torch.hub.set_dir("../pretrained/torch_cache")
    main()
