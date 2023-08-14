import os
import sys
import logging
import importlib
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from guided_diffusion import dist_util
from ood_tester.classifier_model import load_classifier
from ood_tester.dataloader import get_dataloader, get_ood_dataloader
from ood_tester.test_runner import TestRunner
from external.pytorch_grad_cam import grad_cam


def setup(rank, world_size, init_method, port="12345"):
    if init_method.startswith("env://"):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
    # this is important to use dist_util
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method=init_method)


def set_logger(rank, logdir):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    if rank == 0:
        # to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        root.addHandler(handler)
    # to logger
    file_path = os.path.join(logdir, f"test_openood.{rank}.log")
    handler = logging.FileHandler(file_path)
    handler.setFormatter(formatter)
    root.addHandler(handler)


class LocalRandGenerator:
    def __init__(self, seed, device) -> None:
        self.seed = seed
        self.device = device
        self.rand_gen = self.new()

    def new(self):
        rand_gen = torch.Generator(device=self.device)
        rand_gen.manual_seed(self.seed)
        return rand_gen

    def reset(self):
        self.rand_gen = self.new()


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    return LocalRandGenerator(seed, "cpu")


def start_worker(rank, world_size, port, cfg):
    setup(rank, world_size, cfg.init_method, port)
    set_logger(rank, cfg.log_root)
    setattr(cfg, "rank", rank)
    torch.hub.set_dir("../pretrained/torch_cache")

    seed = cfg.seed
    local_rand_gen = fix_seed(seed)
    logging.info(f"{rank} start, world_size={world_size}, seed={seed}")

    # there is no interaction between process within the model, no need for DDP
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
    logging.info("diffusion loaded")

    dataloader = get_dataloader(cfg)
    ood_dataloader = get_ood_dataloader(cfg)
    logging.info("dataset loaded")

    runner = TestRunner(cfg)
    logging.info("start testing")
    runner.run(diffusion_model, classifier, dataloader, ood_dataloader)


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
    logging.info("Current config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)
    port = str(dist_util._find_free_port())
    mp.spawn(start_worker, args=(cfg.num_gpus, port, cfg),
             nprocs=cfg.num_gpus, join=True)


if __name__ == "__main__":
    main()
