from omegaconf import OmegaConf
import os
import sys
import torch
import torchvision
import torchvision.transforms as tvs_trans
import numpy as np
from tqdm import tqdm

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict_with_defaults,
)
from guided_diffusion import dist_util
from ood_tester.dataloader import ImglistDataset, DataLoader
from ood_tester.utils import Convert
from scripts_cls.compare_classifier import load_classifier



def get_q_sample():
    config = OmegaConf.load("testOOD/conf/diffusion/guided-diffusion.yaml")
    config.timestep_respacing = "ddim100"

    _, diffusion = create_model_and_diffusion(
        **args_to_dict_with_defaults(
        config, model_and_diffusion_defaults())
    )
    return diffusion.q_sample



def run_classifier(log_dir):
    size = int(sys.argv[1])
    nor_cls, nor_trans = load_classifier(noisy=False)
    nor_trans = tvs_trans.Compose([
        tvs_trans.CenterCrop(size),
        tvs_trans.Normalize([-1.] * 3, [2.] * 3),  # -1, 1 -> 0, 1
        nor_trans
    ])
    noi_cls, noi_trans = load_classifier(noisy=True)
    noi_trans = tvs_trans.Compose([
        tvs_trans.Normalize([-1.] * 3, [2.] * 3),  # -1, 1 -> 0, 1
        noi_trans
    ])
    bs = 128

    # diffusion
    q_sample = get_q_sample()

    # load data, [-1, 1]
    trans = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(256),
            tvs_trans.CenterCrop(256),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize([0.5] * 3, std=[0.5] * 3),
    ])
    testset = ImglistDataset(
        "imagenet",
        "./useful_data/benchmark_imglist/imagenet/test_imagenet.txt",
        "../data/",
        num_classes=1000,
        preprocessor=trans,
    )
    dataloader = DataLoader(
        testset,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
    )

    os.makedirs(log_dir, exist_ok=True)
    all_ts = [0] + list(range(9, 100, 10))
    print(f"running on {all_ts}")
    
    print("data loaded, start ...")
    labels, preds = [], []
    res_file = open(os.path.join(dir_name, f"both_acc_{size}.csv"), 'w')
    for ti, t in tqdm(enumerate(all_ts), ncols=80, total=len(all_ts)):
        labels.append([])
        preds.append([])
        try:
            for data in tqdm(dataloader, ncols=80, leave=False):
                img = data['data'].to(dist_util.dev())
                label = data['label']

                with torch.no_grad():
                    ts = torch.ones(len(img), dtype=torch.long).to(img.device)
                    ts = ts * t

                    if t == 0:
                        img_t = img
                    else:
                        img_t = q_sample(img, ts)

                    noi_logits = noi_cls(noi_trans(img_t), ts)
                    noi_pred = noi_logits.argmax(dim=-1)

                    nor_logits = nor_cls(nor_trans(img_t))
                    nor_pred = nor_logits.argmax(dim=-1)

                    pred = torch.stack([noi_pred, nor_pred], dim=1)

                labels[ti].append(label.cpu())
                preds[ti].append(pred.cpu())
        except KeyboardInterrupt:
            preds.pop(-1)
            labels.pop(-1)
            break
        else:
            preds[ti] = torch.cat(preds[ti])
            labels[ti] = torch.cat(labels[ti])
            line = f"{t},"
            for ci in range(2):
                acc = (labels[ti] == preds[ti][:, ci]).sum() / len(labels[ti])
                print(f"acc = {acc * 100:.4f} from {len(labels[ti])}")
                line += f"{acc * 100:.4f},"
            assert res_file.write(line[:-1] + "\n")
            res_file.flush()

    res_file.close()
    preds = torch.stack(preds, dim=0).numpy()
    labels = torch.stack(labels, dim=0).numpy()
    full_name = os.path.join(log_dir, f"cls_both_{size}.npz")
    np.savez(full_name, preds=preds, labels=labels)
    print(f"data saved to {full_name}")        
    return full_name


if __name__ == "__main__":
    dir_name = "../DiffGuard-log/"
    torch.hub.set_dir("../pretrained/torch_cache")

    data_path = run_classifier(dir_name)
