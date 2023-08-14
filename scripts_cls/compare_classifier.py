import os
import sys
import yaml
from types import SimpleNamespace
import numpy as np
import torch
import torchvision
import torchvision.transforms as tvs_trans
from tqdm import tqdm
import matplotlib.pyplot as plt

from guided_diffusion.script_util import (
    classifier_defaults,
    create_classifier,
    args_to_dict,
)
from guided_diffusion import dist_util
from ood_tester.classifier_model import load_classifier as load_normal
from ood_tester.dataloader import ImglistDataset, DataLoader
from ood_tester.utils import Convert


def load_classifier(
    cfg_path="testOOD/conf/classifier/resnet50.yaml", noisy=False):
    """load classifier that assume [0, 1] inputs

    Args:
        cfg_path (str, optional): Defaults to "testOOD/conf/classifier/resnet50.yaml".
        noisy (bool, optional): load the noisy cls. Defaults to False.
    """
    if noisy:
        args = SimpleNamespace(**classifier_defaults())
        args.image_size=256
        classifier = create_classifier(
            **args_to_dict(args, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(
                "../pretrained/guided-diffusion/256x256_classifier.pt",
                map_location="cpu",
            )
        )
        trans = torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3)
    else:
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        classifier = load_normal(**cfg["params"])
        trans = lambda x: x
    classifier.to(dist_util.dev())
    classifier.eval()
    return classifier, trans

def run_classifier(cls_type, log_dir):
    if cls_type == "normal":
        classifier, cls_trans = load_classifier(noisy=False)
        bs = 256
    elif cls_type == "noisy":
        classifier, cls_trans = load_classifier(noisy=True)
        bs = 128
    else:
        raise NotImplementedError(f"Unknown {sys.argv}")
    print(f"classifier {cls_type} loaded with trans {cls_trans}")

    # load data
    trans = torchvision.transforms.Compose([
            Convert('RGB'),
            tvs_trans.Resize(256),
            tvs_trans.CenterCrop(256),
            tvs_trans.ToTensor(),
            # tvs_trans.Normalize(mean=self.mean, std=self.std),
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

    print("data loaded, start ...")
    labels, preds = [], []
    try:
        for data in tqdm(dataloader, ncols=80):
            img = data['data'].to(dist_util.dev())
            label = data['label']

            with torch.no_grad():
                if cls_type == "noisy":
                    ts = torch.zeros(len(img), dtype=torch.long).to(img.device)
                    logits = classifier(cls_trans(img), ts)
                else:
                    logits = classifier(cls_trans(img))

                pred = logits.argmax(dim=-1)

            labels.append(label.cpu())
            preds.append(pred.cpu())
    except KeyboardInterrupt:
        pass

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()[:len(preds)]
    print(f"acc = {(labels == preds).sum() / len(preds) * 100:.4f} "
          f"from {len(preds)}")
    os.makedirs(log_dir, exist_ok=True)
    full_name = os.path.join(log_dir, f"cls_{cls_type}.npz")
    np.savez(full_name, preds=preds, labels=labels)
    print(f"data saved to {full_name}")        
    return full_name


def load_data(data_path):
    data = np.load(data_path)
    return torch.from_numpy(data['preds']), torch.from_numpy(data['labels'])

def show_data(npz1, npz2, dir_name):
    noi_preds, noi_labels = load_data(npz1)
    nor_preds, nor_labels = load_data(npz2)

    both_cor = torch.logical_and(
        (noi_preds == noi_labels), (nor_preds == nor_labels)
    )
    both_wrong_same = torch.logical_and(
        torch.logical_and(
            (noi_preds != noi_labels), (nor_preds != nor_labels),
        ), (noi_preds == nor_preds)
    )
    diff_pred = (noi_preds != nor_preds)

    assert both_cor.sum() + both_wrong_same.sum() + diff_pred.sum() == len(noi_preds)
    print(both_cor.sum(), both_wrong_same.sum(), diff_pred.sum())

    total = len(noi_preds)
    data = [
        diff_pred.sum() / total * 100,
        both_wrong_same.sum() / total * 100,
        both_cor.sum() / total * 100,
    ]
    width = 0.35
    plt.bar([0], data[2:3], width=width, label="both correct")
    plt.bar([0], data[1:2], bottom=data[2:3], width=width, label="both wrong")
    plt.bar([0], data[0:1], bottom=[data[1] + data[2]], width=width, label="diff")

    plt.ylabel('Num')
    # plt.xticks([0], ("InD",))
    N = 8
    ind = np.arange(N)  #[ 0  1  2  3  4  5  6  7  8  9 10 11 12]
    plt.xticks(
        ind,
        ['InD'] + ['G2'] * (N - 1),
    )

    plt.legend()
    plt.savefig(os.path.join(dir_name, "cls.pdf"))    


if __name__ == "__main__":
    dir_name = "../DiffGuard-log/"

    # cls_type = sys.argv[1]
    # torch.hub.set_dir("../pretrained/torch_cache")

    # data_path = run_classifier(cls_type, dir_name)

    show_data(
        "../DiffGuard-log/cls_noisy.npz",
        "../DiffGuard-log/cls_normal.npz",
        dir_name,
    )




