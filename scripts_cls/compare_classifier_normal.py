from omegaconf import OmegaConf
import os
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
from ood_tester.cutout import MakeCutouts
from testOOD.test_openood import LocalRandGenerator
from scripts_cls.compare_classifier import load_classifier


def get_q_sample():
    config = OmegaConf.load("testOOD/conf/diffusion/guided-diffusion.yaml")
    config.timestep_respacing = "ddim100"

    model, diffusion = create_model_and_diffusion(
        **args_to_dict_with_defaults(
            config, model_and_diffusion_defaults())
    )
    model.load_state_dict(dist_util.load_state_dict(
        config.model_path, map_location='cpu'))
    if config.use_fp16:
        model.convert_to_fp16()
    model.eval()
    model.to(dist_util.dev())

    @torch.no_grad()
    def get_xstart(xt, t):
        out = diffusion.p_mean_variance(model, xt, t, clip_denoised=True)
        return out['pred_xstart']

    return diffusion.q_sample, get_xstart


def ensemble(cutout_out, bs, aux=False):
    # (r, b, num_classes)
    logits_cutout = cutout_out.reshape(-1, bs, cutout_out.shape[-1])
    pred_cutout_mean = logits_cutout.mean(dim=0).argmax(-1)
    # majority vote, if equal, take the smaller index
    # (r, b) -> (b, r)
    pred_cutout_vote = logits_cutout.argmax(-1).cpu()
    pred_cutout_vote = pred_cutout_vote.transpose(0, 1).mode()[0]
    # max logits
    _max_v, max_ind = logits_cutout.max(-1)
    _max_v_over_cuts = _max_v.argmax(dim=0)
    pred_cutout_max = max_ind[_max_v_over_cuts, torch.arange(bs)]
    if not aux:
        return pred_cutout_mean, pred_cutout_vote, pred_cutout_max
    # for any evaludation
    raw_argmax = logits_cutout.argmax(-1).transpose(0, 1)
    return pred_cutout_mean, pred_cutout_vote, pred_cutout_max, raw_argmax


def judge_raw(raw, label):
    res = []
    for rawi, li in zip(raw, label):
        if li in rawi:
            res.append(li)
        else:
            res.append(torch.ones_like(li) * -1)
    return torch.stack(res, dim=0)


def report(labels, preds, ti, t):
    line = f"{t},"
    this_len = len(labels[ti])
    ci_all = preds[ti].shape[1]
    for ci in range(ci_all):
        acc = (labels[ti] == preds[ti][:this_len, ci]).sum() / this_len
        print(f"acc = {acc * 100:.4f} from {this_len}")
        line += f"{acc * 100:.4f},"
    return line


def run_classifier(log_dir):
    nor_cls, nor_trans = load_classifier(noisy=False)
    nor_trans = tvs_trans.Compose([
        tvs_trans.Normalize([-1.] * 3, [2.] * 3),  # -1, 1 -> 0, 1
        nor_trans
    ])
    bs = 32

    # diffusion
    q_sample, get_xstart = get_q_sample()

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
    # all_ts = [0] + list(range(9, 100, 10))
    all_ts = list(range(9, 100, 10))
    lrg = LocalRandGenerator(42, "cpu")
    cut_n = 8
    cutout = MakeCutouts(224, cut_n, out_size=256, local_rand_gen=lrg)
    print(f"running on {all_ts}")

    print("data loaded, start ...")
    labels, preds = [], []
    res_file = open(os.path.join(dir_name, "ablation_acc.csv"), 'a')
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
                        x0_hat = img
                    else:
                        img_t = q_sample(img, ts)
                        x0_hat = get_xstart(img_t, ts)

                    # x0 only
                    nor_x0_logits = nor_cls(nor_trans(x0_hat))
                    nor_x0_pred = nor_x0_logits.argmax(dim=-1)

                    # use cutout
                    _in = torch.cat([img_t, x0_hat], dim=0)
                    _cutout = cutout(_in)
                    _cutout = _cutout.reshape(cut_n, *_in.shape)
                    (x_cutout, x0_cutout) = _cutout.split(len(img), dim=1)
                    x_cutout = x_cutout.reshape(-1, *img.shape[1:])
                    x0_cutout = x0_cutout.reshape(-1, *img.shape[1:])

                    # cutout only
                    nor_logits_cutout = nor_cls(nor_trans(x_cutout))
                    norp_c_mean, norp_c_vote, norp_c_max, norp_c_raw = \
                        ensemble(nor_logits_cutout, len(img), True)
                    norp_c_raw = judge_raw(norp_c_raw.cpu(), label)

                    # cutout + x0
                    nor_logits_x0_cutout = nor_cls(nor_trans(x0_cutout))
                    norp_x0c_mean, norp_x0c_vote, norp_x0c_max, norp_x0c_raw = \
                        ensemble(nor_logits_x0_cutout, len(img), True)
                    norp_x0c_raw = judge_raw(norp_x0c_raw.cpu(), label)

                    pred = torch.stack([
                        nor_x0_pred.cpu(),
                        norp_c_mean.cpu(),
                        norp_c_vote.cpu(),
                        norp_c_max.cpu(),
                        norp_c_raw.cpu(),
                        norp_x0c_mean.cpu(),
                        norp_x0c_vote.cpu(),
                        norp_x0c_max.cpu(),
                        norp_x0c_raw.cpu(),
                    ], dim=1)

                labels[ti].append(label.cpu())
                preds[ti].append(pred.cpu())
        except KeyboardInterrupt:
            preds[ti] = torch.cat(preds[ti])
            labels[ti] = torch.cat(labels[ti])
            _ = report(labels, preds, ti, t)
            preds.pop(-1)
            labels.pop(-1)
            break
        else:
            preds[ti] = torch.cat(preds[ti])
            labels[ti] = torch.cat(labels[ti])
            line = report(labels, preds, ti, t)
            assert res_file.write(line[:-1] + "\n")
            res_file.flush()

    res_file.close()
    preds = torch.stack(preds, dim=0).numpy()
    labels = torch.stack(labels, dim=0).numpy()
    full_name = os.path.join(log_dir, f"cls_ablation.npz")
    np.savez(full_name, preds=preds, labels=labels)
    print(f"data saved to {full_name}")
    return full_name


if __name__ == "__main__":
    dir_name = "../DiffGuard-log/"
    torch.hub.set_dir("../pretrained/torch_cache")

    data_path = run_classifier(dir_name)
