import os
import logging
import time
import copy
from datetime import timedelta
from functools import partial
from collections import deque
import csv
import torch
import torch.distributed as dist
import torchvision
import numpy as np
import piq

from guided_diffusion import dist_util
from .metrics import *
from .utils import minus_1_1_to_01


class DistReverseWrapper:
    def __init__(self, func, range01=False) -> None:
        super().__init__()
        self.func = func
        self.range01 = range01

    def __call__(self, *args, **kwds):
        if self.range01:
            return 1. - self.func(*args, **kwds)
        else:
            return - self.func(*args, **kwds)


def get_all_metrics(names, ind_scores, ood_socres):
    # ind = 1, ood = 0/-1
    label = np.concatenate(
        [np.ones_like(ind_scores), -np.ones_like(ood_socres)], axis=0)
    scores = np.concatenate([ind_scores, ood_socres], axis=0)
    auroc, aupr_in, aupr_out = auc(scores, label)
    metrics = {
        'AUROC': auroc,
        'AUPR_IN': aupr_in,
        'AUPR_OUT': aupr_out,
    }

    recall = 0.95
    fpr, thresh = fpr_recall(scores, label, recall)
    metrics['FPR@95'] = fpr
    ret_metrics = {k: metrics[k] for k in names}
    return ret_metrics


def get_distance(name):
    dist_funcs = {
        'vif': DistReverseWrapper(piq.VIFLoss(reduction='none').cuda(), True),
        'dists': DistReverseWrapper(piq.DISTS(reduction='none').cuda(), True),
        'mdsi': DistReverseWrapper(piq.MDSILoss(reduction='none').cuda()),
        'fsim': DistReverseWrapper(piq.FSIMLoss(reduction='none').cuda(), True),
        'ssim': DistReverseWrapper(piq.SSIMLoss(reduction='none').cuda(), True),
        'gmsd': DistReverseWrapper(piq.GMSDLoss(reduction='none').cuda()),
        'lpips': DistReverseWrapper(piq.LPIPS(reduction='none').cuda()),
        'l2': DistReverseWrapper(
            lambda x, y: torch.norm(x - y, 2, dim=[1, 2, 3])),
        'logits': None,
    }
    return dist_funcs[name]

    
def a_is_better(value_a, value_b, larger_true):
    return not (value_a > value_b) ^ larger_true


def take_best(dist_metrics_dict):
    larger_better = {
        "AUROC": True,
        "AUPR_IN": True,
        "AUPR_OUT": True,
        "FPR@95": False,
    }
    best_dict = dict()
    for dist_k, dist_v in dist_metrics_dict.items():
        for mk, mv in dist_v.items():
            if mk not in best_dict:
                best_dict[mk] = {"name": dist_k, "value": mv}
            elif a_is_better(mv, best_dict[mk]['value'], larger_better[mk]):
                best_dict[mk] = {"name": dist_k, "value": mv}
    return best_dict


def gather_data(data_tensor):
    gathered_data = [torch.zeros_like(data_tensor)
                     for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_data, data_tensor)
    gathered_data = torch.cat([sample.cpu() for sample in gathered_data])
    return gathered_data


@torch.no_grad()
def run_on_loader(diffusion_model, classifier, loader, dist_funcs, rank,
                  uncond_aux=False, save_feat=False, save_img=False,
                  fig_saver=None, try_run=False, oracle_label=False,
                  oracle_for_ind=False):
    all_pred = torch.Tensor()
    all_label = torch.Tensor()
    all_index = torch.Tensor()
    dist_funcs = copy.deepcopy(dist_funcs)
    if 'logits' in dist_funcs:
        all_distance_dict = {'logits': torch.Tensor()}
        dist_funcs.pop('logits')
    else:
        all_distance_dict = {}
    all_distance_dict.update({k: torch.Tensor() for k in dist_funcs.keys()})
    # start support logits
    all_data = {k: torch.Tensor() for k in ['imgr', 'logits', 'feat']}
    # end support logits
    total_batch = len(loader)
    start_time = last_time = time.time()
    time_used = deque(maxlen=5)  # only track time of last 5 batch
    throughput_ema = deque(maxlen=5)  # only track time of last 5 batch

    def logits_and_feature(img_in):
        cls_in = minus_1_1_to_01(img_in)
        logits_out, feat_out = classifier.model(
            classifier.normalize(cls_in), return_feature=True)
        return logits_out, feat_out

    for batchi, sample in enumerate(loader):
        # key in sample: label, data, soft_label, data_aux
        # data is normalized to [-1, 1], i.e. all 0.5 normalization.
        image = sample['data'].to(dist_util.dev())
        _label = sample['label']  # useless, later to check correctness
        _index = sample['index']

        net_start = time.time()
        # classifier take [0, 1] 224 or 255 inputs, here we assume 224. other
        # data norm is absorbed in cls.
        cls_input = minus_1_1_to_01(image)
        pred = classifier(cls_input)
        pred_label = pred.argmax(-1)

        if oracle_label or oracle_for_ind:
            use_label = _label.to(dist_util.dev())
            if (use_label == -1).any():
                if oracle_for_ind:  # use pred for ood
                    use_label = pred_label
                else:  # use random for ood
                    use_label = torch.randint_like(
                        _label, 0, pred.shape[-1]).to(dist_util.dev())
            logging.debug(f"label for gen {use_label}")
        else:
            use_label = pred_label
        # diffusion take [-1, 1] inputs, input 224 size, diffusion will reize to
        # 255 to fit the model.
        _x_recon = diffusion_model(
            image, use_label, classifier=classifier, uncond_aux=uncond_aux)
        if uncond_aux:
            x_recon = _x_recon[:, 0]  # stacked on dim=1
            x_uncond = _x_recon[:, 1]
        else:
            x_recon = _x_recon

        # start support logits
        # summary distance
        torch.cuda.synchronize()
        if uncond_aux:
            x_for_compare = x_uncond
        else:
            x_for_compare = image
        for k, dist_func in dist_funcs.items():
            distance = dist_func(
                minus_1_1_to_01(x_for_compare), minus_1_1_to_01(x_recon))
            distance = gather_data(distance)
            if rank == 0:
                all_distance_dict[k] = torch.cat(
                    [all_distance_dict[k], distance.detach()], dim=0)
            torch.distributed.barrier()
        net_time = time.time() - net_start

        if save_feat:
            logits_ori, feat_ori = logits_and_feature(image)
            logits_recon, feat_recon = logits_and_feature(x_recon)
            if uncond_aux:
                logits_aux, feat_aux = logits_ori, feat_ori
                logits_comp, feat_comp = logits_and_feature(x_uncond)
            else:
                logits_comp, feat_comp = logits_ori, feat_ori
            # save
            logits_save = [logits_comp, logits_recon]
            feat_save = [feat_comp, feat_recon]
            if uncond_aux:
                logits_save += [logits_aux]
                feat_save += [feat_aux]
            logits_save = torch.stack(logits_save, dim=1)
            feat_save = torch.stack(feat_save, dim=1)
            logits_save = gather_data(logits_save).detach()
            feat_save = gather_data(feat_save).detach()
            if rank == 0:
                all_data['logits'] = torch.cat(
                    [all_data['logits'], logits_save], dim=0)
                all_data['feat'] = torch.cat(
                    [all_data['feat'], feat_save], dim=0)
                # logits distance
                if 'logits' in all_distance_dict:
                    logits_dist = - (
                        logits_save[:, 0] - logits_save[:, 1]).abs().sum(dim=-1)
                    all_distance_dict['logits'] = torch.cat(
                        [all_distance_dict['logits'], logits_dist], dim=0)
        if save_img:
            x_recon_input = minus_1_1_to_01(x_recon)
            if uncond_aux:
                x_uncond_input = minus_1_1_to_01(x_uncond)
                x_recon_input = torch.stack(
                    [x_recon_input, x_uncond_input], dim=1)
            # save
            x_recon_input = gather_data(x_recon_input).detach()
            if rank == 0:
                all_data['imgr'] = torch.cat(
                    [all_data['imgr'], x_recon_input], dim=0)
        # end support logits

        # gather data
        pred_label_g = gather_data(pred_label)
        _label_g = gather_data(_label.cuda())
        _index_g = gather_data(_index.cuda())
        if rank == 0:
            all_pred = torch.cat([all_pred, pred_label_g.cpu()], dim=0)
            all_label = torch.cat([all_label, _label_g], dim=0)
            all_index = torch.cat([all_index, _index_g], dim=0)
        else:
            pass

        # time and log
        time_batch = time.time() - last_time
        time_used.appendleft(time_batch)
        time_total = str(timedelta(seconds=int(time.time() - start_time)))
        time_to_end = str(timedelta(seconds=int(
            np.average(time_used) * (total_batch - batchi - 1))))
        # if we use `uncond_aux`, we actaully double images num for generation
        throughput = len(image) / net_time  # image per second per gpu
        throughput_ema.appendleft(throughput)
        logging.info(
            "+" * 5 +
            f" {batchi}/{len(loader)} done , "
            f"time used {time_total}, time to end {time_to_end}, "
            f"throughput: {np.average(throughput_ema): .2f} images/s/card " +
            "+" * 5 + "\n")

        # save fig
        if rank == 0 and batchi == 0 and fig_saver is not None:
            fig_saver(image, x_recon, _label, pred_label)

        # reset for next
        torch.distributed.barrier()
        last_time = time.time()
        if try_run and batchi > 0:
            break

    all_distance_dict = {k: v.numpy() for k, v in all_distance_dict.items()}
    all_data = {k: v.numpy() for k, v in all_data.items()}
    return all_distance_dict, all_label.numpy(), all_pred.numpy(), \
        all_index.numpy(), all_data


class TestRunner:
    def __init__(self, config) -> None:
        self.distances = {
            name: get_distance(name) for name in config.runner.distances}
        self.metric = config.runner.metrics
        self.log_root = config.log_root
        self.rank = config.rank
        self.try_run = config.try_run
        self.uncond_aux = config.runner.uncond_aux
        self.save_feat = getattr(config.runner, "save_feat", False)
        self.save_img = getattr(config.runner, "save_img", False)
        self.oracle_label = getattr(config.runner, "oracle_label", False)
        self.oracle_for_ind = getattr(config.runner, "oracle_for_ind", False)

    def run(self, diffusion_model, classifier, dataloader, ood_dataloaders):
        # oracle_label: inds use gt labels, oods use random labels
        # oracle_for_ind: only inds use gt labels
        ind_scores, label, pred, im_index, im_data = run_on_loader(
            diffusion_model, classifier, dataloader['test'], self.distances,
            self.rank, self.uncond_aux, self.save_feat, self.save_img,
            fig_saver=partial(self._fig_saver, name_prefix="ind"),
            try_run=self.try_run, oracle_label=self.oracle_label,
            oracle_for_ind=self.oracle_for_ind)
        if self.rank == 0:
            self._save_scores(
                {"scores": ind_scores, "label": label, "pred": pred,
                 "im_index": im_index, **im_data}, "ind")
        ood_scores_dict = dict()
        ood_dist_metrics_dict = dict()
        for split in ood_dataloaders.keys():
            for ood_name, ood_loader in ood_dataloaders[split].items():
                logging.info(f"running on {ood_name}")
                ood_scores, label, pred, im_index, im_data = run_on_loader(
                    diffusion_model, classifier, ood_loader, self.distances,
                    self.rank, self.uncond_aux, self.save_feat, self.save_img,
                    fig_saver=partial(self._fig_saver, name_prefix=ood_name),
                    try_run=self.try_run, oracle_label=self.oracle_label,
                    oracle_for_ind=self.oracle_for_ind)
                if self.rank == 0:
                    self._save_scores(
                        {"scores": ood_scores, "label": label, "pred": pred,
                         "im_index": im_index, **im_data}, ood_name)
                    ood_scores_dict[ood_name] = ood_scores

                # check metrics
                if self.rank == 0:
                    dist_metrics_dict = dict()
                    for dist_k, dist_v in ood_scores.items():
                        metrics_dict = get_all_metrics(
                            self.metric, ind_scores[dist_k], dist_v)
                        self._save_csv(metrics_dict, ood_name, dist_k)
                        dist_metrics_dict[dist_k] = metrics_dict
                    # now we gathered metrics from all distances
                    best_metrics_dict = take_best(dist_metrics_dict)
                    self._save_csv(best_metrics_dict, ood_name, "best")
                    ood_dist_metrics_dict[ood_name] = best_metrics_dict
                torch.distributed.barrier()

            # get mean over all ood datasets
            if self.rank == 0:
                metrics_mean = dict()
                for mkey in self.metric:
                    metrics_mean[mkey] = {
                        "name": "avg",
                        "value": np.mean([
                            ood_dist_metrics_dict[ood_name][mkey]['value']
                            for ood_name in ood_dataloaders[split].keys()])
                    }
                self._save_csv(metrics_mean, split, "best")
            torch.distributed.barrier()

    def _save_csv(self, ood_metrics_dict, dataset_name, dist):
        write_content = {
            'dataset': dataset_name,
            'distance': dist,
        }
        if dist == "best":
            write_content.update({
                k: f"{100 * v['value']:.2f}" for k, v in ood_metrics_dict.items()
            })
        else:
            write_content.update({
                k: f"{100 * v:.2f}" for k, v in ood_metrics_dict.items()
            })

        # print ood metric results
        res_msg = " ".join([f'{k}: {v}' for k, v in write_content.items()])
        logging.info(res_msg)
        print(u'\u2500' * 70, flush=True)
        if dist != "best":
            return

        fieldnames = list(write_content.keys())
        csv_path = os.path.join(self.log_root, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, data_dict, save_name="socres.npz"):
        save_dir = os.path.join(self.log_root, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, f"{save_name}.npz"), **data_dict)

    def _fig_saver(self, image, recon, gt, pred, name_prefix="fig"):
        image_root = os.path.join(self.log_root, "images")
        ori_path = os.path.join(image_root, f"{name_prefix}_ori.png")
        recon_path = os.path.join(image_root, f"{name_prefix}_recon.png")
        # save
        os.makedirs(image_root, exist_ok=True)
        torchvision.utils.save_image(minus_1_1_to_01(image), ori_path)
        torchvision.utils.save_image(minus_1_1_to_01(recon), recon_path)
        # log
        logging.info(f"example image saved to {image_root}, \n"
                     f"gt={[gti.item() for gti in gt]}, \n"
                     f"pred={[predi.item() for predi in pred]}.\n")
