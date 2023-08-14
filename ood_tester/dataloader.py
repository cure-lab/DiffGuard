import logging
import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from .datasets import *
from .preprocessor import *


def get_preprocessor(config, split):
    train_preprocessors = {
        'base': BasePreprocessor,
    }
    test_preprocessors = {
        'base': TestStandardPreProcessor,
    }

    if split == 'train':
        return train_preprocessors[config.preprocessor.name](config)
    else:
        return test_preprocessors[config.preprocessor.name](config)


def get_dataloader(config: DictConfig):
    # prepare a dataloader dictionary
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        # weak augmentation for data_aux
        if config.need_aux:
            data_aux_preprocessor = TestStandardPreProcessor(config)
        else:
            data_aux_preprocessor = None
        CustomDataset = eval(split_config.dataset_class)
        dataset: BaseDataset = CustomDataset(
            name=dataset_config.name + '_' + split,
            imglist_pth=split_config.imglist_pth,
            data_dir=split_config.data_dir,
            num_classes=dataset_config.num_classes,
            preprocessor=preprocessor,
            data_aux_preprocessor=data_aux_preprocessor)
        logging.info(f"{dataset.name} len = {len(dataset)}")
        logging.info(
            f"transform on {dataset.name} is : {dataset.transform_image}")
        sampler = None
        if dataset_config.num_gpus > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=split_config.shuffle)
            split_config.shuffle = False

        dataloader = DataLoader(dataset,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers,
                                sampler=sampler)

        dataloader_dict[split] = dataloader
    return dataloader_dict


def get_ood_dataloader(config: DictConfig):
    # specify custom dataset class
    ood_config = config.ood_dataset
    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        if config.need_aux:
            data_aux_preprocessor = TestStandardPreProcessor(config)
        else:
            data_aux_preprocessor = None
        if split == 'val':
            # validation set
            dataset = CustomDataset(
                name=ood_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            logging.info(f"{dataset.name} len = {len(dataset)}")
            logging.info(
                f"transform on {dataset.name} is : {dataset.transform_image}")
            sampler = None
            if ood_config.num_gpus > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, shuffle=ood_config.shuffle)
                shuffle = False
            else:
                shuffle = ood_config.shuffle
            dataloader = DataLoader(dataset,
                                    batch_size=ood_config.batch_size,
                                    shuffle=shuffle,
                                    num_workers=ood_config.num_workers,
                                    sampler=sampler)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                dataset = CustomDataset(
                    name=ood_config.name + '_' + split + '_' + dataset_name,
                    imglist_pth=dataset_config.imglist_pth,
                    data_dir=dataset_config.data_dir,
                    num_classes=ood_config.num_classes,
                    preprocessor=preprocessor,
                    data_aux_preprocessor=data_aux_preprocessor)
                logging.info(f"{dataset.name} len = {len(dataset)}")
                logging.info(
                    f"transform on {dataset.name} is : {dataset.transform_image}")
                sampler = None
                if ood_config.num_gpus > 1:
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset, shuffle=ood_config.shuffle)
                    shuffle = False
                else:
                    shuffle = ood_config.shuffle
                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=shuffle,
                                        num_workers=ood_config.num_workers,
                                        sampler=sampler)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict
