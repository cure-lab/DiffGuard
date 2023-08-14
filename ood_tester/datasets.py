import ast
import io
import logging
import os
import random
import traceback

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile


class BaseDataset(Dataset):
    def __init__(self, name, pseudo_index=-1, skip_broken=False,
                 new_index='next'):
        super(BaseDataset, self).__init__()
        self.name = name
        self.pseudo_index = pseudo_index
        self.skip_broken = skip_broken
        self.new_index = new_index
        if new_index not in ('next', 'rand'):
            raise ValueError('new_index not one of ("next", "rand")')
        self.preprocessor = None
        self.transform_image = None

    def __getitem__(self, index):
        # in some pytorch versions, input index will be torch.Tensor
        index = int(index)

        # if sampler produce pseudo_index,
        # randomly sample an index, and mark it as pseudo
        if index == self.pseudo_index:
            index = random.randrange(len(self))
            pseudo = 1
        else:
            pseudo = 0

        while True:
            try:
                sample = self.getitem(index)
                break
            except Exception as e:
                if self.skip_broken and not isinstance(e, NotImplementedError):
                    if self.new_index == 'next':
                        new_index = (index + 1) % len(self)
                    else:
                        new_index = random.randrange(len(self))
                    logging.warn(
                        'skip broken index [{}], use next index [{}]'.format(
                            index, new_index))
                    index = new_index
                else:
                    logging.error('index [{}] broken'.format(index))
                    traceback.print_exc()
                    logging.error(e)
                    raise e

        sample['index'] = index
        sample['pseudo'] = pseudo
        return sample

    def getitem(self, index):
        raise NotImplementedError


# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImglistDataset(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor=None,
                 maxlen=None,
                 **kwargs):
        super(ImglistDataset, self).__init__(name, **kwargs)

        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor  # useless
        self.maxlen = maxlen

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 2)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        sample['index'] = index
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        # some preprocessor methods require setup
        if hasattr(self.preprocessor, "setup"):
            self.preprocessor.setup(**kwargs)
        try:
            with open(path, 'rb') as f:
                content = f.read()
            filebytes = content
            with io.BytesIO(filebytes) as buff:
                image = Image.open(buff).convert('RGB')
            sample['data'] = self.transform_image(image)
            if self.transform_aux_image is not None:
                sample['data_aux'] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample
