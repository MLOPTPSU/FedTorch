# -*- coding: utf-8 -*-
import torchvision.datasets as datasets

from logs.logging import log
from components.datasets.preprocess_toolkit import get_transform
from components.datasets.loader.utils import IMDBPT


def define_imagenet_folder(args, name, root, flag, cuda=True):
    is_train = 'train' in root
    transform = get_transform(name, augment=is_train, color_process=False)

    if flag:
        log('load imagenet from lmdb: {}'.format(root), args.debug)
        return IMDBPT(root, transform=transform, is_image=True)
    else:
        log("load imagenet using pytorch's default dataloader.", args.debug)
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=None)