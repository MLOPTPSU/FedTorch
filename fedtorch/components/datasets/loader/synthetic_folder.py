# -*- coding: utf-8 -*-

from fedtorch.logs.logging import log
from fedtorch.components.datasets.loader.utils import IMDBPT


# TODO: to be removed. Use federated_datasets.py 
def define_synthetic_folder(args, root, pattern=False):
    log('load synthetic dataset from lmdb in the form: {}.'.format(root), args.debug)
    return IMDBPT(root, is_image=False, pattern=False)