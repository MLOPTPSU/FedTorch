# -*- coding: utf-8 -*-

from logs.logging import log
from components.datasets.loader.utils import IMDBPT


#TODO: To be removed. Use libsvm_datasets.py
def define_epsilon_or_rcv1_or_MSD_folder(args,root):
    log('load epsilon_or_rcv1_or_MSD from lmdb: {}.'.format(root), args.debug)
    return IMDBPT(root, is_image=False)
