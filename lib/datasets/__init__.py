#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA


from .cifar import CIFAR10, CIFAR100

from ..core.config import Config as cfg
from ..utils.loggers import STDLogger as logger

_MODULE_ = 'dataset'
cfg.register_module(_MODULE_, __name__)
cfg.register_class(_MODULE_, 'cifar10', CIFAR10)
cfg.register_class(_MODULE_, 'cifar100', CIFAR100)

def require_args():

    cfg.add_argument('--%s' % _MODULE_.replace('_', '-'),
                        choices=cfg.get_class_name(_MODULE_),
                        default='cifar10', type=str,
                        help='dataset used for experiment')
    cfg.add_argument('--data-root', default=None, type=str,
                        help='root to data')

def get(root=None, **kwargs):
    root = root if root is not None else cfg.data_root
    dataset = cfg.get_class(_MODULE_, cfg.get(_MODULE_))(root=root, **kwargs)
    logger.debug('Dataset [%s] from directory [%s] is declared and %d samples '
        'are loaded' % (cfg.get(_MODULE_), root, len(dataset)))
    return dataset

def register(name, obj):
    cfg.register_class(_MODULE_, name, obj)