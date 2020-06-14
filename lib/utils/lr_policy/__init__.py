#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

from .fixed import Fixed
from .multistep import MultiStep
from .step import Step

from ...core.config import Config as cfg

_MODULE_ = 'lr_policy'
cfg.register_module(_MODULE_, __name__)
cfg.register_class(_MODULE_, 'fixed', Fixed)
cfg.register_class(_MODULE_, 'multistep', MultiStep)
cfg.register_class(_MODULE_, 'step', Step)

def require_args():

    cfg.add_argument('--%s' % _MODULE_.replace('_', '-'),
                        choices=cfg.get_class_name(_MODULE_),
                        default='fixed', type=str,
                        help='learning rate policy')
    cfg.add_argument('--base-lr', default=1e-1, type=float,
                        help='base learning rate')

def get(lr=None, **kwargs):
    lr = lr if lr is not None else cfg.base_lr
    return cfg.get_class(_MODULE_, cfg.get(_MODULE_))(lr, **kwargs)

def register(name, obj):
    cfg.register_class(_MODULE_, name, obj)