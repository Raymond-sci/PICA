#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

from .sgd import SGD
from .rmsprop import RMSprop
from .adam import Adam

from ...core.config import Config as cfg

_MODULE_ = 'optimizer'
cfg.register_module(_MODULE_, __name__)
cfg.register_class(_MODULE_, 'sgd', SGD)
cfg.register_class(_MODULE_, 'rmsprop', RMSprop)
cfg.register_class(_MODULE_, 'adam', Adam)

def require_args():

    cfg.add_argument('--%s' % _MODULE_.replace('_', '-'), 
                        choices=cfg.get_class_name(_MODULE_),
                        default='sgd', type=str,
                        help='algorithm used for model optimization')
    cfg.add_argument('--optim-weight-decay', default=5e-4, type=float,
                        help='weight decay (L2 penalty)')

def get(weight_decay=None, **kwargs):
    weight_decay = weight_decay if weight_decay is not None else cfg.optim_weight_decay
    return cfg.get_class(_MODULE_, cfg.get(_MODULE_))(weight_decay, **kwargs)

def register(name, obj):
    cfg.register_class(_MODULE_, name, obj)
