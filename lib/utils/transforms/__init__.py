#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

from .default import DefaultTransform

from ...core.config import Config as cfg

_MODULE_ = 'transform'
cfg.register_module(_MODULE_, __name__)
cfg.register_class(_MODULE_, 'default', DefaultTransform)

def require_args():

    cfg.add_argument('--%s' % _MODULE_.replace('_', '-'),
                        choices=cfg.get_class_name(_MODULE_),
                        default='default', type=str,
                        help='transforms for input samples')

def get(train, **kwargs):
    return cfg.get_class(_MODULE_, cfg.get(_MODULE_))(train, **kwargs)

def register(name, obj):
    cfg.register_class(_MODULE_, name, obj)