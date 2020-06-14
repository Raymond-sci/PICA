#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA


from .default import DefaultModel, Tag, Flatten

from ..core.config import Config as cfg
from ..utils.loggers import STDLogger as logger

_MODULE_ = 'network'
cfg.register_module(_MODULE_, __name__)

def require_args():

    cfg.add_argument('--%s' % _MODULE_.replace('_', '-'),
                        choices=cfg.get_class_name(_MODULE_),
                        default='alexnet12', type=str,
                        help='backbone used for experiment')
    cfg.add_argument('--net-cin', default=3, type=int,
                        help='number of input channels')
    cfg.add_argument('--net-cout', default=1000, type=int,
                        help='number of output channels')
    cfg.add_argument('--net-sobel', default=True, type=eval,
                        help='whether to use sobel preprocessing in the network')
    cfg.add_argument('--net-freeze', default=[], type=str, nargs='*',
                        help='name of parameters to be freeze')

def get(cin=None, cout=None, sobel=None, frozen=None, **kwargs):
    cin = cin if cin is not None else cfg.net_cin
    cout = cout if cout is not None else cfg.net_cout
    sobel = sobel if sobel is not None else cfg.net_sobel
    assert (not sobel or cin == 3), ("Sobel descriptor requires cin equal "
        "to 3 but got [%d]" % cin)
    logger.debug('Backbone [%s] is declared with cin [%d] and cout [%d] [%s] sobel' 
        % (cfg.get(_MODULE_), cin, cout, 'with' if sobel else 'without'))
    frozen = frozen if frozen is not None else cfg.net_freeze
    network = cfg.get_class(_MODULE_, cfg.get(_MODULE_))(cin, cout, sobel, **kwargs)
    for scope in cfg.net_freeze:
        network.freeze_scope(scope)
    logger.debug('Number of trainable parameters is [%d]' % len(network.trainable_parameters().keys()))
    frzn_params = network.frozen_parameters().keys()
    logger.debug('Number of frozen parameters is [%d]' % len(frzn_params))
    if len(frzn_params) > 0:
        logger.debug('Name of frozen parameters are: %s' % frzn_params)
    return network

def register(name, obj):
    cfg.register_class(_MODULE_, name, obj)