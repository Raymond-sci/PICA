#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import torch

from ..loggers import STDLogger as logger
from ...core.config import Config as cfg

class SGD(torch.optim.SGD):

    @staticmethod
    def require_args():
            
        cfg.add_argument('--optim-momentum', default=0, type=float,
                            help='momentum factor')
        cfg.add_argument('--optim-dampening', default=0, type=float,
                            help='dampening for momentum')
        cfg.add_argument('--optim-nesterov', action='store_true',
                            help='enables Nesterov momentum')

    def __init__(self, weight_decay, params=None,
                momentum=None, nesterov=None, dampening=None):
        momentum = momentum if momentum is not None else cfg.optim_momentum
        nesterov = nesterov if nesterov is not None else cfg.optim_nesterov
        dampening = dampening if dampening is not None else cfg.optim_dampening
        
        super(SGD, self).__init__(params, weight_decay=weight_decay,
            momentum=momentum, nesterov=nesterov, dampening=dampening, lr=0.1)

        logger.debug('Going to use [SGD] optimizer for training '
            'with momentum %f, dampening %f, weight decay %f %s nesterov' % 
            (momentum, dampening, weight_decay, 
            ('with' if nesterov else 'without')))