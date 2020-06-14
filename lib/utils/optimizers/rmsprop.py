#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import torch

from ..loggers import STDLogger as logger
from ...core.config import Config as cfg

class RMSprop(torch.optim.RMSprop):

    @staticmethod
    def require_args():
            
        cfg.add_argument('--optim-alpha', default=0.99, type=float,
                help='smoothing constant')
        cfg.add_argument('--optim-eps', default=1e-8, type=float,
                help=('term added to the denominator to improve'
                      ' numerical stability'))
        cfg.add_argument('--optim-momentum', default=0, type=float,
                help='momentum factor')
        cfg.add_argument('--optim-centered', action='store_true',
                help='whether to compute the centered RMSProp')

    def __init__(self, weight_decay, params=None,
            alpha=None, eps=None, momentum=None, centered=None):
        alpha = alpha if alpha is not None else cfg.optim_alpha
        eps = eps if eps is not None else cfg.optim_eps
        centered = centered if centered is not None else cfg.optim_centered
        momentum = momentum if momentum is not None else cfg.optim_momentum
        
        super(RMSprop, self).__init__(params, weight_decay=weight_decay,
            alpha=alpha, eps=eps, centered=centered, momentum=momentum, lr=0.1)
        
        logger.debug('Going to use [RMSprop] optimizer for training with alpha %.2f, '
            'eps %f, weight decay %f, momentum %f %s centered' % (alpha,
                eps, weight_decay, momentum,
                ('with' if centered else 'without')))

