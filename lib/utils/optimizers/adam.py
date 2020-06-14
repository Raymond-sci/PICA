#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import torch

from ..loggers import STDLogger as logger
from ...core.config import Config as cfg

class Adam(torch.optim.Adam):

    @staticmethod
    def require_args():
        cfg.add_argument('--optim-beta', default="(0.9, 0.999)", type=eval,
                help='coefficients used for computing running'
                     ' averages of gradient')
        cfg.add_argument('--optim-eps', default=1e-8, type=float,
                help='term added to the denominator to improve numerical stability')
        cfg.add_argument('--optim-amsgrad', action='store_true',
                help='whether to use the AMSGrad variant of this algorithm')

    def __init__(self, weight_decay, params=None,
            betas=None, eps=None, amsgrad=None):
        betas = betas if betas is not None else cfg.optim_beta
        eps = eps if eps is not None else cfg.optim_eps
        amsgrad = amsgrad if amsgrad is not None else cfg.optim_amsgrad
        
        super(Adam, self).__init__(params, weight_decay=weight_decay,
            betas=betas, eps=eps, amsgrad=amsgrad, lr=0.1)

        logger.debug('Going to use [Adam] optimizer for training '
            'with betas %s, eps %f, weight decay %f %s amsgrad' % 
            (betas, eps, weight_decay,
            ('with' if amsgrad else 'without')))