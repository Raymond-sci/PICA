#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-26 20:19:00
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

from ..loggers import STDLogger as logger
from .lr_policy import LRPolicy
from ...core.config import Config as cfg

class Step(LRPolicy):
    """Caffe style step decay learning rate policy
    
    Decay learning rate at every `lr-decay-step` steps after the
    first `lr-decay-offset` ones at the rate of `lr-decay-rate`
    """

    @staticmethod
    def require_args():

        cfg.add_argument('--lr-decay-offset', default=0, type=int,
                        help='learning rate will start to decay at which step')

        cfg.add_argument('--lr-decay-step', default=0, type=int,
                        help='learning rate will decay at every n round')

        cfg.add_argument('--lr-decay-rate', default=0.1, type=float,
                        help='learning rate will decay at what rate')

    def __init__(self, base_lr, offset=None, step=None, rate=None):
        self.base_lr = base_lr
        self.offset = offset if offset is not None else cfg.lr_decay_offset
        self.step = step if step is not None else cfg.lr_decay_step
        self.rate = rate if rate is not None else cfg.lr_decay_rate
        logger.debug('Going to use [step] learning policy for optimization with '
            'base learning rate %.5f, offset %d, step %d and decay rate %f' %
            (self.base_lr, self.offset, self.step, self.rate))

    def _update_(self, steps):
        """decay learning rate according to current step
        
        Decay learning rate at a fixed ratio
        
        Arguments:
            steps {int} -- current steps
        
        Returns:
            int -- updated learning rate
        """

        if steps < self.offset:
            return self.base_lr
        
        return self.base_lr * (self.rate ** ((steps - self.offset) // self.step))
