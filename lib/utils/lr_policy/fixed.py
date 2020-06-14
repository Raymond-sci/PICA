#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

from ..loggers import STDLogger as logger
from .lr_policy import LRPolicy

class Fixed(LRPolicy):

    def __init__(self, base_lr):
        self.base_lr = base_lr
        logger.debug('Going to use [fixed] learning policy for optimization'
            ' with base learning rate [%.5f]' % base_lr)

    def _update_(self, epoch):
        return self.base_lr

