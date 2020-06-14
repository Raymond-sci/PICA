#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA


class LRPolicy:

    def update(self, steps, optimizer=None):
        # get updated learning rate
        lr = self._update_(steps)
        if optimizer is not None:
            self.apply(lr, optimizer)
        return lr

    def apply(self, lr, optimizer):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.)
