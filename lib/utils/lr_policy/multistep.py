#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import os

from ..loggers import STDLogger as logger
from .lr_policy import LRPolicy
from ...core.config import Config as cfg

class MultiStep(LRPolicy):
    """Caffe style step decay learning rate policy
    
    Decay learning rate at every `lr-decay-step` steps after the
    first `lr-decay-offset` ones at the rate of `lr-decay-rate`
    """

    @staticmethod
    def require_args():

        cfg.add_argument('--lr-schedule', default=None, type=str,
                        help='learning rate schedule')

    def __init__(self, base_lr, schedule=None):
        self.base_lr = base_lr
        self.schedule = schedule if schedule is not None else cfg.lr_schedule
        if self.schedule is not None and os.path.exists(self.schedule):
            self.from_file = self.schedule
            self.schedule = self._extract_from_file_(self.schedule)
        else:
            assert (isinstance(eval(self.schedule), (list, tuple)), 
                "Invalid learning rate schedule: %s" % self.schedule)
            self.from_file = None
            self.schedule = eval(self.schedule)
        logger.debug('Going to use [multistep] learning policy for optimization '
            'with base learing rate %.5f and schedule as %s'
            % (self.base_lr, self.schedule))

    def _extract_from_file_(self, path):
        lines = filter(lambda x:not x.startswith('#'),
                    open(path, 'r').readlines())
        assert len(lines) > 0, "Invalid schedule file having no content"
        schedule = []
        for line in lines:
            line = line.split('#')[0]
            anchor, target = line.strip().split(':')
            if target.startswith('-'):
                lr = -1
            else:
                lr = float(target)
            schedule.append((int(anchor), lr))
        return schedule

    def _update_(self, steps):
        """update learning rate
        
        Update learning rate according to current steps and schedule file
        
        Arguments:
            steps {int} -- current steps
        
        Returns:
            float -- updated file
        """
        if self.from_file is not None:
            self.schedule = self._extract_from_file_(self.from_file)
        return next((lr for (anchor, lr) in self.schedule if anchor > steps), 
                        self.schedule[-1][1])


