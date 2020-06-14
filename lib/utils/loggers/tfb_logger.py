#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

from tensorboardX import SummaryWriter as TFBWriter

from ...core.config import Config as cfg

class TFBLogger():

    @staticmethod
    def require_args():
        
        cfg.add_argument('--log-tfb', action='store_true',
                        help='use tensorboard to log training process. ')

    def __init__(self, debugging=False, *args, **kwargs):
        self.debugging = debugging
        if not self.debugging and cfg.log_tfb:
            self.writer = TFBWriter(*args, **kwargs)

    def __getattr__(self,attr):
        if self.debugging or not cfg.log_tfb:
            return do_nothing
        return self.writer.__getattribute__(attr)

def do_nothing(*args, **kwargs):
    pass