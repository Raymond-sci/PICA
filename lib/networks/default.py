#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import math
import re
from easydict import EasyDict as ezdict

import torch
import torch.nn as nn

from ..utils.loggers import STDLogger as logger

class DefaultModel(nn.Module):

    def _initialise_weights_(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is None:
                    continue
                m.bias.data.zero_()

    def _make_sobel_(self):
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        )
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        )
        sobel_filter.bias.data.zero_()
        layers = nn.Sequential(grayscale, sobel_filter)
        for p in layers.parameters():
            p.requires_grad = False
        return layers

    def load_state_dict(self, state_dict):
        # ignore data parallel
        state_dict = {key.replace('module.', ''):val for key,val in state_dict.iteritems()}
        # get valid params
        target = self.state_dict()
        cnt = 0
        for key, val in state_dict.iteritems():
            if key not in target:
                logger.warn('Ignoring pretrained weights [%s]: not found' % key)
            elif val.shape != target[key].shape:
                logger.warn('Ignoring pretrained weights [%s]: invalid shape '
                    '[%s] v.s. [%s]' % (key, val.shape, target[key].shape))
            else:
                target[key] = val
                cnt += 1
        logger.debug('Totally loaded [%d] parameters' % cnt)
        super(DefaultModel, self).load_state_dict(target)


    def scoped_parameters(self, scope):
        names = [name for name, _ in self.named_parameters()]
        names = filter(lambda x:re.match(scope, x) is not None, names)
        return {name:param for name, param in self.named_parameters() if name in names}

    def freeze_scope(self, scope):
        params = self.scoped_parameters(scope)
        for name, param in params.iteritems():
            param.requires_grad = False

    def frozen_parameters(self):
        return {name:param for name, param in self.named_parameters() if not param.requires_grad}

    def trainable_parameters(self):
        return {name:param for name, param in self.named_parameters() if param.requires_grad}

    def data_parallel(self, device_ids):
        return False

    def run(self, x, target=None):
        if target is None:
            return self.forward(x)
        count = 1
        for m in self.modules():
            if not (hasattr(m, 'run') or isinstance(m, nn.Sequential)):
                x = m(x)
            if isinstance(m, Tag):
                if count == target:
                    break
                count += 1
        return x

class Tag(nn.Module):

    def forward(self, x):
        return x

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)

    