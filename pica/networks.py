#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import torch.nn as nn

from lib import Config as cfg
from lib.networks import DefaultModel, Flatten, register
from lib.utils.loggers import STDLogger as logger

__all__ = ['ResNet34']

def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None,
               track_running_stats=None):
    super(BasicBlock, self).__init__()

    assert (track_running_stats is not None)

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class ResNet34(DefaultModel):

    @staticmethod
    def require_args():

        cfg.add_argument('--net-heads', nargs='*', type=int,
                        help='net heads')
        cfg.add_argument('--net-avgpool-size', default=3, type=int, choices=[3, 5, 7],
                        help='Avgpool kernel size determined by inputs size')

    def __init__(self, cin, cout, sobel, net_heads=None, pool_size=None):
        net_heads = net_heads if net_heads is not None else cfg.net_heads
        pool_size = pool_size if pool_size is not None else cfg.net_avgpool_size
        logger.debug('Backbone will be created wit the following heads: %s' % net_heads)
        # do init
        super(ResNet34, self).__init__()
        # build sobel
        self.sobel = self._make_sobel_() if sobel else None
        # build trunk net
        self.inplanes = 64
        self.layer1 = nn.Sequential(nn.Conv2d(2 if sobel else cin, 64, 
                    kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.layer2 = self._make_layer(BasicBlock, 64, 3)
        self.layer3 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer5 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(pool_size, stride=1), Flatten())
        heads = [nn.Sequential(nn.Linear(512 * BasicBlock.expansion, head),
            nn.Softmax(dim=1)) for head in net_heads]
        self.heads = nn.ModuleList(heads)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, 
                        track_running_stats=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                                    track_running_stats=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                    track_running_stats=True))

        return nn.Sequential(*layers)

    def run(self, x, target=None):
        """Function for getting the outputs of intermediate layers
        """
        if target is None or target > 5:
            raise NotImplementedError('Target is expected to be smaller than 6')
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        for layer in layers[:target]:
            x = layer(x)
        return x

    def forward(self, x):
        if self.sobel is not None:
            x = self.sobel(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return map(lambda head:head(x), self.heads)

register('resnet34', ResNet34)
