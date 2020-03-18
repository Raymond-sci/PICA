#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import Config as cfg

class PUILoss(nn.Module):

    def __init__(self, lamda=2.0):
        super(PUILoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.lamda = lamda

    def forward(self, x, y):
        """Partition Uncertainty Index
        
        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]
        
        Returns:
            [Tensor] -- [Loss value]
        """
        assert x.shape == y.shape, ('Inputs are required to have same shape')

        # partition uncertainty index
        pui = torch.mm(F.normalize(x.t(), p=2, dim=1), F.normalize(y, p=2, dim=0))
        loss_ce = self.xentropy(pui, torch.arange(pui.size(0)).to(cfg.device))

        # balance regularisation
        p = x.sum(0).view(-1)
        p /= p.sum()
        loss_ne = math.log(p.size(0)) + (p * p.log()).sum()

        return loss_ce + self.lamda * loss_ne