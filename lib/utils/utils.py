#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import os
import shutil
from datetime import timedelta

import torch

from ..core.config import Config as cfg
from .loggers import STDLogger as logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """Progress Meter"""
    def __init__(self, *meters, **iters):
        self.iters = {key:self._get_iter_fmtstr(key, val) for key, val in iters.iteritems()}
        assert reduce(lambda x,y: x and y, map(lambda x:isinstance(x, 
            AverageMeter), meters), True), "All meters should be in type of AverageMeter"
        self.meters = meters

    def show(self, *iters):
        assert len(self.iters) == len(iters), "Number of tracked variables is invalid"
        entries = map(lambda x:self.iters[x].format(iters[x]), self.iters.keys())
        entries += [str(meter) for meter in self.meters]
        return ' '.join(entries)

    def _get_iter_fmtstr(self, name, niters):
        num_digits = len(str(int(niters // 1)))
        fmt = ('%s: [{:' + str(num_digits) + 'd}/%d]') % (name, niters)
        return fmt

class TimeProgressMeter(ProgressMeter):

    KEY = 'Batch'

    """Time progress meter"""
    def __init__(self, *meters, **iters):
        # get batch num and time meter
        self.batch_num = iters[TimeProgressMeter.KEY]
        self.batch_time = meters[0]

        super(TimeProgressMeter, self).__init__(*meters, **iters)

    def show(self, **iters):
        # format iters
        assert len(self.iters) == len(iters), "Number of tracked variables is invalid"
        entries = map(lambda x:self.iters[x].format(iters[x]), self.iters.keys())
        # format time
        elps_time, est_time = self._estimate(iters[TimeProgressMeter.KEY], self.batch_num, self.batch_time.sum)
        entries.append('Progress: [{}/{}]'.format(elps_time, est_time))
        # format meters
        entries += [str(meter) for meter in self.meters]
        return ' '.join(entries)

    def _estimate(self, elapsed_iters, tot_iters, elapsed_time):
        elapsed_iters += 1
        estimated_time = 1. * tot_iters / elapsed_iters * elapsed_time
        elapsed_time = timedelta(seconds=elapsed_time)
        estimated_time = timedelta(seconds=estimated_time)
        return tuple(map(lambda x:str(x).split('.')[0], [elapsed_time, estimated_time]))

def save_checkpoint(state, is_best, filename='latest.ckpt', root=None):
    root = root if root is not None else cfg.ckpt_dir
    latest, best = map(lambda x:os.path.join(root, x), [filename, 'best.ckpt'])
    torch.save(state, latest)
    if is_best:
        shutil.copyfile(latest, best)

def traverse(net, loader, transform=None, target=None, tencrops=False, device='cpu'):

    if transform is not None:
        bak_transform = loader.dataset.transform
        loader.dataset.transform = transform

    outputs = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            logger.progress('processing %d/%d batch' % (batch_idx, len(loader)))

            inputs = batch[0].to(device, non_blocking=True)
            if tencrops:
                bs, ncrops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)

            out = net.run(inputs, target=target)
            if tencrops:
                out = torch.squeeze(out.view(bs, ncrops, -1).mean(1))

            if outputs is None:
                size = list(out.shape)
                size[0] = len(loader.dataset)
                outputs = torch.zeros(*size).to(device)
            start = batch_idx * loader.batch_size
            end = start + loader.batch_size
            end = min(end, len(loader.dataset))
            outputs[start:end] = out

    if transform is not None:
        loader.dataset.transform = bak_transform

    return outputs

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res