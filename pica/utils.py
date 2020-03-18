#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch._six import int_classes as _int_classes
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset_
from torch.utils.data import Sampler
from torch.utils.data.sampler import RandomSampler as _RandomSampler_

class ConcatDataset(_ConcatDataset_):
    """Dataset as a concatenation of multiple datasets
    
    Wrapper class of Pytorch ConcatDataset to set the labels as an attribute
    
    """

    def __init__(self, *args, **kwargs):
        super(ConcatDataset, self).__init__(*args, **kwargs)
        self.targets = reduce(lambda x,y:x+y.targets, self.datasets, [])

class RepeatSampler(Sampler):
    """repeats samples and arranges in [1, 2, ..., N, 1, 2, ...]
    
    """

    def __init__(self, sampler, batch_size, drop_last=False, nrepeat=1):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.nrepeat = nrepeat

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch * self.nrepeat
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch * self.nrepeat

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class RandomSampler(_RandomSampler_):
    """samples elements randomly, order is fixed once instanced
    
    """

    def __init__(self, *args, **kwargs):
        super(RandomSampler, self).__init__(*args, **kwargs)
        n = len(self.data_source)
        if self.replacement:
            self.indexes = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist()
        else:
            self.indexes = torch.randperm(n).tolist()

    def __iter__(self):
       return iter(self.indexes)

def get_reduced_transform(resize, size, means, stds):
    """Reduced transforms applied to original inputs

    Arguments:
        resize {int} -- resize before cropping
        size {int} -- expected size
        means {list} -- pixel-wise means
        stds {list} -- pixel-wise stds
    """
    tfs = []
    tfs.append(transforms.Resize(size=resize))
    tfs.append(transforms.RandomCrop(size))
    tfs.append(transforms.ToTensor())
    tfs.append(transforms.Normalize(means, stds))
    return transforms.Compose(tfs)